"""Page selection, release manifests, outputs, the job store, the worker and the HTTP API, with a fake reader."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.extract import interpret
from extraction.results import OcrResult
from tablereader.api import create_app
from tablereader.jobs import Job, JobState, JobStore, Worker
from tablereader.output import review_page, write_outputs
from tablereader.reader import Canceled, DocumentResult, Progress, RegionRead, parse_pages
from tablereader.release import CONTRACT_REVISION, MANIFEST, load_release, write_release

TABLE = '# Receipts<sup>1</sup>\n<table><tbody><tr><td data-level="0">Taxes</td><td>100</td></tr></tbody>' \
    '<tfoot><tr><td colspan="2">1/ Provisional. <script>alert(1)</script></td></tr></tfoot></table>'  # fmt: skip
PDF = b"%PDF-1.7\n%fake\n"


def document(path: Path = Path("report.pdf")) -> DocumentResult:
    box = Box(left=10, top=20, right=200, bottom=120)
    prepared = PreparedRegion(
        source=DocumentSource(path=path, sha256="abc"), region=TableRegion(region_id="p0001-t0", page=1, box=box),
        page_size=(600, 800), page_pixels=(2500, 3333), detected_px=box, crop_px=box, crop_box=box,
        image=Image.new("RGB", (40, 20), "white"), settings=PreparationSettings(), renderer="test",
    )  # fmt: skip
    result = interpret("p0001-t0", OcrResult(text=TABLE, model="m"))
    return DocumentResult(
        source=prepared.source, pages=(1,), release_id="test-release", regions=(RegionRead(prepared, result),)
    )


@pytest.mark.parametrize(
    ("spec", "pages"), [(None, [1, 2, 3, 4, 5]), ("2", [2]), ("4-5, 1", [1, 4, 5]), ("2-3,3", [2, 3])]
)
def test_parse_pages(spec: str | None, pages: list[int]) -> None:
    assert parse_pages(spec, 5) == pages


@pytest.mark.parametrize("spec", ["0", "6", "3-2", "x", "1-"])
def test_bad_page_selections(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_pages(spec, 5)


def test_release_manifest_round_trip(tmp_path: Path) -> None:
    written = write_release(
        tmp_path, release_id="r", preparation=PreparationSettings(), base_model="b", training={"run": "r1"}
    )
    loaded = load_release(tmp_path)
    assert loaded == written
    assert loaded.contract_revision == CONTRACT_REVISION
    data = json.loads((tmp_path / MANIFEST).read_text())
    (tmp_path / MANIFEST).write_text(json.dumps(data | {"contract_revision": 99}))
    with pytest.raises(ValueError, match="contract revision"):
        load_release(tmp_path)


def test_outputs(tmp_path: Path) -> None:
    write_outputs(document(), tmp_path)
    (line,) = [json.loads(x) for x in (tmp_path / "tables.jsonl").read_text().splitlines()]
    assert (line["table_id"], line["page"], line["outcome"]) == ("p0001-t0/t0", 1, "ready")
    assert line["table"]["title"]["text"] == "Receipts"
    doc = json.loads((tmp_path / "document.json").read_text())
    assert doc["regions"][0]["box"] == [10, 20, 200, 120]
    assert (tmp_path / "crops/p0001-t0.png").exists()


def test_review_page_escapes_model_output() -> None:
    page = review_page(document())
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "Receipts<sup>1</sup>" in page


class FakeReader:
    release_id = "test-release"

    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self.fail = fail
        self.block = block
        self.calls: list[tuple[Path, str | None]] = []

    def read(self, path: Path, pages: str | None = None, progress: Progress | None = None) -> DocumentResult:
        self.calls.append((path, pages))
        assert progress is not None
        progress("detect", 0, 1)
        if self.block:
            for _ in range(200):
                progress("recognize", 0, 1)  # raises Canceled once a cancel is requested
                time.sleep(0.01)
        if self.fail:
            raise RuntimeError("GPU on fire")
        progress("recognize", 1, 1)
        return document(path)


def test_store_create_is_idempotent_and_survives_restart(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    first = store.create(PDF, filename="a.pdf", pages="1-2", idempotency_key="k")
    assert store.create(PDF, filename="a.pdf", pages="1-2", idempotency_key="k") == first
    store.update(first.job_id, state=JobState.RUNNING)
    reopened = JobStore(tmp_path)
    assert reopened.requeue_interrupted() == 1
    assert reopened.get(first.job_id).state is JobState.QUEUED
    assert reopened.pdf(first.job_id).read_bytes() == PDF


def test_store_rejects_bad_ids(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        JobStore(tmp_path).get("../etc")


def run_one(store: JobStore, reader: FakeReader) -> Job:
    worker = Worker(store, lambda: reader)
    job = store.next_queued()
    assert job is not None
    worker.run_job(job)
    return store.get(job.job_id)


def test_worker_runs_a_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create(PDF, filename="a.pdf", pages="1")
    reader = FakeReader()
    done = run_one(store, reader)
    assert (done.state, done.tables, done.release) == (JobState.SUCCEEDED, 1, "test-release")
    assert reader.calls == [(store.pdf(job.job_id), "1")]
    assert (store.result_dir(job.job_id) / "tables.jsonl").exists()


def test_worker_records_failures(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.create(PDF, filename="a.pdf", pages=None)
    done = run_one(store, FakeReader(fail=True))
    assert done.state is JobState.FAILED
    assert done.error == "RuntimeError: GPU on fire"


def test_cancel_queued_and_running(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    queued = store.create(PDF, filename="a.pdf", pages=None)
    assert store.cancel(queued.job_id).state is JobState.CANCELED
    running = store.create(PDF, filename="b.pdf", pages=None)
    store.update(running.job_id, state=JobState.RUNNING)
    assert store.cancel(running.job_id).cancel_requested

    def progress(stage: str, done: int, total: int) -> None:
        if store.get(running.job_id).cancel_requested:
            raise Canceled

    with pytest.raises(Canceled):
        progress("recognize", 0, 1)


def test_http_api(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    reader = FakeReader()
    app = create_app(store, lambda: reader)
    with TestClient(app) as client:
        assert client.post("/v1/jobs", files={"file": ("x.txt", b"hello")}).status_code == 415
        response = client.post("/v1/jobs", files={"file": ("r.pdf", PDF)}, data={"pages": "1"},
                               headers={"Idempotency-Key": "abc"})  # fmt: skip
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        again = client.post("/v1/jobs", files={"file": ("r.pdf", PDF)}, headers={"Idempotency-Key": "abc"})
        assert again.json()["job_id"] == job_id
        state = "queued"
        for _ in range(200):
            state = client.get(f"/v1/jobs/{job_id}").json()["state"]
            if state == "succeeded":
                break
            time.sleep(0.02)
        assert state == "succeeded"
        (table,) = client.get(f"/v1/jobs/{job_id}/tables").json()
        assert table["table"]["title"]["text"] == "Receipts"
        assert client.get(f"/v1/jobs/{job_id}/document").json()["release"] == "test-release"
        review = client.get(f"/v1/jobs/{job_id}/review").text
        assert f"/v1/jobs/{job_id}/crops/p0001-t0.png" in review
        assert client.get(f"/v1/jobs/{job_id}/crops/p0001-t0.png").headers["content-type"] == "image/png"
        assert client.get(f"/v1/jobs/{job_id}/crops/..%2Fstate.json").status_code == 404
        assert client.get("/v1/jobs/0123/tables").status_code == 404
        assert client.get("/healthz").json()["model_loaded"] is True
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_results_wait_for_success(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    app = create_app(store, FakeReader, start_worker=False)
    with TestClient(app) as client:
        job_id = client.post("/v1/jobs", files={"file": ("r.pdf", PDF)}).json()["job_id"]
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"
        assert client.get(f"/v1/jobs/{job_id}/tables").status_code == 409
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 409
        assert client.post(f"/v1/jobs/{job_id}/cancel").json()["state"] == "canceled"
