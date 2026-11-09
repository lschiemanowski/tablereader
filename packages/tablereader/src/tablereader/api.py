"""The HTTP service.

    POST   /v1/jobs                      upload a PDF (multipart `file`, optional form `pages`, header
                                         `Idempotency-Key`); 202 with the job
    GET    /v1/jobs                      all jobs, oldest first
    GET    /v1/jobs/{id}                 state (queued, running, succeeded, failed, canceled) and progress
    GET    /v1/jobs/{id}/tables          the tables, one object per table (table JSON format 0.3)
    GET    /v1/jobs/{id}/document        the full result: regions, boxes, transcriptions, issues
    GET    /v1/jobs/{id}/review          an HTML page with each crop beside its tables
    GET    /v1/jobs/{id}/crops/{name}    a crop image
    POST   /v1/jobs/{id}/cancel          cancel a queued or running job
    DELETE /v1/jobs/{id}                 delete a finished job and its files
    GET    /healthz                      service and model status

Jobs run one at a time on the GPU (`jobs.Worker`). The service has no authentication: run it on a trusted
network or behind a proxy that adds it.
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from tablereader.jobs import FINISHED, Job, JobState, JobStore, Reader, Worker

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def create_app(store: JobStore, open_reader: Callable[[], Reader], *, start_worker: bool = True) -> FastAPI:
    worker = Worker(store, open_reader)

    @asynccontextmanager  # pyright: ignore[reportDeprecated]
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(title="tablereader", version="0.1.0", lifespan=lifespan)
    app.state.worker = worker

    def find(job_id: str) -> Job:
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "no such job") from None

    def finished(job_id: str) -> Path:
        job = find(job_id)
        if job.state is not JobState.SUCCEEDED:
            raise HTTPException(409, f"job is {job.state.value}, not succeeded")
        return store.result_dir(job_id)

    @app.post("/v1/jobs", status_code=202)
    async def submit(
        file: Annotated[UploadFile, File()],
        pages: Annotated[str | None, Form()] = None,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"PDFs up to {MAX_UPLOAD_BYTES // 2**20} MB are accepted")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(415, "the upload is not a PDF")
        job = store.create(data, filename=file.filename or "upload.pdf", pages=pages or None,
                           idempotency_key=idempotency_key)  # fmt: skip
        worker.notify()
        return job.snapshot()

    @app.get("/v1/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.snapshot() for j in store.jobs()]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return find(job_id).snapshot()

    @app.get("/v1/jobs/{job_id}/tables")
    def tables(job_id: str) -> list[Any]:
        lines = (finished(job_id) / "tables.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines if line]

    @app.get("/v1/jobs/{job_id}/document")
    def document(job_id: str) -> Any:
        return json.loads((finished(job_id) / "document.json").read_text())

    @app.get("/v1/jobs/{job_id}/review", response_class=HTMLResponse)
    def review(job_id: str) -> str:
        page = (finished(job_id) / "review.html").read_text()
        return page.replace('src="crops/', f'src="/v1/jobs/{job_id}/crops/')

    @app.get("/v1/jobs/{job_id}/crops/{name}")
    def crop(job_id: str, name: str) -> FileResponse:
        crops = finished(job_id) / "crops"
        path = crops / name
        if path.parent != crops or path.suffix != ".png" or not path.is_file():
            raise HTTPException(404, "no such crop")
        return FileResponse(path, media_type="image/png")

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        find(job_id)
        return store.cancel(job_id).snapshot()

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete(job_id: str) -> None:
        if find(job_id).state not in FINISHED:
            raise HTTPException(409, "cancel the job before deleting it")
        store.delete(job_id)

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        queued = sum(j.state is JobState.QUEUED for j in store.jobs())
        return {"status": "ok", "model_loaded": worker.ready, "queued": queued, "last_error": worker.last_error}

    return app
