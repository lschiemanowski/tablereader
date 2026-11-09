"""Jobs for the HTTP service: one directory per job, and one worker that owns the GPU (user decision, 2026-11-09).

    <root>/<job_id>/input.pdf
    <root>/<job_id>/state.json    written atomically (temporary file, then rename)
    <root>/<job_id>/result/       what `write_outputs` writes, once the job has succeeded

A job is acknowledged only after its PDF and state are on disk. The worker takes queued jobs oldest first; on
start it puts jobs left `running` by a stopped service back in the queue, so an accepted job is not lost. An
idempotency key maps a repeated submission to the job it created first.
"""

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from tablereader.output import write_outputs
from tablereader.reader import Canceled, DocumentResult, Progress


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


FINISHED = (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED)


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    job_id: str
    state: JobState
    filename: str
    pdf_sha256: str
    pages: str | None
    created: float
    updated: float
    idempotency_key: str | None = None
    stage: str | None = None
    done: int = 0
    total: int = 0
    tables: int | None = None
    error: str | None = None
    cancel_requested: bool = False
    release: str | None = None
    extra: dict[str, Any] = field(default_factory=dict[str, Any])

    def snapshot(self) -> dict[str, Any]:
        """The public view of the job."""
        out = asdict(self)
        out["state"] = self.state.value
        for key in ("idempotency_key", "extra"):
            out.pop(key)
        return out


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _dir(self, job_id: str) -> Path:
        if not job_id or not all(c in "0123456789abcdef" for c in job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def _write(self, job: Job) -> None:
        path = self._dir(job.job_id) / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**asdict(job), "state": job.state.value}))
        os.replace(temporary, path)

    def get(self, job_id: str) -> Job:
        path = self._dir(job_id) / "state.json"
        if not path.exists():
            raise KeyError(job_id)
        data: dict[str, Any] = json.loads(path.read_text())
        data["state"] = JobState(data["state"])
        return Job(**data)

    def jobs(self) -> list[Job]:
        found: list[Job] = []
        for path in self.root.glob("*/state.json"):
            try:
                found.append(self.get(path.parent.name))
            except (KeyError, ValueError):
                continue
        return sorted(found, key=lambda j: j.created)

    def pdf(self, job_id: str) -> Path:
        return self._dir(job_id) / "input.pdf"

    def result_dir(self, job_id: str) -> Path:
        return self._dir(job_id) / "result"

    def create(self, pdf: bytes, *, filename: str, pages: str | None, idempotency_key: str | None = None) -> Job:
        with self._lock:
            if idempotency_key is not None:
                for job in self.jobs():
                    if job.idempotency_key == idempotency_key:
                        return job
            job_id = uuid.uuid4().hex
            directory = self._dir(job_id)
            directory.mkdir()
            (directory / "input.pdf").write_bytes(pdf)
            now = time.time()
            digest = hashlib.sha256(pdf).hexdigest()
            job = Job(job_id=job_id, state=JobState.QUEUED, filename=filename, pdf_sha256=digest, pages=pages,
                      created=now, updated=now, idempotency_key=idempotency_key)  # fmt: skip
            self._write(job)
            return job

    def update(self, job_id: str, **changes: Any) -> Job:
        with self._lock:
            job = replace(self.get(job_id), **changes, updated=time.time())
            self._write(job)
            return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.get(job_id)
            if job.state is JobState.QUEUED:
                job = replace(job, state=JobState.CANCELED, updated=time.time())
            elif job.state is JobState.RUNNING:
                job = replace(job, cancel_requested=True, updated=time.time())
            self._write(job)
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            if self.get(job_id).state is JobState.RUNNING:
                raise ValueError("a running job cannot be deleted; cancel it first")
            shutil.rmtree(self._dir(job_id))

    def requeue_interrupted(self) -> int:
        """Jobs left running by a stopped service go back in the queue."""
        interrupted = [j for j in self.jobs() if j.state is JobState.RUNNING]
        for job in interrupted:
            self.update(job.job_id, state=JobState.QUEUED, stage=None, done=0, total=0)
        return len(interrupted)

    def next_queued(self) -> Job | None:
        return next((j for j in self.jobs() if j.state is JobState.QUEUED), None)


class Reader(Protocol):
    """What the worker needs from a `TableReader`."""

    @property
    def release_id(self) -> str: ...

    def read(self, path: Path, pages: str | None = None, progress: Progress | None = None) -> DocumentResult: ...


class Worker:
    """Runs queued jobs one at a time in a background thread. The reader (and its model) is opened lazily on the
    first job and kept until `stop`."""

    def __init__(self, store: JobStore, open_reader: Callable[[], Reader], *, poll: float = 1.0) -> None:
        self.store = store
        self._open_reader = open_reader
        self._reader: Reader | None = None
        self._poll = poll
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tablereader-worker", daemon=True)
        self.last_error: str | None = None

    def start(self) -> None:
        self.store.requeue_interrupted()
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
        close = getattr(self._reader, "close", None)
        if callable(close):
            close()

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(self._poll)
                self._wake.clear()
                continue
            self.run_job(job)

    def run_job(self, job: Job) -> None:
        store = self.store
        job = store.update(job.job_id, state=JobState.RUNNING, stage="starting")

        def progress(stage: str, done: int, total: int) -> None:
            current = store.update(job.job_id, stage=stage, done=done, total=total)
            if current.cancel_requested or self._stop.is_set():
                raise Canceled

        try:
            if self._reader is None:
                self._reader = self._open_reader()
            result = self._reader.read(store.pdf(job.job_id), job.pages, progress)
            write_outputs(result, store.result_dir(job.job_id))
            store.update(job.job_id, state=JobState.SUCCEEDED, stage=None, tables=len(result.tables),
                         release=self._reader.release_id)  # fmt: skip
        except Canceled:
            if self._stop.is_set() and not store.get(job.job_id).cancel_requested:
                store.update(job.job_id, state=JobState.QUEUED, stage=None)  # shutting down: run it again later
            else:
                store.update(job.job_id, state=JobState.CANCELED, stage=None)
        except ValueError as error:  # bad input, e.g. a page selection outside the document
            store.update(job.job_id, state=JobState.FAILED, stage=None, error=str(error))
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            store.update(job.job_id, state=JobState.FAILED, stage=None, error=self.last_error[:2000])
