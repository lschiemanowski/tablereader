"""Read the tables in a PDF: detect, prepare, recognize, interpret.

    from tablereader import read_pdf
    result = read_pdf("report.pdf", pages="13-20")

`TableReader` keeps one vLLM server for the model release running for its lifetime, so several documents can be
read without reloading the model; `read_pdf` opens one for a single document. Table detection (Docling Heron)
runs in a child process per document so its GPU memory is released before recognition; vLLM is started with
room left for it. Pages are rendered with pdfium, each table is cropped over its title and footnotes with the
release's preparation settings, transcribed by the model and converted to table JSON (format 0.3).
"""

import asyncio
import multiprocessing
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from document_prep.detect import detect_layout, table_regions
from document_prep.models import DocumentSource, LayoutRegion, PreparationSettings, PreparedRegion
from document_prep.prepare import page_count, prepare_page
from extraction.extract import extract_regions
from extraction.results import RegionResult, TableResult
from extraction.vllm import VllmRecognizer, VllmServer
from tablereader.release import DEFAULT_RELEASE, ModelRelease, load_release

Progress = Callable[[str, int, int], None]  # (stage, done, total)


def _quiet(stage: str, done: int, total: int) -> None:
    pass


class Canceled(Exception):
    """Raised from a progress callback to stop reading a document."""


def parse_pages(spec: str | None, count: int) -> list[int]:
    """Pages from a selection such as `"3,7-9"` (one-based, in document order); all pages when `spec` is empty."""
    if not spec:
        return list(range(1, count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if match is None:
            raise ValueError(f"bad page selection {part!r}; use e.g. 3,7-9")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= count:
            raise ValueError(f"pages {part.strip()} are outside 1-{count}")
        pages.update(range(first, last + 1))
    return sorted(pages)


@dataclass(frozen=True, slots=True)
class RegionRead:
    """One detected table region: its prepared crop and what the model made of it."""

    prepared: PreparedRegion
    result: RegionResult


@dataclass(frozen=True, slots=True)
class DocumentResult:
    source: DocumentSource
    pages: tuple[int, ...]
    release_id: str
    regions: tuple[RegionRead, ...]

    @property
    def tables(self) -> list[TableResult]:
        return [t for r in self.regions for t in r.result.tables]


def _detect(
    source: DocumentSource, pages: Sequence[int], settings: PreparationSettings
) -> dict[int, list[LayoutRegion]]:
    """Runs in a child process: Heron on `pages`, boxes in points."""
    from document_prep.heron import HeronLayoutDetector

    return detect_layout(HeronLayoutDetector(), source, pages, settings)


class TableReader:
    """Reads PDFs with one model release. Use as a context manager; it owns a vLLM server while open."""

    def __init__(
        self,
        release: str | Path = DEFAULT_RELEASE,
        *,
        max_in_flight: int = 32,
        gpu_memory_utilization: float = 0.75,
        port: int = 8000,
        log_dir: Path = Path("logs"),
    ) -> None:
        self.release: ModelRelease = load_release(release)
        self.max_in_flight = max_in_flight
        self._server = VllmServer(
            self.release.model_dir,
            name=self.release.release_id,
            log=log_dir / "vllm.log",
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._open = False

    @property
    def release_id(self) -> str:
        return self.release.release_id

    def __enter__(self) -> Self:
        self._server.__enter__()
        self._open = True
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._open:
            self._open = False
            self._server.__exit__(None, None, None)

    def read(self, path: Path, pages: str | None = None, progress: Progress | None = None) -> DocumentResult:
        """Every table on the selected pages of `path`, in reading order."""
        if not self._open:
            raise RuntimeError("open the TableReader (`with TableReader() as reader:`) before reading")
        report: Progress = progress or _quiet
        source = DocumentSource.from_path(path)
        selected = parse_pages(pages, page_count(path))
        settings = self.release.preparation

        report("detect", 0, len(selected))
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            layout = pool.submit(_detect, source, selected, settings).result()
        report("detect", len(selected), len(selected))

        prepared: list[PreparedRegion] = []
        for done, page in enumerate(selected, start=1):
            regions = table_regions(layout[page], page, threshold=self.release.table_threshold)
            if regions:
                prepared.extend(prepare_page(source, page, regions, settings, layout=layout[page]))
            report("prepare", done, len(selected))

        results = asyncio.run(self._recognize(prepared, report))
        reads = tuple(RegionRead(p, results[p.region.region_id]) for p in prepared)
        return DocumentResult(source=source, pages=tuple(selected), release_id=self.release.release_id, regions=reads)

    async def _recognize(self, prepared: Sequence[PreparedRegion], report: Progress) -> dict[str, RegionResult]:
        recognizer = VllmRecognizer(
            model=self.release.release_id,
            base_url=self._server.base_url,
            max_tokens=self.release.max_tokens,
            max_edge=self.release.max_edge,
        )
        results: dict[str, RegionResult] = {}
        report("recognize", 0, len(prepared))
        try:
            async for result in extract_regions(iter(prepared), recognizer, max_in_flight=self.max_in_flight):
                results[result.region_id] = result
                report("recognize", len(results), len(prepared))
        finally:
            await recognizer.aclose()
        return results


def read_pdf(
    path: str | Path,
    pages: str | None = None,
    *,
    release: str | Path = DEFAULT_RELEASE,
    progress: Progress | None = None,
) -> DocumentResult:
    """Read one PDF: starts the model, reads the selected pages (`"3,7-9"`; all when None) and stops the model."""
    with TableReader(release) as reader:
        return reader.read(Path(path), pages, progress)
