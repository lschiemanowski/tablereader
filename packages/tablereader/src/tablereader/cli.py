"""Command line: `tablereader read` for one PDF, `tablereader serve` for the HTTP service.

tablereader read report.pdf --pages 13-20 --out out/report
tablereader serve --jobs jobs/ --port 8080
"""

import argparse
import sys
from pathlib import Path

from tablereader.release import DEFAULT_RELEASE


def _read(args: argparse.Namespace) -> None:
    from tablereader.output import write_outputs
    from tablereader.reader import read_pdf

    def progress(stage: str, done: int, total: int) -> None:
        print(f"\r{stage}: {done}/{total}   ", end="", file=sys.stderr, flush=True)

    out: Path = args.out or Path("out") / args.pdf.stem
    result = read_pdf(args.pdf, args.pages, release=args.release, progress=progress)
    write_outputs(result, out)
    ready = sum(t.outcome.value == "ready" for t in result.tables)
    print(f"\n{len(result.tables)} tables ({ready} ready) on {len(result.pages)} pages; see {out / 'review.html'}",
          file=sys.stderr)  # fmt: skip


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from tablereader.api import create_app
    from tablereader.jobs import JobStore, Reader
    from tablereader.reader import TableReader

    def open_reader() -> Reader:
        return TableReader(args.release, log_dir=args.jobs / "logs", port=args.vllm_port).__enter__()

    uvicorn.run(create_app(JobStore(args.jobs), open_reader), host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tablereader", description="Read the tables in PDFs into structured JSON.")
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="read the tables in one PDF")
    read.add_argument("pdf", type=Path)
    read.add_argument("--pages", help="page selection such as 3,7-9 (default: all)")
    read.add_argument("--out", type=Path, help="output directory (default: out/<pdf name>)")
    read.add_argument("--release", default=DEFAULT_RELEASE, help="model release: Hugging Face repo or directory")
    read.set_defaults(run=_read)
    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--jobs", type=Path, default=Path("jobs"), help="directory for jobs and results")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--vllm-port", type=int, default=8000)
    serve.add_argument("--release", default=DEFAULT_RELEASE)
    serve.set_defaults(run=_serve)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
