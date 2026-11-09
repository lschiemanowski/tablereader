"""Read the tables in PDFs, including scans, into structured JSON."""

from tablereader.reader import DocumentResult, RegionRead, TableReader, read_pdf
from tablereader.release import DEFAULT_RELEASE, ModelRelease, load_release

__all__ = ["DEFAULT_RELEASE", "DocumentResult", "ModelRelease", "RegionRead", "TableReader", "load_release", "read_pdf"]
