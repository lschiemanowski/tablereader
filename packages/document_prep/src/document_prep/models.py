"""Inputs and results of document preparation, and the settings that determine crop content.

Coordinates follow planning/architecture.md: pages are numbered from 1; boxes are in PDF points with a
top-left origin, on the page as displayed (after the PDF's own page rotation, within its crop box). Rendered
images have pixel boxes; `PreparedRegion` keeps both, and the detected box apart from the extended crop.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

POINTS_PER_INCH = 72
ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentSource:
    """A saved PDF and the SHA-256 of its bytes, so results can be tied to the exact file."""

    path: Path
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> "DocumentSource":
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return cls(path=path, sha256=digest.hexdigest())


@dataclass(frozen=True, slots=True, kw_only=True)
class Box:
    """A rectangle (left, top, right, bottom) with a top-left origin, in points or pixels by context."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"box has negative size: {self}")

    def scaled(self, factor: float) -> "Box":
        return Box(
            left=self.left * factor, top=self.top * factor, right=self.right * factor, bottom=self.bottom * factor
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LayoutRegion:
    """A region found by layout detection: its class (Docling Heron's labels, such as `text` or `caption`), the
    detection score and its box."""

    label: str
    score: float
    box: Box


@dataclass(frozen=True, slots=True, kw_only=True)
class TableRegion:
    """A detected (or annotated) table rectangle on one page.

    `rotation_cw` is the clockwise rotation, in degrees, that turns the region's content upright: 90 for a
    table printed sideways that reads bottom to top. It is applied before the crop is extended, because a
    sideways table's title sits beside it on the page, not above it.
    """

    region_id: str
    page: int
    box: Box
    rotation_cw: int = 0

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be nonempty")
        if self.page < 1:
            raise ValueError("pages are numbered from 1")
        if self.rotation_cw not in ROTATIONS:
            raise ValueError(f"rotation_cw must be one of {ROTATIONS}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparationSettings:
    """Rendering resolution and crop-extension parameters. A model release records the settings it was trained on.

    Distances are in pixels at `reference_dpi` (they were tuned on 300-DPI scans) and are scaled to `dpi`.
    """

    dpi: int = 300
    reference_dpi: int = 300
    pad_px: int = 20  # margin added on every side of the extended box
    text_gap_px: int = 60  # a blank band taller than this ends the walk over title or footnote lines
    text_max_px: int = 450  # how far above the table the title, units line and subheadings may reach
    foot_max_px: int = 900  # how far below the table a total line and footnotes may reach
    foot_start_px: int = 40  # between two tables: the upper table's footnotes start within this distance
    foot_line_gap_px: int = 30  # between two tables: footnote lines are at most this far apart
    foot_first_margin: float = 0.04  # ... and start at most this fraction of the width from the left edge
    foot_margin: float = 0.12  # ... with later (wrapped) lines indented at most this fraction
    ink_min_px: int = 3  # a pixel row with at least this many ink pixels across the box holds text
    word_gap_px: int = 40  # title and footnote lines are followed sideways across blank gaps up to this wide
    layout_model: str = "docling-project/docling-layout-heron@8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
    layout_dpi: int = 72  # pages are rendered at this resolution for layout detection, as in Docling
    layout_threshold: float = 0.5
    # Above the table, layout regions with these labels end the walk over title lines: running text, lists,
    # pictures, page furniture, and tables missing from the region list. Captions, titles and section headers
    # do not. Prose-like regions stop it only from `stop_min_lines` lines, so a units line or a short lead-in
    # sentence that Heron labels `text` is still taken with the title above it.
    stop_above: tuple[str, ...] = (
        "table",
        "text",
        "list_item",
        "key_value_region",
        "picture",
        "page_header",
        "page_footer",
        "formula",
        "code",
        "document_index",
        "form",
        "checkbox_selected",
        "checkbox_unselected",
    )
    prose_labels: tuple[str, ...] = ("text", "list_item", "key_value_region")
    stop_min_lines: int = 3
    # Below the table, Heron labels footnotes inconsistently (text, list item, key-value region), so only these
    # end the walk over footnote lines; the gap and distance limits do the rest.
    stop_below: tuple[str, ...] = ("table", "picture", "page_header", "page_footer", "section_header", "title")
    stop_overlap_px: int = 10  # a stopping region may overlap the table edge by this much and still count
    stop_margin_px: int = 8  # layout boxes come from a 72-DPI page and can sit a few pixels inside the ink

    def __post_init__(self) -> None:
        if self.dpi <= 0 or self.reference_dpi <= 0:
            raise ValueError("resolutions must be positive")

    def px(self, reference_px: int) -> int:
        """A reference distance at the rendering resolution."""
        return round(reference_px * self.dpi / self.reference_dpi)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedRegion:
    """An extended crop of one region, ready for recognition, with everything needed to place it on the page.

    `image` is upright (the region's `rotation_cw` applied). `page_size` is the displayed page in points and
    `page_pixels` the rendered page in pixels. `detected_px` and `crop_px` are pixel boxes on the rendered page
    (before the content rotation); `crop_box` is the crop in points. `renderer` names the PDF renderer and its
    version, which together with `settings` determines the pixels.
    """

    source: DocumentSource
    region: TableRegion
    page_size: tuple[float, float]
    page_pixels: tuple[int, int]
    detected_px: Box
    crop_px: Box
    crop_box: Box
    image: Image.Image
    settings: PreparationSettings
    renderer: str

    @property
    def normalized_crop(self) -> Box:
        """The crop as fractions of the displayed page width and height."""
        width, height = self.page_size
        b = self.crop_box
        return Box(left=b.left / width, top=b.top / height, right=b.right / width, bottom=b.bottom / height)
