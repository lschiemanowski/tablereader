"""Rendering, crop extension and rotation, on synthetic pages."""

import hashlib
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
import pytest
from PIL import Image

from document_prep.crop import _count_lines, extend_box, ink_threshold  # pyright: ignore[reportPrivateUsage]
from document_prep.models import Box, DocumentSource, LayoutRegion, PreparationSettings, TableRegion
from document_prep.prepare import prepare_page, render_page, rotate_box

SETTINGS = PreparationSettings()
WIDTH, HEIGHT = 1200, 1800  # pixels at 300 DPI: 4 x 6 inches


def blank() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)


def line(page: np.ndarray, top: int, left: int = 200, right: int = 900, height: int = 12) -> None:
    """A line of text: a dark band."""
    page[top : top + height, left:right] = 0


def table(page: np.ndarray, top: int, bottom: int, left: int = 200, right: int = 1000) -> Box:
    """A ruled table: rows of text between top and bottom. Returns its box."""
    for y in range(top, bottom - 12, 30):
        line(page, y, left, right)
    return Box(left=left, top=top, right=right, bottom=bottom)


def page_with_table() -> tuple[np.ndarray, Box]:
    page = blank()
    line(page, 300, 350, 850)  # an unrelated paragraph, far above
    line(page, 500, 400, 800)  # title
    line(page, 530, 450, 750)  # units line
    box = table(page, 600, 1000)
    line(page, 1030, 200, 900)  # footnote 1
    line(page, 1055, 220, 600)  # footnote 2, wrapped
    line(page, 1400, 300, 900)  # running text, far below
    return page, box


# Crop extension


def test_extends_over_title_and_footnotes_but_not_beyond_gaps() -> None:
    page, box = page_with_table()
    crop = extend_box(page, box, [], SETTINGS)
    pad = SETTINGS.pad_px
    assert crop == Box(left=200 - pad, top=500 - pad, right=1000 + pad, bottom=1055 + 11 + pad)


def test_text_outside_the_table_columns_is_ignored() -> None:
    page = blank()
    box = table(page, 600, 1000, left=600, right=1000)
    line(page, 560, 100, 500)  # beside the table's columns, not above them
    crop = extend_box(page, box, [], SETTINGS)
    assert crop.top == 600 - SETTINGS.pad_px


def test_title_reach_is_limited() -> None:
    page = blank()
    box = table(page, 1000, 1300)
    for y in range(400, 1000, 40):  # a long column of closely spaced lines above the table
        line(page, y)
    crop = extend_box(page, box, [], SETTINGS)
    # the walk stops at text_max_px above the table; the last line it reaches starts inside that limit
    assert 1000 - SETTINGS.text_max_px - SETTINGS.pad_px <= crop.top < 1000 - SETTINGS.text_max_px + 40


def test_neighbour_above_takes_its_footnotes() -> None:
    page = blank()
    upper = table(page, 200, 500)
    line(page, 520, 200, 900)  # upper table's footnote: left-aligned, right under it
    line(page, 640, 450, 750)  # lower table's title: centred
    lower = table(page, 700, 1000)
    crop = extend_box(page, lower, [upper], SETTINGS)
    assert crop.top == 640 - SETTINGS.pad_px
    upper_crop = extend_box(page, upper, [lower], SETTINGS)
    assert upper_crop.bottom == 520 + 11 + SETTINGS.pad_px


def test_neighbour_beside_does_not_limit() -> None:
    page = blank()
    left = table(page, 600, 1000, left=100, right=550)
    right_table = Box(left=650, top=100, right=1100, bottom=400)
    line(page, 560, 150, 500)
    assert extend_box(page, left, [right_table], SETTINGS).top == 560 - SETTINGS.pad_px


def test_crop_is_clipped_to_the_page() -> None:
    page = blank()
    box = table(page, 5, HEIGHT - 5, left=0, right=WIDTH)
    assert extend_box(page, box, [], SETTINGS) == Box(left=0, top=0, right=WIDTH, bottom=HEIGHT)


def test_ink_threshold_adapts_to_faded_ink() -> None:
    page, box = page_with_table()
    faded = np.where(page == 0, 150, 235).astype(np.uint8)
    assert 150 < ink_threshold(faded) < 235
    assert extend_box(faded, box, [], SETTINGS) == extend_box(page, box, [], SETTINGS)


def test_distances_scale_with_resolution() -> None:
    settings = PreparationSettings(dpi=150)
    assert settings.px(60) == 30
    page, box = page_with_table()
    half = np.asarray(Image.fromarray(page).resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.NEAREST))
    crop = extend_box(half, box.scaled(0.5), [], settings)
    assert crop.top == pytest.approx(250 - 10, abs=1)
    assert crop.bottom == pytest.approx((1055 + 11) / 2 + 10, abs=2)


# Rotation


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotate_box_matches_image_rotation(rotation: int) -> None:
    image = Image.new("L", (300, 200), 255)
    image.paste(0, (40, 30, 90, 60))  # a dark box at (40, 30, 90, 60)
    transposes = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
    rotated = image.transpose(transposes[rotation]) if rotation else image
    ys, xs = np.nonzero(np.asarray(rotated) == 0)
    expected = Box(left=int(xs.min()), top=int(ys.min()), right=int(xs.max()) + 1, bottom=int(ys.max()) + 1)
    assert rotate_box(Box(left=40, top=30, right=90, bottom=60), rotation, (300, 200)) == expected


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotate_box_round_trip(rotation: int) -> None:
    box = Box(left=10, top=20, right=50, bottom=80)
    rotated = rotate_box(box, rotation, (300, 200))
    size = (200, 300) if rotation in (90, 270) else (300, 200)
    assert rotate_box(rotated, (360 - rotation) % 360, size) == box


# Rendering and preparing PDF pages


def write_pdf(path: Path, page: np.ndarray, *, page_rotation: int = 0) -> DocumentSource:
    Image.fromarray(page).convert("RGB").save(path, "PDF", resolution=300)
    if page_rotation:
        pdf = pdfium.PdfDocument(path)
        pdf[0].set_rotation(page_rotation)
        rotated = path.with_name("rotated-" + path.name)
        pdf.save(rotated)
        pdf.close()
        path = rotated
    return DocumentSource.from_path(path)


def close_size(got: tuple[int, int], want: tuple[int, int]) -> bool:
    return all(0 <= g - w <= 1 for g, w in zip(got, want, strict=True))


def text(box: Box) -> LayoutRegion:
    return LayoutRegion(label="text", score=0.9, box=box)


def points(box: Box) -> Box:
    return box.scaled(72 / 300)


def test_render_page_size(tmp_path: Path) -> None:
    page, _ = page_with_table()
    source = write_pdf(tmp_path / "p.pdf", page)
    image, size = render_page(source, 1, 300)
    assert close_size(image.size, (WIDTH, HEIGHT))  # pdfium may add a pixel row or column
    assert size == pytest.approx((WIDTH * 72 / 300, HEIGHT * 72 / 300))
    with pytest.raises(ValueError, match="no page 2"):
        render_page(source, 2, 300)


def test_prepare_page(tmp_path: Path) -> None:
    page, box = page_with_table()
    source = write_pdf(tmp_path / "p.pdf", page)
    region = TableRegion(region_id="r1", page=1, box=points(box))
    (prepared,) = prepare_page(source, 1, [region], SETTINGS)
    expected = extend_box(page, box, [], SETTINGS)
    for got, want in zip(
        (prepared.crop_px.left, prepared.crop_px.top, prepared.crop_px.right, prepared.crop_px.bottom),
        (expected.left, expected.top, expected.right, expected.bottom),
        strict=True,
    ):
        assert abs(got - want) <= 2
    assert prepared.image.size == (
        round(prepared.crop_px.right - prepared.crop_px.left),
        round(prepared.crop_px.bottom - prepared.crop_px.top),
    )
    assert prepared.crop_box.top == pytest.approx(prepared.crop_px.top * 72 / 300)
    assert close_size(prepared.page_pixels, (WIDTH, HEIGHT))
    assert prepared.detected_px.top == pytest.approx(600)
    assert prepared.normalized_crop.right <= 1
    assert prepared.renderer.startswith("pypdfium2 ")


def test_prepare_page_respects_other_regions(tmp_path: Path) -> None:
    page = blank()
    upper = table(page, 200, 500)
    line(page, 520, 200, 900)
    line(page, 640, 450, 750)
    lower = table(page, 700, 1000)
    source = write_pdf(tmp_path / "p.pdf", page)
    regions = [
        TableRegion(region_id="a", page=1, box=points(upper)),
        TableRegion(region_id="b", page=1, box=points(lower)),
    ]
    a, b = prepare_page(source, 1, regions, SETTINGS)
    assert a.crop_px.bottom < 640
    assert b.crop_px.top > 531
    (only_b,) = prepare_page(source, 1, regions[1:], SETTINGS, other_boxes=[points(upper)])
    assert only_b.crop_px == b.crop_px


def test_page_rotation_is_applied_by_the_renderer(tmp_path: Path) -> None:
    page, _ = page_with_table()
    upright = write_pdf(tmp_path / "p.pdf", page)
    turned = write_pdf(
        tmp_path / "q.pdf", np.asarray(Image.fromarray(page).transpose(Image.Transpose.ROTATE_90)), page_rotation=90
    )
    image, _ = render_page(turned, 1, 300)
    assert close_size(image.size, (WIDTH, HEIGHT))  # displayed upright again
    a = np.asarray(render_page(upright, 1, 300)[0].convert("L"), dtype=float)[:HEIGHT, :WIDTH]
    b = np.asarray(image.convert("L"), dtype=float)[:HEIGHT, :WIDTH]
    assert np.abs(a - b).mean() < 3


def test_sideways_table_is_cropped_upright(tmp_path: Path) -> None:
    page, box = page_with_table()
    # Print the page sideways: content rotated 90 degrees counter-clockwise, so rotation_cw=90 makes it upright.
    sideways = np.ascontiguousarray(np.asarray(Image.fromarray(page).transpose(Image.Transpose.ROTATE_90)))
    source = write_pdf(tmp_path / "s.pdf", sideways)
    sideways_box = rotate_box(box, 270, (WIDTH, HEIGHT))  # where the table sits on the sideways page
    region = TableRegion(region_id="r1", page=1, box=points(sideways_box), rotation_cw=90)
    (prepared,) = prepare_page(source, 1, [region], SETTINGS)
    expected = extend_box(page, box, [], SETTINGS)
    width, height = prepared.image.size
    assert abs(width - (expected.right - expected.left)) <= 2
    assert abs(height - (expected.bottom - expected.top)) <= 2
    # the crop box on the page is the upright crop rotated back
    assert prepared.crop_px.right - prepared.crop_px.left == pytest.approx(height, abs=1)
    # upright: the title is near the top of the crop
    rows = (np.asarray(prepared.image.convert("L")) < 128).sum(axis=1)
    assert rows[: SETTINGS.pad_px + 15].sum() > 0


def test_document_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "x.pdf"
    path.write_bytes(b"%PDF-1.4 test")
    source = DocumentSource.from_path(path)
    assert source.sha256 == hashlib.sha256(b"%PDF-1.4 test").hexdigest()


def test_regions_must_be_on_the_page(tmp_path: Path) -> None:
    page, box = page_with_table()
    source = write_pdf(tmp_path / "p.pdf", page)
    with pytest.raises(ValueError, match="page 1"):
        prepare_page(source, 1, [TableRegion(region_id="r", page=2, box=points(box))], SETTINGS)


@pytest.mark.parametrize(
    "kwargs",
    [{"region_id": "", "page": 1}, {"region_id": "r", "page": 0}, {"region_id": "r", "page": 1, "rotation_cw": 45}],
)
def test_table_region_checks(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TableRegion(box=Box(left=0, top=0, right=1, bottom=1), **kwargs)  # type: ignore[arg-type]


# Stopping at running text and widening over titles


def test_walk_stops_at_running_text_above() -> None:
    page = blank()
    for y in (460, 485, 510):  # a paragraph, spaced like title lines
        line(page, y, 150, 1050)
    line(page, 540, 400, 800)  # title
    box = table(page, 600, 1000)
    without = extend_box(page, box, [], SETTINGS)
    assert without.top == 460 - SETTINGS.pad_px  # the paragraph is taken for part of the title
    paragraph = Box(left=150, top=460, right=1050, bottom=522)
    # the padding stops at the paragraph (plus a small margin), so no sliver of it shows
    assert extend_box(page, box, [], SETTINGS, [text(paragraph)]).top == 522 + SETTINGS.stop_margin_px


def test_one_line_text_above_is_part_of_the_title() -> None:
    page = blank()
    line(page, 500, 400, 800)  # title
    line(page, 540, 500, 700)  # units line, which layout detection labels as text
    box = table(page, 600, 1000)
    units = Box(left=500, top=540, right=700, bottom=552)
    assert extend_box(page, box, [], SETTINGS, [text(units)]).top == 500 - SETTINGS.pad_px


def test_below_the_table_text_does_not_stop_but_headings_do() -> None:
    page, box = page_with_table()
    footnotes = Box(left=200, top=1030, right=900, bottom=1067)  # labelled text or list item by Heron
    assert extend_box(page, box, [], SETTINGS, [text(footnotes)]).bottom == 1055 + 11 + SETTINGS.pad_px
    heading = LayoutRegion(label="section_header", score=0.9, box=footnotes)
    assert extend_box(page, box, [], SETTINGS, [heading]).bottom == 1000 + SETTINGS.pad_px


def test_low_scores_and_passing_labels_do_not_stop() -> None:
    page = blank()
    for y in (460, 485, 510):
        line(page, y, 150, 1050)
    line(page, 540, 400, 800)
    box = table(page, 600, 1000)
    paragraph = Box(left=150, top=460, right=1050, bottom=522)
    for region in (
        LayoutRegion(label="text", score=0.4, box=paragraph),
        LayoutRegion(label="caption", score=0.9, box=paragraph),
        LayoutRegion(label="section_header", score=0.9, box=paragraph),
    ):
        assert extend_box(page, box, [], SETTINGS, [region]).top == 460 - SETTINGS.pad_px


def test_count_lines() -> None:
    page = blank()
    for y in (100, 125, 150):
        line(page, y)
    assert _count_lines(page, 128, Box(left=200, top=95, right=900, bottom=170), SETTINGS) == 3
    assert _count_lines(page, 128, Box(left=200, top=95, right=900, bottom=115), SETTINGS) == 1
    page[300, 200:900] = 0  # a rule, too thin for text
    assert _count_lines(page, 128, Box(left=200, top=290, right=900, bottom=310), SETTINGS) == 0


def test_stop_region_overlapping_the_table_edge_slightly_still_stops() -> None:
    page, box = page_with_table()
    line(page, 470, 400, 800)  # three lines above the table, in one text region reaching into its top edge
    region = text(Box(left=200, top=460, right=900, bottom=605))
    assert extend_box(page, box, [], SETTINGS, [region]).top == 600


def test_two_line_lead_in_is_taken_with_the_title_above_it() -> None:
    page = blank()
    line(page, 440, 400, 800)  # title
    line(page, 485, 150, 1050)  # lead-in sentence, two lines
    line(page, 510, 150, 700)
    box = table(page, 570, 1000)
    lead_in = text(Box(left=150, top=485, right=1050, bottom=522))
    assert extend_box(page, box, [], SETTINGS, [lead_in]).top == 440 - SETTINGS.pad_px


def test_stop_regions_beside_or_inside_the_table_are_ignored() -> None:
    page, box = page_with_table()
    expected = extend_box(page, box, [], SETTINGS)
    beside = Box(left=1050, top=100, right=1190, bottom=700)
    inside = Box(left=300, top=650, right=700, bottom=900)
    assert extend_box(page, box, [], SETTINGS, [text(beside), text(inside)]) == expected


def test_title_wider_than_the_table_is_kept_whole() -> None:
    page = blank()
    line(page, 540, 100, 300)  # title starts left of the table ...
    line(page, 540, 330, 1100)  # ... and runs past its right edge, with a word gap
    box = table(page, 600, 1000, left=300, right=900)
    crop = extend_box(page, box, [], SETTINGS)
    assert (crop.left, crop.right) == (100 - SETTINGS.pad_px, 1100 + SETTINGS.pad_px)


def test_widening_does_not_jump_wide_gaps_or_into_neighbours() -> None:
    page = blank()
    line(page, 540, 300, 900)  # title within the table's width
    line(page, 540, 20, 200)  # unrelated text far to the left
    line(page, 1030, 300, 1150)  # footnote running right, into a neighbour's column
    box = table(page, 600, 1000, left=300, right=900)
    neighbour = Box(left=1000, top=900, right=1190, bottom=1300)
    crop = extend_box(page, box, [neighbour], SETTINGS)
    assert (crop.left, crop.right) == (300 - SETTINGS.pad_px, 1000 + SETTINGS.pad_px)


def test_widening_ignores_text_beside_the_table_body() -> None:
    page = blank()
    box = table(page, 600, 1000, left=300, right=900)
    line(page, 700, 920, 1150)  # a margin note level with the table rows, not with a title line
    crop = extend_box(page, box, [], SETTINGS)
    assert crop.right == 900 + SETTINGS.pad_px


def test_prepare_page_uses_layout_stops(tmp_path: Path) -> None:
    page = blank()
    for y in (460, 485, 510):
        line(page, y, 150, 1050)
    line(page, 540, 400, 800)
    box = table(page, 600, 1000)
    source = write_pdf(tmp_path / "p.pdf", page)
    region = TableRegion(region_id="r", page=1, box=points(box))
    paragraph = LayoutRegion(label="text", score=0.95, box=points(Box(left=150, top=460, right=1050, bottom=522)))
    caption = LayoutRegion(label="caption", score=0.95, box=points(Box(left=400, top=540, right=800, bottom=552)))
    (with_layout,) = prepare_page(source, 1, [region], SETTINGS, layout=[paragraph, caption])
    (without,) = prepare_page(source, 1, [region], SETTINGS)
    assert abs(with_layout.crop_px.top - (522 + SETTINGS.stop_margin_px)) <= 2
    assert abs(without.crop_px.top - (460 - SETTINGS.pad_px)) <= 2


def test_last_line_of_a_paragraph_detected_apart_still_stops() -> None:
    page = blank()
    for y in (420, 445, 470, 495):
        line(page, y, 150, 1050)
    line(page, 545, 400, 800)  # title
    box = table(page, 600, 1000)
    body = text(Box(left=150, top=420, right=1050, bottom=482))  # three lines
    last = text(Box(left=150, top=495, right=1050, bottom=507))  # the paragraph's last line, a region of its own
    assert extend_box(page, box, [], SETTINGS, [body, last]).top == 545 - SETTINGS.pad_px  # the title only
    assert extend_box(page, box, [], SETTINGS, [last]).top == 420 - SETTINGS.pad_px  # alone it passes


def test_padding_does_not_reach_into_a_heading_below() -> None:
    page, box = page_with_table()
    line(page, 1100, 200, 600)
    heading = LayoutRegion(label="section_header", score=0.9, box=Box(left=200, top=1080, right=600, bottom=1112))
    crop = extend_box(page, box, [], SETTINGS, [heading])
    assert crop.bottom == 1080 - SETTINGS.stop_margin_px


def test_walk_does_not_end_inside_a_line() -> None:
    page, box = page_with_table()
    line(page, 1080, 200, 600)  # a line that a stop cuts through: heading box starts at 1090, inside the line
    heading = LayoutRegion(label="section_header", score=0.9, box=Box(left=200, top=1098, right=600, bottom=1112))
    crop = extend_box(page, box, [], SETTINGS, [heading])
    assert crop.bottom == 1079  # after the footnotes; the padding stops before the cut line
