"""Extend a table box over its title and footnotes, from the rendered page image.

Ported from ttr (`ttr/src/ttr/extract.py`), with its constants moved into `PreparationSettings`. Lines are
found in the image, not in a text layer, so the same code serves scans and born-digital pages: a text line is
a run of pixel rows with ink between the table's left and right edges. Above the table the walk picks up the
title, units line and subheadings; below it, a total line and the footnotes. A neighbouring table above or
below bounds the walk, and the space between two tables is divided where the upper table's footnotes end.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from document_prep.models import Box, LayoutRegion, PreparationSettings

Gray = npt.NDArray[np.uint8]


def ink_threshold(gray: Gray) -> float:
    """Gray level below which a pixel counts as ink, set per page.

    Ink ranges from near black to mid grey on faded scans, so the threshold sits halfway between the paper
    (the median) and the darkest ink (the 0.5th percentile).
    """
    return float((np.median(gray) + np.percentile(gray, 0.5)) / 2)


def extend_box(
    gray: Gray,
    box: Box,
    neighbours: Sequence[Box],
    settings: PreparationSettings,
    layout: Sequence[LayoutRegion] = (),
) -> Box:
    """The crop for `box` (pixels on `gray`): extended up and down over adjacent text lines, widened sideways
    over those lines, then padded.

    `neighbours` are the other table boxes on the page, in the same pixels. The walk never crosses into a
    neighbour that overlaps the table horizontally. `layout` holds layout-detection regions in the same pixels:
    above the table the walk ends at running text of three or more lines (so a paragraph is not taken for the
    title), below it at page furniture, pictures and headings (see `PreparationSettings`).
    """
    height, width = gray.shape
    threshold = ink_threshold(gray)
    x0, y0, x1, y1 = _pixels(box)
    tolerance = settings.px(settings.stop_overlap_px)
    floors = [0, y0 - settings.px(settings.text_max_px)]
    ceilings = [height, y1 + settings.px(settings.foot_max_px)]
    for n in neighbours:
        nx0, ny0, nx1, ny1 = _pixels(n)
        if nx0 >= x1 or nx1 <= x0:
            continue
        lo, hi = max(x0, nx0), min(x1, nx1)
        if ny1 <= y0:
            floors.append(_split_between(gray, threshold, lo, hi, ny1, y0, settings))
        elif ny0 >= y1:
            ceilings.append(_split_between(gray, threshold, lo, hi, y1, ny0, settings))
    margin = settings.px(settings.stop_margin_px)
    usable = [r for r in layout if r.score >= settings.layout_threshold]
    blocking = _blocking_regions(gray, threshold, usable, settings)
    obstacles = [*neighbours, *(r.box for r in blocking)]
    for region in blocking:
        sx0, sy0, sx1, sy1 = _pixels(region.box)
        if sx0 < x1 and sx1 > x0 and sy0 < y0 and sy1 <= y0 + tolerance:
            floors.append(min(sy1 + margin, y0))
    for region in usable:
        sx0, sy0, sx1, sy1 = _pixels(region.box)
        if region.label in settings.stop_below and sx0 < x1 and sx1 > x0 and sy1 > y1 and sy0 >= y1 - tolerance:
            obstacles.append(region.box)
            ceilings.append(max(sy0 - margin, y1))
    floor, ceiling = max(floors), min(ceilings)
    top = _text_edge(gray, threshold, x0, x1, y0, floor - 1, settings)
    bottom = _text_edge(gray, threshold, x0, x1, y1, ceiling, settings)
    left, right = _widen(gray, threshold, (x0, top, x1, bottom), (y0, y1), obstacles, settings)
    pad = settings.px(settings.pad_px)
    # The padding never reaches past a stop or into a further text line, so no sliver of excluded text shows.
    pad_up = _blank_rows(gray, threshold, x0, x1, top, -1, pad, settings)
    pad_down = _blank_rows(gray, threshold, x0, x1, bottom, 1, pad, settings)
    return Box(
        left=max(left - pad, 0),
        top=max(top - pad_up, floor, 0),
        right=min(right + pad, width),
        bottom=min(bottom + pad_down, ceiling, height),
    )


def _blank_rows(
    gray: Gray, threshold: float, x0: int, x1: int, edge: int, step: int, pad: int, settings: PreparationSettings
) -> int:
    """How many rows beyond `edge` (up to `pad`) are free of text."""
    for distance in range(1, pad + 1):
        y = edge + step * distance
        if not 0 <= y < gray.shape[0]:
            return pad  # the page edge clips the crop instead
        if _is_text_row(gray, threshold, y, x0, x1, settings):
            return distance - 1
    return pad


def _blocking_regions(
    gray: Gray, threshold: float, regions: Sequence[LayoutRegion], settings: PreparationSettings
) -> list[LayoutRegion]:
    """The regions that end the walk over title lines.

    A short prose region that directly continues a blocking one, such as a paragraph's last line detected as a
    region of its own, blocks too.
    """
    blocking = [r for r in regions if _blocks_above(gray, threshold, r, settings)]
    short_prose = [r for r in regions if r.label in settings.prose_labels and r not in blocking]
    gap = settings.px(settings.text_gap_px)
    grown = True
    while grown:
        grown = False
        for region in short_prose:
            if region not in blocking and any(_continues(b.box, region.box, gap) for b in blocking):
                blocking.append(region)
                grown = True
    return blocking


def _continues(upper: Box, lower: Box, gap: int) -> bool:
    """Whether `lower` starts right below `upper` (within `gap` pixels) and overlaps it horizontally."""
    return lower.left < upper.right and lower.right > upper.left and 0 <= lower.top - upper.bottom <= gap


def _blocks_above(gray: Gray, threshold: float, region: LayoutRegion, settings: PreparationSettings) -> bool:
    """Whether a layout region counts as non-title content when it lies above (or beside) a table's title."""
    if region.label not in settings.stop_above:
        return False
    return region.label not in settings.prose_labels or _count_lines(gray, threshold, region.box, settings) >= (
        settings.stop_min_lines
    )


def _count_lines(gray: Gray, threshold: float, box: Box, settings: PreparationSettings) -> int:
    """Text lines in `box`: runs of pixel rows with ink, ignoring runs too short to be text."""
    height, width = gray.shape
    x0, y0, x1, y1 = _pixels(box)
    x0, x1 = max(x0, 0), min(x1, width)
    y0, y1 = max(y0, 0), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return 0
    rows = ((gray[y0:y1, x0:x1] < threshold).sum(axis=1) >= settings.ink_min_px).tolist()
    min_height = settings.px(6)
    lines, run, gap = 0, 0, 0
    for has_ink in rows:
        if not has_ink:
            gap += 1
            continue
        if run and gap <= 1:  # a one-row gap inside a line
            run += gap
        elif run:
            lines += run >= min_height
            run = 0
        run += 1
        gap = 0
    return lines + (run >= min_height)


def _widen(
    gray: Gray,
    threshold: float,
    extent: tuple[int, int, int, int],
    table_rows: tuple[int, int],
    obstacles: Sequence[Box],
    settings: PreparationSettings,
) -> tuple[int, int]:
    """Left and right edges that take in the full width of the title and footnote lines.

    A title is often wider than the table below it. Ink in the rows above and below the table is followed
    sideways from the table's edges across gaps up to `word_gap_px`, but never into a neighbouring table or a
    stopping region beside those rows.
    """
    x0, top, x1, bottom = extent
    y0, y1 = table_rows
    bands = [(top, y0), (y1 + 1, bottom + 1)]
    rows = [r for a, b in bands if b > a for r in range(a, b)]
    if not rows:
        return x0, x1
    columns = (gray[rows] < threshold).sum(axis=0) >= 2
    width = gray.shape[1]
    left_limit, right_limit = 0, width
    for o in obstacles:
        ox0, oy0, ox1, oy1 = _pixels(o)
        if not any(oy0 < b and oy1 > a for a, b in bands):
            continue
        if ox1 <= x0:
            left_limit = max(left_limit, ox1)
        elif ox0 >= x1:
            right_limit = min(right_limit, ox0)
    gap = settings.px(settings.word_gap_px)
    left = x0
    for c in range(x0 - 1, left_limit - 1, -1):
        if columns[c]:
            left = c
        elif left - c > gap:
            break
    right = x1
    for c in range(x1, right_limit):
        if columns[c]:
            right = c + 1
        elif c - right >= gap:
            break
    return left, right


def _pixels(box: Box) -> tuple[int, int, int, int]:
    return round(box.left), round(box.top), round(box.right), round(box.bottom)


def _is_text_row(gray: Gray, threshold: float, y: int, x0: int, x1: int, settings: PreparationSettings) -> bool:
    return int((gray[y, x0:x1] < threshold).sum()) >= settings.ink_min_px


def _text_edge(
    gray: Gray, threshold: float, x0: int, x1: int, start: int, limit: int, settings: PreparationSettings
) -> int:
    """Walk from a table edge towards `limit` (exclusive) over closely spaced text lines; return the far edge.

    If the walk reaches `limit` inside a line (a stop or neighbour boundary that cuts through text), it returns
    the edge of the last complete line instead, so no half line ends up in the crop.
    """
    step = -1 if limit < start else 1
    max_gap = settings.px(settings.text_gap_px)
    edge, gap, complete = start, 0, start
    for y in range(start + step, limit, step):
        if _is_text_row(gray, threshold, y, x0, x1, settings):
            edge, gap = y, 0
        else:
            if gap == 0:
                complete = edge
            gap += 1
            if gap > max_gap:
                return edge
    if gap == 0 and 0 <= limit < gray.shape[0] and _is_text_row(gray, threshold, limit, x0, x1, settings):
        return complete
    return edge


def _split_between(
    gray: Gray, threshold: float, x0: int, x1: int, upper_bottom: int, lower_top: int, settings: PreparationSettings
) -> int:
    """Where to divide the space between a table ending at `upper_bottom` and one starting at `lower_top` below it.

    The upper table's footnotes start right under it and are left-aligned (wrapped lines indented a little);
    the lower table's title and units lines are centred. So the upper table gets the run of left-aligned,
    closely spaced lines that starts near its bottom, and the lower one the rest.
    """
    b, t = upper_bottom, lower_top
    if t <= b or x1 <= x0:
        return (b + t) // 2
    ink = gray[b:t, x0:x1] < threshold
    rows = ink.sum(axis=1) >= settings.ink_min_px
    lines: list[tuple[int, int]] = []
    y = 0
    while y < len(rows):
        if rows[y]:
            y1 = y
            while y1 < len(rows) and rows[y1]:
                y1 += 1
            lines.append((y, y1))
            y = y1
        else:
            y += 1
    width, end = x1 - x0, 0
    for i, (y0, y1) in enumerate(lines):
        margin = int(ink[y0:y1].any(axis=0).argmax()) / width
        max_gap = settings.px(settings.foot_start_px if i == 0 else settings.foot_line_gap_px)
        max_margin = settings.foot_first_margin if i == 0 else settings.foot_margin
        if y0 - end > max_gap or margin > max_margin:
            return b + (end + y0) // 2
        end = y1
    return b + (end + len(rows)) // 2
