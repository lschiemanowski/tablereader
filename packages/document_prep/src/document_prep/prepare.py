"""Render a page and cut extended, upright crops of the table regions on it."""

from collections.abc import Sequence

import numpy as np
from PIL import Image

from document_prep._pdfium import RENDERER as RENDERER  # re-exported: the renderer name saved with crops
from document_prep._pdfium import render
from document_prep.crop import Gray, extend_box
from document_prep.models import (
    POINTS_PER_INCH,
    Box,
    DocumentSource,
    LayoutRegion,
    PreparationSettings,
    PreparedRegion,
    TableRegion,
)

_TRANSPOSE = {
    90: Image.Transpose.ROTATE_270,  # PIL rotates counter-clockwise
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def render_page(source: DocumentSource, page: int, dpi: int) -> tuple[Image.Image, tuple[float, float]]:
    """The page as displayed (its PDF rotation and crop box applied), as RGB at `dpi`, and its size in points."""
    return render(source.path, page, dpi / POINTS_PER_INCH)


def prepare_page(
    source: DocumentSource,
    page: int,
    regions: Sequence[TableRegion],
    settings: PreparationSettings,
    *,
    other_boxes: Sequence[Box] = (),
    layout: Sequence[LayoutRegion] = (),
) -> list[PreparedRegion]:
    """Prepare every region on `page`, rendering the page once.

    Each region's crop stops at the other regions, at `other_boxes` (further detections on the page that are
    not prepared) and at `layout` regions as `extend_box` describes. Boxes are in points.
    """
    if any(r.page != page for r in regions):
        raise ValueError(f"all regions must be on page {page}")
    image, page_size = render_page(source, page, settings.dpi)
    scale = settings.dpi / POINTS_PER_INCH
    gray_cache: dict[int, Gray] = {}
    prepared: list[PreparedRegion] = []
    for region in regions:
        rotation = region.rotation_cw
        if rotation not in gray_cache:
            upright = image.transpose(_TRANSPOSE[rotation]) if rotation else image
            gray_cache[rotation] = np.asarray(upright.convert("L"))
        gray = gray_cache[rotation]
        size = image.size
        detected_px = region.box.scaled(scale)
        others = [r.box for r in regions if r is not region] + list(other_boxes)
        neighbours = [rotate_box(b.scaled(scale), rotation, size) for b in others]
        regions_px = [
            LayoutRegion(label=r.label, score=r.score, box=rotate_box(r.box.scaled(scale), rotation, size))
            for r in layout
        ]
        crop_upright = extend_box(gray, rotate_box(detected_px, rotation, size), neighbours, settings, regions_px)
        crop_px = rotate_box(crop_upright, (360 - rotation) % 360, (gray.shape[1], gray.shape[0]))
        cut = image.crop((round(crop_px.left), round(crop_px.top), round(crop_px.right), round(crop_px.bottom)))
        prepared.append(
            PreparedRegion(
                source=source,
                region=region,
                page_size=page_size,
                page_pixels=size,
                detected_px=detected_px,
                crop_px=crop_px,
                crop_box=crop_px.scaled(1 / scale),
                image=cut.transpose(_TRANSPOSE[rotation]) if rotation else cut,
                settings=settings,
                renderer=RENDERER,
            )
        )
    return prepared


def rotate_box(box: Box, rotation_cw: int, size: tuple[int, int]) -> Box:
    """`box` on an image of `size` (width, height), mapped onto that image rotated clockwise by `rotation_cw`."""
    width, height = size
    x0, y0, x1, y1 = box.left, box.top, box.right, box.bottom
    if rotation_cw == 0:
        return box
    if rotation_cw == 90:
        return Box(left=height - y1, top=x0, right=height - y0, bottom=x1)
    if rotation_cw == 180:
        return Box(left=width - x1, top=height - y1, right=width - x0, bottom=height - y0)
    if rotation_cw == 270:
        return Box(left=y0, top=width - x1, right=y1, bottom=width - x0)
    raise ValueError(f"unsupported rotation {rotation_cw}")
