"""Contact sheets of recrops for checking by eye: each tile is a recrop with the detected (v1) box in red.

uv run python scripts/recrop_sheets.py data/recrops --per-source 1 --tiles 6 --out data/recrop-sheets
uv run python scripts/recrop_sheets.py data/recrops --where 'up_at_limit' --out data/recrop-sheets/limit
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H, CAPTION = 800, 900, 34


def upright_detected(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """The detected box inside the (upright) crop image, in crop pixels."""
    d, c, rot = row["detected_px"], row["crop_px"], row["rotation_cw"]
    left, top, right, bottom = d[0] - c[0], d[1] - c[1], d[2] - c[0], d[3] - c[1]
    w, h = c[2] - c[0], c[3] - c[1]
    if rot == 90:
        return h - bottom, left, h - top, right
    if rot == 180:
        return w - right, h - bottom, w - left, h - top
    if rot == 270:
        return top, w - right, bottom, w - left
    return left, top, right, bottom


def extension(row: dict[str, Any], pad: int = 20) -> tuple[float, float]:
    _, top, _, bottom = upright_detected(row)
    height = row["image_size"][1]
    return top - pad, height - bottom - pad


def tile(row: dict[str, Any], root: Path) -> Image.Image:
    image = Image.open(root / row["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(upright_detected(row), outline=(230, 0, 0), width=max(3, image.width // 400))
    image.thumbnail((TILE_W, TILE_H - CAPTION))
    out = Image.new("RGB", (TILE_W, TILE_H), (200, 200, 200))
    out.paste(image, (0, CAPTION))
    up, down = extension(row)
    caption = f"{row['table_id'][:58]}  {row['medium']}  {row['label']}  up {up:.0f} down {down:.0f}"
    ImageDraw.Draw(out).text((6, 6), caption, fill=(0, 0, 0), font=ImageFont.load_default(18))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recrops", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/recrop-sheets"))
    parser.add_argument("--per-source", type=int, default=1)
    parser.add_argument(
        "--where", choices=["all", "up_at_limit", "down_at_limit", "scan_or_mixed", "rotated"], default="all"
    )
    parser.add_argument("--max", type=int, default=48)
    parser.add_argument("--tiles", type=int, default=6, help="tiles per sheet (2 columns)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--same-as", type=Path, help="sheet the examples listed in another sheet set's sample.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.recrops / "manifest.jsonl").open()]
    if args.where == "up_at_limit":
        rows = [r for r in rows if extension(r)[0] > 410]
    elif args.where == "down_at_limit":
        rows = [r for r in rows if extension(r)[1] > 840]
    elif args.where == "scan_or_mixed":
        rows = [r for r in rows if r["medium"] == "scan_or_mixed"]
    elif args.where == "rotated":
        rows = [r for r in rows if r["rotation_cw"]]
    if args.same_as:
        wanted = json.loads(args.same_as.read_text())
        by_id = {r["table_id"]: r for r in rows}
        sample = [by_id[t] for t in wanted if t in by_id]
        write_sheets(sample, args)
        return
    rng = random.Random(args.seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source_id"]].append(r)
    sample = [
        r for src in sorted(by_source) for r in rng.sample(by_source[src], min(args.per_source, len(by_source[src])))
    ]
    rng.shuffle(sample)
    sample = sample[: args.max]
    write_sheets(sample, args)


def write_sheets(sample: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(sample), args.tiles):
        chunk = sample[k : k + args.tiles]
        rows_n = (len(chunk) + 1) // 2
        sheet = Image.new("RGB", (2 * TILE_W + 10, rows_n * (TILE_H + 10)), (255, 255, 255))
        for i, row in enumerate(chunk):
            sheet.paste(tile(row, args.recrops), ((i % 2) * (TILE_W + 10), (i // 2) * (TILE_H + 10)))
        sheet.save(args.out / f"sheet-{k // args.tiles + 1:02d}.png")
    (args.out / "sample.json").write_text(json.dumps([r["table_id"] for r in sample], indent=1))
    print(f"{len(sample)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
