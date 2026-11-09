"""Build the Hugging Face dataset of the rebuilt benchmark locally (no upload).

One row per recropped part with an `include` decision in table_benchmark's crop rights and privacy audit
(`data/full/hf-audit-work/crop-decisions.jsonl`; the 348 held crops are left out). Each row carries the crop PNG
as recropped by `document_prep`, its transcription under contract revision 1 where one exists, the table JSON
(format 0.3) proposed from it, the v1 split and label, and the document's rights basis and attribution.

The recrops extend the v1 crops over titles and footnotes, which the audit screened as nearby page text, not as
crop pixels. As an extra screen, transcriptions whose text suggests third-party credit or personal data are
written to `<out>/flags.jsonl` and left out until cleared (`--cleared`, one region id per line).

    uv run --group train --with datasets==5.1.0 python scripts/build_hf_dataset.py ~/workspace/table_benchmark
"""

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from table_core.codec import encode_table
from training.data import Recrops, read_jsonl, region_id, reinterpret, rejection, training_target

SCHEMA_VERSION = "0.1"
# user decision, 2026-11-08
ANNOTATIONS_LICENSE = (
    "The annotations (`transcription`, `tables` and the label columns) are released under the Creative Commons "
    "Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The crop "
    "images remain under their source documents' terms."
)
SHARD_BYTES = 500_000_000
SCREEN = re.compile(
    r"©|\(c\)\s*\d{4}|copyright|reproduced (?:with|by) (?:kind )?permission|courtesy of|all rights reserved"
    r"|signature|signed\b|date of birth|\bd\.o\.b\b|national insurance|passport|home address|next of kin",
    re.IGNORECASE,
)
DOCUMENT_FIELDS = ("source_id", "series", "title", "publisher", "published", "rights_basis", "rights_url",
                   "attribution", "pdf_url", "metadata_url")  # fmt: skip


def features() -> Any:
    from datasets import Features, Image, Sequence, Value

    return Features({
        "region_id": Value("string"), "table_id": Value("string"), "part": Value("int32"), "split": Value("string"),
        "image": Image(), "image_sha256": Value("string"), "rotation_cw": Value("int32"),
        "document_id": Value("string"), "source_id": Value("string"), "medium": Value("string"),
        "page": Value("int32"), "crop_box": Sequence(Value("float32")), "page_size": Sequence(Value("float32")),
        "v1_label": Value("string"),
        "transcription": Value("string"), "transcription_status": Value("string"),
        "transcription_model": Value("string"), "region_status": Value("string"), "tables": Value("string"),
        "rights_basis": Value("string"), "attribution": Value("string"), "source_url": Value("string"),
    })  # fmt: skip


def label_fields(label: dict[str, Any] | None) -> dict[str, Any]:
    if label is None:
        return {"transcription": None, "transcription_status": "unlabeled", "transcription_model": None,
                "region_status": None, "tables": None}  # fmt: skip
    if label["response"] is None:
        return {"transcription": None, "transcription_status": "rejected", "transcription_model": None,
                "region_status": label["status"], "tables": None}  # fmt: skip
    result = reinterpret(label)
    tables = [encode_table(t.candidate) for t in result.tables if t.candidate is not None]
    return {
        "transcription": training_target(label["response"]["text"]),
        "transcription_status": "rejected" if rejection(label) else "accepted",
        "transcription_model": label["response"]["model"],
        "region_status": result.status.value,
        "tables": json.dumps(tables, ensure_ascii=False) if tables else None,
    }


def write_card(args: argparse.Namespace, summary: dict[str, Any], full: Path) -> None:
    """README (dataset card), LICENSE and the format documents next to the data."""
    repo = Path(__file__).resolve().parent.parent
    for source in ("docs/transcription-format.md", "docs/table-format.md",
                   "packages/table_core/src/table_core/table.schema.json"):  # fmt: skip
        shutil.copy(repo / source, args.out / Path(source).name)
    shutil.copy(full / "hf-dev-v1/LICENSE", args.out / "LICENSE")
    split_rows = []
    for split in ("train", "validation", "test"):
        info = summary["splits"].get(split, {"parts": 0, "transcriptions": {}})
        counts = info["transcriptions"]
        split_rows.append(f"| {split} | {info['parts']:,} | {counts.get('accepted', 0):,} | "
                          f"{counts.get('rejected', 0):,} | {counts.get('unlabeled', 0):,} |")  # fmt: skip
    card = (repo / "scripts/hf_card.md").read_text()
    card = card.replace("{documents}", str(summary["documents"])).replace("{schema_version}", SCHEMA_VERSION)
    card = card.replace("{split_rows}", "\n".join(split_rows)).replace("{flagged}", str(summary["flagged"]))
    card = card.replace("{held}", str(sum(summary["held_by_audit"].values())))
    card = card.replace("{annotations_license}", args.annotations_license)
    (args.out / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--recrops", type=Path, default=Path("data/recrops"))
    parser.add_argument("--labels", type=Path, nargs="+", default=[Path("data/labels/terra/results.jsonl")],
                        help="label files, earlier ones taking precedence")  # fmt: skip
    parser.add_argument("--cleared", type=Path, help="region ids cleared after reviewing their flags")
    parser.add_argument("--holds", type=Path, help="JSON lines of region ids held after review, with reasons")
    parser.add_argument("--out", type=Path, default=Path("data/hf/tablereader-benchmark"))
    parser.add_argument("--annotations-license", default=ANNOTATIONS_LICENSE, help="paragraph for the dataset card")
    args = parser.parse_args()

    full = args.benchmark / "data/full"
    decisions = {r["crop_key"]: r["decision"] for r in read_jsonl(full / "hf-audit-work/crop-decisions.jsonl")}
    documents = {d["document_id"]: d for d in read_jsonl(full / "hf-dev-v1/documents.jsonl")}
    recrops = Recrops.load(args.recrops)
    labels: dict[str, dict[str, Any]] = {}
    for path in reversed(args.labels):
        labels |= {r["region_id"]: r for r in read_jsonl(path)}
    cleared = set(args.cleared.read_text().split()) if args.cleared else set[str]()
    # held by crop: every part of a held table stays out
    holds = {r["region_id"].rsplit("-", 1)[0]: r["reason"] for r in read_jsonl(args.holds)} if args.holds else {}

    held: Counter[str] = Counter()
    flags: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recrops.rows:
        rid = region_id(row)
        if decisions.get(row["crop_key"]) != "include" or row["document_id"] not in documents:
            held[row["split"]] += 1
            continue
        if (reason := holds.get(row["crop_key"])) is not None:
            held[f"{row['split']} ({reason})"] += 1
            continue
        fields = label_fields(labels.get(rid))
        if (text := fields["transcription"]) and rid not in cleared and (found := SCREEN.findall(text)):
            matches = sorted({m.lower() for m in found})
            flags.append({"region_id": rid, "table_id": row["table_id"], "split": row["split"], "matches": matches,
                          "image": str(recrops.image_path(row))})  # fmt: skip
            continue
        document = documents[row["document_id"]]
        rows[row["split"]].append({
            "region_id": rid, "table_id": row["table_id"], "part": row["part"], "split": row["split"],
            "image": str(recrops.image_path(row)), "image_sha256": row["sha256"], "rotation_cw": row["rotation_cw"],
            "document_id": row["document_id"], "source_id": row["source_id"], "medium": row["medium"],
            "page": row["page"], "crop_box": row["crop_box"], "page_size": row["page_size"], "v1_label": row["label"],
            **fields,
            "rights_basis": document["rights_basis"], "attribution": document["attribution"],
            "source_url": document["pdf_url"],
        })  # fmt: skip

    from datasets import Dataset

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "data").mkdir(parents=True)
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "held_by_audit": dict(held), "flagged": len(flags),
                               "splits": {}}  # fmt: skip
    for split, members in rows.items():

        def generate(members: list[dict[str, Any]] = members) -> Iterator[dict[str, Any]]:
            for m in members:  # PNG bytes embedded, named by region id
                yield m | {"image": {"bytes": Path(m["image"]).read_bytes(), "path": f"{m['region_id']}.png"}}

        dataset = Dataset.from_generator(generate, features=features(), cache_dir=str(args.out / ".cache"))
        size = sum(Path(m["image"]).stat().st_size for m in members)
        shards = max(1, round(size / SHARD_BYTES))
        for i in range(shards):
            name = f"{split}-{i:05d}-of-{shards:05d}.parquet"
            dataset.shard(num_shards=shards, index=i, contiguous=True).to_parquet(str(args.out / "data" / name))
        summary["splits"][split] = {
            "parts": len(members),
            "transcriptions": dict(Counter(m["transcription_status"] for m in members)),
            "region_status": dict(Counter(m["region_status"] or "unlabeled" for m in members)),
            "shards": shards,
        }
    shutil.rmtree(args.out / ".cache", ignore_errors=True)
    used = {m["document_id"] for members in rows.values() for m in members}
    with (args.out / "documents.jsonl").open("w") as out:
        for document_id in sorted(used):
            document = documents[document_id]
            out.write(json.dumps({"document_id": document_id, **{k: document.get(k) for k in DOCUMENT_FIELDS}},
                                 ensure_ascii=False) + "\n")  # fmt: skip
    summary["documents"] = len(used)
    (args.out / "flags.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flags))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_card(args, summary, full)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
