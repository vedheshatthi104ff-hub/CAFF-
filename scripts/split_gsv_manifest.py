"""Create a place-disjoint validation split and small internal retrieval manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _digest(seed: int, label: str, place_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{place_id}".encode("utf-8")).digest()


def _temp_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{target.stem}-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    return Path(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "gsv_cities" / "manifest.csv")
    parser.add_argument("--eval-manifest", type=Path, help="Default: internal_eval_manifest.csv beside --manifest")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-place-limit", type=int, default=512, help="0 uses every eligible held-out place")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.eval_place_limit < 0:
        parser.error("--eval-place-limit cannot be negative")

    manifest = args.manifest.resolve()
    eval_manifest = (args.eval_manifest or manifest.with_name("internal_eval_manifest.csv")).resolve()
    counts: Counter[str] = Counter()
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"image_path", "place_id"}.issubset(reader.fieldnames):
            parser.error("manifest must contain image_path and place_id columns")
        fields = list(reader.fieldnames)
        if "split" not in fields:
            fields.append("split")
        for row in reader:
            counts[row["place_id"]] += 1
    if not counts:
        parser.error("manifest is empty")

    validation_places = {
        place_id
        for place_id, count in counts.items()
        if count >= 4
        and int.from_bytes(_digest(args.seed, "split", place_id)[:8], "big") / 2**64
        < args.validation_fraction
    }
    if len(validation_places) < 2:
        parser.error("fewer than two eligible validation places; use another seed/fraction")
    eligible_eval = validation_places
    if args.eval_place_limit:
        eligible_eval = set(
            sorted(validation_places, key=lambda place_id: _digest(args.seed, "eval", place_id))[
                : args.eval_place_limit
            ]
        )

    query_index = {
        place_id: int.from_bytes(_digest(args.seed, "query", place_id)[:8], "big") % counts[place_id]
        for place_id in eligible_eval
    }
    place_index: defaultdict[str, int] = defaultdict(int)
    train_temp = _temp_path(manifest)
    eval_temp = _temp_path(eval_manifest)
    train_rows = validation_rows = query_rows = gallery_rows = 0
    try:
        with manifest.open("r", newline="", encoding="utf-8-sig") as source, train_temp.open(
            "w", newline="", encoding="utf-8"
        ) as train_stream, eval_temp.open("w", newline="", encoding="utf-8") as eval_stream:
            reader = csv.DictReader(source)
            train_writer = csv.DictWriter(train_stream, fieldnames=fields)
            eval_writer = csv.DictWriter(eval_stream, fieldnames=fields)
            train_writer.writeheader()
            eval_writer.writeheader()
            for row in reader:
                place_id = row["place_id"]
                if place_id in validation_places:
                    row["split"] = "validation"
                    validation_rows += 1
                else:
                    row["split"] = "train"
                    train_rows += 1
                train_writer.writerow(row)

                if place_id in eligible_eval:
                    index = place_index[place_id]
                    row["split"] = "query" if index == query_index[place_id] else "gallery"
                    eval_writer.writerow(row)
                    if row["split"] == "query":
                        query_rows += 1
                    else:
                        gallery_rows += 1
                    place_index[place_id] += 1
        os.replace(train_temp, manifest)
        os.replace(eval_temp, eval_manifest)
    except Exception:
        train_temp.unlink(missing_ok=True)
        eval_temp.unlink(missing_ok=True)
        raise

    train_places = len(counts) - len(validation_places)
    audit_path = manifest.with_name("internal_split_audit.json")
    audit = {
        "created_on": date.today().isoformat(),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "training_manifest": str(manifest),
        "training_places": train_places,
        "training_images": train_rows,
        "validation_places": len(validation_places),
        "validation_images": validation_rows,
        "internal_retrieval_protocol": "place-disjoint GSV-Cities holdout; one query per selected place, remaining images in that place form gallery",
        "internal_eval_manifest": str(eval_manifest),
        "internal_eval_places": len(eligible_eval),
        "internal_eval_queries": query_rows,
        "internal_eval_gallery_images": gallery_rows,
        "internal_eval_place_limit": args.eval_place_limit,
        "note": "This is a project-created internal validation split, not an official EfficientVPR benchmark split or paper result.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
