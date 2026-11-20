"""Check a canonical dataset manifest and image readability."""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = tomllib.loads((ROOT / "configs" / "datasets.toml").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-images", type=int, default=0, help="Validate only the first N rows; 0 means all")
    args = parser.parse_args()
    data_root = args.root or Path(DEFAULTS[args.dataset]["root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if args.manifest:
        manifest = args.manifest
    else:
        manifest_value = Path(DEFAULTS[args.dataset]["manifest"])
        if DEFAULTS[args.dataset].get("manifest_root") == "project":
            manifest = ROOT / manifest_value
        else:
            manifest = data_root / manifest_value
    if not manifest.is_file():
        print(f"MISSING: manifest not found: {manifest}")
        return 2
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or [])
        if not {"image_path", "place_id"}.issubset(headers):
            print("INVALID: canonical manifest requires image_path and place_id columns")
            return 2
        rows = list(reader)
    if args.max_images > 0:
        rows = rows[: args.max_images]
    errors: list[str] = []
    places = Counter()
    for index, row in enumerate(rows, start=2):
        image_path = Path(row.get("image_path", ""))
        image_path = image_path if image_path.is_absolute() else (data_root / image_path)
        places[row.get("place_id", "")] += 1
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            errors.append(f"row {index}: {image_path}: {error}")
    print(f"dataset={args.dataset} rows_checked={len(rows)} unique_places={len(places)}")
    if args.max_images and len(rows) == args.max_images:
        print("scope=sampled rows only; this is not a full validation")
    if errors:
        print(f"unreadable_or_missing={len(errors)}")
        for error in errors[:20]:
            print(f"ERROR {error}")
        return 1
    print("image_files=readable for every checked row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

