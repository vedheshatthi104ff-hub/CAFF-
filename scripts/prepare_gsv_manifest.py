"""Build a canonical CAFF-EfficientVPR manifest from an official GSV-Cities tree.

Expected layout::

    <root>/Images/<city>/<image files>
    <root>/Dataframes/<city>.csv

The official GSV-Cities filename format is reconstructed from each annotation
row. Place IDs are prefixed with city_id to avoid collisions across city CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"}


def _expected_filename(row: dict[str, str]) -> str:
    return (
        f"{row['city_id']}_{int(row['place_id']):07d}_{int(row['year']):04d}_"
        f"{int(row['month']):02d}_{int(row['northdeg']):03d}_{row['lat']}_"
        f"{row['lon']}_{row['panoid']}.JPG"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="GSV-Cities archive root containing Images/ and Dataframes/")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "gsv_cities" / "manifest.csv",
        help="Canonical manifest output path",
    )
    parser.add_argument("--verify-images", action="store_true", help="Open and verify every referenced image while building the manifest")
    args = parser.parse_args()

    data_root = args.root.resolve()
    images_root = data_root / "Images"
    dataframes_root = data_root / "Dataframes"
    if not images_root.is_dir() or not dataframes_root.is_dir():
        parser.error(f"Expected Images/ and Dataframes/ under {data_root}")
    annotation_files = sorted(dataframes_root.glob("*.csv"))
    if not annotation_files:
        parser.error(f"No city CSV files found under {dataframes_root}")

    image_cls = None
    if args.verify_images:
        try:
            from PIL import Image
        except ImportError as error:
            raise SystemExit("Pillow is required for --verify-images; install requirements.txt") from error
        image_cls = Image

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="manifest-", suffix=".csv.tmp", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    total_rows = 0
    total_images_verified = 0
    place_counts: Counter[str] = Counter()
    city_stats: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    conflicting_images: list[str] = []
    duplicate_image_rows = 0

    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=("image_path", "place_id", "split", "city_id", "source_panoid"),
            )
            writer.writeheader()
            for csv_path in annotation_files:
                city_rows = 0
                city_places: set[str] = set()
                seen_images: dict[str, str] = {}
                with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    headers = set(reader.fieldnames or [])
                    if not REQUIRED_COLUMNS.issubset(headers):
                        raise ValueError(
                            f"{csv_path.name} is missing required columns: "
                            f"{', '.join(sorted(REQUIRED_COLUMNS - headers))}"
                        )
                    for line_number, row in enumerate(reader, start=2):
                        city_id = row["city_id"].strip()
                        panoid = row["panoid"].strip()
                        if not city_id or not panoid:
                            raise ValueError(f"{csv_path.name}:{line_number}: empty city_id or panoid")
                        filename = _expected_filename(row)
                        image_path = images_root / city_id / filename
                        if not image_path.is_file():
                            # Most files use JPG; account for archives that differ only by extension case.
                            alternatives = (
                                image_path.with_suffix(".jpg"),
                                image_path.with_suffix(".jpeg"),
                                image_path.with_suffix(".JPEG"),
                            )
                            image_path = next((candidate for candidate in alternatives if candidate.is_file()), image_path)
                        if not image_path.is_file():
                            missing.append(f"{csv_path.name}:{line_number}:{image_path}")
                            continue
                        raw_place_id = str(int(row["place_id"]))
                        canonical_place_id = f"{city_id}:{raw_place_id}"
                        image_key = image_path.relative_to(data_root).as_posix().casefold()
                        previous_place = seen_images.get(image_key)
                        if previous_place is not None:
                            if previous_place != canonical_place_id:
                                conflicting_images.append(
                                    f"{csv_path.name}:{line_number}:{image_key}:{previous_place}!={canonical_place_id}"
                                )
                            else:
                                duplicate_image_rows += 1
                            continue
                        seen_images[image_key] = canonical_place_id
                        if image_cls is not None:
                            try:
                                with image_cls.open(image_path) as image:
                                    image.verify()
                            except Exception as error:
                                raise ValueError(f"Unreadable image {image_path}: {error}") from error
                            total_images_verified += 1

                        relative_image_path = image_path.relative_to(data_root).as_posix()
                        writer.writerow(
                            {
                                "image_path": relative_image_path,
                                "place_id": canonical_place_id,
                                "split": "train",
                                "city_id": city_id,
                                "source_panoid": panoid,
                            }
                        )
                        place_counts[canonical_place_id] += 1
                        city_places.add(canonical_place_id)
                        city_rows += 1
                        total_rows += 1

                city_stats[csv_path.stem] = {"rows": city_rows, "places": len(city_places)}
                print(
                    f"{csv_path.stem}: rows={city_rows:,} places={len(city_places):,}",
                    flush=True,
                )

        if missing or conflicting_images:
            details = missing[:10] + conflicting_images[:10]
            raise ValueError(
                f"Dataset join failed: missing_images={len(missing)}, "
                f"conflicting_image_labels={len(conflicting_images)}; first issues: {details}"
            )
        if total_rows == 0:
            raise ValueError("No annotation rows were written")

        os.replace(temporary_path, output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    fewer_than_four = sum(count < 4 for count in place_counts.values())
    audit = {
        "dataset": "GSV-Cities",
        "selected_version": "530k images / 62k places (user-selected)",
        "archive_root": str(data_root),
        "source_annotations": str(dataframes_root),
        "verified_on": date.today().isoformat(),
        "official_format_source": "https://github.com/amaralibey/gsv-cities",
        "annotation_files": len(annotation_files),
        "annotation_rows_joined": total_rows,
        "duplicate_annotation_rows_skipped": duplicate_image_rows,
        "unique_city_scoped_places": len(place_counts),
        "minimum_images_per_place": min(place_counts.values()),
        "places_with_fewer_than_four_images": fewer_than_four,
        "image_files_verified": total_images_verified if args.verify_images else None,
        "manifest": str(output),
        "place_id_policy": "city_id:source_place_id to prevent cross-city label collisions",
        "city_stats": city_stats,
    }
    audit_path = output.with_name("manifest_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"COMPLETE rows={total_rows:,} places={len(place_counts):,} "
        f"min_images_per_place={min(place_counts.values())} "
        f"places_with_<4={fewer_than_four:,} output={output}",
        flush=True,
    )
    if fewer_than_four:
        print(
            "WARNING: some places have fewer than four annotated images; "
            "the place-balanced training sampler will exclude those places.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
