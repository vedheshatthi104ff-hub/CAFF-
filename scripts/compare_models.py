"""Compare completed evaluation JSON files without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.results.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "query_metrics" in record:
            records.append((path, record))
    if not records:
        print("No completed evaluation JSON files found. Experiments remain NOT RUN.")
        return 2
    print("file,model,descriptor_dim,R@1,R@5,R@10,query_ms_per_image")
    for path, record in records:
        metrics = record["query_metrics"]["recall"]
        print(",".join(map(str, [path.name, record.get("model"), record.get("descriptor_dim"), metrics.get("R@1"), metrics.get("R@5"), metrics.get("R@10"), record.get("query_extraction_ms_per_image")])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

