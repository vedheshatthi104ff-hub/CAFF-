"""Zip configs, logs, metric files, manifests and (optionally) final checkpoints, with a SHA-256 manifest.

  python scripts/package_results.py                       # logs/configs/metrics only; says checkpoints are omitted
  python scripts/package_results.py --include-checkpoints # adds model-only final checkpoints (needs torch)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = ["configs/*.toml", "results/**/*", "checkpoints/**/*.jsonl", "data/gsv_cities/*audit*.json",
         "data/gsv_cities/internal_eval_manifest.csv", "data/gsv_cities/accuracy_eval_*.csv",
         "docs/*.md", "README.md", "THIRD_PARTY_NOTICES.md", "scripts/*", "src/**/*.py", "tests/*.py", "requirements.txt"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--include-checkpoints", action="store_true")
    ap.add_argument("--output", type=Path, default=ROOT / "results_package.zip"); args = ap.parse_args()
    files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    manifest = {"checkpoints_included": args.include_checkpoints, "files": {}}
    extra = {}
    if args.include_checkpoints:
        import torch
        for ckpt in sorted((ROOT / "checkpoints").glob("**/*.pt")):
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            slim = ckpt.parent / (ckpt.stem + ".model_only.pt")
            torch.save({k: state[k] for k in ("model", "epoch", "profile", "config", "record") if k in state}, slim)
            extra[slim] = slim
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + list(extra):
            rel = p.relative_to(ROOT).as_posix(); z.write(p, rel); manifest["files"][rel] = sha(p)
        note = ("Final checkpoints are INCLUDED as *.model_only.pt (optimizer state removed)." if args.include_checkpoints
                else "Final checkpoints are OMITTED from this package (size). Re-create them with scripts/run_fair_comparison.sh; "
                     "checkpoint SHA-256 values are recorded in each results/fair/*.json.")
        z.writestr("REPRODUCIBILITY_MANIFEST.json", json.dumps({**manifest, "note": note}, indent=2))
    print(note, "->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
