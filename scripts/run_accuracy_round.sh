#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "Set DATA_ROOT to the directory containing Images/ and Dataframes/." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use a CUDA-enabled PyTorch environment.")
print(f"Training on {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
PY

"$PYTHON_BIN" scripts/validate_datasets.py \
  --dataset gsv_cities \
  --root "$DATA_ROOT" \
  --manifest data/gsv_cities/manifest.csv \
  --max-images 100

while read -r tag lr; do
  checkpoint="checkpoints/accuracy/baseline_lr_${tag}.pt"
  result="results/accuracy_lr_${tag}_512.json"
  echo "Baseline sweep: tag=${tag}, backbone_lr=${lr}"
  if ! "$PYTHON_BIN" scripts/train.py \
    --config configs/accuracy_baseline.toml \
    --profile accuracy_baseline \
    --lr-backbone "$lr" \
    --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT"; then
    echo "Training failed for ${tag}; continuing to the next learning rate." >&2
    continue
  fi
  if ! "$PYTHON_BIN" scripts/evaluate.py \
    --config configs/accuracy_baseline.toml \
    --manifest data/gsv_cities/internal_eval_manifest.csv \
    --checkpoint "$checkpoint" \
    --output "$result" \
    --data-root "$DATA_ROOT"; then
    echo "Evaluation failed for ${tag}; excluding it from selection." >&2
  fi
done <<'TRIALS'
2e-5 0.00002
2e-4 0.0002
2e-3 0.002
2e-2 0.02
TRIALS

best_checkpoint="$("$PYTHON_BIN" - <<'PY'
import json
import math
from pathlib import Path

best = None
for result in sorted(Path("results").glob("accuracy_lr_*_512.json")):
    try:
        record = json.loads(result.read_text(encoding="utf-8"))
        score = float(record["query_metrics"]["recall"]["R@1"])
        if record["query_metrics"]["valid_queries"] and math.isfinite(score):
            tag = result.name.removeprefix("accuracy_lr_").removesuffix("_512.json")
            checkpoint = Path("checkpoints/accuracy") / f"baseline_lr_{tag}.pt"
            log_path = Path("checkpoints/accuracy") / f"baseline_lr_{tag}.jsonl"
            log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            losses_finite = bool(log_rows) and all(
                math.isfinite(float(row["train_loss"]))
                and math.isfinite(float(row["validation_loss"]))
                for row in log_rows
            )
            if checkpoint.is_file() and losses_finite and (best is None or score > best[0]):
                best = (score, checkpoint)
    except (KeyError, ValueError, OSError):
        continue
if best is None:
    raise SystemExit("No valid baseline candidate was evaluated.")
print(best[1])
PY
)"
cp "$best_checkpoint" checkpoints/accuracy/baseline_best.pt
echo "Selected baseline checkpoint: $best_checkpoint"

"$PYTHON_BIN" scripts/evaluate.py \
  --config configs/accuracy_baseline.toml \
  --manifest data/gsv_cities/accuracy_eval_2048.csv \
  --checkpoint checkpoints/accuracy/baseline_best.pt \
  --output results/accuracy_baseline_2048.json \
  --data-root "$DATA_ROOT"
echo "B0 ready: checkpoints/accuracy/baseline_best.pt. Next: DATA_ROOT=$DATA_ROOT bash scripts/run_fair_comparison.sh"
