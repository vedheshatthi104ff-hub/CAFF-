#!/usr/bin/env bash
# Fair comparison of baseline vs CAFF (equal / static / learned gate).
# All arms start from the SAME baseline checkpoint B0 and use the same split, preprocessing,
# seed, data order, optimizer and update budget (10 epochs x 50 updates, 4 places x 4 images).
# Trainable set is identical in every arm (everything); only the gate differs.
#
#   DATA_ROOT=/path/to/gsv BASE_CKPT=checkpoints/accuracy/baseline_best.pt bash scripts/run_fair_comparison.sh
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the folder containing Images/ and Dataframes/}"
BASE_CKPT="${BASE_CKPT:-checkpoints/accuracy/baseline_best.pt}"
LR_BACKBONE="${LR_BACKBONE:-0.00002}"
PY="${PYTHON_BIN:-python}"
# Extra training arguments applied IDENTICALLY to every arm, e.g. TRAIN_ARGS="--epochs 20 --updates-per-epoch 250 --places-per-batch 8"
read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
PROFILE="${PROFILE:-fair_arm}"
[[ -f "$BASE_CKPT" ]] || { echo "Baseline checkpoint $BASE_CKPT not found (run scripts/run_accuracy_round.sh first, or set BASE_CKPT)." >&2; exit 2; }

"$PY" scripts/fetch_dinov2_weights.py
"$PY" -m pytest -q tests/test_models.py

mkdir -p results/fair checkpoints/fair
eval_all () {  # arm config checkpoint
  local arm="$1" cfg="$2" ckpt="$3"
  for pair in "512 data/gsv_cities/internal_eval_manifest.csv" "2048 data/gsv_cities/accuracy_eval_2048.csv" "1536 data/gsv_cities/accuracy_eval_1536_unseen.csv"; do
    set -- $pair
    "$PY" scripts/evaluate.py --config "$cfg" --manifest "$2" --checkpoint "$ckpt" \
      --arm-name "$arm" --output "results/fair/${arm}_$1.json" --data-root "$DATA_ROOT"
  done
}

# Reference: B0 itself, no extra training.
eval_all baseline_b0 configs/accuracy_baseline.toml "$BASE_CKPT"

for arm in baseline_continue caff_equal caff_static caff_learned; do
  cfg="configs/fair_${arm}.toml"; ckpt="checkpoints/fair/${arm}.pt"
  rm -f "checkpoints/fair/${arm}.jsonl"
  extra=(); [[ "$arm" == caff_learned ]] && extra=(--log-gates)
  "$PY" scripts/train.py --config "$cfg" --profile "$PROFILE" --init-checkpoint "$BASE_CKPT" \
    --checkpoint "$ckpt" --lr-backbone "$LR_BACKBONE" --data-root "$DATA_ROOT" ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"} ${extra[@]+"${extra[@]}"}
  eval_all "$arm" "$cfg" "$ckpt"
done

"$PY" scripts/summarize_fair.py
echo "Done. See results/fair/summary.md. Package with: python scripts/package_results.py [--include-checkpoints]"
