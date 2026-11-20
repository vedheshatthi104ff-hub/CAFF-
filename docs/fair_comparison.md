# Fair comparison protocol (baseline vs CAFF v2)

**Start point B0:** the baseline checkpoint selected by `scripts/run_accuracy_round.sh` (`checkpoints/accuracy/baseline_best.pt`).

**Arms** (all start from B0; same GSV-Cities place-disjoint split, 266 px training / 322 px test preprocessing, seed 42,
same data order, Adam, 10 epochs x 50 updates of 4 places x 4 images = 500 updates, all parameters trainable):

| Arm | Config | What differs |
|---|---|---|
| `baseline_b0` | `configs/accuracy_baseline.toml` | reference, no extra training |
| `baseline_continue` | `configs/fair_baseline_continue.toml` | no gate; absorbs the effect of 500 more updates |
| `caff_equal` | `configs/fair_caff_equal.toml` | gate fixed at 0.5/0.5 (no-op control; should match `baseline_continue`) |
| `caff_static` | `configs/fair_caff_static.toml` | two learned, image-independent weights |
| `caff_learned` | `configs/fair_caff_learned.toml` | per-image gate (the proposed CAFF), 150,148 gate parameters |

Reading the table: `caff_learned` vs `caff_static` isolates the value of *conditioning*; `caff_static` vs `caff_equal`
isolates *re-weighting*; everything vs `baseline_continue` removes the benefit of extra training.

**Evaluation:** Recall@1/5/10 on (a) the 512-query set, (b) the 2,048-query holdout, (c) the 1,536 holdout queries not used to
select B0 (`data/gsv_cities/accuracy_eval_1536_unseen.csv`, same 15,131-image gallery). These are internal GSV-Cities checks, not
official benchmarks. Use a single seed only as a first look; repeat with other seeds before claiming a difference.

**Run:** `DATA_ROOT=/path/to/gsv-cities bash scripts/run_fair_comparison.sh` -> `results/fair/*.json|csv`, `results/fair/summary.md`,
logs in `checkpoints/fair/*.jsonl`. **Package:** `python scripts/package_results.py [--include-checkpoints]`.

**Metadata:** every result JSON records the arm, trained parameter counts and breakdown (from the training log, not the eval-time
`requires_grad` count), gate mode, training config, init-checkpoint and checkpoint SHA-256, and `evaluate.py` refuses a
`--config` that differs from the config the checkpoint was trained with.

## Longer training for higher accuracy

The default `fair_arm` profile (500 updates of 16 images) is only a first look. For real gains use the `long_run` profile
(20 epochs x 250 updates x 8 places x 4 images by default) and apply the **same** overrides to every arm:

```bash
TRAIN_ARGS="--epochs 20 --updates-per-epoch 250 --places-per-batch 8 --images-per-place 4 --snapshot-every 5" \
PROFILE=long_run DATA_ROOT=D:/archive bash scripts/run_fair_comparison.sh
```

Training options: mixed precision (fp16, on by default on CUDA), `--accum-steps N` (N micro-batches per optimizer update; the
loss is averaged, but metric-learning mining still sees only one micro-batch, so it is not identical to one large batch),
`--snapshot-every N` (model-only snapshot every N epochs), a `.best.pt` model-only file chosen by validation loss, and the
latest resumable checkpoint every epoch (`--resume` now restores the scheduler and AMP scaler too). If you run out of memory,
lower `--places-per-batch` and raise `--accum-steps`, identically for all arms. Train the starting baseline B0 with the same
profile before running the arms.
