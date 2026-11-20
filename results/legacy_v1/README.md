# Legacy pilot results (CAFF v1) — NOT results for the current CAFF

These files come from an earlier 25-update pilot and are kept only for the record.

| File | What produced it | Trained parameters (from its training log) | R@1 / R@5 / R@10 (512-query set) |
|---|---|---|---|
| `pilot_baseline_512.*` | baseline, profile `gpu_laptop` (25 updates, 2 places x 2 images), config `configs/baseline.toml` | 29,437,705 (all) | 51.8 / 70.3 / 76.4 |
| `pilot_caff_v1_512.*` | **CAFF v1 head** (mean-pool -> 128-D projections -> linear to 3456-D, no CLS in descriptor, OLE-only SGFS K/V), profile `gpu_laptop_caff` (25 updates, base frozen), initialised from the pilot baseline | **1,036,802** | 39.1 / 60.7 / 67.8 |

Metadata correction: the `parameter_counts.trainable` field inside `pilot_caff_v1_512.json` (8,417,931) is the
evaluation-time `requires_grad` count of the rebuilt model, not the number of parameters that were trained.
The trained count is 1,036,802, as recorded in `checkpoints/legacy_v1/pilot_caff_v1.jsonl`.

The v1 CAFF module no longer exists in the code. The current (v2) gate has 150,148 parameters
(2,304 LayerNorm + 147,584 + 258 MLP + 2 static logits), so **v1 numbers say nothing about v2**.
The v1 config file `configs/caff.toml` has since been rewritten for v2; the v1 checkpoint is not included
and cannot be loaded into v2 (different `caff.*` keys).
