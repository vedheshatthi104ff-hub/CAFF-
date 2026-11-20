# Experiment ledger

The software checks ran on 2026-10-28. Dataset preparation and pilot model runs used the supplied GSV-Cities archive on 2026-10-29. Internal retrieval results below use the project-created 512-place, place-disjoint GSV holdout; they are not official EfficientVPR benchmark results.

| ID | Model | Dataset/split | Status | Metrics |
|---|---|---|---|---|
| E0 | Baseline, CAFF, OLE-concat, and 64-prompt forward checks | Random 266×266 tensor, official DINOv2-S/14 weights, CPU | **PASS** — finite outputs, shape `(1, 3456)` | Shape checks only; not retrieval metrics |
| E1 | Paper-based baseline pilot | GSV-Cities place-disjoint internal holdout; 1 epoch, 25 updates, batch 4 | **PASS** — train loss 0.6716, validation loss 0.9584 | 512/512 queries; R@1 0.5176, R@5 0.7031, R@10 0.7637 |
| E2 | Baseline + paper SGFS | Official benchmark splits | **NOT RUN** — benchmark datasets absent | None |
| E3 | CAFF, learned gates pilot | Same GSV-Cities internal holdout; initialized from baseline, 1 epoch, 25 updates, batch 4 | **PASS** — train loss 0.4971, validation loss 0.4664; mean gates 0.5482 / 0.4518 | 512/512 queries; R@1 0.3906, R@5 0.6074, R@10 0.6777 |
| E4 | CAFF vs baseline pilot | Matched internal holdout and descriptor size | **CAFF lower in this short pilot** | R@1 difference: -0.1270 absolute (-12.7 percentage points) |
| E5 | CAFF without CLS context | Not implemented; needs controlled design | **NOT RUN** | None |
| E6 | Global-only/local-only | Not implemented; needs controlled design | **NOT RUN** | None |
| E7 | Prompt count 8/16/32/64 | Pitts250k-test, AmsterTime, SVOX-Overcast | **NOT RUN** — configs added; data/training missing | None |
| E8 | OLE direct concat vs orthogonal | Pitts250k-test, AmsterTime, SVOX-Overcast | **NOT RUN** — config and mode added; data/training missing | None |
| E9 | Training/checkpoint smoke | Four generated images, two places, CPU | **PASS** — 1 epoch; checkpoint saved, reloaded, and forward-checked | Synthetic loss is not a research metric |
| E10 | Accuracy-improvement round: 4 places × 4 images, 500 updates, backbone LR sweep; CAFF fine-tune | GSV-Cities seed-42 place split; expanded 2,048-query holdout prepared | **NOT RUN** — CUDA PyTorch runtime unavailable; large official wheel transfer did not complete | No new metrics |

The `gpu_laptop` profile used 2 places × 2 images (batch 4), 25 optimizer updates, and a backbone LR of 2e-5; the CAFF run initialized from the baseline and froze all non-CAFF parameters. Each one-epoch run is only a short slice of the 424,338-image training split, not a full data epoch. Validation loss used ten batches. The existing unit suite passed: **10 passed**. E1 and E3 are measured internal retrieval results, but the short schedule and project-created split do not establish a paper result or support a final CAFF efficacy claim.

E10's expanded manifest has 2,048 queries and 15,131 gallery images. A direct place-ID and query-path comparison confirmed that its 512-query subset matches the existing internal evaluation manifest. The training recipe and blocker are recorded in `docs/accuracy_experiments.md`.

The paper's reported values are literature results and are not copied into this table. New evaluation scripts write one JSON and CSV per completed run. `compare_models.py` reads only those result JSONs.

