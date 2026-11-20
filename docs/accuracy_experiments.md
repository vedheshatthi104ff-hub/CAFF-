# Accuracy improvement round

This round is designed to improve retrieval accuracy while keeping comparisons controlled. It uses only the supplied GSV-Cities archive and an unchanged train/validation assignment from seed 42.

**Status: prepared, not run.** This session has no attached remote GPU, and its local Python runtime has no PyTorch installation. The official CUDA-enabled Windows wheel is 2.75 GB; its download did not complete, so no new model metrics or checkpoints are claimed. The training profiles, controlled learning-rate override, expanded evaluation manifest, and Linux remote-GPU runner are ready for a CUDA host.

## Data and evaluation

- Training: 424,338 images from 51,880 places (`data/gsv_cities/manifest.csv`, split `train`).
- Place-disjoint validation: 105,168 images from 12,514 places (same manifest, split `validation`).
- Development retrieval set: the existing 512-query manifest.
- Expanded retrieval set: `data/gsv_cities/accuracy_eval_2048.csv`, with 2,048 queries and 15,131 gallery images. Its 512-query subset matches the existing seed-42 internal set.
- Both retrieval sets are project-created internal GSV-Cities evaluations, not official EfficientVPR benchmark results.

## Training comparison

- Baseline profile: 4 places × 4 images, 50 updates per epoch for 10 epochs (500 updates total), image size 266, fixed seed 42.
- Backbone learning-rate sweep: `2e-5`, `2e-4`, `2e-3`, and `2e-2`. Other settings and the 500-update budget remain fixed; the highest value is an exploratory transcription of the paper setting and may be unstable.
- Select the strongest stable candidate by Recall@1 on the fixed 512-query development set; report its R@1/R@5/R@10 on the expanded 2,048-query set.
- **Superseded CAFF protocol:** an earlier version of this plan trained CAFF for 500 *additional* updates on top of the 500-update baseline with the base frozen, which gives CAFF 1,000 total updates and a different trainable set. That is not a fair comparison. Use `docs/fair_comparison.md` / `scripts/run_fair_comparison.sh` instead (same start B0, same budget, same trainable set, equal-gate control). `scripts/run_accuracy_round.sh` is now only the baseline learning-rate sweep that produces B0 (`baseline_best.pt`); its trailing CAFF steps are legacy.
- Selection note: B0 is chosen by R@1 on the 512-query set, which is also inside the 2,048 set. Report the 1,536 queries not used for selection (`accuracy_eval_1536_unseen.csv`) as the cleanest number.
- A CAFF result that is lower than baseline remains a valid result; it will not be described as an accuracy gain.

Pilot checkpoint files are omitted; their logs and records are in `checkpoints/legacy_v1/` and `results/legacy_v1/`. New runs use separate names under `checkpoints/accuracy/` and write their own logs and result files.

The 6 GB RTX 4050 and smaller 4-place batch are a resource-limited deviation from the paper's 160-place batch and 30-epoch schedule. The four learning-rate trials and CAFF run are development experiments, not a full reproduction. No official benchmark data beyond the supplied GSV-Cities archive is included in this round.

## Commands for a CUDA-ready machine

On a Linux cloud/GPU host, stage the project ZIP and the GSV-Cities archive, install a CUDA-compatible PyTorch build and the project requirements, then set `DATA_ROOT` to the mounted folder containing `Images/` and `Dataframes/` and run:

```bash
DATA_ROOT=/path/to/gsv-cities bash scripts/run_accuracy_round.sh
```

The runner executes the baseline learning-rate sweep, selects a stable baseline by 512-query Recall@1, trains CAFF from that checkpoint, and evaluates both on the 2,048-query holdout. The dataset at `D:\archive` is local to this Windows machine; it must be staged on the remote host separately.

The commands below show the same steps for a CUDA-ready Windows machine.

Run the same baseline budget for each learning rate, then evaluate each checkpoint on the fixed development set:

```powershell
$trials = @(
    @{ tag = '2e-5';  lr = '0.00002' },
    @{ tag = '2e-4';  lr = '0.0002' },
    @{ tag = '2e-3';  lr = '0.002' },
    @{ tag = '2e-2';  lr = '0.02' }
)
foreach ($trial in $trials) {
    $checkpoint = "checkpoints/accuracy/baseline_lr_$($trial.tag).pt"
    python scripts/train.py --config configs/accuracy_baseline.toml --profile accuracy_baseline --lr-backbone $trial.lr --checkpoint $checkpoint
    python scripts/evaluate.py --config configs/accuracy_baseline.toml --manifest data/gsv_cities/internal_eval_manifest.csv --checkpoint $checkpoint --output "results/accuracy_lr_$($trial.tag)_512.json"
}
```

After selecting the strongest stable baseline by R@1, train CAFF from it and evaluate both models on the expanded set:

```powershell
python scripts/train.py --config configs/accuracy_caff.toml --profile accuracy_caff --init-checkpoint checkpoints/accuracy/baseline_lr_2e-4.pt --checkpoint checkpoints/accuracy/caff.pt --log-gates
python scripts/evaluate.py --config configs/accuracy_baseline.toml --manifest data/gsv_cities/accuracy_eval_2048.csv --checkpoint checkpoints/accuracy/baseline_lr_2e-4.pt --output results/accuracy_baseline_2048.json
python scripts/evaluate.py --config configs/accuracy_caff.toml --manifest data/gsv_cities/accuracy_eval_2048.csv --checkpoint checkpoints/accuracy/caff.pt --output results/accuracy_caff_2048.json
```

Replace `baseline_lr_2e-4.pt` in these example commands with the selected trial's checkpoint. Training/evaluation commands require a matching CUDA PyTorch install and the supplied images at `D:\archive`.
