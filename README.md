# CAFF-EfficientVPR

A paper-based PyTorch reimplementation of EfficientVPR with a separate research extension, Condition-Aware Adaptive Feature Fusion (CAFF). The authors' repository was checked on 2026-10-19 and contained only a short README and `.gitignore`; this project is not the authors' code and includes no EfficientVPR-trained checkpoint. It includes Meta's official DINOv2-S/14 no-register backbone checkpoint under Apache-2.0; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Status

- Source audit and paper-based architecture implementation: created.
- Existing unit suite: **10 passed** with CPU PyTorch.
- Official DINOv2-S/14 no-register weights loaded successfully; baseline, CAFF, OLE-concat, and 64-prompt CPU forward paths returned finite 3,456-D descriptors at 266×266.
- Synthetic smoke training: **PASS** — one CPU epoch on four generated images; the temporary checkpoint reloaded and returned a finite descriptor.
- GSV-Cities at `D:\archive`: **VALIDATED** — 529,506 annotation rows across 23 cities matched and passed image readability checks; 64,394 city-scoped place labels were found.
- Place-disjoint GSV split: **PREPARED** — 424,338 train images / 51,880 places and 105,168 validation images / 12,514 places; a separate internal retrieval sample has 512 queries and 3,718 gallery images.
- Real-data pilot training: **PASS** on the RTX 4050 with a one-epoch, 25-update slice per model using 4 images per batch. Pilot checkpoints are not packaged; their logs and metric records are in `checkpoints/legacy_v1/` and `results/legacy_v1/`.
- Legacy pilot (CAFF **v1**, now removed): internal 512-query R@1/5/10 baseline 51.8/70.3/76.4%, v1 CAFF 39.1/60.7/67.8% after 25 updates. These are kept in `results/legacy_v1/` for the record only and say nothing about the current CAFF (v2: gate inside the baseline path, 150,148 gate parameters; v1 trained 1,036,802). See `results/legacy_v1/README.md`.
- **Current CAFF (v2): no trained results yet.** Run `scripts/run_fair_comparison.sh` (baseline B0, baseline-continue, CAFF equal/static/learned from the same start, same split/seed/budget; evaluated on the 512, 2,048 and 1,536-unseen query sets). See `docs/fair_comparison.md`.
- Accuracy-improvement round: **PREPARED, NOT RUN**. Added 500-update baseline/CAFF profiles, backbone learning-rate sweep controls, a 2,048-query internal holdout (15,131 gallery images), and a Linux remote-GPU runner. The expanded holdout includes the same 512 queries used by the pilot. No new accuracy results are claimed because this session has no attached remote GPU and its Python runtime has no PyTorch. See [accuracy experiments](docs/accuracy_experiments.md).
- Official paper benchmark retrieval metrics: **NOT RUN**. Pitts250k, MSLS, AmsterTime, Eynsham, and SVOX evaluation data are not present.
- The 6 GB RTX 4050 Laptop GPU is materially smaller than the paper's A800. The full 640-image batch and 30-epoch schedule were not run; the pilot results are not a full paper reproduction. The ZIP includes the DINOv2 starting weights (`checkpoints/dinov2_vits14_pretrain.pth`, SHA-256 in `THIRD_PARTY_NOTICES.md`); `python scripts/fetch_dinov2_weights.py` re-downloads and verifies them if the file is missing. Baseline/CAFF pilot checkpoints are omitted.

See [the research audit](docs/research_audit.md), [the architecture notes](docs/architecture.md), [the data plan](docs/dataset_plan.md), and [limitations](docs/limitations.md) for evidence and unresolved details.

## What is implemented

- Official DINOv2-S/14 loading from the bundled checkpoint (or from the documented PyTorch Hub path if the file is absent); random initialization is rejected.
- SceneVPT prompt insertion and attention-based prompt filtering, using learnable thresholds as described in the paper.
- MsIA multi-scale token construction, paper OLE orthogonal fusion plus the supplement's direct-concatenation ablation, and SGFS prompt self/cross-attention.
- Separate configs for the supplement's 8/16/32/64 prompt-count ablation.
- Baseline descriptor and optional CAFF gate (v2: rescales the MsIA and OLE streams inside the baseline path; learned, static and equal gate modes). The baseline path is selected with `use_caff=False` and remains separate.
- CSV manifest dataset loader, place-balanced sampler, multi-similarity loss, training/checkpoint loop, generic retrieval evaluator, and dataset presence validator.
- Remote training and evaluation accept `--data-root`; `scripts/run_accuracy_round.sh` runs the controlled sweep on a Linux CUDA host after the GSV-Cities archive has been staged there.

Several paper details are unspecified, so the implementation marks those choices as inferred in the docs and configuration comments. Shape-compatible code is not experimental validation.

## Environment setup

Use a PyTorch build compatible with the machine's CUDA driver, following the current [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install this project's Python requirements:

```powershell
python -m pip install -r requirements.txt
```

The project ZIP includes `checkpoints/dinov2_vits14_pretrain.pth` (verify or re-download with `python scripts/fetch_dinov2_weights.py`), the official no-register ViT-S/14 backbone weights. The model loads Meta's official DINOv2 architecture through PyTorch Hub and applies that checkpoint locally. The first use still needs network access to fetch the Hub source code. If the checkpoint is removed, the loader falls back to PyTorch Hub's pretrained download. Randomly initialized DINOv2 is not used as a substitute.

## Data preparation

The supplied archive at `D:\archive` has the official `Images/<city>` and `Dataframes/<city>.csv` layout. The manifest builder joins each annotation row to the full official image filename, including bearing and panoid, prefixes place IDs with city names, and can verify every image. The observed archive has 529,506 images, 64,394 city-scoped place labels, and 1,880 places with fewer than four images. Those small groups remain in the manifest; the paper's K=4 training sampler skips them. The EfficientVPR supplement reports 570k / 67k, so report this data-version difference.

The scripts create a canonical manifest and reproducible place-disjoint validation split:

```powershell
python scripts/prepare_gsv_manifest.py --root D:\archive --verify-images
python scripts/split_gsv_manifest.py --manifest data/gsv_cities/manifest.csv --eval-place-limit 512 --seed 42
python scripts/validate_datasets.py --dataset gsv_cities --root D:\archive --manifest data/gsv_cities/manifest.csv --max-images 100
```

The first command performs a full image-readability pass. The split assigns entire places to train or validation. Its 512-query retrieval set is a project-created internal check, not an official EfficientVPR benchmark. Configs currently point to `D:/archive`; change `data_root` if you move the dataset. The [VPR Datasets Downloader](https://github.com/gmberton/VPR-datasets-downloader) handles its listed evaluation datasets and does not fetch GSV-Cities.

For benchmark evaluation, create a CSV with columns `image_path,split,place_id`; optionally supply `latitude,longitude` for 25 m GPS ground truth. Use `query` and `gallery` as the split values. For AmsterTime's single-ground-truth protocol, use `ground_truth_path` in the query rows and keep the official pairing/leave-one-out protocol documented alongside the manifest. Do not silently infer benchmark splits from image order.

## Commands

All commands below correspond to scripts in this project. Use the bundled scripts after installing the dependencies and supplying data.
Old pilot checkpoints (`baseline.pt`, v1 `caff.pt`) are not packaged, and v1 `caff.pt` cannot be loaded into the current CAFF.

```powershell
python scripts/validate_datasets.py --dataset gsv_cities --root D:\archive --manifest data/gsv_cities/manifest.csv
python scripts/train.py --config configs/baseline.toml --profile gpu_laptop
# Fair comparison (recommended): DATA_ROOT=... bash scripts/run_fair_comparison.sh
python scripts/train.py --config configs/accuracy_baseline.toml --profile accuracy_baseline --lr-backbone 0.0002 --checkpoint checkpoints/accuracy/baseline_lr_2e4.pt
python scripts/train.py --config configs/baseline.toml --profile smoke_test
python scripts/train.py --config configs/baseline.toml --profile quick_experiment
python scripts/train.py --config configs/baseline.toml --profile research_run
python scripts/train.py --config configs/prompt_count_08.toml --profile research_run
python scripts/train.py --config configs/prompt_count_16.toml --profile research_run
python scripts/train.py --config configs/prompt_count_32.toml --profile research_run
python scripts/train.py --config configs/prompt_count_64.toml --profile research_run
python scripts/train.py --config configs/ole_orthogonal.toml --profile research_run
python scripts/train.py --config configs/ole_concat.toml --profile research_run
python scripts/compare_models.py --results results
```

Training/evaluation commands require the PyTorch stack and the specified local datasets. The bundled DINOv2 weights and synthetic smoke training have been verified on CPU. Real dataset training and retrieval evaluation have not been run. `smoke_test` is a software check, not a paper reproduction.

## Reproduction and CAFF experiments

Paper values are recorded in the configs and [reproduction plan](docs/reproduction_plan.md). `research_run` follows the paper's reported batch structure and schedule; it remains unverified and exceeds what should be attempted on the detected GPU without reducing batch size. `quick_experiment` and the CAFF profiles are development settings and must be labelled as deviations in any results.

Primary comparison: baseline descriptor versus CAFF descriptor, same pretrained checkpoint, data split, preprocessing, output dimension, and retrieval code. Also compare learned gates against fixed 0.5/0.5 gates. Log gates with `--log-gates` or set `model.return_aux=true` in config. No experimental conclusion about CAFF is available yet.

## Licenses and citation

This repository's newly written code is offered under MIT in `LICENSE`. The included DINOv2-S/14 backbone weights retain their Apache-2.0 license; see `LICENSES/DINOv2-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`. Dataset terms remain separate. The GSV-Cities images are not included in the ZIP; the manifest contains relative paths to the supplied archive. Pilot checkpoints are included and must not be presented as full paper-reproduction weights.

```bibtex
@InProceedings{Tang_2026_CVPR,
  author    = {Tang, Wenjing and Yang, Chuanguang and An, Zhulin and Huang, Libo and Xu, Yongjun},
  title     = {EfficientVPR: Toward Efficient Visual Place Recognition via Scene-Aware Prompt Tuning and Adaptive Feature Enhancement},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

