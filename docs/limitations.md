# Limitations and blockers

## Current blockers

1. **Official EfficientVPR implementation is unavailable.** The author repo contained only a minimal README and `.gitignore`; no official code/checkpoint was found. This project is consequently a paper-based reimplementation.
2. **Only short pilot training has run.** The official no-register DINOv2-S/14 weights loaded on the RTX 4050, and one-epoch/25-update baseline and CAFF pilots produced checkpoints and internal retrieval metrics. This is far below the paper's 30-epoch, 640-image schedule. The CUDA runtime used for these pilots is temporary and is not included in the archive.
3. **GSV-Cities is present; official test benchmarks are still absent.** The supplied archive was fully matched to 529,506 readable images and split by place into training and validation. The 512-query retrieval sample is an internal check, not an official benchmark. MSLS requires user login/download, Pitts250k requires author permission per the VPR dataset downloader, and AmsterTime, Eynsham, and SVOX are also not present.
4. **GPU is smaller than the paper hardware.** The detected 6 GB RTX 4050 Laptop GPU is much smaller than the paper's A800. A 4-image batch completed, but the 640-image paper batch was not attempted. The one-epoch pilot settings are recorded in `docs/experiments.md` and the checkpoint logs.

## Scientific implementation uncertainties

- Official EfficientVPR module code is absent, so conv padding, residual placement, precise layer/head settings, GeM and MLP details, augmentation, weight decay, random seed, and some attention normalization choices cannot be source-verified. They are marked inferred in code/docs.
- The paper gives main LR 2e-4 and pretrained fine-tuning LR 2e-2. This unusual 100× ratio is transcribed; verify if official code appears before large training.
- The paper's own SGFS cross-attends prompts to concatenated MsIA and OLE tokens. CAFF uses the shared SGFS weights with OLE-only K/V for the proposed local branch. This is a documented architecture adaptation requiring ablations.
- CAFF arms must begin from a compatible baseline checkpoint (enforced by the training script). In the fair comparison every arm trains all parameters from the same start, so extra updates cannot be credited to the gate.
- The generic benchmark evaluator does not replace official dataset formatting or every benchmark's ground-truth implementation. In particular, AmsterTime's leave-one-out protocol must be validated from official files before its metrics are accepted.
- The only CAFF numbers so far (R@1 39.1% vs 51.8%) are from the removed v1 head after 25 updates (`results/legacy_v1/`). The current v2 CAFF has not been trained or evaluated; no conclusion about it is possible until `scripts/run_fair_comparison.sh` has been run and, ideally, official benchmarks are added.
- The supplemental prompt-count and OLE ablations now have configs and code paths, but have not been trained or evaluated because benchmark data is unavailable.

## Deferred demo

The Streamlit UI is deferred because no trained EfficientVPR/CAFF checkpoint or valid reference-image database exists. An interface now would not demonstrate a measured place-retrieval system. `app/README.md` records this stage gate.

