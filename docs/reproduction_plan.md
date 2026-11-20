# Reproduction plan

## Paper-reported settings

| Setting | Paper value | Implementation status |
|---|---:|---|
| Backbone | DINOv2-Small | Official pretrained DINOv2-S/14 Hub loader executed on CUDA for the pilot. |
| Training set | GSV-Cities selected release | Supplied at `D:\archive`; 529,506 images / 64,394 city-scoped places. A reproducible 80/20 place-disjoint split is prepared; this count differs from the supplement's 570k/67k. |
| Loss | Multi-Similarity, α=1, β=50, λ=0 | Implemented; mining margin 0.1 is an inferred default because the paper does not report it. |
| Optimizer | Adam | Implemented with framework defaults; betas/epsilon/weight decay are not reported by the paper. Weight decay 1e-4 is an inferred config default. |
| Main/pretrained LR | 2e-4 / 2e-2 | Configured as transcribed from the paper. This is a 100× ratio; verify against author code if it becomes available before a full run. |
| Schedule | ×0.8 every 5 epochs; up to 30 epochs | Implemented with StepLR; full schedule not run. Pilot used 1 epoch / 25 updates. |
| Training batch | 160 places × 4 images; 266×266 | Implemented place-balanced sampler/config. Pilot used 2 places × 2 images. Full 640-image batch not attempted; 6 GB GPU is smaller than paper's A800. |
| Test resize | 322×322 | Evaluator ran at 322×322 on an internal place-disjoint GSV holdout; official benchmark evaluation remains unrun. |
| Augmentation | Not fully specified by EfficientVPR | Default disabled. Optional horizontal flip is a project choice, not claimed as paper protocol. |
| Evaluation | Recall@K; mostly 25 m, AmsterTime single ground truth | Generic evaluator supports place ID, GPS radius, and explicit pair path. Official split/pair conversion remains unverified. |

## Run profiles

- `smoke_test`: 1 epoch, 2 places × 2 images, up to 2 batches, frozen pretrained DINO body. Checks only a few forward/backward/checkpoint steps if dependencies and data exist; it is not a paper reproduction.
- `quick_experiment`: 2 epochs, 8 places × 4 images, at most 50 batches/epoch. Development experiment only.
- `research_run`: config's paper-reported 30 epochs, 160 × 4, 266 input. Requires external storage and GPU memory profiling; it has not been run and should not be launched directly on the detected 6 GB GPU.
- `caff_finetune`: development profile, loads a baseline checkpoint with `--init-checkpoint`, freezes the full baseline, and trains the CAFF head only by default. It refuses to run without a compatible baseline initialization or CAFF resume checkpoint. This is a proposed staged comparison, not a paper setting.

Every reduced setting is a deviation and must be recorded with results. The script saves model/optimizer/scheduler state, epoch record, profile, and config at the configured checkpoint path, with a JSONL epoch log.

## Evaluation and ablations

1. Baseline SGFS descriptor (3456-D).
2. CAFF fixed equal gates (0.5/0.5), same branches and descriptor size.
3. CAFF learned image-conditioned gates.
4. CAFF with CLS gating disabled only after defining a valid controlled alternative (e.g. learned constant input); not implemented yet.
5. Global-only / local-only branch control only if same output projection and training protocol are retained.

### Supplement-specific ablations

- Prompt count: configs `prompt_count_08.toml`, `prompt_count_16.toml`, `prompt_count_32.toml`, and `prompt_count_64.toml`; use 266×266 evaluation for Pitts250k-test, AmsterTime, and SVOX-Overcast as in Supplement Table 11.
- OLE fusion: `ole_orthogonal.toml` selects paper OLE and `ole_concat.toml` selects direct feature concatenation. Use 266×266 evaluation on the same three datasets as in Supplement Table 12.
- Cross-scale Tables 9–10 require PCA projections to 3456-D and 2048-D, plus the compared methods' official code/checkpoints. PCA and competitor integrations are not part of the CAFF core implementation.

Report Recall@1/5/10 per official dataset split, descriptor extraction time separately for query/gallery, descriptor size, actual parameter counts, device, and GPU peak allocation. The generic search-time field is currently `null`; it is not measured separately. For latency comparisons, use the same device, resolution, checkpoint, batch size, warmup, and repetitions, then add repeated timing before claiming speed differences.

## Environment and current execution

- Python 3.12.14 available in the app runtime; Pillow, NumPy, CPU PyTorch 2.14.1, torchvision 0.29.1, and pytest 9.1.1 were available in the temporary work environment for verification. The project uses TOML and does not require PyYAML.
- The unit suite passed (10 tests). The official DINOv2-S/14 checkpoint loaded; baseline, CAFF, OLE-concat, and 64-prompt paths each produced finite `(1, 3456)` descriptors for 266×266 random inputs on CPU.
- A synthetic smoke run completed one CPU epoch on four generated images; it saved and reloaded a temporary checkpoint. This validates the training path only, not learning quality or benchmark performance.
- NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB total memory reported; roughly 5.9 GiB free at inspection. Paper used a single NVIDIA A800.
- No GSV-Cities or benchmark data and no trained EfficientVPR/CAFF checkpoint are present.
- The official DINOv2-S/14 checkpoint is bundled with its Apache-2.0 license notice. PyTorch Hub still needs network access on first use to fetch the architecture source code.

No GSV-Cities training, real-image descriptor extraction, benchmark metric, latency, or GPU memory measurement was executed. See [experiment ledger](experiments.md).

