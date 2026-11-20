# Research audit

Audit date: 2026-10-19. This is a source audit and implementation plan, not a reproduction result. The official GitHub repository was rechecked on 2026-10-19 and still showed only `.gitignore` and `README.md`, with no releases listed.

Dataset decision and inspection update (2026-10-29): use the user-selected GSV-Cities 530k / 62k release. The supplied archive has 529,506 matched/readable images and 64,394 city-scoped place labels. It differs from the 570k / 67k counts in the EfficientVPR supplement; see [dataset plan](dataset_plan.md). The VPR Datasets Downloader does not list or supply GSV-Cities.

## Findings and claim labels

| Claim | Status | Evidence / qualification |
|---|---|---|
| EfficientVPR is described in the CVPR 2026 paper. | **Paper-specified** | Official CVF paper and its supplement were retrieved and inspected. See [source links](source_links.md). |
| Authors' GitHub project contains runnable model code, checkpoint, or release. | **Unverified / unavailable** | The live `WiniTang/EfficientVPR` repository showed only `README.md` and `.gitignore`, with three commits and no release artifacts on 2026-10-18. The README is only a title. |
| Newly authored files here are the authors' implementation. | **False; repository-verified** | They are an independent, paper-based reimplementation because no official implementation was found. |
| Backbone is DINOv2-Small /14. | **Paper-specified** | EfficientVPR paper; official DINOv2 Hub exposes `dinov2_vits14`. Standard model is loaded with official code and pretrained weights by default. |
| SceneVPT uses 32 prompts and a learnable threshold. | **Paper-specified** | Supplement reports 32 prompts as the selected setting and ablates 8/16/32/64; paper defines the learnable γ threshold and CLS-to-prompt attention filtering. |
| MsIA forms feature maps at scales 1, 2, and 4 and uses backbone tokens as Q, multi-scale tokens as K/V. | **Paper-specified** | Section 3.2.1 and Fig. 2. Exact conv padding and some block settings are absent. |
| OLE uses horizontal/vertical asymmetric convolutions and orthogonalization. | **Paper-specified** | Section 3.2.2. GeM + MLP summarizes vertical features, then the horizontal branch is projected orthogonally before fusion. |
| SGFS applies prompt self-attention then cross-attention over concatenated MsIA and OLE tokens. | **Paper-specified** | Section 3.2.3. Thus SGFS is already a sample-conditioned selection/fusion module; CAFF must be tested as a distinct output-level gate. |
| Reported descriptor dimension is 3456. | **Paper-specified** | Paper reports 3456. In the reported DINOv2-S width of 384, the paper's SGFS compression to 8 tokens plus CLS gives (8+1)×384=3456. |
| Training is GSV-Cities with MS loss (α=1, β=50, λ=0), Adam, 30 epochs maximum, 160 places × 4 images, 266-pixel train and 322-pixel test resize. | **Paper-specified** | Section 4.2. Main-parameter LR 2e-4 and pretrained fine-tuning LR 2e-2 decay by 0.8 every 5 epochs. Paper does not specify weight decay, exact augmentation, seed, or several optimizer details. |
| Official GSV code defines the exact EfficientVPR augmentation. | **Unverified** | EfficientVPR paper says it follows GSV-Cities training protocol but does not state exact augmentation. GSV reference code has its own transforms; they are not silently treated as exact EfficientVPR transforms. |
| Supplied GSV-Cities annotations match readable images. | **Dataset-verified** | On 2026-10-29, all 529,506 rows from 23 city CSVs matched a readable image under `D:\archive\Images`; the archive contains 64,394 city-scoped place labels. 1,880 labels have fewer than four images. |
| Supplied archive's exact counts equal the EfficientVPR supplement. | **Known version difference** | Supplied archive: 529,506 images / 64,394 city-scoped places. Supplement: 570k / 67k. Record the archive's observed counts in all experiment reports. |
| VPR Datasets Downloader supplies GSV-Cities training data. | **Repository-verified: no** | Its current README lists benchmark datasets and procedures but does not include GSV-Cities. Obtain the selected training archive from the GSV-Cities project/listing separately. |
| MSLS can be downloaded without user action. | **Unverified / unavailable** | Official Mapillary page requires login and explicit acceptance of dataset terms, including confidentiality conditions. The account holder must review and accept them. |
| Pitts250k can be fetched automatically. | **Unverified / unavailable** | VPR-datasets-downloader directs users to ask the dataset authors for permission for Pitts250k. No data downloaded. |
| AmsterTime is publicly listed. | **Repository-verified** | Official project identifies its evaluation code and 1,231 verified pairs; Kaggle listing was observed with a CC0 label and a download listing. No data downloaded. Paper's single-ground-truth evaluation needs to be preserved. |
| Eynsham and SVOX downloader instructions exist. | **Repository-verified** | VPR-datasets-downloader documents Eynsham rsync and an SVOX script. Dataset licenses/split compatibility still need validation after acquisition. |
| CAFF improves retrieval. | **Project-proposed hypothesis; pilot not supportive** | On the short matched internal GSV holdout, baseline R@1 was 0.5176 and CAFF R@1 was 0.3906. This small run does not establish the final outcome; see [experiment ledger](experiments.md). |
| CAFF's local branch can simply reuse baseline SGFS output. | **Inferred / adapted** | Baseline SGFS cross-attends to both MsIA and OLE, so it is not a local-only feature. In CAFF mode, the implementation reuses the SGFS weights with OLE tokens as K/V to form an OLE-conditioned local branch; the baseline path continues to use concatenated MsIA+OLE K/V unchanged. This adaptation is documented for ablation. |

## Architecture and protocol extracted

The paper's baseline is DINOv2-Small with SceneVPT, MsIA, OLE, and SGFS. SceneVPT inserts learnable prompt tokens and uses averaged CLS-to-prompt attention from the preceding layer to mix prior prompt outputs with the next layer's learned prompts. MsIA builds three scales (stride 1/2/4), concatenates them, and uses backbone features as multi-head attention queries. OLE uses 1×3 and 3×1 convolutions, GeM+MLP summary, orthogonal projection, concatenation, and 1×1 fusion. SGFS self-attends prompt features, cross-attends them to concatenated MsIA/OLE tokens, applies normalization and an FC compression to eight prompt features, then combines them with CLS.

The paper evaluates Pitts250k-test, MSLS-val, AmsterTime, Eynsham, and three SVOX conditions. It uses Recall@N; a 25 m matching radius applies to most datasets while AmsterTime uses single-ground-truth evaluation. Dataset splits must come from the official files / benchmark scripts, not be recreated by sorting or random splitting.

## Scope achieved in this workspace

This project implements the model, loader, loss, training/evaluation commands, and tests as code. A real forward pass and all execution stages that require Torch, weights, or licensed datasets remain unverified. See [experiments](experiments.md) and [limitations](limitations.md); all experimental rows remain `NOT RUN`.

