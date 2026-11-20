# Source links and access status

Checked 2026-10-29. Links below are primary project sources where available. Access and license terms can change; recheck before use.

| Resource | Link | Observed contents / status |
|---|---|---|
| EfficientVPR CVPR 2026 paper | [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.html) · [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.pdf) | Paper inspected, including figures, equations, training section, data, and evaluation. A local audit copy is in the task's `work/` folder, not part of the deliverable. |
| EfficientVPR supplement | [CVF supplement PDF](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tang_EfficientVPR_Toward_Efficient_CVPR_2026_supplemental.pdf) | Inspected; supports 32-prompt setting and model details. |
| Authors' EfficientVPR repository | [WiniTang/EfficientVPR](https://github.com/WiniTang/EfficientVPR) | Live repo checked; only `README.md` and `.gitignore`; no code, checkpoint, or release found. |
| Author profile | [WiniTang](https://github.com/WiniTang) | One public repository was relevant; no separate official implementation/checkpoint found. |
| DINOv2 official repository | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Official Hub exposes `dinov2_vits14` (without registers); official source uses DINOv2-S width 384. The repository licenses DINOv2 code and model weights under Apache-2.0. |
| DINOv2-S/14 checkpoint | [Official no-register checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | Downloaded, SHA-256 recorded in `THIRD_PARTY_NOTICES.md`, and included in `checkpoints/`. The register-token checkpoint is incompatible with this project. |
| DINOv2 Hub source | [hub/backbones.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py) · [vision transformer](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) | Primary references used for the DINOv2 interface. |
| GSV-Cities project | [GitHub](https://github.com/amaralibey/gsv-cities) | Describes approximately 530k images / more than 62k places, the CSV schema, full image filename convention, and reference code. The supplied archive was joined and fully checked on 2026-10-29: 529,506 readable images / 64,394 city-scoped places. |
| GSV-Cities dataset | [Kaggle](https://www.kaggle.com/datasets/amaralibey/gsv-cities) | The user supplied the selected archive at `D:\archive`; the project does not need to fetch it. The listing's data terms remain separate and must be honored. |
| VPR Datasets Downloader | [gmberton/VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) | MIT-licensed tooling / formats for its listed benchmark datasets. Its current README list does not include GSV-Cities; it is not the GSV training archive downloader. README directs Pitts250k users to ask authors, requires MSLS download through Mapillary, and provides Eynsham/SVOX procedures. Dataset licenses remain separate. |
| Pitts/NetVLAD source | [CVUT NetVLAD data page](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/) | VPR downloader requests permission from authors for Pitts variants; no download performed. |
| Mapillary Street-Level Sequences | [Official download page](https://www.mapillary.com/dataset/places) · [GitHub](https://github.com/mapillary/mapillary_sls) | Download requires account login, explicit acceptance of dataset license and Mapillary terms, and confidentiality obligations. User action is required; no acceptance performed. |
| AmsterTime | [Official project](https://github.com/seyrankhademi/AmsterTime) · [Kaggle listing](https://www.kaggle.com/datasets/byildiz/amstertime) | Project describes 1,231 verified pairs and retrieval protocol. Kaggle listing observed as CC0; no files downloaded. Preserve official leave-one-out/single-ground-truth behavior. |
| Eynsham / SVOX | [VPR Datasets Downloader](https://github.com/gmberton/VPR-datasets-downloader) | Eynsham has a public rsync command; SVOX has a downloader script. This project does not yet wrap those source scripts or assert the resulting split/license. |

## Access actions needed

- **GSV-Cities 530k / 62k (selected):** user needs to access the GSV-Cities source, review CC BY-NC-ND terms, and place the approximately 23.36 GB archive on storage with enough capacity. The VPR Datasets Downloader does not provide this training data. This version differs from the supplement's 570k / 67k counts.
- **MSLS:** user must log into Mapillary and accept the terms on the official page, then provide the downloaded archive locally.
- **Pitts250k:** ask the dataset authors for permission as instructed by the downloader, then place the authorized files locally.
- **AmsterTime:** download from the project/Kaggle page if desired; confirm current access terms and preserve the official benchmark pairing.
- **Eynsham/SVOX:** after checking current data terms and storage, use the source repository's procedure and validate the resulting files/splits locally.

