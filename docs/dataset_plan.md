# Dataset acquisition and validation plan

The selected training set is GSV-Cities at `D:\archive`. Its 23 city CSVs were joined against their image folders on 2026-10-29; all 529,506 annotation rows matched a readable image. The generated manifest and place-disjoint split are in `data/gsv_cities/`.

## Current access state

| Dataset | Intended role / paper split | Verified access state | Required user action |
|---|---|---|---|
| GSV-Cities selected release | Training | User-supplied archive at `D:\archive`; 23 city CSVs, 529,506 matched/readable images, 64,394 city-scoped places. 1,880 places have fewer than four images. | Dataset access is resolved. Continue to honor its terms; use other datasets for the official paper benchmarks. |
| Pitts250k | Benchmark `test` | VPR-datasets-downloader says to request permission from dataset authors. | Obtain permission and authorized files. |
| MSLS | `val` | Official Mapillary portal requires login and explicit acceptance of dataset terms and confidentiality requirements. | User must personally review/accept the terms and download. |
| AmsterTime | Single-ground-truth retrieval | Official project describes 1,231 verified pairs; Kaggle listing observed as CC0; download was not attempted. | Obtain the current files and preserve official `image_pairs.csv`/leave-one-out behavior. |
| Eynsham | Paper benchmark | VPR-datasets-downloader documents a public rsync procedure. Dataset license and exact split still need checking. | Review terms, download through documented source procedure, validate. |
| SVOX Night / Overcast / Snow | Three paper conditions | VPR-datasets-downloader documents a helper script. Dataset terms/splits need checking. | Review terms, use source procedure, validate. |

The user selected the 530k-image / 62k-place GSV-Cities release. The official project describes approximately 530k images and more than 62k places; the supplied archive contains 529,506 images and 64,394 unique city-scoped place IDs. The EfficientVPR supplement reports 570,000 training images from 67,000 locations, so experiments on this archive must record a data-version difference and must not be described as an exact data reproduction of the supplement. The archive has 1,880 places with fewer than four annotated images, despite the project description's stated minimum; these groups are retained in the manifest and excluded by the K=4 sampler.

The [VPR Datasets Downloader](https://github.com/gmberton/VPR-datasets-downloader) is for the benchmark datasets and formats listed in its README; its current dataset list does not include GSV-Cities. It is not the GSV-Cities training-data download source. Get GSV-Cities separately from its [project page](https://github.com/amaralibey/gsv-cities) / [dataset listing](https://www.kaggle.com/datasets/amaralibey/gsv-cities).

See [source links](source_links.md) for full source URLs and exact observed access requirements. This project does not accept portal terms, use credentials, request author permission, or bypass gating.

## Canonical manifest contracts

Training manifest CSV:

```csv
image_path,place_id,split
images/example.jpg,place-001,train
```

`image_path` can be absolute or relative to `data_root`. The builder emits relative paths rooted at `D:\archive`, uses `city_id:place_id` to avoid cross-city label collisions, and preserves `source_panoid` and `city_id` as provenance columns. Rows need at least four images per place for the paper's K=4 sampler. If `split` exists, training selects rows labelled `train`.

Generic benchmark manifest:

```csv
image_path,split,place_id,latitude,longitude,ground_truth_path
images/query.jpg,query,place-001,52.0,4.0,images/reference.jpg
images/reference.jpg,gallery,place-001,52.0,4.0,
```

Use `place_id` only when that is the benchmark's official ground truth. For radius-based data, supply valid `latitude,longitude`; evaluation uses the documented 25 m default. For AmsterTime, use official explicit query/reference pair fields and single-ground-truth matching. The generic evaluator does not reconstruct the official AmsterTime leave-one-out gallery: prepare the manifest from official project files and verify pair semantics first. Official benchmark split files are authoritative.

## Commands

```powershell
python scripts/prepare_gsv_manifest.py --root D:\archive --verify-images
python scripts/split_gsv_manifest.py --manifest data/gsv_cities/manifest.csv --eval-place-limit 512 --seed 42
python scripts/validate_datasets.py --dataset gsv_cities --root D:\archive --manifest data/gsv_cities/manifest.csv
```

`prepare_gsv_manifest.py --verify-images` checks every referenced image. `split_gsv_manifest.py` assigns entire places to train or validation and prepares a 512-place internal retrieval manifest. This is an internal protocol, not one of the paper's official benchmark splits. `scripts/download_datasets.py --dataset gsv_cities` remains informational and does not download data.

## Preservation and provenance

Keep untouched source archives separate from manifests/derived outputs. Record source URL, download date, version, source license, any acceptance action, number of files/places, split, image checks, and conversion command alongside each local dataset. Large data and weights are excluded from Git.

