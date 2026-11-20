"""Canonical CSV manifest loading and place-balanced training batches."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path: str | Path, *, require_place_id: bool = True) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path"} | ({"place_id"} if require_place_id else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    for index, row in enumerate(rows, start=2):
        if not row.get("image_path") or (require_place_id and not row.get("place_id")):
            raise ValueError(f"Manifest row {index} is missing a required field")
    return rows


class ImageManifestDataset(Dataset):
    """Read canonical rows with ``image_path`` and ``place_id`` columns."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path = ".",
        image_size: int = 266,
        *,
        split: str | None = None,
        augment: bool = False,
        require_place_id: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.data_root = Path(data_root).resolve()
        rows = read_manifest(self.manifest_path, require_place_id=require_place_id)
        if split is not None:
            if not all("split" in row for row in rows):
                raise ValueError("A split was requested but manifest has no split column")
            rows = [row for row in rows if row.get("split", "").lower() == split.lower()]
        if not rows:
            raise ValueError(f"No manifest rows for split={split!r}")
        self.rows = rows
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def resolved_path(self, index: int) -> Path:
        raw = Path(self.rows[index]["image_path"])
        return raw if raw.is_absolute() else (self.data_root / raw).resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.resolved_path(index)
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            if self.augment and random.random() < 0.5:
                image = ImageOps.mirror(image)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
        std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
        return {
            "image": (tensor - mean) / std,
            "place_id": str(row.get("place_id", "")),
            "path": str(path),
            "row": row,
        }


class PlaceBatchSampler(Sampler[list[int]]):
    """Sample P places and K images per place, as reported in the paper."""

    def __init__(
        self,
        dataset: ImageManifestDataset,
        places_per_batch: int,
        images_per_place: int,
        *,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.places_per_batch = int(places_per_batch)
        self.images_per_place = int(images_per_place)
        self.seed = int(seed)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            groups[str(row["place_id"])].append(index)
        self.groups = {key: indices for key, indices in groups.items() if len(indices) >= self.images_per_place}
        if len(self.groups) < self.places_per_batch:
            raise ValueError(
                f"Need {self.places_per_batch} places with at least {self.images_per_place} images; "
                f"found {len(self.groups)} eligible places"
            )
        self.batches_per_epoch = batches_per_epoch or max(1, len(self.groups) // self.places_per_batch)

    def __len__(self) -> int:
        return int(self.batches_per_epoch)

    def __iter__(self):
        rng = random.Random(self.seed)
        place_ids = list(self.groups)
        for _ in range(len(self)):
            chosen_places = rng.sample(place_ids, self.places_per_batch)
            batch: list[int] = []
            for place_id in chosen_places:
                batch.extend(rng.sample(self.groups[place_id], self.images_per_place))
            yield batch

