from __future__ import annotations

import csv

import torch
from PIL import Image

from caff_efficientvpr.data.manifest import ImageManifestDataset, PlaceBatchSampler
from caff_efficientvpr.losses import MultiSimilarityLoss


def test_manifest_image_loading_and_place_sampler(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    for place in ("a", "b"):
        for index in range(2):
            name = f"{place}-{index}.jpg"
            Image.new("RGB", (20, 12), color=(index * 30, 40, 90)).save(root / name)
            rows.append((f"images/{name}", place))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "place_id"])
        writer.writerows(rows)
    dataset = ImageManifestDataset(manifest, tmp_path, image_size=16)
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    sampler = PlaceBatchSampler(dataset, places_per_batch=2, images_per_place=2, batches_per_epoch=1)
    batch = next(iter(sampler))
    labels = [dataset.rows[index]["place_id"] for index in batch]
    assert len(batch) == 4
    assert len(set(labels)) == 2
    assert all(labels.count(label) == 2 for label in set(labels))


def test_multi_similarity_loss_is_finite_and_differentiable():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = MultiSimilarityLoss(alpha=1, beta=50, lam=0)(embeddings, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()

