"""Generic Recall@K evaluation for canonical query/gallery manifests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def is_ground_truth(query: dict[str, Any], gallery: dict[str, Any], radius_m: float) -> bool:
    resolved = query.get("_ground_truth_resolved")
    gallery_path = gallery.get("_resolved_path")
    if resolved and gallery_path:
        return str(Path(str(resolved)).resolve()) == str(Path(str(gallery_path)).resolve())
    explicit = query.get("ground_truth_path", "")
    if explicit:
        return Path(str(explicit)).name == Path(str(gallery.get("image_path", ""))).name
    if query.get("place_id") and gallery.get("place_id"):
        return str(query["place_id"]) == str(gallery["place_id"])
    coordinates = ("latitude", "longitude")
    if all(query.get(key) not in (None, "") and gallery.get(key) not in (None, "") for key in coordinates):
        distance = _haversine_m(
            float(query["latitude"]), float(query["longitude"]),
            float(gallery["latitude"]), float(gallery["longitude"]),
        )
        return distance <= radius_m
    return False


def recall_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_rows: list[dict[str, Any]],
    gallery_rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5),
    radius_m: float = 25.0,
) -> dict[str, Any]:
    if query_embeddings.shape[0] != len(query_rows) or gallery_embeddings.shape[0] != len(gallery_rows):
        raise ValueError("Embedding count does not match manifest rows")
    if not query_rows or not gallery_rows:
        raise ValueError("Both query and gallery sets must be non-empty")
    q = torch.nn.functional.normalize(query_embeddings.float(), dim=-1)
    g = torch.nn.functional.normalize(gallery_embeddings.float(), dim=-1)
    similarities = q @ g.transpose(0, 1)
    recalls = {k: 0 for k in ks}
    valid_queries = 0
    for index, query in enumerate(query_rows):
        eligible = [
            j for j, row in enumerate(gallery_rows)
            if str(query.get("image_path", "")) != str(row.get("image_path", ""))
        ]
        matches = {j for j in eligible if is_ground_truth(query, gallery_rows[j], radius_m)}
        if not matches:
            continue
        valid_queries += 1
        if not eligible:
            continue
        eligible_tensor = torch.tensor(eligible, dtype=torch.long)
        top_count = min(max(ks), len(eligible))
        top_indices = eligible_tensor[torch.topk(similarities[index, eligible_tensor], k=top_count).indices].tolist()
        for k in ks:
            if any(j in matches for j in top_indices[:k]):
                recalls[k] += 1
    if valid_queries == 0:
        raise ValueError("No query has a matching gallery item; check the manifest ground-truth fields")
    return {
        "valid_queries": valid_queries,
        "total_queries": len(query_rows),
        "recall": {f"R@{k}": recalls[k] / valid_queries for k in ks},
    }

