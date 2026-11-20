"""Extract descriptors and evaluate Recall@K on a canonical benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as error:  # Keep the dependency blocker actionable.
    raise SystemExit("PyTorch is required. Install the matching official PyTorch build first.") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caff_efficientvpr.config import load_config
from caff_efficientvpr.data.manifest import ImageManifestDataset
from caff_efficientvpr.models import EfficientVPR
from caff_efficientvpr.retrieval import recall_at_k
from caff_efficientvpr.training import sha256_file


def _collate(items):
    return torch.stack([item["image"] for item in items]), [item["row"] for item in items], [item["path"] for item in items]


def _extract(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)
    model.eval()
    embeddings, rows = [], []
    with torch.inference_mode():
        # Warm-up is excluded from reported extraction timing.
        first = next(iter(loader), None)
        if first is None:
            raise ValueError("Dataset split is empty")
        warmup = first[0].to(device)
        model(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        count = 0
        for images, batch_rows, paths in loader:
            images = images.to(device, non_blocking=True)
            output = model(images).float()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            embeddings.append(output.cpu())
            for row, path in zip(batch_rows, paths):
                row = dict(row)
                row["image_path"] = path
                row["_resolved_path"] = path
                gt = row.get("ground_truth_path", "")
                if gt:
                    gt_path = Path(gt)
                    if not gt_path.is_absolute():
                        gt_path = dataset.data_root / gt_path
                    row["_ground_truth_resolved"] = str(gt_path.resolve())
                rows.append(row)
            count += len(images)
        elapsed = time.perf_counter() - started
    return torch.cat(embeddings), rows, elapsed, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm-name", help="Label written into the result record (e.g. caff_learned)")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--data-root", type=Path, help="Override the config dataset root (for a mounted remote dataset)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["data_root"] = str(args.data_root)
    model = EfficientVPR(config["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # Refuse to label a checkpoint with a config that does not match how it was trained.
    check_keys = ("use_caff", "caff_gate_mode", "prompt_count", "ole_fusion", "descriptor_dim", "caff_hidden_dim")
    trained_model_cfg = (checkpoint.get("config") or {}).get("model", {})
    mismatches = {
        key: (trained_model_cfg.get(key), config["model"].get(key))
        for key in check_keys
        if key in trained_model_cfg and trained_model_cfg.get(key) != config["model"].get(key)
    }
    if mismatches and not args.allow_config_mismatch:
        raise SystemExit(f"--config does not match the config this checkpoint was trained with: {mismatches}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    root = Path(config["data"].get("data_root", "data"))
    if not root.is_absolute():
        root = ROOT / root
    common = dict(image_size=int(config["data"].get("test_image_size", 322)), augment=False)
    query_set = ImageManifestDataset(args.manifest, root, split="query", require_place_id=False, **common)
    gallery_set = ImageManifestDataset(args.manifest, root, split="gallery", require_place_id=False, **common)
    query_embeddings, query_rows, query_seconds, query_count = _extract(
        model, query_set, device, args.batch_size, int(config["data"].get("num_workers", 0))
    )
    gallery_embeddings, gallery_rows, gallery_seconds, gallery_count = _extract(
        model, gallery_set, device, args.batch_size, int(config["data"].get("num_workers", 0))
    )
    metrics = recall_at_k(query_embeddings, gallery_embeddings, query_rows, gallery_rows, (1, 5, 10), args.radius_m)
    counts = model.parameter_counts()  # eval-time view: NOT the number of parameters that were trained
    train_record = checkpoint.get("record") or {}
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "CAFF" if model.use_caff else "baseline",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "descriptor_dim": int(query_embeddings.shape[1]),
        "arm": args.arm_name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_profile": checkpoint.get("profile"),
        "checkpoint_epochs_completed": checkpoint.get("epoch"),
        "trained_parameter_counts": train_record.get("parameter_counts"),
        "trained_parameter_breakdown": train_record.get("trainable_breakdown"),
        "training_config_path": train_record.get("config_path"),
        "training_init_checkpoint_sha256": train_record.get("init_checkpoint_sha256"),
        "caff_gate_mode": model.caff.gate_mode if model.use_caff else None,
        "evaluation_config_matches_training": not mismatches,
        "model_parameter_counts_total": counts["total"],
        "parameter_counts": counts,  # kept for compatibility; 'trainable' here is NOT the trained count
        "query_metrics": metrics,
        "query_extraction_ms_per_image": query_seconds * 1000 / max(1, query_count),
        "gallery_extraction_ms_per_image": gallery_seconds * 1000 / max(1, gallery_count),
        "descriptor_storage_bytes": int((query_embeddings.numel() + gallery_embeddings.numel()) * 4),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "database_search_time_seconds": None,
        "note": "Recall uses manifest place_id or GPS radius; explicit AmsterTime ground_truth_path rows use single-ground-truth matching.",
    }
    output_path = args.output or (ROOT / "results" / f"{record['model'].lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "descriptor_dim", "valid_queries", "R@1", "R@5", "R@10", "query_ms_per_image", "gallery_ms_per_image"])
        recalls = metrics["recall"]
        writer.writerow([record["model"], record["descriptor_dim"], metrics["valid_queries"], recalls["R@1"], recalls["R@5"], recalls["R@10"], record["query_extraction_ms_per_image"], record["gallery_extraction_ms_per_image"]])
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

