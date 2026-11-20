"""Reproducible training loop and checkpoint helpers."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.manifest import ImageManifestDataset, PlaceBatchSampler
from .losses import MultiSimilarityLoss
from .retrieval import recall_at_k


PROFILES: dict[str, dict[str, Any]] = {
    "smoke_test": {"epochs": 1, "places_per_batch": 2, "images_per_place": 2, "max_steps": 2, "freeze_backbone": True},
    "quick_experiment": {"epochs": 2, "places_per_batch": 8, "images_per_place": 4, "max_steps": 50},
    "gpu_laptop": {
        "epochs": 1,
        "places_per_batch": 2,
        "images_per_place": 2,
        "max_steps": 25,
        "lr_backbone": 0.00002,
    },
    "gpu_laptop_caff": {
        "epochs": 1,
        "places_per_batch": 2,
        "images_per_place": 2,
        "max_steps": 25,
        "freeze_base_for_caff": True,
    },
    # Accuracy-development runs use smaller batches than the paper, but train
    # for 500 updates and retain the paper's four views per place.
    "accuracy_baseline": {
        "epochs": 10,
        "places_per_batch": 4,
        "images_per_place": 4,
        "max_steps": 50,
    },
    "accuracy_caff": {
        "epochs": 10,
        "places_per_batch": 4,
        "images_per_place": 4,
        "max_steps": 50,
        "freeze_base_for_caff": True,
    },
    # Fair-comparison arm: identical budget for baseline and every CAFF variant.
    # Whether the base is frozen comes from the config (fair_* configs keep it trainable).
    "fair_arm": {"epochs": 10, "places_per_batch": 4, "images_per_place": 4, "max_steps": 50},
    # Longer run for real accuracy gains. Starting point only: override any value from the command line
    # (--epochs, --updates-per-epoch, --places-per-batch, --images-per-place, --accum-steps).
    "long_run": {"epochs": 20, "places_per_batch": 8, "images_per_place": 4, "max_steps": 250},
    "research_run": {"max_steps": None},
    "caff_finetune": {"max_steps": 50, "freeze_base_for_caff": True},
}


def sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_breakdown(model) -> dict[str, int]:
    groups = {"caff": 0, "backbone": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = "caff" if name.startswith("caff.") else "backbone" if name.startswith("scene_vpt.backbone.") else "other"
        groups[key] += parameter.numel()
    return groups


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _collate(items):
    return {
        "images": torch.stack([item["image"] for item in items]),
        "place_ids": [item["place_id"] for item in items],
        "paths": [item["path"] for item in items],
        "rows": [item["row"] for item in items],
    }


def _batch_loss(model, loader, loss_fn, device, scaler, optimizer=None, amp=False, max_steps=None, log_gates=False, accum_steps=1):
    """Run one pass. ``max_steps`` limits micro-batches; with ``accum_steps`` > 1 one optimizer
    update uses ``accum_steps`` micro-batches (loss is averaged over them)."""
    training = optimizer is not None
    accum_steps = max(1, int(accum_steps)) if training else 1
    model.train(training)
    losses: list[float] = []
    gate_values: list[torch.Tensor] = []
    pending = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["images"].to(device, non_blocking=True)
        label_ids = {label: index for index, label in enumerate(sorted(set(batch["place_ids"])))}
        labels = torch.tensor([label_ids[label] for label in batch["place_ids"]], device=device, dtype=torch.long)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if log_gates and model.use_caff:
                    descriptors, aux = model(images, return_aux=True)
                    gate_values.append(aux["gate_weights"].detach().float().cpu())
                else:
                    descriptors = model(images)
                loss = loss_fn(descriptors.float(), labels)
            if training:
                scaler.scale(loss / accum_steps).backward()
                pending += 1
                if pending == accum_steps:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
        losses.append(float(loss.detach().cpu()))
    if training and pending:  # flush a trailing partial group
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if not losses:
        raise RuntimeError("No training batches were produced; check the manifest and sampler.")
    gates = torch.cat(gate_values).mean(0).tolist() if gate_values else None
    return float(np.mean(losses)), gates


def _validation_loss(model, config, training_config, device, loss_fn, amp):
    manifest_value = config["data"].get("validation_manifest")
    if not manifest_value:
        return None
    root = Path(config["data"].get("data_root", "."))
    if not root.is_absolute():
        root = Path(config["_project_root"]) / root
    manifest = Path(manifest_value)
    if not manifest.is_absolute():
        manifest = Path(config["_project_root"]) / manifest
    dataset = ImageManifestDataset(
        manifest, root, int(config["data"].get("image_size", 266)), split="validation", augment=False
    )
    sampler = PlaceBatchSampler(
        dataset,
        int(training_config.get("places_per_batch", 8)),
        int(training_config.get("images_per_place", 4)),
        batches_per_epoch=min(10, max(1, len(dataset) // max(1, int(training_config.get("places_per_batch", 8)) * int(training_config.get("images_per_place", 4))))),
        seed=int(training_config.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=_collate)
    value, _ = _batch_loss(model, loader, loss_fn, device, None, optimizer=None, amp=amp, max_steps=10)
    return value


def train(
    model,
    config: dict[str, Any],
    *,
    profile: str,
    resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    log_gates: bool = False,
) -> list[dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}")
    if profile in {"caff_finetune", "gpu_laptop_caff"} and not model.use_caff:
        raise ValueError("The caff_finetune profile requires a model configured with use_caff=true")
    if profile in {"caff_finetune", "gpu_laptop_caff"} and not init_checkpoint and not resume:
        raise ValueError("CAFF fine-tuning requires --init-checkpoint from the baseline or --resume from CAFF")
    data_cfg = config["data"]
    train_cfg = dict(config["training"])
    train_cfg.update(PROFILES[profile])
    overrides = dict(train_cfg.get("overrides") or {})
    train_cfg.update(overrides)  # command-line overrides win over the profile
    accum_steps = max(1, int(train_cfg.get("accum_steps", 1)))
    data_root = Path(data_cfg.get("data_root", "."))
    manifest = Path(data_cfg["manifest"])
    project_root = Path(config["_project_root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not manifest.is_absolute():
        manifest = project_root / manifest
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)
    dataset = ImageManifestDataset(
        manifest,
        data_root,
        int(data_cfg.get("image_size", 266)),
        split="train" if "split" in read_csv_header(manifest) else None,
        augment=bool(data_cfg.get("augment", False)),
    )
    sampler = PlaceBatchSampler(
        dataset,
        int(train_cfg.get("places_per_batch", 160)),
        int(train_cfg.get("images_per_place", 4)),
        batches_per_epoch=(int(train_cfg["max_steps"]) * accum_steps if train_cfg.get("max_steps") is not None else None),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if resume and init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location=device, weights_only=False)
        initial_state = initial.get("model", initial)
        incompat = model.load_state_dict(initial_state, strict=False)
        if incompat.unexpected_keys or any(not key.startswith("caff.") for key in incompat.missing_keys):
            raise ValueError(
                "Initialization checkpoint is not a compatible baseline checkpoint: "
                f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
            )
        if not model.use_caff and incompat.missing_keys:
            raise ValueError("A baseline model must load all parameters from its initialization checkpoint")
    if train_cfg.get("freeze_base_for_caff") and model.use_caff:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("caff.")
    if train_cfg.get("freeze_backbone"):
        for parameter in model.scene_vpt.backbone.parameters():
            parameter.requires_grad = False

    backbone_params, main_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_params if name.startswith("scene_vpt.backbone.") else main_params).append(parameter)
    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": float(train_cfg.get("lr_main", 2e-4))})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(train_cfg.get("lr_backbone", 2e-2))})
    if not groups:
        raise ValueError("No trainable parameters remain for this profile")
    optimizer = torch.optim.Adam(groups, weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_decay_every", 5)),
        gamma=float(train_cfg.get("lr_decay_factor", 0.8)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"))
    loss_fn = MultiSimilarityLoss(
        float(train_cfg.get("ms_alpha", 1.0)),
        float(train_cfg.get("ms_beta", 50.0)),
        float(train_cfg.get("ms_lambda", 0.0)),
        float(train_cfg.get("ms_margin", 0.1)),
    )

    init_sha = sha256_file(init_checkpoint) if init_checkpoint else None
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])

    checkpoint_path = Path(train_cfg.get("checkpoint", "checkpoints/model.pt"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_path.with_suffix(".jsonl")
    history: list[dict[str, Any]] = []
    epochs = int(train_cfg.get("epochs", 1))
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    best_validation = float("inf")
    for epoch in range(start_epoch, epochs):
        sampler.seed = seed + epoch
        began = time.perf_counter()
        train_loss, gates = _batch_loss(
            model, loader, loss_fn, device, scaler, optimizer,
            amp=use_amp, log_gates=log_gates, accum_steps=accum_steps,
            max_steps=(int(train_cfg["max_steps"]) * accum_steps if train_cfg.get("max_steps") is not None else None),
        )
        validation_loss = _validation_loss(model, config, train_cfg, device, loss_fn, use_amp)
        scheduler.step()
        record = {
            "profile": profile,
            "epoch": epoch + 1,
            "updates_this_epoch": int(train_cfg.get("max_steps") or len(loader)),
            "images_per_micro_batch": int(train_cfg.get("places_per_batch", 160)) * int(train_cfg.get("images_per_place", 4)),
            "accum_steps": accum_steps,
            "images_per_update": int(train_cfg.get("places_per_batch", 160)) * int(train_cfg.get("images_per_place", 4)) * accum_steps,
            "overrides": overrides,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mean_gate_weights": gates,
            "epoch_seconds": time.perf_counter() - began,
            "device": str(device),
            "parameter_counts": model.parameter_counts(),
            "trainable_breakdown": trainable_breakdown(model),
            "caff_gate_mode": model.caff.gate_mode if model.use_caff else None,
            "config_path": config.get("_config_path"),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "init_checkpoint_sha256": init_sha,
            "seed": seed,
            "lr_main": float(train_cfg.get("lr_main", 2e-4)),
            "lr_backbone": float(train_cfg.get("lr_backbone", 2e-2)),
            "freeze_base_for_caff": bool(train_cfg.get("freeze_base_for_caff")) and model.use_caff,
            "status": "completed_epoch",
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch + 1,
            "config": config,
            "profile": profile,
            "record": record,
        }
        torch.save(payload, checkpoint_path)  # latest, fully resumable (overwritten each epoch)
        slim = {k: payload[k] for k in ("model", "epoch", "config", "profile", "record")}
        if validation_loss is not None and validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(slim, checkpoint_path.with_name(checkpoint_path.stem + ".best.pt"))
        snapshot_every = int(train_cfg.get("snapshot_every", 0))
        if snapshot_every and (epoch + 1) % snapshot_every == 0:
            torch.save(slim, checkpoint_path.with_name(f"{checkpoint_path.stem}.epoch{epoch + 1:03d}.pt"))
    return history


def read_csv_header(path: Path) -> set[str]:
    import csv
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return set(next(csv.reader(stream), []))

