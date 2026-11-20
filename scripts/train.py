"""Train the baseline or CAFF using a local canonical CSV manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caff_efficientvpr.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke_test", "quick_experiment", "gpu_laptop", "gpu_laptop_caff", "accuracy_baseline", "accuracy_caff", "fair_arm", "long_run", "research_run", "caff_finetune"),
        required=True,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize from a compatible baseline checkpoint before CAFF fine-tuning")
    parser.add_argument("--log-gates", action="store_true")
    parser.add_argument("--lr-backbone", type=float, help="Override the config backbone learning rate for controlled sweeps")
    parser.add_argument("--checkpoint", type=Path, help="Write this run's checkpoint here instead of the config default")
    parser.add_argument("--data-root", type=Path, help="Override the config dataset root (for a mounted remote dataset)")
    parser.add_argument("--epochs", type=int, help="Override profile epochs")
    parser.add_argument("--updates-per-epoch", type=int, help="Override optimizer updates per epoch")
    parser.add_argument("--places-per-batch", type=int)
    parser.add_argument("--images-per-place", type=int)
    parser.add_argument("--accum-steps", type=int, help="Gradient accumulation: micro-batches per optimizer update")
    parser.add_argument("--snapshot-every", type=int, help="Also keep a model-only snapshot every N epochs")
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit("PyTorch is required. Install the matching official PyTorch build first.") from error
    from caff_efficientvpr.models import EfficientVPR
    from caff_efficientvpr.training import train

    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    if args.lr_backbone is not None:
        config["training"]["lr_backbone"] = args.lr_backbone
    if args.checkpoint is not None:
        config["training"]["checkpoint"] = str(args.checkpoint)
    if args.data_root is not None:
        config["data"]["data_root"] = str(args.data_root)
    overrides = {k: v for k, v in {
        "epochs": args.epochs, "max_steps": args.updates_per_epoch, "places_per_batch": args.places_per_batch,
        "images_per_place": args.images_per_place, "accum_steps": args.accum_steps, "snapshot_every": args.snapshot_every,
    }.items() if v is not None}
    if overrides:
        config["training"]["overrides"] = overrides
    model = EfficientVPR(config["model"])
    history = train(
        model,
        config,
        profile=args.profile,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        log_gates=args.log_gates,
    )
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

