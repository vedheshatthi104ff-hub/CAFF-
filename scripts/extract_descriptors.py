"""Extract and save descriptors from one manifest split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
except ImportError as error:
    raise SystemExit("PyTorch and NumPy are required. Install the project requirements first.") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from caff_efficientvpr.config import load_config
from caff_efficientvpr.data.manifest import ImageManifestDataset
from caff_efficientvpr.models import EfficientVPR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("query", "gallery", "train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = EfficientVPR(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    root = Path(cfg["data"].get("data_root", "data"))
    if not root.is_absolute():
        root = ROOT / root
    dataset = ImageManifestDataset(args.manifest, root, int(cfg["data"].get("test_image_size", 322)), split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    descriptors, paths = [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"]
            descriptors.append(model(images.to(device)).cpu().numpy())
            paths.extend(batch["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, descriptors=np.concatenate(descriptors), paths=np.asarray(paths))
    print(f"saved={args.output} count={len(paths)} dimension={descriptors[0].shape[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

