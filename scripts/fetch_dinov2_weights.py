"""Verify (or download) the official DINOv2 ViT-S/14 no-register weights.

The file ships inside the project ZIP at checkpoints/dinov2_vits14_pretrain.pth. If it is missing or
corrupt, this script downloads it from Meta's official URL and checks the SHA-256 recorded in
THIRD_PARTY_NOTICES.md. Run:  python scripts/fetch_dinov2_weights.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
TARGET = Path(__file__).resolve().parents[1] / "checkpoints" / "dinov2_vits14_pretrain.pth"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if TARGET.is_file() and digest(TARGET) == SHA256:
        print(f"OK: {TARGET} matches the expected SHA-256.")
        return 0
    print(f"Weights missing or checksum mismatch; downloading {URL}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix(".part")
    urllib.request.urlretrieve(URL, partial)
    if digest(partial) != SHA256:
        partial.unlink(missing_ok=True)
        print("Downloaded file does not match the expected SHA-256; refusing to use it.", file=sys.stderr)
        return 1
    partial.replace(TARGET)
    print(f"Downloaded and verified {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
