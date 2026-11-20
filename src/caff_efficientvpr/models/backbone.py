"""Official DINOv2-S/14 loading and a paper-based SceneVPT implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def load_dinov2_small(
    *,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """Load the official DINOv2-S/14 Hub model.

    Pretrained weights are required by default. Random initialization is not a
    valid substitute for the paper's pretrained backbone.
    """
    if not pretrained:
        raise ValueError("Random DINOv2 initialization is disabled for this project.")
    project_root = Path(__file__).resolve().parents[3]
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else (
        project_root / "checkpoints" / "dinov2_vits14_pretrain.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if checkpoint.is_file():
        # Load the official architecture from Meta's Hub repo, then apply the
        # separately distributed official no-register state dict locally.
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
            raise ValueError(f"Unexpected DINOv2 checkpoint structure: {checkpoint}")
        backbone.load_state_dict(state_dict, strict=True)
        return backbone
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)


def _flatten_blocks(blocks: nn.ModuleList) -> list[nn.Module]:
    flat: list[nn.Module] = []
    for item in blocks:
        if isinstance(item, (nn.ModuleList, nn.Sequential)):
            flat.extend(_flatten_blocks(item))
        elif not isinstance(item, nn.Identity):
            flat.append(item)
    return flat


class SceneVPT(nn.Module):
    """Deep visual prompt tuning with CLS-to-prompt adaptive filtering.

    The prompt threshold initialization is an implementation choice: 1/Np,
    which places a uniform initial attention score near the sigmoid midpoint.
    """

    def __init__(
        self,
        backbone: nn.Module,
        prompt_count: int = 32,
        init_std: float = 0.02,
        threshold_init: float | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.blocks = _flatten_blocks(backbone.blocks)
        if not self.blocks:
            raise ValueError("DINOv2 backbone exposes no transformer blocks.")
        if getattr(backbone, "num_register_tokens", 0):
            raise ValueError("SceneVPT expects the standard DINOv2-S/14 model without register tokens.")
        self.prompt_count = int(prompt_count)
        self.embed_dim = int(backbone.embed_dim)
        if self.prompt_count <= 0:
            raise ValueError("prompt_count must be positive")
        self.prompts = nn.Parameter(torch.randn(len(self.blocks), self.prompt_count, self.embed_dim) * init_std)
        if threshold_init is None:
            threshold_init = 1.0 / self.prompt_count
        self.gamma = nn.Parameter(torch.tensor(float(threshold_init)))

    def _cls_prompt_attention(self, block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Return head-averaged CLS-to-prompt attention from a block's QKV."""
        attn = block.attn
        normed = block.norm1(tokens)
        batch, seq_len, dim = normed.shape
        heads = int(getattr(attn, "num_heads", getattr(self.backbone, "num_heads", 0)))
        if not heads or dim % heads:
            raise ValueError("Cannot infer valid DINOv2 attention head dimensions.")
        head_dim = dim // heads
        qkv = attn.qkv(normed).reshape(batch, seq_len, 3, heads, head_dim)
        q, k = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2)
        scale = float(getattr(attn, "scale", head_dim**-0.5))
        logits = torch.matmul(q[:, :, :1], k.transpose(-2, -1)).squeeze(2) * scale
        weights = logits.softmax(dim=-1).mean(dim=1)
        return weights[:, 1 : 1 + self.prompt_count]

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.backbone.prepare_tokens_with_masks(images)
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError("Unexpected DINOv2 token tensor shape")

        previous_prompts: torch.Tensor | None = None
        previous_scores: torch.Tensor | None = None
        for layer_index, block in enumerate(self.blocks):
            if layer_index == 0:
                current_prompts = self.prompts[layer_index].unsqueeze(0).expand(images.shape[0], -1, -1)
                tokens = torch.cat((tokens[:, :1], current_prompts, tokens[:, 1:]), dim=1)
            else:
                assert previous_prompts is not None and previous_scores is not None
                scores = previous_scores / previous_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                alpha = torch.sigmoid(scores - self.gamma).unsqueeze(-1)
                new_prompts = self.prompts[layer_index].unsqueeze(0).expand_as(previous_prompts)
                current_prompts = alpha * previous_prompts + (1.0 - alpha) * new_prompts
                patch_offset = 1 + self.prompt_count
                tokens = torch.cat((tokens[:, :1], current_prompts, tokens[:, patch_offset:]), dim=1)

            scores = self._cls_prompt_attention(block, tokens)
            tokens = block(tokens)
            previous_prompts = tokens[:, 1 : 1 + self.prompt_count]
            previous_scores = scores

        normalized = self.backbone.norm(tokens)
        return {
            "cls": normalized[:, 0],
            "prompts": normalized[:, 1 : 1 + self.prompt_count],
            "patches": normalized[:, 1 + self.prompt_count :],
            "tokens": normalized,
        }

