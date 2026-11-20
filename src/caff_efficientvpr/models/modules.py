"""MsIA, OLE and SGFS modules reconstructed from the EfficientVPR paper."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleInteractionAttention(nn.Module):
    """Build stride 1/2/4 feature tokens and attend from backbone patch tokens."""

    def __init__(self, channels: int, heads: int = 6) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.scales = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
            for stride in (1, 2, 4)
        )
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch, count, channels = patch_tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"MsIA currently expects a square patch grid, got {count} tokens")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, channels, side, side)
        multi_scale = [
            layer(feature_map).flatten(2).transpose(1, 2)
            for layer in self.scales
        ]
        key_value = torch.cat(multi_scale, dim=1)
        attended, _ = self.attention(patch_tokens, key_value, key_value, need_weights=False)
        # Residual + LayerNorm is inferred from the architecture diagram.
        return self.norm(patch_tokens + attended)


class OrientationAwareLocalEnhancer(nn.Module):
    """Asymmetric local filters with the paper OLE or its concat ablation."""

    def __init__(
        self,
        channels: int,
        gem_p: float = 3.0,
        eps: float = 1e-6,
        fusion_mode: str = "orthogonal",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"orthogonal", "concat"}:
            raise ValueError("OLE fusion_mode must be 'orthogonal' or 'concat'")
        self.horizontal = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.vertical = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.vertical_mlp = (
            nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
            if fusion_mode == "orthogonal"
            else None
        )
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gem_p = float(gem_p)
        self.eps = float(eps)
        self.fusion_mode = fusion_mode

    @staticmethod
    def orthogonalize(horizontal: torch.Tensor, guide: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Remove the projection of each horizontal location onto its guide."""
        guide_map = guide[:, :, None, None]
        denominator = guide.square().sum(dim=1, keepdim=True).clamp_min(eps)[:, :, None, None]
        coefficient = (horizontal * guide_map).sum(dim=1, keepdim=True) / denominator
        return horizontal - coefficient * guide_map

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch, count, channels = patch_tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"OLE currently expects a square patch grid, got {count} tokens")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, channels, side, side)
        horizontal = self.horizontal(feature_map)
        vertical = self.vertical(feature_map)
        if self.fusion_mode == "orthogonal":
            assert self.vertical_mlp is not None
            gem = vertical.clamp_min(self.eps).pow(self.gem_p).mean(dim=(-2, -1)).pow(1.0 / self.gem_p)
            vertical_summary = self.vertical_mlp(gem)
            horizontal = self.orthogonalize(horizontal, vertical_summary, self.eps)
        fused = self.fuse(torch.cat((horizontal, vertical), dim=1))
        return fused.flatten(2).transpose(1, 2)


class SemanticGuidedFeatureSelection(nn.Module):
    """Prompt self-attention, local cross-attention and prompt-axis compression."""

    def __init__(self, channels: int, prompt_count: int = 32, compressed_tokens: int = 8, heads: int = 6) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.prompt_self_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.prompt_norm = nn.LayerNorm(channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.prompt_count = int(prompt_count)
        self.compressed_tokens = int(compressed_tokens)
        self.compress_prompt_axis = nn.Linear(self.prompt_count, self.compressed_tokens)

    def forward(self, prompts: torch.Tensor, local_tokens: torch.Tensor) -> torch.Tensor:
        refined, _ = self.prompt_self_attention(prompts, prompts, prompts, need_weights=False)
        refined = self.prompt_norm(prompts + refined)
        selected, _ = self.cross_attention(refined, local_tokens, local_tokens, need_weights=False)
        selected = self.cross_norm(selected)
        return self.compress_prompt_axis(selected.transpose(1, 2)).transpose(1, 2)


class ConditionAwareAdaptiveFeatureFusion(nn.Module):
    """Image-conditioned gate over the global (MsIA) and local (OLE) token streams.

    The gate sits *inside* the baseline descriptor path: it only rescales the two
    token streams that SGFS consumes, so the CLS token, the SGFS compression and the
    (compressed tokens + CLS) x channels descriptor layout stay exactly as in the
    baseline. The last gate layer is zero-initialised, so softmax gives 0.5/0.5 and
    the scale ``2 * w`` is 1.0 for both streams: at initialisation CAFF is identical
    to the baseline, and training can only move away from it if that helps.

    Gate modes: ``learned`` (per-image), ``static`` (two learned image-independent
    weights, the control that isolates the benefit of conditioning), ``equal``
    (fixed 0.5/0.5, a no-op equal to the baseline).
    """

    GATE_MODES = ("learned", "static", "equal")

    def __init__(self, channels: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.condition_norm = nn.LayerNorm(3 * channels)
        self.gate_network = nn.Sequential(
            nn.Linear(3 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)
        self.static_logits = nn.Parameter(torch.zeros(2))
        self.gate_mode = "learned"

    def gate_weights(
        self, cls: torch.Tensor, global_tokens: torch.Tensor, local_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.gate_mode == "equal":
            return cls.new_full((cls.shape[0], 2), 0.5)
        if self.gate_mode == "static":
            return self.static_logits.softmax(dim=-1).unsqueeze(0).expand(cls.shape[0], -1)
        if self.gate_mode == "learned":
            condition = torch.cat((cls, global_tokens.mean(dim=1), local_tokens.mean(dim=1)), dim=-1)
            return self.gate_network(self.condition_norm(condition)).softmax(dim=-1)
        raise ValueError(f"Unsupported CAFF gate mode: {self.gate_mode}")

    def forward(
        self,
        cls: torch.Tensor,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.gate_weights(cls, global_tokens, local_tokens)
        scale = 2.0 * weights  # 1.0 for both streams at initialisation
        return (
            global_tokens * scale[:, 0].view(-1, 1, 1),
            local_tokens * scale[:, 1].view(-1, 1, 1),
            weights,
        )
