"""EfficientVPR baseline and opt-in CAFF extension."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import SceneVPT, load_dinov2_small
from .modules import (
    ConditionAwareAdaptiveFeatureFusion,
    MultiScaleInteractionAttention,
    OrientationAwareLocalEnhancer,
    SemanticGuidedFeatureSelection,
)


class EfficientVPR(nn.Module):
    """Paper-based baseline; ``use_caff`` selects an independent CAFF head.

    A supplied backbone is intended for small tests and research adapters. Normal
    use loads official pretrained DINOv2-S/14 weights through PyTorch Hub.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        use_caff: bool | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        self.model_config = cfg
        prompt_count = int(cfg.get("prompt_count", 32))
        pretrained = bool(cfg.get("pretrained", True))
        backbone = backbone if backbone is not None else load_dinov2_small(
            pretrained=pretrained,
            checkpoint_path=cfg.get("pretrained_checkpoint"),
        )
        self.scene_vpt = SceneVPT(
            backbone,
            prompt_count=prompt_count,
            init_std=float(cfg.get("scene_vpt_init_std", 0.02)),
            threshold_init=cfg.get("scene_vpt_threshold_init"),
        )
        channels = self.scene_vpt.embed_dim
        self.msia = MultiScaleInteractionAttention(channels, int(cfg.get("msia_heads", 6)))
        self.ole = OrientationAwareLocalEnhancer(
            channels,
            float(cfg.get("ole_gem_p", 3.0)),
            fusion_mode=str(cfg.get("ole_fusion", "orthogonal")),
        )
        self.sgfs = SemanticGuidedFeatureSelection(
            channels,
            prompt_count=prompt_count,
            compressed_tokens=int(cfg.get("sgfs_compressed_tokens", 8)),
            heads=int(cfg.get("sgfs_heads", cfg.get("msia_heads", 6))),
        )
        self.descriptor_dim = int(cfg.get("descriptor_dim", 3456))
        expected_dim = (self.sgfs.compressed_tokens + 1) * channels
        if self.descriptor_dim != expected_dim:
            raise ValueError(
                f"Descriptor is (compressed tokens + CLS) x channels = {expected_dim}; "
                f"config requested {self.descriptor_dim}."
            )
        if use_caff is None:
            use_caff = bool(cfg.get("use_caff", False))
        self.use_caff = bool(use_caff)
        self.caff = None
        if self.use_caff:
            self.caff = ConditionAwareAdaptiveFeatureFusion(
                channels,
                hidden_dim=int(cfg.get("caff_hidden_dim", 128)),
                dropout=float(cfg.get("caff_dropout", 0.0)),
            )
            self.caff.gate_mode = str(cfg.get("caff_gate_mode", "learned"))
        for parameter in self.scene_vpt.backbone.parameters():
            parameter.requires_grad = bool(cfg.get("backbone_trainable", True))

    def forward(self, images: torch.Tensor, *, return_aux: bool = False):
        backbone_features = self.scene_vpt(images)
        cls = backbone_features["cls"]
        msia_tokens = self.msia(backbone_features["patches"])
        ole_tokens = self.ole(backbone_features["patches"])
        prompts = backbone_features["prompts"]

        gate_weights = None
        if self.use_caff:
            # CAFF only rescales the two streams; SGFS input and descriptor layout
            # are identical to the baseline.
            assert self.caff is not None
            msia_tokens, ole_tokens, gate_weights = self.caff(cls, msia_tokens, ole_tokens)
        fused_tokens = self.sgfs(prompts, torch.cat((msia_tokens, ole_tokens), dim=1))
        descriptor = F.normalize(torch.cat((cls.unsqueeze(1), fused_tokens), dim=1).flatten(1), p=2, dim=-1)
        if return_aux:
            return descriptor, {"gate_weights": gate_weights, "msia_tokens": msia_tokens, "ole_tokens": ole_tokens}
        return descriptor

    def set_gate_mode(self, mode: str) -> None:
        if mode not in ConditionAwareAdaptiveFeatureFusion.GATE_MODES:
            raise ValueError("gate mode must be 'learned', 'static' or 'equal'")
        if self.caff is None:
            raise RuntimeError("This model was constructed with use_caff=False")
        self.caff.gate_mode = mode

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable}

