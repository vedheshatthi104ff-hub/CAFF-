"""Multi-Similarity loss used by the EfficientVPR training setup."""

from __future__ import annotations

import torch
from torch import nn


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity loss with paper-reported α/β/λ.

    The hard-mining margin (0.1 by default) is an implementation choice because
    the EfficientVPR paper does not state its value.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 50.0, lam: float = 0.0, margin: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        similarities = embeddings @ embeddings.transpose(0, 1)
        labels = labels.reshape(-1)
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
        positives = same & ~eye
        negatives = ~same
        anchor_losses = []
        valid_anchor_count = 0
        for index in range(labels.numel()):
            pos = similarities[index][positives[index]]
            neg = similarities[index][negatives[index]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            valid_anchor_count += 1
            hard_pos = pos[neg.max() + self.margin > pos]
            hard_neg = neg[neg > pos.min() - self.margin]
            if hard_pos.numel() == 0 and hard_neg.numel() == 0:
                continue
            positive_loss = (
                torch.log1p(torch.exp(-self.alpha * (hard_pos - self.lam)).sum()) / self.alpha
                if hard_pos.numel()
                else embeddings.new_zeros(())
            )
            negative_loss = (
                torch.log1p(torch.exp(self.beta * (hard_neg - self.lam)).sum()) / self.beta
                if hard_neg.numel()
                else embeddings.new_zeros(())
            )
            anchor_losses.append(positive_loss + negative_loss)
        if valid_anchor_count == 0:
            raise ValueError("MultiSimilarityLoss needs at least one anchor with positive and negative pairs")
        # Easy batches are valid: the mining rule selected no pairs, so their
        # loss is zero. Keep the scalar connected to the graph for backward().
        loss = torch.stack(anchor_losses).mean() if anchor_losses else embeddings.sum() * 0.0
        if not torch.isfinite(loss):
            raise FloatingPointError("MultiSimilarityLoss produced a non-finite value")
        return loss

