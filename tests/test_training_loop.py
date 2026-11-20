from __future__ import annotations

import torch
from torch import nn

from caff_efficientvpr.losses import MultiSimilarityLoss
from caff_efficientvpr.training import _batch_loss


class CountingSGD(torch.optim.SGD):
    steps = 0

    def step(self, *args, **kwargs):
        type(self).steps += 1
        return super().step(*args, **kwargs)


class Tiny(nn.Module):
    use_caff = False

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 4)

    def forward(self, images):
        return nn.functional.normalize(self.fc(images.flatten(1)), dim=-1)


def _loader(n_batches):
    for i in range(n_batches):
        yield {"images": torch.randn(4, 6), "place_ids": ["a", "a", "b", "b"]}


class _Scaler:  # plain (non-AMP) stand-in with the GradScaler interface
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def test_gradient_accumulation_counts_updates_and_flushes_partial_group():
    torch.manual_seed(0)
    model = Tiny()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    CountingSGD.steps = 0
    _batch_loss(model, _loader(4), MultiSimilarityLoss(1.0, 50.0, 0.0, 0.1), torch.device("cpu"),
                _Scaler(), optimizer, amp=False, max_steps=4, accum_steps=2)
    assert CountingSGD.steps == 2  # 4 micro-batches / accumulation 2
    CountingSGD.steps = 0
    _batch_loss(model, _loader(3), MultiSimilarityLoss(1.0, 50.0, 0.0, 0.1), torch.device("cpu"),
                _Scaler(), optimizer, amp=False, max_steps=3, accum_steps=2)
    assert CountingSGD.steps == 2  # one full group + a flushed partial group
