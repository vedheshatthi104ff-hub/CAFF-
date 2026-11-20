from __future__ import annotations

import torch
from torch import nn

from caff_efficientvpr.models import EfficientVPR
from caff_efficientvpr.models.modules import OrientationAwareLocalEnhancer


class TinyAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, count, dim = x.shape
        qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = (q @ k.transpose(-2, -1) * self.scale).softmax(-1) @ v
        return self.proj(values.transpose(1, 2).reshape(batch, count, dim))


class TinyBlock(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 32
        self.num_heads = 4
        self.num_register_tokens = 0
        self.patch_embed = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.norm = nn.LayerNorm(32)

    def prepare_tokens_with_masks(self, images):
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        return torch.cat((self.cls_token.expand(images.shape[0], -1, -1), patches), dim=1)


def make_model(use_caff=False, prompt_count=4, ole_fusion="orthogonal"):
    return EfficientVPR(
        {
            "prompt_count": prompt_count,
            "scene_vpt_init_std": 0.02,
            "msia_heads": 4,
            "descriptor_dim": 288,
            "sgfs_compressed_tokens": 8,
            "ole_fusion": ole_fusion,
            "caff_hidden_dim": 12,
            "backbone_trainable": True,
            "use_caff": use_caff,
        },
        backbone=TinyBackbone(),
    )


def test_baseline_descriptor_shape_finite_and_normalized():
    model = make_model(use_caff=False).eval()
    descriptor = model(torch.randn(2, 3, 32, 32))
    assert descriptor.shape == (2, 288)
    assert torch.isfinite(descriptor).all()
    torch.testing.assert_close(descriptor.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_caff_gate_weights_sum_to_one_and_descriptor_shape():
    model = make_model(use_caff=True).eval()
    descriptor, auxiliary = model(torch.randn(3, 3, 32, 32), return_aux=True)
    gates = auxiliary["gate_weights"]
    assert descriptor.shape == (3, 288)
    assert torch.isfinite(descriptor).all()
    assert torch.all(gates >= 0)
    torch.testing.assert_close(gates.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(descriptor.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_equal_gate_control_and_caff_backpropagation():
    model = make_model(use_caff=True)
    model.set_gate_mode("equal")
    descriptor, auxiliary = model(torch.randn(2, 3, 32, 32), return_aux=True)
    torch.testing.assert_close(auxiliary["gate_weights"], torch.full((2, 2), 0.5))
    model.set_gate_mode("learned")
    model.zero_grad(set_to_none=True)
    descriptor = model(torch.randn(2, 3, 32, 32))
    # Zero-initialised last gate layer: only that layer gets gradient on step 0.
    (descriptor[:, :8] * torch.randn(2, 8)).sum().backward()
    assert model.caff.gate_network[-1].weight.grad is not None


def test_caff_matches_baseline_at_initialization():
    torch.manual_seed(0)
    baseline = make_model(use_caff=False).eval()
    caff = make_model(use_caff=True).eval()
    caff.load_state_dict(baseline.state_dict(), strict=False)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = baseline(images)
        for mode in ("learned", "static", "equal"):
            caff.set_gate_mode(mode)
            descriptor, aux = caff(images, return_aux=True)
            torch.testing.assert_close(descriptor, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(aux["gate_weights"], torch.full((2, 2), 0.5))


def test_static_gate_is_image_independent_and_trainable():
    model = make_model(use_caff=True)
    model.set_gate_mode("static")
    _, aux = model(torch.randn(3, 3, 32, 32), return_aux=True)
    torch.testing.assert_close(aux["gate_weights"][0], aux["gate_weights"][1])
    model(torch.randn(2, 3, 32, 32))[:, 0].sum().backward()
    assert model.caff.static_logits.grad is not None


def test_checkpoint_state_round_trip(tmp_path):
    model = make_model(use_caff=True)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    restored = make_model(use_caff=True)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_baseline_checkpoint_initializes_caff_without_missing_base_weights():
    baseline = make_model(use_caff=False)
    caff = make_model(use_caff=True)
    incompat = caff.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompat.unexpected_keys
    assert incompat.missing_keys
    assert all(key.startswith("caff.") for key in incompat.missing_keys)


def test_ole_orthogonal_projection_and_concat_ablation():
    horizontal = torch.randn(2, 8, 5, 5)
    guide = torch.randn(2, 8)
    projected = OrientationAwareLocalEnhancer.orthogonalize(horizontal, guide)
    residual_dot = (projected * guide[:, :, None, None]).sum(dim=1)
    torch.testing.assert_close(residual_dot, torch.zeros_like(residual_dot), atol=1e-5, rtol=1e-5)

    tokens = torch.randn(2, 16, 8)
    paper_ole = OrientationAwareLocalEnhancer(8, fusion_mode="orthogonal")
    concat_ole = OrientationAwareLocalEnhancer(8, fusion_mode="concat")
    paper_output = paper_ole(tokens)
    concat_output = concat_ole(tokens)
    assert paper_output.shape == concat_output.shape == (2, 16, 8)
    assert torch.isfinite(paper_output).all() and torch.isfinite(concat_output).all()
    assert concat_ole.vertical_mlp is None
    assert make_model(ole_fusion="concat").ole.fusion_mode == "concat"


def test_supplement_prompt_count_settings_forward():
    for prompt_count in (8, 16, 32, 64):
        model = make_model(prompt_count=prompt_count).eval()
        assert model.scene_vpt.prompt_count == prompt_count
        with torch.inference_mode():
            descriptor = model(torch.randn(1, 3, 32, 32))
        assert descriptor.shape == (1, 288)
        assert torch.isfinite(descriptor).all()

