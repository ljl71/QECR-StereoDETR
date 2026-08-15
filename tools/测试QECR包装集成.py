"""Integration-test the QECR wrapper with a minimal fake StereoDETR."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_runtime_module():
    name = "lib.models.monodetr.query_epipolar"
    path = (
        ROOT / "lib" / "models" / "monodetr" / "query_epipolar_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module


def load_wrapper_module():
    load_runtime_module()
    fake_monodetr = ModuleType("lib.models.monodetr")
    fake_monodetr.build_stereodetr = lambda _cfg: (None, None)
    sys.modules["lib.models.monodetr"] = fake_monodetr
    name = "qecr_model_standalone"
    path = ROOT / "lib" / "models" / "qecr_model.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTransformer(nn.Module):
    def forward(self, decoder_features):
        return (decoder_features,)


class FakeStereoDETR(nn.Module):
    def __init__(self, hidden_dim=8, layers=3, queries=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.queries = queries
        self.input_proj = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Conv2d(3, hidden_dim, kernel_size=1),
            ]
        )
        self.depthaware_transformer = FakeTransformer()
        self.query_features = nn.Parameter(
            torch.randn(layers, 1, queries, hidden_dim) * 0.1
        )

    def _layer_output(self, batch_size, layer_index, device):
        query_points = torch.rand(
            batch_size, self.queries, 2, device=device
        ) * 0.4 + 0.3
        bins = torch.tensor(
            [8.0, 10.0, 12.0, 14.0],
            device=device,
        ).view(1, 1, -1).expand(batch_size, self.queries, -1)
        logits = torch.tensor(
            [-1.0, 1.0, 0.5, -1.0],
            device=device,
        ).view(1, 1, -1).expand(batch_size, self.queries, -1)
        probabilities = logits.softmax(dim=-1)
        expected = (probabilities * bins).sum(dim=-1, keepdim=True)
        return {
            "pred_logits": torch.zeros(
                batch_size, self.queries, 4, device=device
            ),
            "pred_boxes": torch.full(
                (batch_size, self.queries, 6),
                0.2,
                device=device,
            ),
            "pred_3d_dim": torch.full(
                (batch_size, self.queries, 3),
                1.5,
                device=device,
            ),
            "pred_sample_points": query_points,
            "pred_depth": torch.cat(
                [expected, torch.zeros_like(expected)],
                dim=-1,
            ),
            "pred_depth_dist_logits": logits,
            "pred_depth_dist_probs": probabilities,
            "pred_depth_dist_bins": bins,
            "pred_depth_entropy": torch.full(
                (batch_size, self.queries, 1),
                0.5,
                device=device,
            ),
            "pred_depth_background_probability": torch.full(
                (batch_size, self.queries, 1),
                0.1,
                device=device,
            ),
            "pred_query_depth_mean": expected,
            "pred_point_depth_variance": torch.full(
                (batch_size, self.queries, 1),
                0.2,
                device=device,
            ),
        }

    def forward(
        self,
        images,
        calibs,
        targets,
        img_sizes,
        img_sizes_ori,
        img_sizes_upper,
        dn_args=None,
    ):
        batch_size = images.shape[0]
        stereo = torch.cat([images[:, :3], images[:, 3:]], dim=0)
        self.input_proj[2](stereo)
        decoder_features = self.query_features.expand(
            -1, batch_size, -1, -1
        )
        self.depthaware_transformer(decoder_features)
        layers = [
            self._layer_output(batch_size, index, images.device)
            for index in range(self.layers)
        ]
        output = layers[-1]
        output["aux_outputs"] = layers[:-1]
        return output


def main():
    module = load_wrapper_module()
    base = FakeStereoDETR()
    model = module.QECRStereoDETR(
        base,
        {
            "feature_level": 2,
            "num_samples": 5,
            "search_radius": 2.0,
            "bidirectional": True,
            "use_geometry_gate": True,
        },
    )
    batch_size = 2
    images = torch.randn(batch_size, 6, 8, 16, requires_grad=True)
    calibs = torch.zeros(batch_size, 3, 4)
    calibs[:, 0, 0] = 700.0
    targets = [
        {
            "stereo_fb": torch.tensor(390.0),
            "stereo_direction": torch.tensor(-1.0),
        }
        for _ in range(batch_size)
    ]
    img_sizes = torch.tensor([[16.0, 8.0], [16.0, 8.0]])
    outputs = model(
        images,
        calibs,
        targets,
        img_sizes,
        img_sizes.clone(),
        torch.zeros(batch_size),
    )
    for key in (
        "pred_epipolar_depth",
        "pred_epipolar_confidence",
        "pred_epipolar_cycle_error",
        "pred_depth_gate",
        "pred_stereo_depth",
        "pred_geometry_depth",
    ):
        assert key in outputs
        assert torch.isfinite(outputs[key]).all(), key
    assert len(outputs["aux_outputs"]) == 2
    assert outputs["pred_depth"].shape == (batch_size, 6, 2)
    outputs["pred_depth"].mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    print("QECR 包装、三层输出、极线校正与几何门控集成测试通过")


if __name__ == "__main__":
    main()

