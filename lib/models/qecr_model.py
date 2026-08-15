"""Non-distillation QECR wrapper for StereoDETR.

This file intentionally keeps the official/previous StereoDETR implementation
untouched.  It captures the already-computed left/right projected backbone
features, refines each decoder layer's query-depth distribution with sparse
epipolar matching, and optionally applies an epipolar-aware geometry gate.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lib.models.monodetr import build_stereodetr
from lib.models.monodetr.query_epipolar import (
    QueryEpipolarConsistencyRefiner,
)


def _capture_module_output(
    module: nn.Module,
    _inputs: Tuple[torch.Tensor, ...],
    output,
) -> None:
    """Store an output on the module replica itself (DataParallel-safe hook)."""
    module._qecr_captured_output = output


class EpipolarUncertaintyGeometryGate(nn.Module):
    """Fuse refined stereo and geometry depth using seven reliability cues."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_gate_dim: int = 32,
        initial_stereo_weight: float = 0.9,
        max_correction: float = 10.0,
    ) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_gate_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        initial_stereo_weight = min(
            max(float(initial_stereo_weight), 1.0e-4),
            1.0 - 1.0e-4,
        )
        initial_bias = math.log(
            initial_stereo_weight / (1.0 - initial_stereo_weight)
        )
        nn.init.constant_(self.gate[-1].bias, initial_bias)

    def forward(
        self,
        query_features: torch.Tensor,
        stereo_depth: torch.Tensor,
        geometry_depth: torch.Tensor,
        entropy: torch.Tensor,
        background_probability: torch.Tensor,
        log_variance: torch.Tensor,
        point_depth_variance: torch.Tensor,
        epipolar_confidence: torch.Tensor,
        epipolar_cycle_error: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        stereo_depth = stereo_depth.clamp_min(1.0e-3)
        geometry_depth = geometry_depth.clamp_min(1.0e-3)
        relative_disagreement = (
            (stereo_depth - geometry_depth).abs()
            / stereo_depth.detach().clamp_min(1.0e-3)
        ).clamp(max=5.0)
        epipolar_uncertainty = 1.0 - epipolar_confidence.clamp(0.0, 1.0)
        normalized_cycle_error = epipolar_cycle_error.clamp(
            min=0.0,
            max=5.0,
        ) / 5.0
        reliability = torch.cat(
            [
                entropy,
                background_probability,
                relative_disagreement,
                log_variance.clamp(-5.0, 5.0),
                point_depth_variance.sqrt().clamp(max=20.0),
                epipolar_uncertainty,
                normalized_cycle_error,
            ],
            dim=-1,
        )
        stereo_weight = torch.sigmoid(
            self.gate(torch.cat([query_features, reliability], dim=-1))
        )
        correction = (geometry_depth - stereo_depth).clamp(
            min=-self.max_correction,
            max=self.max_correction,
        )
        fused_depth = stereo_depth + (1.0 - stereo_weight) * correction
        return {
            "depth": fused_depth,
            "stereo_weight": stereo_weight,
            "geometry_depth": geometry_depth,
            "stereo_depth": stereo_depth,
        }


class QECRStereoDETR(nn.Module):
    """Add sparse epipolar refinement to an existing StereoDETR model."""

    def __init__(self, base_model: nn.Module, cfg: Dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.cfg = dict(cfg)
        self.feature_level = int(self.cfg.get("feature_level", 2))
        if not 0 <= self.feature_level < len(self.base_model.input_proj):
            raise ValueError("query_epipolar.feature_level is out of range")

        self.epipolar_refiner = QueryEpipolarConsistencyRefiner(
            num_samples=int(self.cfg.get("num_samples", 5)),
            search_radius=float(self.cfg.get("search_radius", 2.0)),
            matching_temperature=float(
                self.cfg.get("matching_temperature", 0.07)
            ),
            depth_temperature=float(self.cfg.get("depth_temperature", 1.0)),
            fusion_weight=float(self.cfg.get("fusion_weight", 0.5)),
            bidirectional=bool(self.cfg.get("bidirectional", True)),
            cycle_threshold=float(self.cfg.get("cycle_threshold", 1.0)),
            detach_points=bool(self.cfg.get("detach_points", True)),
            detach_depth_center=bool(
                self.cfg.get("detach_depth_center", True)
            ),
            align_corners=bool(self.cfg.get("align_corners", True)),
            min_disparity_pixels=float(
                self.cfg.get("min_disparity_pixels", 0.25)
            ),
        )

        self.use_geometry_gate = bool(
            self.cfg.get("use_geometry_gate", False)
        )
        if self.use_geometry_gate:
            self.geometry_gate = EpipolarUncertaintyGeometryGate(
                hidden_dim=int(self.base_model.hidden_dim),
                hidden_gate_dim=int(self.cfg.get("gate_hidden_dim", 32)),
                initial_stereo_weight=float(
                    self.cfg.get("initial_stereo_weight", 0.9)
                ),
                max_correction=float(
                    self.cfg.get("max_geometry_correction", 10.0)
                ),
            )

        self._feature_handle = self.base_model.input_proj[
            self.feature_level
        ].register_forward_hook(_capture_module_output)
        self._transformer_handle = (
            self.base_model.depthaware_transformer.register_forward_hook(
                _capture_module_output
            )
        )

    @staticmethod
    def _metadata_tensor(
        targets,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(targets, (list, tuple)):
            values = []
            for target in targets:
                if key not in target:
                    raise KeyError(
                        "QECR requires '{}' in every training target".format(key)
                    )
                value = target[key]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                values.append(value.reshape(-1)[0])
            tensor = torch.stack(values)
        elif isinstance(targets, dict):
            if key not in targets:
                raise KeyError(
                    "QECR requires '{}' in evaluation targets".format(key)
                )
            tensor = targets[key]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
        else:
            raise TypeError(
                "QECR requires targets to carry stereo calibration metadata"
            )
        return tensor.to(device=device, dtype=dtype).reshape(-1)

    def _refine_layer(
        self,
        layer_output: Dict[str, torch.Tensor],
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        stereo_fb: torch.Tensor,
        image_width: torch.Tensor,
        stereo_direction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        required = {
            "pred_depth_dist_logits",
            "pred_depth_dist_probs",
            "pred_depth_dist_bins",
            "pred_sample_points",
            "pred_depth",
        }
        missing = required.difference(layer_output)
        if missing:
            raise KeyError(
                "query-level depth must be enabled before QECR; missing {}".format(
                    sorted(missing)
                )
            )

        local_probabilities = layer_output["pred_depth_dist_probs"]
        local_bin_count = local_probabilities.shape[-1]
        local_indices = torch.arange(
            local_bin_count,
            device=local_probabilities.device,
            dtype=torch.long,
        ).view(1, 1, -1).expand_as(local_probabilities)

        refined = self.epipolar_refiner(
            left_features=left_features,
            right_features=right_features,
            query_points=layer_output["pred_sample_points"][..., :2],
            local_logits=layer_output["pred_depth_dist_logits"],
            local_bins=layer_output["pred_depth_dist_bins"],
            local_indices=local_indices,
            full_probabilities=local_probabilities,
            stereo_fb=stereo_fb,
            image_width=image_width,
            stereo_direction=stereo_direction,
        )

        layer_output["pred_depth_dist_logits"] = refined["logits"]
        layer_output["pred_depth_dist_probs"] = refined["probabilities"]
        layer_output["pred_depth_entropy"] = refined["entropy"]
        layer_output["pred_query_depth_mean"] = refined["expected_depth"]
        updated_depth = layer_output["pred_depth"].clone()
        updated_depth[..., 0:1] = refined["expected_depth"]
        layer_output["pred_depth"] = updated_depth
        return refined

    @staticmethod
    def _geometry_depth(
        outputs: Dict[str, torch.Tensor],
        calibs: torch.Tensor,
        img_sizes: torch.Tensor,
        img_sizes_ori: torch.Tensor,
        img_sizes_upper: torch.Tensor,
    ) -> torch.Tensor:
        size3d = outputs["pred_3d_dim"]
        boxes = outputs["pred_boxes"]
        scales = img_sizes_ori[:, 1:2] / (
            img_sizes[:, 1:2] + img_sizes_upper.unsqueeze(1)
        ).clamp_min(1.0)
        box_height_norm = boxes[:, :, 4] + boxes[:, :, 5]
        box_height = (
            box_height_norm * img_sizes[:, 1:2] * scales
        ).clamp_min(1.0)
        return (
            size3d[:, :, 0] / box_height * calibs[:, 0, 0].unsqueeze(1)
            + size3d[:, :, 2] / 2.0
        ).unsqueeze(-1)

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
        outputs = self.base_model(
            images,
            calibs,
            targets,
            img_sizes,
            img_sizes_ori,
            img_sizes_upper,
            dn_args=dn_args,
        )

        projected_stereo = getattr(
            self.base_model.input_proj[self.feature_level],
            "_qecr_captured_output",
            None,
        )
        transformer_output = getattr(
            self.base_model.depthaware_transformer,
            "_qecr_captured_output",
            None,
        )
        if projected_stereo is None or transformer_output is None:
            raise RuntimeError("QECR hooks did not capture model features")

        batch_size = images.shape[0]
        if projected_stereo.shape[0] != 2 * batch_size:
            raise RuntimeError("captured stereo feature batch has invalid size")
        left_features = projected_stereo[:batch_size]
        right_features = projected_stereo[batch_size:]
        decoder_features = transformer_output[0]

        stereo_fb = self._metadata_tensor(
            targets,
            "stereo_fb",
            left_features.device,
            left_features.dtype,
        )
        stereo_direction = self._metadata_tensor(
            targets,
            "stereo_direction",
            left_features.device,
            left_features.dtype,
        )
        image_width = img_sizes[:, 0].to(
            device=left_features.device,
            dtype=left_features.dtype,
        )

        layer_outputs: Iterable[Dict[str, torch.Tensor]] = [
            *outputs.get("aux_outputs", []),
            outputs,
        ]
        layer_outputs = list(layer_outputs)
        if len(layer_outputs) != decoder_features.shape[0]:
            raise RuntimeError(
                "decoder feature count and output layer count do not match"
            )

        final_epipolar = None
        for layer_output in layer_outputs:
            final_epipolar = self._refine_layer(
                layer_output,
                left_features,
                right_features,
                stereo_fb,
                image_width,
                stereo_direction,
            )
        assert final_epipolar is not None

        outputs.update(
            {
                "pred_epipolar_depth": final_epipolar["epipolar_depth"],
                "pred_epipolar_confidence": final_epipolar["confidence"],
                "pred_epipolar_cycle_error": final_epipolar["cycle_error"],
                "pred_epipolar_valid": final_epipolar["valid"],
                "pred_epipolar_forward_confidence": final_epipolar[
                    "forward_confidence"
                ],
                "pred_epipolar_reverse_confidence": final_epipolar[
                    "reverse_confidence"
                ],
                "pred_refined_disparity": final_epipolar[
                    "refined_disparity_pixels"
                ],
            }
        )

        if self.use_geometry_gate:
            geometry_depth = self._geometry_depth(
                outputs,
                calibs,
                img_sizes,
                img_sizes_ori,
                img_sizes_upper,
            )
            gate_output = self.geometry_gate(
                query_features=decoder_features[-1],
                stereo_depth=outputs["pred_query_depth_mean"],
                geometry_depth=geometry_depth,
                entropy=outputs["pred_depth_entropy"],
                background_probability=outputs[
                    "pred_depth_background_probability"
                ],
                log_variance=outputs["pred_depth"][..., 1:2],
                point_depth_variance=outputs["pred_point_depth_variance"],
                epipolar_confidence=outputs["pred_epipolar_confidence"],
                epipolar_cycle_error=outputs["pred_epipolar_cycle_error"],
            )
            gated_depth = outputs["pred_depth"].clone()
            gated_depth[..., 0:1] = gate_output["depth"]
            outputs["pred_depth"] = gated_depth
            outputs["pred_depth_gate"] = gate_output["stereo_weight"]
            outputs["pred_stereo_depth"] = gate_output["stereo_depth"]
            outputs["pred_geometry_depth"] = gate_output["geometry_depth"]

        self.base_model.input_proj[
            self.feature_level
        ]._qecr_captured_output = None
        self.base_model.depthaware_transformer._qecr_captured_output = None
        return outputs


class QECRCriterion(nn.Module):
    """Extend the original criterion with GT-only QECR supervision."""

    def __init__(self, base_criterion: nn.Module, cfg: Dict) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.cfg = dict(cfg)
        self.weight_dict = dict(base_criterion.weight_dict)
        self.epipolar_loss_coef = float(
            self.cfg.get("epipolar_loss_coef", 0.2)
        )
        self.use_geometry_gate = bool(
            self.cfg.get("use_geometry_gate", False)
        )
        if self.epipolar_loss_coef > 0:
            self.weight_dict["loss_epipolar_depth"] = self.epipolar_loss_coef
        if self.use_geometry_gate:
            self.weight_dict["loss_depth_gate"] = float(
                self.cfg.get("gate_loss_coef", 0.2)
            )

    def set_training_epoch(self, epoch):
        if hasattr(self.base_criterion, "set_training_epoch"):
            self.base_criterion.set_training_epoch(epoch)

    @staticmethod
    def _source_indices(indices):
        batch_index = torch.cat(
            [
                torch.full_like(source, index)
                for index, (source, _) in enumerate(indices)
            ]
        )
        source_index = torch.cat([source for source, _ in indices])
        return batch_index, source_index

    def _match(self, outputs, targets):
        output_for_match = {
            key: value
            for key, value in outputs.items()
            if key != "aux_outputs"
        }
        group_num = (
            self.base_criterion.group_num
            if self.training
            else 1
        )
        return self.base_criterion.matcher(
            output_for_match,
            targets,
            group_num=group_num,
        )

    def _epipolar_depth_loss(
        self,
        outputs,
        targets,
        indices_filted,
    ) -> torch.Tensor:
        source_index = self._source_indices(indices_filted)
        predicted = outputs["pred_epipolar_depth"][source_index].view(-1)
        valid = outputs["pred_epipolar_valid"][source_index].view(-1) > 0.5
        confidence = outputs["pred_epipolar_confidence"][
            source_index
        ].view(-1)
        if source_index[0].numel() == 0:
            return predicted.sum() * 0.0
        target_depth = torch.cat(
            [
                target["depth"][target_index]
                for target, (_, target_index) in zip(
                    targets,
                    indices_filted,
                )
            ],
            dim=0,
        ).view(-1)
        valid = (
            valid
            & torch.isfinite(predicted)
            & torch.isfinite(target_depth)
            & (target_depth > 0.0)
        )
        if not valid.any():
            return predicted.sum() * 0.0
        robust_log_error = F.smooth_l1_loss(
            predicted[valid].clamp_min(1.0e-3).log(),
            target_depth[valid].clamp_min(1.0e-3).log(),
            reduction="none",
        )
        reliability = 0.25 + 0.75 * confidence[valid].detach()
        return (
            robust_log_error * reliability
        ).sum() / reliability.sum().clamp_min(1.0)

    def _gate_loss(
        self,
        outputs,
        targets,
        indices_filted,
    ) -> torch.Tensor:
        source_index = self._source_indices(indices_filted)
        gate = outputs["pred_depth_gate"][source_index].view(-1)
        if source_index[0].numel() == 0:
            return gate.sum() * 0.0
        stereo_depth = outputs["pred_stereo_depth"][source_index].view(-1)
        geometry_depth = outputs["pred_geometry_depth"][source_index].view(-1)
        target_depth = torch.cat(
            [
                target["depth"][target_index]
                for target, (_, target_index) in zip(
                    targets,
                    indices_filted,
                )
            ],
            dim=0,
        ).view(-1)
        stereo_error = (stereo_depth - target_depth).abs()
        geometry_error = (geometry_depth - target_depth).abs()
        oracle = torch.sigmoid(
            (geometry_error - stereo_error)
            / max(float(self.cfg.get("gate_oracle_temperature", 1.0)), 1.0e-6)
        ).detach()
        return F.binary_cross_entropy(
            gate.clamp(1.0e-6, 1.0 - 1.0e-6),
            oracle,
        )

    def forward(self, outputs, targets, mask_dict=None):
        losses = self.base_criterion(outputs, targets, mask_dict)
        if self.epipolar_loss_coef <= 0 and not self.use_geometry_gate:
            return losses
        _, indices_filted = self._match(outputs, targets)
        if self.epipolar_loss_coef > 0:
            losses["loss_epipolar_depth"] = self._epipolar_depth_loss(
                outputs,
                targets,
                indices_filted,
            )
        if self.use_geometry_gate:
            losses["loss_depth_gate"] = self._gate_loss(
                outputs,
                targets,
                indices_filted,
            )
        return losses


def build_qecr_model(cfg: Dict):
    """Build the original StereoDETR and wrap it only when QECR is enabled."""
    if cfg.get("query_depth", {}).get("offline_teacher", False):
        raise ValueError("QECR-StereoDETR does not support teacher distillation")
    if cfg.get("query_depth", {}).get("geometry_gate", False):
        raise ValueError(
            "disable the legacy query_depth.geometry_gate; use "
            "query_epipolar.use_geometry_gate for the epipolar-aware gate"
        )

    base_model, base_criterion = build_stereodetr(cfg)
    epipolar_cfg = cfg.get("query_epipolar", {})
    if not epipolar_cfg.get("enabled", False):
        return base_model, base_criterion
    if not cfg.get("query_depth", {}).get("enabled", False):
        raise ValueError("query_depth.enabled must be true when QECR is enabled")

    model = QECRStereoDETR(base_model, epipolar_cfg)
    criterion = QECRCriterion(base_criterion, epipolar_cfg)
    criterion.to(torch.device(cfg["device"]))
    return model, criterion

