"""Model routing for the non-distillation QECR experiment."""

from lib.models.qecr_model import build_qecr_model
from lib.models.monodetr import build_monodetr


def build_model(cfg):
    if cfg["model_type"] == "stereodetr":
        return build_qecr_model(cfg)
    if cfg["model_type"] == "monodetr":
        return build_monodetr(cfg)
    raise NotImplementedError(
        "Model type '{}' not recognized".format(cfg["model_type"])
    )

