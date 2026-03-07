"""ResNet backbone wrappers for SSL pretraining."""

import torch.nn as nn
from torchvision import models

_BACKBONE_REGISTRY: dict[str, tuple[type, int]] = {}


def _register(name: str, model_fn: type, feat_dim: int):
    _BACKBONE_REGISTRY[name] = (model_fn, feat_dim)


# Register supported backbones
_register("resnet18", models.resnet18, 512)
_register("resnet50", models.resnet50, 2048)


def get_backbone(name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return (backbone_without_fc, feature_dim) for the given name.

    The returned module outputs a flat feature vector of size `feature_dim`.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(_BACKBONE_REGISTRY)}")

    model_fn, feat_dim = _BACKBONE_REGISTRY[name]
    weights = "IMAGENET1K_V1" if pretrained else None
    base = model_fn(weights=weights)

    # Remove the final fully-connected layer, keep everything up to avgpool
    layers = list(base.children())[:-1]  # drop fc
    backbone = nn.Sequential(*layers, nn.Flatten())
    return backbone, feat_dim
