"""CNN architecture factory for binary deepfake classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

SUPPORTED_ARCHITECTURES = ("efficientnet_b0", "artifact_cnn")


class ArtifactCNN(nn.Module):
    """Compact face-artifact CNN.

    Trained on real photographs versus synthetic faces. It is small enough to
    train and run on CPU while still separating this dataset's two classes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.2), nn.Linear(192, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_classifier(architecture: str, pretrained_backbone: bool = False) -> nn.Module:
    """Build a 2-class classifier. ImageNet weights are a starting point for training only."""
    name = architecture.lower()
    if name == "artifact_cnn":
        if pretrained_backbone:
            raise ValueError("artifact_cnn has no ImageNet backbone. Omit --pretrained-backbone.")
        return ArtifactCNN()
    if name != "efficientnet_b0":
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {SUPPORTED_ARCHITECTURES}"
        )
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 * 1024), 2)


def resolve_device(preference: str) -> torch.device:
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    return torch.device("cpu")
