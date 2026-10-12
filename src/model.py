"""
model.py
========
EfficientNet-B0 (or ResNet-18) based clothing classifier for Wardrobe Circle.
Uses pretrained ImageNet weights; only the final classification head is trained
initially. Call `unfreeze_top_layers()` for fine-tuning.
"""

import ssl
import certifi

# Fix macOS Python SSL certificate verification issue
# Python 3.10 on macOS doesn't use system certs by default.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights



def build_model(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a transfer-learning classifier.

    Parameters
    ----------
    num_classes : int   — number of Wardrobe Circle categories
    backbone    : str   — "efficientnet_b0" or "resnet18"
    pretrained  : bool  — use ImageNet pretrained weights

    Returns
    -------
    model with backbone frozen; only the classifier head is trainable.
    """
    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace final FC layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'efficientnet_b0' or 'resnet18'.")

    return model


def unfreeze_top_layers(model: nn.Module, backbone: str, n_layers: int = 2) -> None:
    """
    Unfreeze the last `n_layers` blocks of the backbone for fine-tuning.
    Call this after initial head-only training.
    """
    if backbone == "efficientnet_b0":
        # EfficientNet has features[0..8]; unfreeze from the end
        blocks = list(model.features.children())
        for block in blocks[-n_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    elif backbone == "resnet18":
        # ResNet-18 layers: layer1, layer2, layer3, layer4
        layers = [model.layer4, model.layer3]
        for layer in layers[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
