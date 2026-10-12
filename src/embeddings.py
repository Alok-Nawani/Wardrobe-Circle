"""
embeddings.py
=============
Wardrobe Circle Phase 2 — Visual feature extraction.

Extracts a 1280-dimensional L2-normalised visual embedding from any clothing
image using the EfficientNet-B0 backbone that was already loaded for
classification in Phase 1.

The embedding is derived from the `model.features` + `model.avgpool` layers —
i.e., everything *before* the final classification head.  This gives a rich
visual representation of the image without discarding information.

Embedding dimension: 1280  (EfficientNet-B0 avgpool output)

The embedding is used for:
- cosine-similarity based visual retrieval
- building a reference index across the dataset
- future outfit compatibility features (Phase 3+)

It does NOT encode semantic outfit compatibility or fashion trends.
It is a general visual representation.

Usage
-----
    from embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor(model, device)
    vec = extractor.extract("shirt.jpg")   # np.ndarray shape (1280,)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Reuse the val transform from Phase 1 (correct ImageNet normalisation)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_val_transform

EMBEDDING_DIM = 1280  # EfficientNet-B0 feature dimension


class EmbeddingExtractor:
    """
    Wraps a loaded EfficientNet-B0 model and exposes `.extract()`.

    The extractor hooks into `model.features` + `model.avgpool` and bypasses
    the `model.classifier` head — so it returns raw visual features, not
    classification logits.

    Parameters
    ----------
    model   : the loaded EfficientNet-B0 nn.Module (from Phase 1)
    device  : torch.device

    Notes
    -----
    - The model is kept in eval mode.
    - No gradient computation.
    - Thread-safe for sequential calls; use separate instances for parallel.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.device  = device
        self.transform = get_val_transform()

        # Build a feature extractor: features → avgpool → flatten
        # Works for EfficientNet-B0; adapts gracefully for ResNet-18 too.
        self._extractor = _build_feature_extractor(model)
        self._extractor.to(device)
        self._extractor.eval()

    def extract(
        self,
        source: Union[str, Path, bytes, Image.Image],
    ) -> np.ndarray:
        """
        Extract a normalised 1280-dim embedding from one image.

        Parameters
        ----------
        source : image path, raw bytes, or PIL Image

        Returns
        -------
        np.ndarray  shape (1280,)  float32  L2-normalised
        """
        image = _load_image(source)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self._extractor(tensor)          # (1, 1280)
            feat = feat.squeeze(0).cpu().numpy()    # (1280,)

        # L2 normalise
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm

        return feat.astype(np.float32)

    def extract_batch(
        self,
        sources: list[Union[str, Path, bytes, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings for a list of images efficiently.

        Returns
        -------
        np.ndarray  shape (N, 1280)  float32  L2-normalised
        """
        all_feats = []
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i : i + batch_size]
            tensors = []
            for src in batch_sources:
                img = _load_image(src)
                tensors.append(self.transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)  # (B, 3, 224, 224)

            with torch.no_grad():
                feats = self._extractor(batch_tensor)             # (B, 1280)
                feats = feats.cpu().numpy()

            # L2 normalise each row
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            feats = feats / norms
            all_feats.append(feats)

        return np.vstack(all_feats).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_extractor(model: nn.Module) -> nn.Module:
    """
    Build a forward-pass module that returns the pooled feature vector.

    EfficientNet-B0 structure:
        model.features   →  (B, 1280, H, W)
        model.avgpool    →  (B, 1280, 1, 1)
        Flatten          →  (B, 1280)

    ResNet-18 fallback:
        everything except model.fc
    """
    # Detect backbone type by inspecting the model
    if hasattr(model, "features") and hasattr(model, "avgpool"):
        # EfficientNet-B0
        return _EfficientNetFeatures(model)
    elif hasattr(model, "layer4") and hasattr(model, "fc"):
        # ResNet-18
        return _ResNetFeatures(model)
    else:
        raise ValueError(
            "Unsupported backbone. Expected EfficientNet-B0 or ResNet-18."
        )


class _EfficientNetFeatures(nn.Module):
    """Feature extractor for EfficientNet-B0."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool  = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 1280, 7, 7)
        x = self.avgpool(x)        # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)   # (B, 1280)
        return x


class _ResNetFeatures(nn.Module):
    """Feature extractor for ResNet-18 (512-dim output)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # All layers except the final FC
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)       # (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # (B, 512)
        return x


def _load_image(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return Image.open(str(source)).convert("RGB")
