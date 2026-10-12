"""
predict.py
==========
Step 8 — Run inference on a single clothing image.

Usage:
    python src/predict.py --image path/to/shirt.jpg

Output:
    Predicted category : Tops
    Confidence         : 94.2%

    Top predictions:
    1. Tops        94.2%
    2. Outerwear    3.1%
    3. Dresses      1.4%
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from dataset import get_val_transform
from model import build_model


def load_model(model_path: str, classes_path: str):
    """Load model and class names. Returns (model, class_names, device)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Class names file not found: {classes_path}")

    device = _get_device()

    with open(classes_path) as f:
        class_names: list[str] = json.load(f)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(
        num_classes=len(class_names),
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(
    image_path: str,
    model,
    class_names: list[str],
    device: torch.device,
    top_k: int = 3,
) -> dict:
    """
    Run inference on a single image file.

    Returns
    -------
    {
        "category": "Tops",
        "confidence": 0.942,
        "top_predictions": [
            {"category": "Tops", "confidence": 0.942},
            ...
        ]
    }
    """
    transform = get_val_transform()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(top_k)

    top_predictions = [
        {
            "category": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "category": top_predictions[0]["category"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="Wardrobe Circle — clothing category predictor")
    parser.add_argument("--image", required=True, help="Path to the clothing image")
    parser.add_argument(
        "--model",
        default=os.path.join(base, "models", "clothing_classifier_best.pth"),
        help="Path to saved .pth checkpoint",
    )
    parser.add_argument(
        "--classes",
        default=os.path.join(base, "models", "class_names.json"),
        help="Path to class_names.json",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    model, class_names, device = load_model(
        os.path.abspath(args.model),
        os.path.abspath(args.classes),
    )

    result = predict_image(
        image_path=os.path.abspath(args.image),
        model=model,
        class_names=class_names,
        device=device,
        top_k=args.top_k,
    )

    print(f"\nPredicted category : {result['category']}")
    print(f"Confidence         : {result['confidence'] * 100:.1f}%")
    print(f"\nTop predictions:")
    for i, pred in enumerate(result["top_predictions"], 1):
        print(f"  {i}. {pred['category']:15s}  {pred['confidence'] * 100:.1f}%")
