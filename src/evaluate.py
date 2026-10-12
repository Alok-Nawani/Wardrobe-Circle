"""
evaluate.py
===========
Evaluate a trained Wardrobe Circle classifier on the test set.
Generates metrics.json, confusion_matrix.png.

Run:
    python src/evaluate.py \
        --model models/clothing_classifier_best.pth \
        --classes models/class_names.json \
        --test data/processed/test.csv
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from dataset import WardrobeDataset, get_val_transform
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(
    model_path: str,
    classes_path: str,
    test_csv: str,
    results_dir: str = "results",
    batch_size: int = 32,
) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Load class names ─────────────────────────────────────────────────────
    with open(classes_path) as f:
        class_names: list[str] = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}  (backbone={backbone})")

    # ── Dataset & loader ─────────────────────────────────────────────────────
    dataset = WardrobeDataset(test_csv, class_names, get_val_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Metrics ───────────────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    accuracy = report["accuracy"]
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    metrics = {
        "test_accuracy": round(accuracy, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1_score": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in class_names
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall": round(report["macro avg"]["recall"], 4),
            "f1_score": round(report["macro avg"]["f1-score"], 4),
        },
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall": round(report["weighted avg"]["recall"], 4),
            "f1_score": round(report["weighted avg"]["f1-score"], 4),
        },
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize by true label count for readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes - 1)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix (normalized)", fontsize=14)
    plt.tight_layout()

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {cm_path}")

    return metrics


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(base, "models", "clothing_classifier_best.pth"))
    parser.add_argument("--classes", default=os.path.join(base, "models", "class_names.json"))
    parser.add_argument("--test", default=os.path.join(base, "data", "processed", "test.csv"))
    parser.add_argument("--results", default=os.path.join(base, "results"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate(
        model_path=os.path.abspath(args.model),
        classes_path=os.path.abspath(args.classes),
        test_csv=os.path.abspath(args.test),
        results_dir=os.path.abspath(args.results),
        batch_size=args.batch_size,
    )
