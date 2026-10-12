"""
train.py
========
Step 6 — Train the Wardrobe Circle clothing classifier.

Run (quick experiment — 500 samples/class, 5 epochs):
    python src/train.py --max_per_class 500 --epochs 5

Run (full dataset):
    python src/train.py --max_per_class 0 --epochs 20

Key flags:
    --backbone      efficientnet_b0 | resnet18
    --batch_size    default 32
    --lr            default 1e-3
    --epochs        default 15
    --max_per_class 0 = no limit
    --finetune      unfreeze top backbone layers after head training
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(__file__))

from dataset import WardrobeDataset, get_train_transform, get_val_transform
from model import build_model, count_trainable, count_total, unfreeze_top_layers


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: train
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(loader)}  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: evaluate
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Plot training history
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history: dict, out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train acc", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", label="Val acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training history → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_csv = args.train_csv or os.path.join(base, "data", "processed", "train.csv")
    val_csv = args.val_csv or os.path.join(base, "data", "processed", "val.csv")
    test_csv = args.test_csv or os.path.join(base, "data", "processed", "test.csv")
    models_dir = args.models_dir or os.path.join(base, "models")
    results_dir = args.results_dir or os.path.join(base, "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = get_device()

    # ── Load class names from training CSV ────────────────────────────────────
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    class_names: list[str] = sorted(train_df["wardrobe_category"].unique().tolist())
    num_classes = len(class_names)
    print(f"\nWardrobe categories ({num_classes}): {class_names}")

    # Save class names
    classes_path = os.path.join(models_dir, "class_names.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names → {classes_path}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = WardrobeDataset(train_csv, class_names, get_train_transform())
    val_dataset = WardrobeDataset(val_csv, class_names, get_val_transform())

    print(f"Train samples : {len(train_dataset):,}")
    print(f"Val samples   : {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
    )
    model.to(device)

    print(f"\nBackbone   : {args.backbone}")
    print(f"Trainable  : {count_trainable(model):,} params")
    print(f"Total      : {count_total(model):,} params")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_model_path = os.path.join(models_dir, "clothing_classifier_best.pth")

    phase = "head"  # first train only the classification head
    print(f"\n{'=' * 55}")
    print(f"Phase: {phase.upper()} — training classification head only")
    print(f"{'=' * 55}")

    for epoch in range(1, args.epochs + 1):
        # Optional fine-tuning phase
        if args.finetune and epoch == args.finetune_epoch and phase == "head":
            phase = "finetune"
            print(f"\n{'=' * 55}")
            print(f"Phase: FINE-TUNE — unfreezing top {args.finetune_layers} backbone layers")
            print(f"{'=' * 55}")
            unfreeze_top_layers(model, args.backbone, n_layers=args.finetune_layers)
            # Re-create optimizer with lower lr for backbone
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,
                weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(args.epochs - epoch + 1),
                eta_min=1e-7,
            )
            print(f"Trainable params now: {count_trainable(model):,}")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:3d}/{args.epochs}  "
            f"[{phase}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc * 100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc * 100:.2f}%  "
            f"({elapsed:.1f}s)"
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": class_names,
                    "backbone": args.backbone,
                    "num_classes": num_classes,
                },
                best_model_path,
            )
            print(f"  ★ New best model saved  (val_acc={val_acc * 100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
                break

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")

    # ── Save training history plot ─────────────────────────────────────────────
    plot_history(history, os.path.join(results_dir, "training_history.png"))

    # ── Save history to results ────────────────────────────────────────────────
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Running final evaluation on test set …")
    print("=" * 55)

    # Import here to avoid circular issues when run standalone
    from evaluate import evaluate as run_evaluate
    metrics = run_evaluate(
        model_path=best_model_path,
        classes_path=classes_path,
        test_csv=test_csv,
        results_dir=results_dir,
        batch_size=args.batch_size,
    )

    print(f"\n✓ Training complete.")
    print(f"  Best val accuracy  : {best_val_acc * 100:.2f}%")
    print(f"  Test accuracy      : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Saved model        : {best_model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="Train Wardrobe Circle clothing classifier")

    # Paths
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--models_dir", default=None)
    parser.add_argument("--results_dir", default=None)

    # Model
    parser.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18"],
        help="Backbone architecture (default: efficientnet_b0)",
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience")

    # Fine-tuning
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Unfreeze top backbone layers after head training",
    )
    parser.add_argument(
        "--finetune_epoch",
        type=int,
        default=5,
        help="Epoch at which to start fine-tuning (default: 5)",
    )
    parser.add_argument(
        "--finetune_layers",
        type=int,
        default=2,
        help="Number of top backbone blocks to unfreeze (default: 2)",
    )

    args = parser.parse_args()
    train(args)
