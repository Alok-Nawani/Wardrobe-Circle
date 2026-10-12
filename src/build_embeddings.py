"""
build_embeddings.py
===================
Wardrobe Circle Phase 2 — Build the reference embedding index.

Runs once (or whenever new images are added) to generate visual embeddings
for all images in the dataset and save them to:

    data/embeddings/index.npz

The index contains:
    embeddings  : np.ndarray  shape (N, 1280)  float32
    image_paths : list[str]   absolute paths to source images
    categories  : list[str]   wardrobe category for each image

Usage
-----
    python src/build_embeddings.py

    # Rebuild even if index already exists:
    python src/build_embeddings.py --force
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from predict import load_model
from embeddings import EmbeddingExtractor, EMBEDDING_DIM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "clothing_classifier_best.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
INDEX_PATH     = os.path.join(EMBEDDINGS_DIR, "index.npz")


def build_index(
    model_path: str = MODEL_PATH,
    classes_path: str = CLASSES_PATH,
    processed_dir: str = PROCESSED_DIR,
    embeddings_dir: str = EMBEDDINGS_DIR,
    force: bool = False,
    batch_size: int = 64,
) -> str:
    """
    Build the reference embedding index.

    Returns the path to the saved index file.
    """
    index_path = os.path.join(embeddings_dir, "index.npz")

    if os.path.exists(index_path) and not force:
        print(f"Index already exists at {index_path}")
        print("Use --force to rebuild.")
        return index_path

    os.makedirs(embeddings_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    model, class_names, device = load_model(model_path, classes_path)
    extractor = EmbeddingExtractor(model, device)
    print(f"Extractor ready. Embedding dim: {EMBEDDING_DIM}. Device: {device}")

    # ── Collect all images from train + val + test CSVs ──────────────────────
    frames = []
    for split in ("train", "val", "test"):
        p = os.path.join(processed_dir, f"{split}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, dtype=str))

    if not frames:
        raise FileNotFoundError(f"No processed CSVs found in {processed_dir}")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Total images to embed: {len(df):,}")

    image_paths = df["image_path"].tolist()
    categories  = df["wardrobe_category"].tolist()

    # ── Extract embeddings in batches ─────────────────────────────────────────
    t0 = time.time()
    all_embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        feats = extractor.extract_batch(batch_paths, batch_size=batch_size)
        all_embeddings.append(feats)

        done = min(start + batch_size, len(image_paths))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(image_paths) - done) / rate if rate > 0 else 0
        print(
            f"  {done:4d}/{len(image_paths)}  "
            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
            flush=True,
        )

    embeddings = np.vstack(all_embeddings)  # (N, 1280)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez_compressed(
        index_path,
        embeddings=embeddings,
        image_paths=np.array(image_paths),
        categories=np.array(categories),
    )

    elapsed_total = time.time() - t0
    print(f"\nSaved index → {index_path}")
    print(f"  Shape      : {embeddings.shape}")
    print(f"  Time       : {elapsed_total:.1f}s")
    print(f"  Images/sec : {len(image_paths) / elapsed_total:.1f}")

    return index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build visual embedding index")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    build_index(force=args.force, batch_size=args.batch_size)
