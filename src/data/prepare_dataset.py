"""
prepare_dataset.py
==================
Step 3 — Create clean train/val/test splits from the Fashion Product Images
dataset (archive.zip), using the category mapping produced by inspect_dataset.py.

Run:
    python src/data/prepare_dataset.py \
        --zip /path/to/archive.zip \
        --mapping data/category_mapping.json \
        --out data/processed \
        --images_dir data/raw/images \
        --max_per_class 500

Outputs
-------
data/raw/images/          (extracted JPEGs — only files used in splits)
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
# SPLIT_TEST  = 0.15  (remainder)

RANDOM_SEED = 42


def load_metadata(zip_path: str) -> tuple[pd.DataFrame, set[str]]:
    """Load styles.csv and collect the set of image IDs present in the ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Locate styles.csv
        csv_candidates = [n for n in all_names if n.endswith("styles.csv")]
        if not csv_candidates:
            raise FileNotFoundError("styles.csv not found inside the ZIP.")

        with zf.open(csv_candidates[0]) as f:
            df = pd.read_csv(f, on_bad_lines="skip", dtype=str, engine="python")

        image_ids: set[str] = {
            os.path.splitext(os.path.basename(n))[0]
            for n in all_names
            if n.startswith("images/") and n.endswith(".jpg")
        }

    df["id"] = df["id"].str.strip()
    return df, image_ids


def prepare(
    zip_path: str,
    mapping_path: str,
    out_dir: str,
    images_dir: str,
    max_per_class: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("Loading category mapping …")
    with open(mapping_path) as f:
        mapping_data = json.load(f)
    article_to_wardrobe: dict[str, str] = mapping_data["mapping"]

    print("Loading metadata from ZIP …")
    df, image_ids = load_metadata(zip_path)
    print(f"  Metadata rows : {len(df):,}")
    print(f"  Images in ZIP : {len(image_ids):,}")

    # ── Filter to rows that have a matching image ────────────────────────────
    df = df[df["id"].isin(image_ids)].copy()
    print(f"  Matched rows  : {len(df):,}")

    # ── Apply category mapping ───────────────────────────────────────────────
    df["wardrobe_category"] = df["articleType"].map(article_to_wardrobe)
    before = len(df)
    df = df[df["wardrobe_category"].notna()].copy()
    print(f"  After mapping : {len(df):,}  (dropped {before - len(df):,} unmappable rows)")

    # Drop "Other" — too noisy for classification
    df = df[df["wardrobe_category"] != "Other"].copy()
    print(f"  After dropping 'Other': {len(df):,}")

    # ── Remove duplicate IDs (same product ID appearing twice) ───────────────
    df = df.drop_duplicates(subset=["id"]).copy()
    print(f"  After dedup   : {len(df):,}")

    # ── Build image paths ────────────────────────────────────────────────────
    df["image_path"] = df["id"].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))

    # ── Cap per-class samples ────────────────────────────────────────────────
    if max_per_class and max_per_class > 0:
        print(f"  Capping at {max_per_class} samples per wardrobe category …")
        sampled_parts = []
        for cat, group in df.groupby("wardrobe_category"):
            if len(group) > max_per_class:
                sampled_parts.append(
                    group.sample(n=max_per_class, random_state=RANDOM_SEED)
                )
            else:
                sampled_parts.append(group)
        df = pd.concat(sampled_parts).reset_index(drop=True)
        print(f"  After cap     : {len(df):,}")

    # ── Print final class distribution ──────────────────────────────────────
    print("\nFinal wardrobe category distribution:")
    dist = df["wardrobe_category"].value_counts()
    for cat, cnt in dist.items():
        print(f"  {cat:15s}  {cnt:6,}")
    print()

    # ── Stratified train / val / test split ──────────────────────────────────
    # Drop any class with fewer than 3 samples (can't stratify-split)
    min_samples = 3
    valid_classes = dist[dist >= min_samples].index
    df = df[df["wardrobe_category"].isin(valid_classes)].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - SPLIT_TRAIN),
        stratify=df["wardrobe_category"],
        random_state=RANDOM_SEED,
    )
    val_ratio_of_temp = SPLIT_VAL / (1 - SPLIT_TRAIN)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_of_temp),
        stratify=temp_df["wardrobe_category"],
        random_state=RANDOM_SEED,
    )

    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")

    # ── Columns to save ──────────────────────────────────────────────────────
    keep_cols = [
        "id", "image_path", "original_category", "wardrobe_category",
        "gender", "baseColour", "season", "productDisplayName",
    ]
    # "original_category" doesn't exist yet — use articleType
    df.rename(columns={"articleType": "original_category"}, inplace=True)
    train_df.rename(columns={"articleType": "original_category"}, inplace=True, errors="ignore")
    val_df.rename(columns={"articleType": "original_category"}, inplace=True, errors="ignore")
    test_df.rename(columns={"articleType": "original_category"}, inplace=True, errors="ignore")

    available_cols = [c for c in keep_cols if c in train_df.columns]

    train_df[available_cols].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df[available_cols].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df[available_cols].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"\nSaved CSVs to {out_dir}/")

    # ── Extract only needed images ────────────────────────────────────────────
    all_ids_needed = set(
        train_df["id"].tolist()
        + val_df["id"].tolist()
        + test_df["id"].tolist()
    )
    # Rename column back for extraction
    needed_zip_paths = {f"images/{pid}.jpg" for pid in all_ids_needed}

    print(f"\nExtracting {len(needed_zip_paths):,} images to {images_dir} …")
    already = sum(
        1 for pid in all_ids_needed
        if os.path.exists(os.path.join(images_dir, f"{pid}.jpg"))
    )
    print(f"  Already on disk: {already:,}")

    if already < len(needed_zip_paths):
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = set(zf.namelist())
            to_extract = needed_zip_paths & members
            extracted = 0
            for member in to_extract:
                filename = os.path.basename(member)
                dest = os.path.join(images_dir, filename)
                if not os.path.exists(dest):
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    extracted += 1
                    if extracted % 1000 == 0:
                        print(f"  … extracted {extracted:,} / {len(to_extract):,}")
        print(f"  Done. Extracted {extracted:,} new images.")
    else:
        print("  All images already on disk. Skipping extraction.")

    print("\n✓ Dataset preparation complete.")


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    parser = argparse.ArgumentParser(description="Prepare Wardrobe Circle training dataset")
    parser.add_argument(
        "--zip",
        default=os.path.join(base, "..", "archive.zip"),
        help="Path to archive.zip",
    )
    parser.add_argument(
        "--mapping",
        default=os.path.join(base, "data", "category_mapping.json"),
        help="Path to category_mapping.json",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(base, "data", "processed"),
        help="Output directory for CSV splits",
    )
    parser.add_argument(
        "--images_dir",
        default=os.path.join(base, "data", "raw", "images"),
        help="Directory where extracted images are stored",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=500,
        help="Maximum samples per wardrobe category (0 = no limit)",
    )
    args = parser.parse_args()

    prepare(
        zip_path=os.path.abspath(args.zip),
        mapping_path=os.path.abspath(args.mapping),
        out_dir=os.path.abspath(args.out),
        images_dir=os.path.abspath(args.images_dir),
        max_per_class=args.max_per_class,
    )
