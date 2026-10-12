"""
attributes.py
=============
Wardrobe Circle Phase 2 — Attribute lookup from dataset metadata.

This module provides two things:

1. MetadataIndex — loads the train/val/test CSVs once and allows fast lookup
   of metadata (gender, season, baseColour, original_category) by image ID.
   Used when an image is already part of the dataset.

2. infer_attributes_from_category() — returns rule-based attribute hints
   derived purely from the predicted wardrobe category.  These are NOT
   neural-network predictions — they are conservative, clearly labelled
   heuristics.  We explicitly flag them as "inferred" not "predicted".

Honesty contract
----------------
We never fabricate attributes.  If an attribute cannot be determined
reliably, it is returned as None and omitted from API responses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Metadata index
# ─────────────────────────────────────────────────────────────────────────────
class MetadataIndex:
    """
    Builds an in-memory lookup from image ID → metadata row.

    Source: data/processed/train.csv, val.csv, test.csv
    Columns used: id, gender, baseColour, season, original_category,
                  wardrobe_category, productDisplayName
    """

    def __init__(self, processed_dir: str) -> None:
        frames = []
        for split in ("train", "val", "test"):
            p = os.path.join(processed_dir, f"{split}.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p, dtype=str))
        if not frames:
            raise FileNotFoundError(
                f"No processed CSVs found in {processed_dir}. "
                "Run src/data/prepare_dataset.py first."
            )
        df = pd.concat(frames, ignore_index=True)
        df["id"] = df["id"].str.strip()
        # Keep one row per ID (no duplication across splits)
        df = df.drop_duplicates(subset=["id"])
        self._df = df.set_index("id")

    def lookup(self, image_path: str) -> dict:
        """
        Look up metadata for a given image path.

        Returns a dict with keys: gender, season, baseColour, original_category,
        productDisplayName (all may be None if the image is not in the dataset).
        """
        stem = Path(image_path).stem          # filename without extension
        if stem in self._df.index:
            row = self._df.loc[stem]
            return {
                "gender":             _clean(row.get("gender")),
                "season":             _clean(row.get("season")),
                "base_colour":        _clean(row.get("baseColour")),
                "article_type":       _clean(row.get("original_category")),
                "product_name":       _clean(row.get("productDisplayName")),
                "source":             "dataset_metadata",
            }
        return {
            "gender":         None,
            "season":         None,
            "base_colour":    None,
            "article_type":   None,
            "product_name":   None,
            "source":         "not_in_dataset",
        }

    def __len__(self) -> int:
        return len(self._df)


def _clean(val) -> Optional[str]:
    """Return None for NaN / 'nan' / empty strings."""
    if val is None:
        return None
    s = str(val).strip()
    return None if s.lower() in ("nan", "", "none") else s


# ─────────────────────────────────────────────────────────────────────────────
# Conservative category-level attribute hints
# These are used ONLY for images not in our dataset.
# They are coarse inferences, clearly labelled as such.
# ─────────────────────────────────────────────────────────────────────────────
_CATEGORY_HINTS: dict[str, dict] = {
    "Tops":       {"typical_usage": "Casual / Formal / Sport (varies by item)"},
    "Bottoms":    {"typical_usage": "Casual / Formal (varies by item)"},
    "Outerwear":  {"typical_usage": "Casual / Formal (varies by item)"},
    "Dresses":    {"typical_usage": "Casual / Formal / Party (varies by item)"},
    "Footwear":   {"typical_usage": "Casual / Sport / Formal (varies by item)"},
    "Accessories":{"typical_usage": "Varies by item"},
}


def get_category_hints(wardrobe_category: str) -> dict:
    """
    Return conservative, non-fabricated category-level hints.
    These are clearly flagged as 'inferred' — not individual-item predictions.
    """
    hints = _CATEGORY_HINTS.get(wardrobe_category, {})
    return {**hints, "note": "Category-level hint only. Not an item-level prediction."}
