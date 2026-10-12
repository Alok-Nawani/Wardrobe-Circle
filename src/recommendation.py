"""
src/recommendation.py
=====================
Wardrobe Circle Phase 3 — Public recommendation interface.

This module is the single entry point for Phase 3 recommendations.
It wraps CandidateGenerator and exposes two clean functions:

    recommend_outfits(items, top_k=5)
    recommend_missing_item(outfit_items, candidate_items, top_k=5)

It also handles lazy-loading of the CompatibilityModel so the
model is only loaded once (cached at module level).

Structured for future extension:
    recommend_outfits(items, top_k=5, season=None, weather=None)
These optional parameters are reserved but not used yet.

Usage
-----
    from recommendation import recommend_outfits, recommend_missing_item
    from candidate_generation import WardrobeItem

    wardrobe = [
        WardrobeItem("t1", "Tops",     dominant_color="black"),
        WardrobeItem("b1", "Bottoms",  dominant_color="blue"),
        WardrobeItem("f1", "Footwear", dominant_color="white"),
    ]

    results = recommend_outfits(wardrobe, top_k=5)
    # [{"items": [...], "compatibility_score": 0.81, "categories": [...]}, ...]
"""

from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from candidate_generation import CandidateGenerator, WardrobeItem
from polyvore.compatibility import CompatibilityModel

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "compatibility_model.pkl")

# ── Module-level singleton (loaded once) ─────────────────────────────────────
_compat_model:    Optional[CompatibilityModel]    = None
_generator:       Optional[CandidateGenerator]    = None


def _ensure_loaded() -> None:
    global _compat_model, _generator
    if _compat_model is None:
        _compat_model = CompatibilityModel.load(MODEL_PATH)
        _generator    = CandidateGenerator(_compat_model)


def recommend_outfits(
    items: list[WardrobeItem],
    top_k: int = 5,
    min_size: int = 2,
    max_size: int = 5,
    # Future parameters (not implemented yet):
    season:  Optional[str] = None,   # e.g., "Summer" — reserved for Phase 4
    weather: Optional[str] = None,   # e.g., "rainy"  — reserved for Phase 4
) -> list[dict]:
    """
    Recommend the top-K compatible outfits from a user's wardrobe.

    Parameters
    ----------
    items    : list of WardrobeItem objects (the user's wardrobe)
    top_k    : number of outfits to return (default 5)
    min_size : minimum items per outfit (default 2)
    max_size : maximum items per outfit (default 5)
    season   : reserved for future use — ignored in Phase 3
    weather  : reserved for future use — ignored in Phase 3

    Returns
    -------
    list of dicts:
    [
        {
            "items":               [{"item_id": ..., "category": ..., ...}],
            "compatibility_score": float,    # model-generated, not objective
            "categories":          [str, ...]
        },
        ...
    ]

    Note
    ----
    Compatibility scores are model-generated ranking signals based on
    learned category co-occurrence patterns from the 2017 Polyvore dataset.
    They do not represent objective fashion quality or current trends.
    """
    _ensure_loaded()
    return _generator.recommend_outfits(
        wardrobe=items,
        top_k=top_k,
        min_size=min_size,
        max_size=max_size,
    )


def recommend_missing_item(
    outfit_items: list[WardrobeItem],
    candidate_items: list[WardrobeItem],
    top_k: int = 5,
) -> list[dict]:
    """
    Rank candidate items by how well they complete a partial outfit.

    Analogous to the Polyvore Fill-In-The-Blank task.

    Parameters
    ----------
    outfit_items    : existing items in the partial outfit
    candidate_items : items to score as potential additions
    top_k           : number of ranked candidates to return

    Returns
    -------
    list of dicts:
    [
        {
            "item":               {"item_id": ..., "category": ..., ...},
            "compatibility_score": float
        },
        ...
    ]
    """
    _ensure_loaded()
    return _generator.recommend_missing_item(
        outfit_items=outfit_items,
        candidate_items=candidate_items,
        top_k=top_k,
    )


def get_model_info() -> dict:
    """Return metadata about the loaded compatibility model."""
    _ensure_loaded()
    return {
        "type":              "LogisticRegression on pairwise category co-occurrence",
        "training_outfits":  _compat_model.train_outfits,
        "unique_categories": _compat_model.unique_categories,
        "training_pairs":    _compat_model.training_pairs,
        "note": (
            "Scores are model-generated signals based on 2017 Polyvore outfit data. "
            "Not objective fashion advice."
        ),
    }


def is_model_loaded() -> bool:
    return _compat_model is not None
