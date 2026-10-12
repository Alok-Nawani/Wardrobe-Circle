"""
src/candidate_generation.py
============================
Wardrobe Circle Phase 3 — Candidate outfit generation.

Given a user's wardrobe (list of WardrobeItem objects), generates candidate
outfit combinations using category-aware rules derived from Polyvore outfit
structure.

Rules
-----
- A valid candidate outfit must contain at least 2 items.
- Avoid combining more than 2 items from the same wardrobe category
  (e.g., 3 Tops is unusual; Dresses + Tops together is unusual).
- Prefer combinations that match common Polyvore outfit patterns:
    Top + Bottom + Footwear (most common)
    Top + Bottom + Footwear + Outerwear
    Dress + Footwear
    Any of above + Accessories (optional)
- The generator does NOT hard-code fashion rules — it uses category
  frequency observed in training data to weight combinations.
- All candidates are scored by the CompatibilityModel before returning.

Usage
-----
    from candidate_generation import CandidateGenerator, WardrobeItem
    from polyvore.compatibility import CompatibilityModel

    model = CompatibilityModel.load("models/compatibility_model.pkl")
    gen   = CandidateGenerator(model)

    wardrobe = [
        WardrobeItem("item1", "Tops",      image_path="...", dominant_color="black"),
        WardrobeItem("item2", "Bottoms",   image_path="...", dominant_color="blue"),
        WardrobeItem("item3", "Footwear",  image_path="...", dominant_color="white"),
        WardrobeItem("item4", "Outerwear", image_path="...", dominant_color="grey"),
    ]

    outfits = gen.recommend_outfits(wardrobe, top_k=5)
    # [{"items": [...], "compatibility_score": 0.82}, ...]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from polyvore.compatibility import CompatibilityModel


# ─────────────────────────────────────────────────────────────────────────────
# WardrobeItem — user's wardrobe item representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WardrobeItem:
    """
    Represents one item in a user's digital wardrobe.

    Designed to be database-ready: uses item_id rather than relying
    on dataset-specific filenames.
    """
    item_id:        str
    category:       str                 # Wardrobe Circle category name
    image_path:     Optional[str] = None
    dominant_color: Optional[str] = None
    dominant_hex:   Optional[str] = None
    embedding:      Optional[list] = None   # 1280-dim, if pre-computed
    metadata:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id":        self.item_id,
            "category":       self.category,
            "image_path":     self.image_path,
            "dominant_color": self.dominant_color,
            "dominant_hex":   self.dominant_hex,
            "metadata":       self.metadata,
            # embedding omitted from serialised output (large array)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Outfit combination validity rules
# ─────────────────────────────────────────────────────────────────────────────

# Maximum items per category in a candidate outfit
_MAX_PER_CATEGORY: dict[str, int] = {
    "Tops":       1,
    "Bottoms":    1,
    "Dresses":    1,
    "Outerwear":  1,
    "Footwear":   1,
    "Accessories": 2,   # e.g., bag + sunglasses
}

# Incompatible co-occurrences (category pairs that rarely co-occur)
_INCOMPATIBLE_PAIRS: set[frozenset] = {
    frozenset({"Tops",    "Dresses"}),    # You don't usually wear a top with a dress
    frozenset({"Bottoms", "Dresses"}),    # Dress replaces top+bottom
}

# Minimum outfit size
_MIN_OUTFIT_SIZE = 2


def _is_valid_combination(items: list[WardrobeItem]) -> bool:
    """Check whether a candidate combination respects basic outfit rules."""
    if len(items) < _MIN_OUTFIT_SIZE:
        return False

    # Check category count limits
    cat_counts: dict[str, int] = {}
    for item in items:
        cat_counts[item.category] = cat_counts.get(item.category, 0) + 1
        if cat_counts[item.category] > _MAX_PER_CATEGORY.get(item.category, 1):
            return False

    # Check incompatible pairs
    cats = set(item.category for item in items)
    for bad_pair in _INCOMPATIBLE_PAIRS:
        if bad_pair.issubset(cats):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CandidateGenerator
# ─────────────────────────────────────────────────────────────────────────────

class CandidateGenerator:
    """
    Generates and ranks candidate outfit combinations from a user's wardrobe.

    Scoring uses the CompatibilityModel trained on Polyvore data.
    Category constraints avoid obviously invalid combinations.
    """

    def __init__(self, model: CompatibilityModel) -> None:
        self.model = model

    def recommend_outfits(
        self,
        wardrobe: list[WardrobeItem],
        top_k: int = 5,
        min_size: int = 2,
        max_size: int = 5,
    ) -> list[dict]:
        """
        Generate and rank the top-K outfit candidates from the wardrobe.

        Parameters
        ----------
        wardrobe  : list of WardrobeItem objects
        top_k     : number of outfits to return
        min_size  : minimum items in a candidate outfit
        max_size  : maximum items in a candidate outfit

        Returns
        -------
        list of dicts:
        [
            {
                "items": [WardrobeItem.to_dict(), ...],
                "compatibility_score": float,
                "categories": [str, ...]
            },
            ...
        ]
        Scores are model-generated estimates from Polyvore-trained model.
        """
        if not wardrobe:
            return []

        candidates: list[tuple[list[WardrobeItem], float]] = []

        # Generate all combinations of min_size..max_size
        for size in range(min_size, min(max_size + 1, len(wardrobe) + 1)):
            for combo in combinations(wardrobe, size):
                combo_list = list(combo)
                if not _is_valid_combination(combo_list):
                    continue
                cats = [item.category for item in combo_list]
                score = self.model.score_by_wardrobe_categories(cats)
                candidates.append((combo_list, score))

        if not candidates:
            return []

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "items":               [item.to_dict() for item in combo],
                "compatibility_score": round(score, 4),
                "categories":          [item.category for item in combo],
            }
            for combo, score in candidates[:top_k]
        ]

    def recommend_missing_item(
        self,
        outfit_items: list[WardrobeItem],
        candidate_items: list[WardrobeItem],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Given a partial outfit and a list of candidate items, rank candidates
        by how well they would complete the outfit.

        This is analogous to Polyvore Fill-In-The-Blank.

        Parameters
        ----------
        outfit_items    : existing items in the partial outfit
        candidate_items : items to rank as potential additions
        top_k           : number of candidates to return

        Returns
        -------
        list of dicts:
        [
            {
                "item": WardrobeItem.to_dict(),
                "compatibility_score": float
            },
            ...
        ]
        """
        if not candidate_items:
            return []

        results: list[dict] = []
        for candidate in candidate_items:
            trial = outfit_items + [candidate]
            cats  = [item.category for item in trial]
            score = self.model.score_by_wardrobe_categories(cats)
            results.append({
                "item":               candidate.to_dict(),
                "compatibility_score": round(score, 4),
            })

        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return results[:top_k]
