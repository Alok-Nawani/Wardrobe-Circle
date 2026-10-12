"""
src/polyvore/compatibility.py
==============================
Wardrobe Circle Phase 3 — Compatibility model.

Architecture
------------
Since Polyvore images are unavailable (defunct CDN), this module builds
compatibility scoring from **category co-occurrence statistics** learned
from the 17,316 training outfits.

Approach
--------
1. Extract all ordered pairs (cat_a, cat_b) from every training outfit.
2. Count how often each pair co-occurs in POSITIVE outfits vs NEGATIVE outfits
   using the fashion_compatibility_prediction.txt labels.
3. Compute a log-odds co-occurrence score for each pair:
       score(a, b) = log( (pos_count + 1) / (neg_count + 1) )
4. Score a new outfit = mean of pairwise log-odds scores across all item pairs.
5. Apply sigmoid to map to [0, 1].

Additionally, a Logistic Regression is trained directly on the compatibility
binary labels using aggregate pairwise features per outfit — giving a proper
sklearn model that can be evaluated and saved as a .pkl file.

Evaluation
----------
- Compatibility prediction: accuracy, precision, recall, F1
  (binary classification on 7,076 labeled outfits from compatibility file)
- FITB: top-1 accuracy (standard metric for Polyvore FITB task)

Important limitations
---------------------
- This model captures category-level co-occurrence patterns from 2017 Polyvore
  data. It does NOT capture visual style, colour, or current fashion trends.
- Scores are model-generated ranking signals, not objective fashion advice.
- The model was trained on historical data from the Polyvore platform.

Usage
-----
    from polyvore.compatibility import CompatibilityModel

    model = CompatibilityModel.load("models/compatibility_model.pkl")
    score = model.score_by_categories([11, 237, 43])  # [top_cat, bottom_cat, shoe_cat]
    # → float in [0, 1]
"""

from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from .loader import PolyvoreDataset, CompatibilityEntry, FITBQuestion


# ─────────────────────────────────────────────────────────────────────────────
# Category vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Map Polyvore category IDs → our 6 Wardrobe Circle categories.
# Based on top-frequency category IDs identified during inspection +
# item name heuristics. Transparent and stored in the saved model.
POLYVORE_TO_WARDROBE: dict[int, str] = {
    # Tops
    11:  "Tops",    17: "Tops",    19: "Tops",    21: "Tops",
    22:  "Tops",    26: "Tops",    27: "Tops",    29: "Tops",
    # Bottoms
    237: "Bottoms", 238: "Bottoms", 239: "Bottoms", 240: "Bottoms",
    241: "Bottoms", 242: "Bottoms", 243: "Bottoms", 244: "Bottoms",
    # Dresses
    4:   "Dresses", 5: "Dresses",  6: "Dresses",   7: "Dresses",
    8:   "Dresses", 9: "Dresses",
    # Outerwear
    24:  "Outerwear", 25: "Outerwear", 236: "Outerwear",
    # Footwear
    43:  "Footwear", 46: "Footwear", 49: "Footwear", 261: "Footwear",
    # Accessories
    37:  "Accessories", 38: "Accessories", 57: "Accessories",
    62:  "Accessories", 64: "Accessories", 65: "Accessories",
    106: "Accessories",
}

# Polyvore category IDs that have no strong clothing mapping (backgrounds,
# makeup, home décor, etc.) — scored neutrally
_DECOR_CATS: set[int] = {200, 1967, 2654}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _pairs(cats: list[int]) -> list[tuple[int, int]]:
    """Return all ordered pairs (a, b) with a <= b from a list of category IDs."""
    return [(min(a, b), max(a, b)) for a, b in combinations(cats, 2)]


# ─────────────────────────────────────────────────────────────────────────────
# CompatibilityModel
# ─────────────────────────────────────────────────────────────────────────────

class CompatibilityModel:
    """
    Category co-occurrence based outfit compatibility model.

    The model is trained on the Polyvore compatibility dataset and saves
    two components:
    1. `pair_logodds`: dict mapping (cat_a, cat_b) → log-odds score
    2. `lr_model`:     sklearn LogisticRegression trained on outfit features

    Scoring is done by the logistic regression on outfit-level features
    derived from pairwise log-odds scores.

    Do not interpret scores as objective fashion quality.
    Scores are learned from 2017 Polyvore outfit data.
    """

    def __init__(self) -> None:
        self.pair_logodds:  dict[tuple[int, int], float] = {}
        self.lr_model:      Optional[LogisticRegression] = None
        self.scaler:        Optional[StandardScaler] = None
        self.wardrobe_map:  dict[int, str] = POLYVORE_TO_WARDROBE.copy()
        self.train_outfits: int = 0
        self.unique_categories: int = 0
        self.training_pairs: int = 0

    # ── Training ────────────────────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        dataset: PolyvoreDataset,
        verbose: bool = True,
    ) -> "CompatibilityModel":
        """
        Train from a loaded PolyvoreDataset.

        Steps:
        1. Build positive co-occurrence counts from all training outfits.
        2. Load compatibility labels → positive + negative pair counts.
        3. Compute log-odds for each (cat_a, cat_b) pair.
        4. Build outfit-level feature vectors and train LogisticRegression.
        """
        model = cls()

        if verbose:
            print("[CompatibilityModel] Building co-occurrence statistics …")

        # Step 1: Count co-occurrences from training outfits (ALL positive)
        pos_counts: dict[tuple[int, int], int] = defaultdict(int)
        cat_set: set[int] = set()

        train_outfits = dataset.train_outfits()
        model.train_outfits = len(train_outfits)

        for outfit in train_outfits:
            cats = outfit.category_ids()
            cat_set.update(cats)
            for pair in _pairs(cats):
                pos_counts[pair] += 1

        if verbose:
            print(f"  Training outfits: {len(train_outfits):,}")
            print(f"  Unique categories: {len(cat_set):,}")
            print(f"  Unique co-occurrence pairs: {len(pos_counts):,}")

        model.unique_categories = len(cat_set)

        # Step 2: Load compatibility labels → separate pos/neg pair counts
        entries = dataset.load_compatibility()
        compat_pos_counts: dict[tuple[int, int], int] = defaultdict(int)
        compat_neg_counts: dict[tuple[int, int], int] = defaultdict(int)

        for entry in entries:
            cats = [dataset.item_registry[ref].category_id for ref in entry.item_refs]
            for pair in _pairs(cats):
                if entry.label == 1:
                    compat_pos_counts[pair] += 1
                else:
                    compat_neg_counts[pair] += 1

        # Step 3: Compute log-odds
        all_pairs = set(pos_counts) | set(compat_pos_counts) | set(compat_neg_counts)
        for pair in all_pairs:
            # Combine training outfit counts with compatibility label counts
            total_pos = pos_counts.get(pair, 0) + compat_pos_counts.get(pair, 0)
            total_neg = compat_neg_counts.get(pair, 0)
            log_odds = math.log((total_pos + 1.0) / (total_neg + 1.0))
            model.pair_logodds[pair] = log_odds

        model.training_pairs = len(model.pair_logodds)

        if verbose:
            print(f"  Pairs with log-odds: {len(model.pair_logodds):,}")

        # Step 4: Train Logistic Regression on outfit-level features
        if verbose:
            print("[CompatibilityModel] Training logistic regression …")

        X, y = [], []
        for entry in entries:
            cats = [dataset.item_registry[ref].category_id for ref in entry.item_refs]
            feats = model._outfit_features(cats)
            X.append(feats)
            y.append(entry.label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        model.scaler = StandardScaler()
        X_scaled = model.scaler.fit_transform(X)

        model.lr_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model.lr_model.fit(X_scaled, y)

        train_acc = accuracy_score(y, model.lr_model.predict(X_scaled))
        if verbose:
            print(f"  LR training accuracy (on compatibility set): {train_acc:.4f}")

        return model

    # ── Scoring ─────────────────────────────────────────────────────────────

    def score_by_categories(self, category_ids: list[int]) -> float:
        """
        Score an outfit given a list of Polyvore category IDs.

        Returns a float in [0, 1] — higher means more compatible
        according to the model trained on Polyvore data.

        Not a claim of objective fashion quality.
        """
        if len(category_ids) < 2:
            return 0.5   # Not enough items to evaluate

        feats = self._outfit_features(category_ids)
        feats_arr = np.array([feats], dtype=np.float32)

        if self.lr_model is not None and self.scaler is not None:
            feats_scaled = self.scaler.transform(feats_arr)
            prob = self.lr_model.predict_proba(feats_scaled)[0][1]
            return float(round(prob, 4))

        # Fallback: sigmoid of mean log-odds
        return float(round(self._mean_logodds_score(category_ids), 4))

    def score_by_wardrobe_categories(self, wardrobe_categories: list[str]) -> float:
        """
        Score an outfit given Wardrobe Circle category names
        (e.g., ["Tops", "Bottoms", "Footwear"]).

        Maps to a representative Polyvore category ID for each category,
        then scores using the compatibility model.

        This is an approximation — the mapping from 6 broad wardrobe
        categories to 363 Polyvore IDs is lossy.
        """
        # Representative category IDs for each wardrobe category
        _REPRESENTATIVE: dict[str, int] = {
            "Tops":       11,
            "Bottoms":    237,
            "Dresses":    4,
            "Outerwear":  24,
            "Footwear":   43,
            "Accessories": 64,
        }
        cat_ids = [_REPRESENTATIVE.get(c, 0) for c in wardrobe_categories if c in _REPRESENTATIVE]
        if len(cat_ids) < 2:
            return 0.5
        return self.score_by_categories(cat_ids)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate_compatibility(
        self,
        entries: list[CompatibilityEntry],
        dataset: PolyvoreDataset,
    ) -> dict:
        """
        Evaluate binary compatibility prediction on labeled entries.

        Returns
        -------
        dict with accuracy, precision, recall, f1, report
        """
        y_true, y_pred = [], []
        for entry in entries:
            cats = [dataset.item_registry[ref].category_id for ref in entry.item_refs]
            score = self.score_by_categories(cats)
            y_pred.append(1 if score >= 0.5 else 0)
            y_true.append(entry.label)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = float(accuracy_score(y_true, y_pred))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            "accuracy":  round(acc, 4),
            "precision": round(report["1"]["precision"], 4),
            "recall":    round(report["1"]["recall"],    4),
            "f1":        round(report["1"]["f1-score"],  4),
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "n_negative": int((1 - y_true).sum()),
        }

    def evaluate_fitb(
        self,
        questions: list[FITBQuestion],
        dataset: PolyvoreDataset,
        top_k: list[int] = (1, 3),
    ) -> dict:
        """
        Evaluate Fill-In-The-Blank accuracy.

        For each question:
        - Score each candidate by adding it to the question items.
        - Rank candidates by compatibility score.
        - Check if correct answer (answers[0]) is in top-K.

        Returns
        -------
        dict with top-1 and top-3 accuracy.
        """
        hits = {k: 0 for k in top_k}
        total = 0

        for q in questions:
            q_cats = [dataset.item_registry[r].category_id for r in q.question_refs]
            correct_cat = dataset.item_registry[q.correct_ref].category_id

            # Score each answer
            scores: list[tuple[str, float]] = []
            for ref in q.answer_refs:
                ans_cat = dataset.item_registry[ref].category_id
                trial_cats = q_cats + [ans_cat]
                score = self.score_by_categories(trial_cats)
                scores.append((ref, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_refs = [r for r, _ in scores]

            for k in top_k:
                if q.correct_ref in ranked_refs[:k]:
                    hits[k] += 1
            total += 1

        return {
            f"top_{k}_accuracy": round(hits[k] / total, 4) if total > 0 else 0.0
            for k in top_k
        } | {"n_questions": total}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model to a .pkl file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[CompatibilityModel] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CompatibilityModel":
        """Load model from a .pkl file."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Compatibility model not found: {path}\n"
                "Run: python src/polyvore/train_compatibility.py"
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[CompatibilityModel] Loaded from {path}")
        return model

    # ── Internal ────────────────────────────────────────────────────────────

    def _outfit_features(self, category_ids: list[int]) -> list[float]:
        """
        Compute outfit-level feature vector from category IDs.

        Features:
        - mean pairwise log-odds score
        - min pairwise log-odds score
        - max pairwise log-odds score
        - std pairwise log-odds score
        - number of items
        - number of unique categories
        """
        pairs = _pairs(category_ids)
        if not pairs:
            return [0.0, 0.0, 0.0, 0.0, len(category_ids), len(set(category_ids))]

        pair_scores = [self.pair_logodds.get(p, 0.0) for p in pairs]
        return [
            float(np.mean(pair_scores)),
            float(np.min(pair_scores)),
            float(np.max(pair_scores)),
            float(np.std(pair_scores)),
            float(len(category_ids)),
            float(len(set(category_ids))),
        ]

    def _mean_logodds_score(self, category_ids: list[int]) -> float:
        """Fallback scorer using mean log-odds → sigmoid."""
        pairs = _pairs(category_ids)
        if not pairs:
            return 0.5
        scores = [self.pair_logodds.get(p, 0.0) for p in pairs]
        return _sigmoid(float(np.mean(scores)))
