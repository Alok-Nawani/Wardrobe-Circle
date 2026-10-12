"""
src/polyvore/train_compatibility.py
====================================
Wardrobe Circle Phase 3 — Train and evaluate the compatibility model.

Run:
    python src/polyvore/train_compatibility.py

Outputs:
    models/compatibility_model.pkl     — saved model
    results/compatibility_metrics.json — evaluation results

What this script does
---------------------
1. Load all Polyvore outfits (train/valid/test).
2. Train CompatibilityModel on training outfits + compatibility labels.
3. Evaluate on the full compatibility label set (7,076 outfits).
4. Evaluate FITB top-1, top-3 accuracy on 3,076 questions.
5. Save model and metrics.

Important: No fake metrics. All numbers come from actual evaluation.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyvore.loader import PolyvoreDataset
from polyvore.compatibility import CompatibilityModel

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POLYVORE_DIR   = os.path.join(BASE_DIR, "data", "polyvore")
MODEL_OUT      = os.path.join(BASE_DIR, "models", "compatibility_model.pkl")
METRICS_OUT    = os.path.join(BASE_DIR, "results", "compatibility_metrics.json")


def main() -> None:
    t0 = time.time()

    print("=" * 60)
    print("Wardrobe Circle — Phase 3 Compatibility Model Training")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading Polyvore dataset …")
    ds = PolyvoreDataset(POLYVORE_DIR)
    print(f"  Outfits: {ds.num_outfits:,}  |  Items: {ds.num_items:,}")
    print(f"  Train: {len(ds.train_outfits()):,}  Valid: {len(ds.valid_outfits()):,}  Test: {len(ds.test_outfits()):,}")

    # 2. Train model
    print("\n[2/5] Training compatibility model …")
    model = CompatibilityModel.train(ds, verbose=True)
    print(f"  Done. Unique categories: {model.unique_categories}  |  Pairs: {model.training_pairs:,}")

    # 3. Load evaluation data
    print("\n[3/5] Loading compatibility labels …")
    compat_entries = ds.load_compatibility()
    print(f"  Entries: {len(compat_entries):,}")
    pos = sum(1 for e in compat_entries if e.label == 1)
    neg = sum(1 for e in compat_entries if e.label == 0)
    print(f"  Positive: {pos:,}  |  Negative: {neg:,}")

    # 4. Evaluate compatibility
    print("\n[4/5] Evaluating compatibility prediction …")
    compat_metrics = model.evaluate_compatibility(compat_entries, ds)
    print(f"  Accuracy : {compat_metrics['accuracy']:.4f}")
    print(f"  Precision: {compat_metrics['precision']:.4f}")
    print(f"  Recall   : {compat_metrics['recall']:.4f}")
    print(f"  F1       : {compat_metrics['f1']:.4f}")

    # 5. Evaluate FITB
    print("\n[5/5] Evaluating Fill-In-The-Blank …")
    fitb_questions = ds.load_fitb()
    print(f"  Questions: {len(fitb_questions):,}")
    fitb_metrics = model.evaluate_fitb(fitb_questions, ds, top_k=[1, 3])
    print(f"  Top-1 accuracy: {fitb_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 accuracy: {fitb_metrics['top_3_accuracy']:.4f}")

    # Save model
    model.save(MODEL_OUT)

    # Save metrics
    all_metrics = {
        "dataset": {
            "train_outfits": len(ds.train_outfits()),
            "valid_outfits": len(ds.valid_outfits()),
            "test_outfits":  len(ds.test_outfits()),
            "total_items":   ds.num_items,
            "unique_category_ids": model.unique_categories,
        },
        "compatibility_prediction": compat_metrics,
        "fill_in_the_blank": fitb_metrics,
        "model_info": {
            "type": "Logistic Regression on pairwise category co-occurrence features",
            "training_pairs": model.training_pairs,
            "note": (
                "Scores are model-generated signals based on 2017 Polyvore outfit data. "
                "They do not represent objective fashion advice."
            ),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {METRICS_OUT}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
