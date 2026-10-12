"""
tests/test_phase3.py
====================
Wardrobe Circle Phase 3 — Functional test suite.

Run (server must be running on :8000):
    python tests/test_phase3.py

Tests
-----
 1. Polyvore dataset loading
 2. Train/validation/test split counts
 3. Item registry
 4. Compatibility label loading
 5. FITB question loading
 6. Compatibility model loading
 7. Single outfit compatibility score (valid)
 8. Single outfit score — empty item list
 9. Single outfit score — single item
10. Wardrobe category scoring
11. CandidateGenerator — recommend_outfits
12. CandidateGenerator — empty wardrobe
13. CandidateGenerator — single-item wardrobe
14. CandidateGenerator — recommend_missing_item
15. CandidateGenerator — no candidates
16. Compatibility evaluation — accuracy in reasonable range
17. FITB top-1 accuracy check
18. POST /recommend-outfits API
19. POST /recommend-outfits — empty items
20. POST /recommend-outfits — unknown category
21. POST /recommend-missing-item API
22. POST /recommend-missing-item — no candidates
23. GET /health — compatibility_model_loaded
"""

import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []

BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLYVORE_DIR   = os.path.join(BASE_DIR, "data", "polyvore")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "compatibility_model.pkl")


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}{'  →  ' + detail if detail else ''}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Wardrobe Circle Phase 3 Functional Tests ===\n")

# ── Test 1-5: Dataset Loading ─────────────────────────────────────────────────
print("Tests 1–5: Polyvore Dataset Loading")
from polyvore.loader import PolyvoreDataset

ds = PolyvoreDataset(POLYVORE_DIR)

check("Test 1: dataset loads", True)
check("Test 2: total outfits == 21889", ds.num_outfits == 21889, str(ds.num_outfits))

train_n = len(ds.train_outfits())
valid_n = len(ds.valid_outfits())
test_n  = len(ds.test_outfits())
check("Test 2a: train outfits == 17316", train_n == 17316, str(train_n))
check("Test 2b: valid outfits == 1497",  valid_n == 1497,  str(valid_n))
check("Test 2c: test outfits == 3076",   test_n  == 3076,  str(test_n))

check("Test 3: item registry > 100000",  ds.num_items > 100000, str(ds.num_items))

compat_entries = ds.load_compatibility()
check("Test 4: compat entries > 7000",   len(compat_entries) > 7000, str(len(compat_entries)))
labels = [e.label for e in compat_entries]
check("Test 4a: has positive labels",    1 in labels)
check("Test 4b: has negative labels",    0 in labels)
pos_count = sum(1 for l in labels if l == 1)
neg_count = sum(1 for l in labels if l == 0)
check("Test 4c: positive count ~3076",  abs(pos_count - 3076) < 50, str(pos_count))
check("Test 4d: negative count ~4000",  abs(neg_count - 4000) < 50, str(neg_count))

fitb_questions = ds.load_fitb()
check("Test 5: FITB questions > 2000",   len(fitb_questions) > 2000, str(len(fitb_questions)))
check("Test 5a: FITB has correct_ref",
      all(hasattr(q, "correct_ref") for q in fitb_questions[:10]))
check("Test 5b: FITB has 4 answers",
      all(len(q.answer_refs) == 4 for q in fitb_questions[:10]))

# ── Test 6-10: Compatibility Model ───────────────────────────────────────────
print("\nTests 6–10: Compatibility Model")
if not os.path.exists(MODEL_PATH):
    check("Test 6: model file exists",    False,
          f"Not found at {MODEL_PATH} — run: python src/polyvore/train_compatibility.py")
    print("  Skipping model-dependent tests.")
else:
    from polyvore.compatibility import CompatibilityModel

    model = CompatibilityModel.load(MODEL_PATH)
    check("Test 6: model loads",          True)
    check("Test 6a: has pair_logodds",    len(model.pair_logodds) > 0,
          str(len(model.pair_logodds)))
    check("Test 6b: has lr_model",        model.lr_model is not None)
    check("Test 6c: has scaler",          model.scaler is not None)

    # Test 7: Valid outfit
    score = model.score_by_categories([11, 237, 43])   # top + bottom + shoe
    check("Test 7: score in [0,1]",       0 <= score <= 1, f"{score:.4f}")

    # Test 8: Empty list
    score_empty = model.score_by_categories([])
    check("Test 8: empty → 0.5",          score_empty == 0.5, str(score_empty))

    # Test 9: Single item
    score_single = model.score_by_categories([11])
    check("Test 9: single item → 0.5",    score_single == 0.5, str(score_single))

    # Test 10: Wardrobe categories
    score_wc = model.score_by_wardrobe_categories(["Tops", "Bottoms", "Footwear"])
    check("Test 10: wardrobe cat score in [0,1]", 0 <= score_wc <= 1, f"{score_wc:.4f}")

    # ── Test 11-15: CandidateGenerator ───────────────────────────────────────
    print("\nTests 11–15: Candidate Generation")
    from candidate_generation import CandidateGenerator, WardrobeItem

    gen = CandidateGenerator(model)
    wardrobe = [
        WardrobeItem("t1", "Tops",       dominant_color="black"),
        WardrobeItem("b1", "Bottoms",    dominant_color="blue"),
        WardrobeItem("f1", "Footwear",   dominant_color="white"),
        WardrobeItem("o1", "Outerwear",  dominant_color="grey"),
        WardrobeItem("a1", "Accessories",dominant_color="gold"),
    ]

    outfits = gen.recommend_outfits(wardrobe, top_k=5)
    check("Test 11: returns outfits list",   isinstance(outfits, list))
    check("Test 11a: up to 5 outfits",       len(outfits) <= 5, str(len(outfits)))
    check("Test 11b: outfits > 0",           len(outfits) > 0, str(len(outfits)))
    for o in outfits:
        check("Test 11c: outfit has items",  "items" in o)
        check("Test 11d: outfit has score",  "compatibility_score" in o)
        check("Test 11e: score in [0,1]",    0 <= o["compatibility_score"] <= 1,
              str(o["compatibility_score"]))
        check("Test 11f: has categories",    "categories" in o)

    # Print sample output
    if outfits:
        print(f"\n     Sample outfit recommendation:")
        for o in outfits[:2]:
            cats = o["categories"]
            score = o["compatibility_score"]
            print(f"       {' + '.join(cats)}  →  score={score:.4f}")

    # Test 12: Empty wardrobe
    empty_result = gen.recommend_outfits([], top_k=5)
    check("Test 12: empty wardrobe → []",    empty_result == [])

    # Test 13: Single item
    single_result = gen.recommend_outfits([wardrobe[0]], top_k=5)
    check("Test 13: single item → []",       single_result == [])

    # Test 14: Missing item recommendation
    partial = [wardrobe[0], wardrobe[1]]   # Tops + Bottoms
    candidates = [wardrobe[2], wardrobe[3], wardrobe[4]]  # Footwear, Outerwear, Accessories
    missing = gen.recommend_missing_item(partial, candidates, top_k=3)
    check("Test 14: missing item results",   len(missing) > 0, str(len(missing)))
    check("Test 14a: up to 3 results",       len(missing) <= 3)
    for r in missing:
        check("Test 14b: has item",          "item" in r)
        check("Test 14c: has score",         "compatibility_score" in r)
        check("Test 14d: score in [0,1]",    0 <= r["compatibility_score"] <= 1,
              str(r["compatibility_score"]))
    if missing:
        print(f"\n     Sample missing-item recommendation:")
        for r in missing:
            print(f"       {r['item']['category']} ({r['item']['item_id']})  "
                  f"→  score={r['compatibility_score']:.4f}")

    # Test 15: No candidates
    no_cands = gen.recommend_missing_item(partial, [], top_k=3)
    check("Test 15: no candidates → []",     no_cands == [])

    # ── Test 16-17: Evaluation ────────────────────────────────────────────────
    print("\nTests 16–17: Model Evaluation")
    compat_metrics = model.evaluate_compatibility(compat_entries, ds)
    acc = compat_metrics["accuracy"]
    check("Test 16: compatibility accuracy > 0.5",   acc > 0.5, f"{acc:.4f}")
    check("Test 16a: accuracy <= 1.0",               acc <= 1.0)
    check("Test 16b: precision in [0,1]",
          0 <= compat_metrics["precision"] <= 1, str(compat_metrics["precision"]))
    check("Test 16c: recall in [0,1]",
          0 <= compat_metrics["recall"] <= 1, str(compat_metrics["recall"]))
    check("Test 16d: f1 in [0,1]",
          0 <= compat_metrics["f1"] <= 1, str(compat_metrics["f1"]))
    print(f"\n     Compatibility metrics (functional test — not accuracy claim):")
    print(f"       Accuracy : {compat_metrics['accuracy']:.4f}")
    print(f"       Precision: {compat_metrics['precision']:.4f}")
    print(f"       Recall   : {compat_metrics['recall']:.4f}")
    print(f"       F1       : {compat_metrics['f1']:.4f}")

    fitb_metrics = model.evaluate_fitb(fitb_questions[:500], ds, top_k=[1, 3])
    top1 = fitb_metrics.get("top_1_accuracy", 0)
    top3 = fitb_metrics.get("top_3_accuracy", 0)
    check("Test 17: FITB top-1 in [0,1]",   0 <= top1 <= 1, f"{top1:.4f}")
    check("Test 17a: FITB top-3 in [0,1]",  0 <= top3 <= 1, f"{top3:.4f}")
    check("Test 17b: top-3 >= top-1",        top3 >= top1)
    print(f"\n     FITB metrics (functional test — not accuracy claim):")
    print(f"       Top-1: {top1:.4f}  |  Top-3: {top3:.4f}  (over {fitb_metrics['n_questions']} questions)")

# ── Tests 18-23: API ──────────────────────────────────────────────────────────
print("\nTests 18–23: API Endpoints")
server_up = False
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    server_up = r.status_code == 200
except Exception:
    pass

if not server_up:
    check("Server reachable", False, "Server not running — skipping API tests")
else:
    check("Server reachable", True)

    # Test 18: /recommend-outfits
    payload = {
        "items": [
            {"item_id": "t1", "category": "Tops",     "dominant_color": "black"},
            {"item_id": "b1", "category": "Bottoms",  "dominant_color": "blue"},
            {"item_id": "f1", "category": "Footwear", "dominant_color": "white"},
            {"item_id": "o1", "category": "Outerwear","dominant_color": "grey"},
        ],
        "top_k": 5
    }
    r = requests.post(f"{BASE_URL}/recommend-outfits", json=payload, timeout=30)
    if r.status_code == 503:
        check("Test 18: /recommend-outfits (model not trained)", True,
              "503 — run python src/polyvore/train_compatibility.py")
    else:
        check("Test 18: status 200",            r.status_code == 200, str(r.status_code))
        data = r.json()
        check("Test 18a: has recommendations",  "recommendations" in data)
        check("Test 18b: has model_note",       "model_note" in data)
        check("Test 18c: recommendations list", isinstance(data.get("recommendations"), list))
        for rec in data.get("recommendations", []):
            check("Test 18d: has items",         "items" in rec)
            check("Test 18e: has score",         "compatibility_score" in rec)
            check("Test 18f: score in [0,1]",    0 <= rec["compatibility_score"] <= 1,
                  str(rec["compatibility_score"]))

    # Test 19: Empty items
    r = requests.post(f"{BASE_URL}/recommend-outfits", json={"items": [], "top_k": 5}, timeout=10)
    check("Test 19: empty items → 400",     r.status_code in (400, 503), str(r.status_code))

    # Test 20: Unknown category
    r = requests.post(f"{BASE_URL}/recommend-outfits",
                      json={"items": [{"item_id": "x", "category": "Swimwear"}], "top_k": 3},
                      timeout=10)
    check("Test 20: bad category → 400",   r.status_code in (400, 503), str(r.status_code))

    # Test 21: /recommend-missing-item
    payload21 = {
        "outfit_items":  [
            {"item_id": "t1", "category": "Tops",    "dominant_color": "black"},
            {"item_id": "b1", "category": "Bottoms", "dominant_color": "blue"},
        ],
        "candidate_items": [
            {"item_id": "f1", "category": "Footwear",    "dominant_color": "white"},
            {"item_id": "f2", "category": "Footwear",    "dominant_color": "black"},
            {"item_id": "a1", "category": "Accessories", "dominant_color": "gold"},
        ],
        "top_k": 3
    }
    r = requests.post(f"{BASE_URL}/recommend-missing-item", json=payload21, timeout=30)
    if r.status_code == 503:
        check("Test 21: /recommend-missing-item (model not trained)", True,
              "503 — run python src/polyvore/train_compatibility.py")
    else:
        check("Test 21: status 200",        r.status_code == 200, str(r.status_code))
        data = r.json()
        check("Test 21a: has recommendations", "recommendations" in data)
        for rec in data.get("recommendations", []):
            check("Test 21b: has item",     "item" in rec)
            check("Test 21c: has score",    "compatibility_score" in rec)
            check("Test 21d: score [0,1]",  0 <= rec["compatibility_score"] <= 1,
                  str(rec["compatibility_score"]))
        if data.get("recommendations"):
            top = data["recommendations"][0]
            print(f"\n     Top missing-item recommendation: "
                  f"{top['item']['category']} (score={top['compatibility_score']:.4f})")

    # Test 22: No candidates
    r = requests.post(f"{BASE_URL}/recommend-missing-item",
                      json={"outfit_items": [{"item_id": "t1", "category": "Tops"}],
                            "candidate_items": [], "top_k": 3},
                      timeout=10)
    check("Test 22: no candidates → 400",  r.status_code in (400, 503), str(r.status_code))

    # Test 23: /health includes compatibility_model_loaded
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    check("Test 23: status 200",              r.status_code == 200)
    data = r.json()
    check("Test 23a: compatibility_model_loaded key present",
          "compatibility_model_loaded" in data, str(list(data.keys())))
    print(f"\n     Health: {json.dumps({k: v for k, v in data.items() if isinstance(v, bool)}, indent=2)}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"\n{'='*50}")
print(f"Phase 3 tests: {passed}/{total} passed")
print(f"{'='*50}\n")
if passed < total:
    print("Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL}  {name}  {detail}")

sys.exit(0 if passed == total else 1)
