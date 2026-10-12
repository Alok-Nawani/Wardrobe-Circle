"""
tests/test_embeddings.py
========================
Functional tests for src/embeddings.py and src/similarity.py

Run:
    python tests/test_embeddings.py

Tests
-----
1. EmbeddingExtractor produces correct shape
2. Output is L2-normalised
3. Identical images produce identical embeddings
4. Different images produce different embeddings
5. Cosine similarity of identical embeddings = 1.0
6. Cosine similarity is in [0, 1]
7. Batch extraction produces same results as single
8. SimilarityIndex search returns correct structure
9. Similarity values are in [0, 1]
10. top_k is respected
"""

import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "clothing_classifier_best.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
INDEX_PATH  = os.path.join(BASE_DIR, "data", "embeddings", "index.npz")


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}{'  →  ' + detail if detail else ''}")


def _solid_image(r: int, g: int, b: int) -> Image.Image:
    return Image.new("RGB", (224, 224), color=(r, g, b))


print("\n=== Embedding & Similarity Functional Tests ===\n")

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run: python src/train.py --epochs 5  first.")
    sys.exit(1)

from predict import load_model
from embeddings import EmbeddingExtractor, EMBEDDING_DIM

print("Loading model …")
model, class_names, device = load_model(MODEL_PATH, CLASSES_PATH)
extractor = EmbeddingExtractor(model, device)
print(f"Model loaded. Device: {device}\n")

# ── Test 1: Correct shape ─────────────────────────────────────────────────────
print("Test 1: Embedding shape")
emb = extractor.extract(_solid_image(100, 100, 100))
check("shape is (1280,)",        emb.shape == (EMBEDDING_DIM,), str(emb.shape))
check("dtype is float32",        emb.dtype == np.float32, str(emb.dtype))

# ── Test 2: L2 normalisation ──────────────────────────────────────────────────
print("\nTest 2: L2 normalisation")
norm = float(np.linalg.norm(emb))
check("norm is ~1.0",            abs(norm - 1.0) < 1e-5, f"{norm:.6f}")

# ── Test 3: Identical images → identical embeddings ───────────────────────────
print("\nTest 3: Deterministic output for identical inputs")
emb_a = extractor.extract(_solid_image(50, 120, 200))
emb_b = extractor.extract(_solid_image(50, 120, 200))
check("identical images → identical embeddings",
      np.allclose(emb_a, emb_b, atol=1e-5))

# ── Test 4: Different images → different embeddings ───────────────────────────
print("\nTest 4: Different images → different embeddings")
emb_red   = extractor.extract(_solid_image(220, 20, 20))
emb_blue  = extractor.extract(_solid_image(20, 20, 220))
cosine_diff = float(np.dot(emb_red, emb_blue))
check("different images → cosine < 1.0",
      cosine_diff < 0.999, f"cosine={cosine_diff:.4f}")

# ── Test 5: Self-similarity = 1.0 ─────────────────────────────────────────────
print("\nTest 5: Self-similarity")
self_sim = float(np.dot(emb_a, emb_a))
check("self cosine similarity ≈ 1.0",
      abs(self_sim - 1.0) < 1e-5, f"{self_sim:.6f}")

# ── Test 6: Real image embeddings ─────────────────────────────────────────────
print("\nTest 6: Real dataset images")
img_paths = [
    os.path.join(BASE_DIR, "data", "raw", "images", f)
    for f in ["8833.jpg", "52969.jpg", "27207.jpg"]
    if os.path.exists(os.path.join(BASE_DIR, "data", "raw", "images", f))
]
for img_path in img_paths[:2]:
    emb_real = extractor.extract(img_path)
    norm_real = float(np.linalg.norm(emb_real))
    check(f"{os.path.basename(img_path)}: shape OK",  emb_real.shape == (EMBEDDING_DIM,))
    check(f"{os.path.basename(img_path)}: norm ≈ 1.0", abs(norm_real - 1.0) < 1e-5,
          f"{norm_real:.6f}")

# ── Test 7: Batch vs single consistency ───────────────────────────────────────
print("\nTest 7: Batch extraction = single extraction")
images = [_solid_image(200, 50, 50), _solid_image(50, 200, 50)]
batch_embs = extractor.extract_batch(images, batch_size=2)
single_0   = extractor.extract(images[0])
single_1   = extractor.extract(images[1])
check("batch[0] ≈ single[0]",  np.allclose(batch_embs[0], single_0, atol=1e-4))
check("batch[1] ≈ single[1]",  np.allclose(batch_embs[1], single_1, atol=1e-4))

# ── Test 8: SimilarityIndex ───────────────────────────────────────────────────
print("\nTest 8: SimilarityIndex search")
if os.path.exists(INDEX_PATH):
    from similarity import SimilarityIndex
    index = SimilarityIndex(INDEX_PATH)
    check("index loaded, size > 0",    len(index) > 0, str(len(index)))

    query = extractor.extract(_solid_image(100, 100, 200))
    results_sim = index.search(query, top_k=5)

    check("returns 5 results",         len(results_sim) == 5, str(len(results_sim)))
    for r in results_sim:
        check("has 'image_path'",      "image_path" in r)
        check("has 'category'",        "category" in r)
        check("has 'similarity'",      "similarity" in r)
        check("similarity in [0,1]",   0.0 <= r["similarity"] <= 1.0,
              str(r["similarity"]))

    # Test top_k parameter
    results_k3 = index.search(query, top_k=3)
    check("top_k=3 returns 3",         len(results_k3) == 3, str(len(results_k3)))

    # Similarity ordering
    sims = [r["similarity"] for r in results_sim]
    check("results sorted descending", sims == sorted(sims, reverse=True),
          str(sims))
else:
    check("index.npz exists",  False,
          f"Not found at {INDEX_PATH} — run: python src/build_embeddings.py")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"\n{'='*45}")
print(f"Embedding tests: {passed}/{total} passed")
print(f"{'='*45}\n")
if passed < total:
    print("Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL}  {name}  {detail}")

sys.exit(0 if passed == total else 1)
