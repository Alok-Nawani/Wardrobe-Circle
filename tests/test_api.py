"""
tests/test_api.py
=================
Functional tests for the FastAPI endpoints (Phase 1 + Phase 2).

Run with the server already started:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Then in another terminal:
    python tests/test_api.py

Or start + test in one go (the script starts the server itself):
    python tests/test_api.py --start-server
"""

import argparse
import json
import os
import subprocess
import sys
import time
import io

import requests
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_IMAGES = os.path.join(BASE_DIR, "data", "raw", "images")

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []

_server_proc = None


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}{'  →  ' + detail if detail else ''}")


def _make_image_bytes(r: int, g: int, b: int, size: int = 150) -> bytes:
    img = Image.new("RGB", (size, size), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _real_image_bytes(filename: str) -> bytes | None:
    path = os.path.join(RAW_IMAGES, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _post_predict(image_bytes: bytes, top_k: int = 3) -> dict:
    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"top_k": top_k},
        timeout=30,
    )
    return resp


def _post_analyze(image_bytes: bytes, n_colors: int = 3) -> requests.Response:
    resp = requests.post(
        f"{BASE_URL}/analyze-clothing",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"n_colors": n_colors},
        timeout=30,
    )
    return resp


def _post_similar(image_bytes: bytes, top_k: int = 5) -> requests.Response:
    resp = requests.post(
        f"{BASE_URL}/similar-clothing",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"top_k": top_k},
        timeout=30,
    )
    return resp


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Wardrobe Circle API Functional Tests ===\n")

# ── Wait for server ───────────────────────────────────────────────────────────
print("Checking server availability …")
for attempt in range(15):
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"  Server is up.\n")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("ERROR: Server not reachable at http://localhost:8000")
    print("Start it with: uvicorn api.main:app --host 0.0.0.0 --port 8000")
    sys.exit(1)

# ── Test 1: /health ───────────────────────────────────────────────────────────
print("Test 1: GET /health")
r = requests.get(f"{BASE_URL}/health", timeout=10)
check("status_code == 200",      r.status_code == 200, str(r.status_code))
data = r.json()
check("status == 'ok'",          data.get("status") == "ok")
check("model_loaded == True",    data.get("model_loaded") == True)
check("num_classes > 0",         (data.get("num_classes") or 0) > 0,
      str(data.get("num_classes")))
print(f"     → classes: {data.get('classes')}")
print(f"     → index_size: {data.get('index_size')}")

# ── Test 2: Phase 1 /predict ──────────────────────────────────────────────────
print("\nTest 2: POST /predict — black image")
r = _post_predict(_make_image_bytes(10, 10, 10))
check("status_code == 200",      r.status_code == 200, str(r.status_code))
data = r.json()
check("category is str",         isinstance(data.get("category"), str),
      str(data.get("category")))
check("confidence in (0,1)",     0 < data.get("confidence", -1) <= 1,
      str(data.get("confidence")))
check("top_predictions is list", isinstance(data.get("top_predictions"), list))

# ── Test 3: /predict — real dataset image ─────────────────────────────────────
print("\nTest 3: POST /predict — real dataset image (Sweater)")
img_bytes = _real_image_bytes("8833.jpg")
if img_bytes:
    r = _post_predict(img_bytes)
    check("status_code == 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    check("category is str",    isinstance(data.get("category"), str),
          data.get("category"))
    check("confidence in (0,1)",0 < data.get("confidence", -1) <= 1)
else:
    check("image 8833.jpg available", False, "skipping")

# ── Test 4: /analyze-clothing — black image ───────────────────────────────────
print("\nTest 4: POST /analyze-clothing — black image")
r = _post_analyze(_make_image_bytes(10, 10, 10))
check("status_code == 200",        r.status_code == 200, str(r.status_code))
data = r.json()
check("category is str",           isinstance(data.get("category"), str))
check("confidence in (0,1)",       0 < data.get("confidence", -1) <= 1)
check("dominant_color is str",     isinstance(data.get("dominant_color"), str),
      str(data.get("dominant_color")))
check("dominant_hex starts with #",
      isinstance(data.get("dominant_hex"), str) and
      data.get("dominant_hex", "").startswith("#"))
check("color_palette is list",     isinstance(data.get("color_palette"), list))
check("color_distribution is list",isinstance(data.get("color_distribution"), list))
check("attributes is dict",        isinstance(data.get("attributes"), dict))
check("attributes.source is str",
      isinstance((data.get("attributes") or {}).get("source"), str))

# ── Test 5: /analyze-clothing — white image ───────────────────────────────────
print("\nTest 5: POST /analyze-clothing — white image")
r = _post_analyze(_make_image_bytes(245, 245, 245))
check("status_code == 200",        r.status_code == 200, str(r.status_code))
data = r.json()
check("dominant_color is str",     isinstance(data.get("dominant_color"), str),
      str(data.get("dominant_color")))

# ── Test 6: /analyze-clothing — coloured image (blue) ────────────────────────
print("\nTest 6: POST /analyze-clothing — blue image")
r = _post_analyze(_make_image_bytes(30, 80, 200))
check("status_code == 200",        r.status_code == 200, str(r.status_code))
data = r.json()
check("dominant_color is str",     isinstance(data.get("dominant_color"), str),
      str(data.get("dominant_color")))

# ── Test 7: /analyze-clothing — multicolour ──────────────────────────────────
print("\nTest 7: POST /analyze-clothing — multicolour image (3 colors requested)")
arr = np.zeros((150, 150, 3), dtype=np.uint8)
arr[:50, :, :] = [220, 20, 20]
arr[50:100, :, :] = [20, 150, 20]
arr[100:, :, :] = [20, 20, 220]
multi_img = Image.fromarray(arr)
buf = io.BytesIO()
multi_img.save(buf, format="JPEG")
r = _post_analyze(buf.getvalue(), n_colors=3)
check("status_code == 200",        r.status_code == 200, str(r.status_code))
data = r.json()
check("color_palette has <= 3",    len(data.get("color_palette", [])) <= 3,
      str(len(data.get("color_palette", []))))

# ── Test 8: Real image analyze ────────────────────────────────────────────────
print("\nTest 8: POST /analyze-clothing — real dataset image")
img_bytes = _real_image_bytes("8833.jpg")
if img_bytes:
    r = _post_analyze(img_bytes)
    check("status_code == 200", r.status_code == 200)
    data = r.json()
    check("category is str",    isinstance(data.get("category"), str),
          data.get("category"))
    check("dominant_color str", isinstance(data.get("dominant_color"), str),
          data.get("dominant_color"))
    check("palette has items",  len(data.get("color_palette", [])) > 0)
    print(f"     → category: {data.get('category')} ({data.get('confidence'):.2f})")
    print(f"     → color: {data.get('dominant_color')}  hex: {data.get('dominant_hex')}")
    print(f"     → palette: {data.get('color_palette')}")
    print(f"     → attributes: {data.get('attributes')}")
else:
    check("image 8833.jpg available", False, "skipping")

# ── Test 9: Invalid file ───────────────────────────────────────────────────────
print("\nTest 9: POST /predict — invalid (text) file")
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("bad.txt", b"this is not an image", "text/plain")},
    timeout=10,
)
check("status_code == 400",  r.status_code == 400, str(r.status_code))
check("error detail present", "detail" in r.json())

# ── Test 10: Empty upload ──────────────────────────────────────────────────────
print("\nTest 10: POST /predict — empty file")
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("empty.jpg", b"", "image/jpeg")},
    timeout=10,
)
check("status_code == 400", r.status_code == 400, str(r.status_code))

# ── Test 11: /similar-clothing ─────────────────────────────────────────────────
print("\nTest 11: POST /similar-clothing")
img_bytes = _real_image_bytes("8833.jpg")
if img_bytes:
    r = _post_similar(img_bytes, top_k=5)
    if r.status_code == 503:
        check("/similar-clothing responds (index not built)",
              True, "index not built — run python src/build_embeddings.py")
    else:
        check("status_code == 200",    r.status_code == 200, str(r.status_code))
        data = r.json()
        check("results is list",       isinstance(data.get("results"), list))
        check("index_size > 0",        data.get("index_size", 0) > 0,
              str(data.get("index_size")))
        for item in data.get("results", []):
            check("item has image_path",  "image_path" in item)
            check("item has category",    "category" in item)
            check("similarity in [0,1]",  0 <= item.get("similarity", -1) <= 1,
                  str(item.get("similarity")))
        if data.get("results"):
            print(f"     → top match: {data['results'][0]['image_name']}  "
                  f"({data['results'][0]['category']}, "
                  f"sim={data['results'][0]['similarity']:.3f})")
else:
    check("image 8833.jpg available", False, "skipping")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"\n{'='*50}")
print(f"API tests: {passed}/{total} passed")
print(f"{'='*50}\n")
if passed < total:
    print("Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL}  {name}  {detail}")

sys.exit(0 if passed == total else 1)
