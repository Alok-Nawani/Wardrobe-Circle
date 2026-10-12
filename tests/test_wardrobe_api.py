"""
tests/test_wardrobe_api.py
==========================
Tests for the wardrobe CRUD endpoints added for the frontend.

Run with the server already started:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Then:
    python tests/test_wardrobe_api.py
"""

import io
import os
import sys
import time

import requests
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
passed = 0
failed = 0
_created_ids = []


def _make_image_bytes(color=(100, 150, 200), size=(64, 64)) -> bytes:
    """Create a minimal solid-colour JPEG in memory."""
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def ok(name):
    global passed
    passed += 1
    print(f"  ✓  {name}")


def fail(name, reason=""):
    global failed
    failed += 1
    print(f"  ✗  {name}  —  {reason}")


def check(name, condition, reason="assertion failed"):
    if condition:
        ok(name)
    else:
        fail(name, reason)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health_includes_status():
    r = requests.get(f"{BASE_URL}/health")
    check("GET /health returns 200", r.status_code == 200)
    data = r.json()
    check("health.status == ok", data.get("status") == "ok")


def test_list_items_empty_or_list():
    """GET /wardrobe/items must always return items list + count."""
    r = requests.get(f"{BASE_URL}/wardrobe/items")
    check("GET /wardrobe/items returns 200", r.status_code == 200)
    data = r.json()
    check("/wardrobe/items has 'items' key", "items" in data)
    check("/wardrobe/items has 'count' key", "count" in data)
    check("/wardrobe/items items is a list", isinstance(data["items"], list))
    check("/wardrobe/items count >= 0", data["count"] >= 0)


def test_add_item():
    """POST /wardrobe/items with a synthetic image."""
    img_bytes = _make_image_bytes(color=(200, 100, 50))
    files = {"file": ("test_item.jpg", img_bytes, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/wardrobe/items", files=files)
    check("POST /wardrobe/items returns 201", r.status_code == 201,
          f"got {r.status_code}: {r.text[:200]}")
    if r.status_code != 201:
        return

    data = r.json()
    item_id = data.get("id")
    _created_ids.append(item_id)

    check("response has id",            bool(item_id))
    check("response has category",      bool(data.get("category")))
    check("response has dominant_hex",  bool(data.get("dominant_hex")))
    check("response has image_url",     "image_url" in data)
    check("image_url starts with /",    str(data.get("image_url", "")).startswith("/"))
    check("response has created_at",    bool(data.get("created_at")))
    check("response has attributes",    "attributes" in data)
    check("confidence is float or null",
          data.get("confidence") is None or isinstance(data.get("confidence"), float))


def test_get_item():
    """GET /wardrobe/items/{id} for a previously created item."""
    if not _created_ids:
        fail("GET /wardrobe/items/{id}", "no item created yet — skipping")
        return
    item_id = _created_ids[-1]
    r = requests.get(f"{BASE_URL}/wardrobe/items/{item_id}")
    check("GET /wardrobe/items/{id} returns 200", r.status_code == 200)
    data = r.json()
    check("returned item id matches", data.get("id") == item_id)


def test_get_item_not_found():
    r = requests.get(f"{BASE_URL}/wardrobe/items/nonexistent-uuid-12345")
    check("GET /wardrobe/items/bad_id returns 404", r.status_code == 404)


def test_list_contains_added_item():
    """After adding, list must contain the item."""
    if not _created_ids:
        fail("List contains added item", "no item created yet — skipping")
        return
    r = requests.get(f"{BASE_URL}/wardrobe/items")
    data = r.json()
    ids = [it["id"] for it in data["items"]]
    check("List contains added item", _created_ids[-1] in ids)


def test_image_served():
    """GET /wardrobe/images/<filename> must return the image bytes."""
    if not _created_ids:
        fail("Image served correctly", "no item created yet — skipping")
        return
    item_id = _created_ids[-1]
    r_item = requests.get(f"{BASE_URL}/wardrobe/items/{item_id}")
    image_url = r_item.json().get("image_url", "")
    if not image_url:
        fail("Image URL present", "image_url is empty")
        return
    r = requests.get(f"{BASE_URL}{image_url}")
    check("GET image URL returns 200", r.status_code == 200)
    check("Image content-type is image/*",
          r.headers.get("content-type", "").startswith("image/"))


def test_delete_item():
    """DELETE /wardrobe/items/{id} removes the item."""
    if not _created_ids:
        fail("DELETE /wardrobe/items/{id}", "no item created yet — skipping")
        return
    item_id = _created_ids.pop()
    r = requests.delete(f"{BASE_URL}/wardrobe/items/{item_id}")
    check("DELETE /wardrobe/items/{id} returns 200", r.status_code == 200)
    data = r.json()
    check("DELETE response has deleted key", "deleted" in data)
    check("deleted id matches", data.get("deleted") == item_id)

    # Confirm it's gone
    r2 = requests.get(f"{BASE_URL}/wardrobe/items/{item_id}")
    check("After delete, GET returns 404", r2.status_code == 404)


def test_delete_nonexistent():
    r = requests.delete(f"{BASE_URL}/wardrobe/items/nonexistent-uuid-99999")
    check("DELETE non-existent returns 404", r.status_code == 404)


def test_add_invalid_file():
    """Sending a text file must return 400."""
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    r = requests.post(f"{BASE_URL}/wardrobe/items", files=files)
    check("POST /wardrobe/items with non-image returns 400", r.status_code == 400,
          f"got {r.status_code}")


def test_add_empty_file():
    """Empty file must return 400."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    r = requests.post(f"{BASE_URL}/wardrobe/items", files=files)
    check("POST /wardrobe/items with empty file returns 400", r.status_code == 400,
          f"got {r.status_code}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(max_wait=30):
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("Wardrobe API Tests")
    print("=" * 54)

    if not wait_for_server():
        print("ERROR: Could not connect to server at", BASE_URL)
        sys.exit(1)

    tests = [
        test_health_includes_status,
        test_list_items_empty_or_list,
        test_add_item,
        test_get_item,
        test_get_item_not_found,
        test_list_contains_added_item,
        test_image_served,
        test_delete_item,
        test_delete_nonexistent,
        test_add_invalid_file,
        test_add_empty_file,
    ]

    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            fail(t.__name__, str(e))

    total = passed + failed
    print("\n" + "=" * 54)
    print(f"Wardrobe API Tests: {passed}/{total} passed")
    print("=" * 54)
    sys.exit(0 if failed == 0 else 1)
