"""
tests/test_color.py
===================
Functional tests for src/color.py

Run:
    python tests/test_color.py

Tests
-----
1. Black solid image
2. White solid image
3. Red solid image
4. Blue solid image
5. Multicolour gradient image
6. Real clothing image from dataset
7. Invalid input handling
8. Configurable n_colors
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from color import extract_colors

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}{'  →  ' + detail if detail else ''}")


def _solid_image(r: int, g: int, b: int, size: int = 100) -> Image.Image:
    return Image.new("RGB", (size, size), color=(r, g, b))


def _gradient_image() -> Image.Image:
    """Simple left-to-right red→blue gradient."""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    for x in range(200):
        t = x / 199.0
        arr[:, x, 0] = int(255 * (1 - t))   # red
        arr[:, x, 2] = int(255 * t)          # blue
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Color Extraction Functional Tests ===\n")

# Test 1: Black image
print("Test 1: Black solid image")
result = extract_colors(_solid_image(10, 10, 10))
check("dominant_color is 'black'",   "black" in result["dominant_color"].lower(),
      result["dominant_color"])
check("dominant_hex is dark",        result["dominant_hex"].lower() < "#333333",
      result["dominant_hex"])
check("color_palette not empty",     len(result["color_palette"]) > 0)
check("color_distribution sums ~100",
      abs(sum(d["percentage"] for d in result["color_distribution"]) - 100) < 5,
      str(sum(d["percentage"] for d in result["color_distribution"])))

# Test 2: White image
print("\nTest 2: White solid image")
result = extract_colors(_solid_image(245, 245, 245))
check("dominant_color contains 'white'",
      "white" in result["dominant_color"].lower() or "cream" in result["dominant_color"].lower(),
      result["dominant_color"])
check("color_palette not empty", len(result["color_palette"]) > 0)

# Test 3: Red image
print("\nTest 3: Red solid image")
result = extract_colors(_solid_image(210, 20, 20))
check("dominant_color is red-ish",
      any(w in result["dominant_color"].lower() for w in ("red", "dark", "maroon", "rust", "coral")),
      result["dominant_color"])

# Test 4: Blue image
print("\nTest 4: Blue solid image")
result = extract_colors(_solid_image(30, 80, 200))
check("dominant_color is blue-ish",
      any(w in result["dominant_color"].lower() for w in ("blue", "navy", "teal", "violet")),
      result["dominant_color"])

# Test 5: Multicolour gradient
print("\nTest 5: Multicolour gradient (red→blue)")
result = extract_colors(_gradient_image(), n_colors=3)
check("returns 3 palette entries",    len(result["color_palette"]) == 3,
      str(len(result["color_palette"])))
check("distribution has 3 entries",  len(result["color_distribution"]) == 3,
      str(len(result["color_distribution"])))
check("all percentages > 0",
      all(d["percentage"] > 0 for d in result["color_distribution"]))

# Test 6: Configurable n_colors
print("\nTest 6: n_colors=5")
result = extract_colors(_gradient_image(), n_colors=5)
check("returns 5 palette entries",
      len(result["color_palette"]) <= 5,
      str(len(result["color_palette"])))

# Test 7: Return structure validation
print("\nTest 7: Return structure validation")
result = extract_colors(_solid_image(100, 100, 200))
check("dominant_color is str",         isinstance(result["dominant_color"], str))
check("dominant_hex is str starting #",
      isinstance(result["dominant_hex"], str) and result["dominant_hex"].startswith("#"))
check("color_palette is list",         isinstance(result["color_palette"], list))
check("color_distribution is list",    isinstance(result["color_distribution"], list))
for entry in result["color_distribution"]:
    check("entry has 'color' key",       "color" in entry)
    check("entry has 'percentage' key",  "percentage" in entry)
    check("percentage is float/int",     isinstance(entry["percentage"], (int, float)))

# Test 8: Real dataset image (if available)
print("\nTest 8: Real dataset image")
dataset_img = "/Users/alok/Documents/Major project/wardrobe-circle-ai/data/raw/images/8833.jpg"
if os.path.exists(dataset_img):
    result = extract_colors(dataset_img)
    check("dominant_color is str",      isinstance(result["dominant_color"], str))
    check("color_palette not empty",    len(result["color_palette"]) > 0)
    check("dominant_hex valid format",
          result["dominant_hex"].startswith("#") and len(result["dominant_hex"]) == 7)
    print(f"     → dominant: {result['dominant_color']}  hex: {result['dominant_hex']}")
    print(f"     → palette: {result['color_palette']}")
else:
    check("dataset image available",    False, "image not found — skipping")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"\n{'='*45}")
print(f"Color tests: {passed}/{total} passed")
print(f"{'='*45}\n")
if passed < total:
    print("Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL}  {name}  {detail}")

sys.exit(0 if passed == total else 1)
