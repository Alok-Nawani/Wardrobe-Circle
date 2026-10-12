"""
color.py
========
Dominant color extraction for Wardrobe Circle Phase 2.

Approach
--------
1. Centre-crop the image to focus on the clothing region (avoids margins/background).
2. Convert the cropped region to CIELAB colour space for perceptually uniform distances.
3. Run K-Means clustering to find dominant colour clusters.
4. Map each cluster centroid back to an RGB hex value.
5. Match the dominant hex value against a curated palette of named colours.

No neural network is trained here.  This is pure computer-vision / colour science.

Usage
-----
    from color import extract_colors

    result = extract_colors("shirt.jpg", n_colors=3)
    # {
    #   "dominant_color": "navy blue",
    #   "dominant_hex":   "#1a237e",
    #   "color_palette":  ["#1a237e", "#546e7a", "#e8eaf6"],
    #   "color_distribution": [
    #       {"color": "#1a237e", "percentage": 68.4},
    #       {"color": "#546e7a", "percentage": 21.1},
    #       {"color": "#e8eaf6", "percentage": 10.5},
    #   ]
    # }
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans

# ─────────────────────────────────────────────────────────────────────────────
# Named colour palette
# Covers the most common clothing colours with their approximate Lab centroids.
# Matching is done in Lab space for better perceptual accuracy.
# ─────────────────────────────────────────────────────────────────────────────
NAMED_COLOURS: list[tuple[str, tuple[int, int, int]]] = [
    # (name, (R, G, B))
    ("black",       (20,  20,  20)),
    ("white",       (240, 240, 240)),
    ("grey",        (128, 128, 128)),
    ("light grey",  (192, 192, 192)),
    ("dark grey",   (64,  64,  64)),
    ("red",         (200, 30,  30)),
    ("dark red",    (120, 10,  10)),
    ("pink",        (255, 153, 180)),
    ("light pink",  (255, 200, 215)),
    ("orange",      (230, 100, 20)),
    ("yellow",      (240, 210, 30)),
    ("light yellow",(255, 245, 150)),
    ("green",       (30,  130, 50)),
    ("dark green",  (0,   70,  30)),
    ("olive",       (100, 110, 30)),
    ("mint",        (150, 220, 190)),
    ("blue",        (30,  80,  200)),
    ("dark blue",   (10,  30,  120)),
    ("navy blue",   (15,  20,  80)),
    ("light blue",  (100, 170, 240)),
    ("sky blue",    (135, 200, 240)),
    ("purple",      (120, 40,  160)),
    ("violet",      (80,  0,   160)),
    ("lavender",    (190, 160, 220)),
    ("brown",       (100, 55,  20)),
    ("light brown", (160, 110, 60)),
    ("tan",         (200, 165, 110)),
    ("beige",       (220, 200, 160)),
    ("cream",       (240, 230, 200)),
    ("off white",   (245, 240, 230)),
    ("maroon",      (110, 10,  30)),
    ("teal",        (0,   130, 130)),
    ("turquoise",   (50,  180, 180)),
    ("cyan",        (0,   200, 220)),
    ("magenta",     (200, 30,  160)),
    ("gold",        (200, 170, 30)),
    ("silver",      (192, 192, 192)),
    ("khaki",       (190, 175, 100)),
    ("rust",        (175, 70,  20)),
    ("peach",       (255, 190, 155)),
    ("coral",       (240, 100, 80)),
    ("charcoal",    (55,  55,  55)),
    ("denim",       (70,  105, 150)),
]


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert Nx3 uint8 RGB array → Nx3 float32 CIELAB array (D65 illuminant)."""
    # Normalise to [0, 1]
    rgb_f = rgb.astype(np.float32) / 255.0

    # Linearise (inverse sRGB gamma)
    mask = rgb_f > 0.04045
    rgb_f[mask] = ((rgb_f[mask] + 0.055) / 1.055) ** 2.4
    rgb_f[~mask] = rgb_f[~mask] / 12.92

    # RGB → XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = rgb_f @ M.T

    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    # XYZ → Lab
    eps = 0.008856
    kappa = 903.3

    def _f(t: np.ndarray) -> np.ndarray:
        out = np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)
        return out

    fx, fy, fz = _f(xyz[:, 0]), _f(xyz[:, 1]), _f(xyz[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)

    return np.stack([L, a, b_ch], axis=1)


def _closest_named_colour(rgb: tuple[float, float, float]) -> str:
    """Find the perceptually closest named colour using Lab distance."""
    query = np.array([[rgb[0], rgb[1], rgb[2]]], dtype=np.uint8)
    query_lab = _rgb_to_lab(query)[0]

    best_name = "unknown"
    best_dist = float("inf")

    for name, ref_rgb in NAMED_COLOURS:
        ref_arr = np.array([[ref_rgb[0], ref_rgb[1], ref_rgb[2]]], dtype=np.uint8)
        ref_lab = _rgb_to_lab(ref_arr)[0]
        dist = float(np.sqrt(np.sum((query_lab - ref_lab) ** 2)))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def _load_image(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return Image.open(str(source)).convert("RGB")


def _centre_crop(img: Image.Image, fraction: float = 0.6) -> Image.Image:
    """
    Crop the central `fraction` × `fraction` region of the image.
    Typical product photos place the clothing in the centre.
    """
    w, h = img.size
    left   = int(w * (1 - fraction) / 2)
    top    = int(h * (1 - fraction) / 2)
    right  = int(w * (1 + fraction) / 2)
    bottom = int(h * (1 + fraction) / 2)
    return img.crop((left, top, right, bottom))


def _filter_near_white_black(pixels: np.ndarray,
                               white_threshold: int = 240,
                               black_threshold: int = 15) -> np.ndarray:
    """
    Remove pixels that are near-white (likely background) or near-black
    (likely shadows) from the pixel array before clustering.
    Returns the filtered array. Falls back to the original if too many
    pixels are removed.
    """
    brightness = pixels.mean(axis=1)
    mask = (brightness >= black_threshold) & (brightness <= white_threshold)
    filtered = pixels[mask]
    # Only use filtered if we keep at least 20 % of original pixels
    return filtered if len(filtered) >= max(100, int(len(pixels) * 0.20)) else pixels


def extract_colors(
    source: Union[str, Path, bytes, Image.Image],
    n_colors: int = 3,
    sample_pixels: int = 4000,
    centre_crop_fraction: float = 0.65,
    filter_background: bool = True,
) -> dict:
    """
    Extract dominant colours from a clothing image.

    Parameters
    ----------
    source              : image path, raw bytes, or PIL Image
    n_colors            : number of dominant colours to return (default 3)
    sample_pixels       : max pixels to feed into K-Means (random subsample)
    centre_crop_fraction: fraction of image to centre-crop (0 < f <= 1)
    filter_background   : if True, remove near-white / near-black pixels first

    Returns
    -------
    {
        "dominant_color":   str   e.g. "navy blue"
        "dominant_hex":     str   e.g. "#1a237e"
        "color_palette":    list[str]   hex values, sorted by cluster size
        "color_distribution": list[{"color": str, "percentage": float}]
    }
    """
    img = _load_image(source)

    # 1. Centre-crop to focus on clothing region
    img = _centre_crop(img, fraction=centre_crop_fraction)

    # 2. Resize so K-Means is fast (max 256px on the longest side)
    img.thumbnail((256, 256), Image.LANCZOS)

    # 3. Flatten to pixel array
    pixels = np.array(img).reshape(-1, 3)  # (N, 3) uint8

    # 4. Optionally filter near-white/black background pixels
    if filter_background:
        pixels = _filter_near_white_black(pixels)

    # 5. Random subsample if too many pixels
    if len(pixels) > sample_pixels:
        idx = np.random.choice(len(pixels), sample_pixels, replace=False)
        pixels = pixels[idx]

    # 6. K-Means in Lab space
    pixels_lab = _rgb_to_lab(pixels.astype(np.uint8))

    n_clusters = min(n_colors, len(pixels))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=3,
        random_state=42,
        max_iter=100,
    )
    labels = kmeans.fit_predict(pixels_lab)

    # 7. Compute cluster sizes and centres (back in RGB)
    distribution = []
    for k in range(n_clusters):
        mask = labels == k
        count = int(mask.sum())
        if count == 0:
            continue
        cluster_pixels = pixels[mask]
        mean_rgb = cluster_pixels.mean(axis=0)        # (3,) float
        hex_val = _rgb_to_hex(mean_rgb[0], mean_rgb[1], mean_rgb[2])
        distribution.append({
            "color": hex_val,
            "percentage": round(count / len(labels) * 100, 1),
            "_rgb": tuple(mean_rgb.tolist()),
        })

    # Sort by percentage descending
    distribution.sort(key=lambda x: x["percentage"], reverse=True)

    # 8. Name the dominant colour
    dominant_rgb = distribution[0]["_rgb"]
    dominant_name = _closest_named_colour(dominant_rgb)
    dominant_hex  = distribution[0]["color"]

    # 9. Clean up internal key
    for d in distribution:
        d.pop("_rgb", None)

    return {
        "dominant_color":     dominant_name,
        "dominant_hex":       dominant_hex,
        "color_palette":      [d["color"] for d in distribution],
        "color_distribution": distribution,
    }
