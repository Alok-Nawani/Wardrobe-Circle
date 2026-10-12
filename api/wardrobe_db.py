"""
api/wardrobe_db.py
==================
Wardrobe Circle — Minimal wardrobe persistence layer.

Stores wardrobe items as a JSON file + uploaded images on disk.
Designed to migrate cleanly to PostgreSQL in a future phase.

No ML models are imported here.
No embeddings are stored (they are recomputed on demand).

Schema (one item in wardrobe.json):
{
    "id":            str (uuid4),
    "category":      str,
    "confidence":    float | null,
    "dominant_color": str,
    "dominant_hex":  str,
    "color_palette": [str, ...],
    "color_distribution": [{color, percentage}, ...],
    "attributes": {
        "gender": str|null, "season": str|null,
        "base_colour": str|null, "article_type": str|null,
        "product_name": str|null, "source": str
    },
    "image_filename": str,       # basename only, served via /wardrobe/images/{id}
    "created_at":    str (ISO)
}
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR     = Path(__file__).parent.parent
WARDROBE_DIR = BASE_DIR / "data" / "wardrobe"
UPLOADS_DIR  = WARDROBE_DIR / "uploads"
DB_FILE      = WARDROBE_DIR / "wardrobe.json"

WARDROBE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_db() -> list[dict]:
    if not DB_FILE.exists():
        return []
    try:
        with open(DB_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_db(items: list[dict]) -> None:
    tmp = str(DB_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, DB_FILE)   # atomic on POSIX


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def add_item(
    *,
    image_bytes: bytes,
    image_ext: str,             # ".jpg" | ".png"
    category: str,
    confidence: Optional[float],
    dominant_color: str,
    dominant_hex: str,
    color_palette: list[str],
    color_distribution: list[dict],
    attributes: dict,
) -> dict:
    """
    Persist a new wardrobe item.
    Returns the full item record.
    """
    item_id = str(uuid.uuid4())
    filename = f"{item_id}{image_ext}"
    dest = UPLOADS_DIR / filename

    with open(dest, "wb") as f:
        f.write(image_bytes)

    record = {
        "id":               item_id,
        "category":         category,
        "confidence":       confidence,
        "dominant_color":   dominant_color,
        "dominant_hex":     dominant_hex,
        "color_palette":    color_palette,
        "color_distribution": color_distribution,
        "attributes":       attributes,
        "image_filename":   filename,
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }

    items = _load_db()
    items.append(record)
    _save_db(items)
    return record


def list_items() -> list[dict]:
    """Return all wardrobe items, newest first."""
    items = _load_db()
    return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)


def get_item(item_id: str) -> Optional[dict]:
    """Return a single item by id, or None."""
    for item in _load_db():
        if item["id"] == item_id:
            return item
    return None


def delete_item(item_id: str) -> bool:
    """Delete item record + uploaded image. Returns True if found."""
    items = _load_db()
    to_delete = next((it for it in items if it["id"] == item_id), None)
    if to_delete is None:
        return False

    # Remove image file
    img_path = UPLOADS_DIR / to_delete.get("image_filename", "")
    if img_path.exists():
        img_path.unlink(missing_ok=True)

    remaining = [it for it in items if it["id"] != item_id]
    _save_db(remaining)
    return True


def image_path(item_id: str) -> Optional[Path]:
    """Return the full filesystem path to the uploaded image, or None."""
    item = get_item(item_id)
    if item is None:
        return None
    p = UPLOADS_DIR / item["image_filename"]
    return p if p.exists() else None
