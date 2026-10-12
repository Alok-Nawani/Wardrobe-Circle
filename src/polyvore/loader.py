"""
src/polyvore/loader.py
======================
Wardrobe Circle Phase 3 — Polyvore dataset loader.

Loads and parses all five Polyvore files:

  train_no_dup.json                  — 17,316 training outfits
  valid_no_dup.json                  — 1,497 validation outfits
  test_no_dup.json                   — 3,076 test outfits
  fashion_compatibility_prediction.txt — 7,076 compatibility labels
  fill_in_blank_test.json            — 3,076 fill-in-the-blank questions

Key findings from dataset inspection
-------------------------------------
- All item images are remote URLs (defunct Polyvore CDN). No local images.
- Items carry: index, name, price, likes, categoryid
- Compatibility file format: "<label> <setid>_<idx> <setid>_<idx> ..."
  where label=1 (compatible) or label=0 (incompatible).
- FITB format: question=[item_refs], answers=[4 candidates], blank_position=N
  answer[0] is always the correct one.

Usage
-----
    from polyvore.loader import PolyvoreDataset

    ds = PolyvoreDataset("data/polyvore")
    print(ds.num_outfits)           # 21889
    print(ds.num_items)             # 142480
    compat = ds.load_compatibility() # list of (label, [item_refs])
    fitb   = ds.load_fitb()          # list of FITB questions
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolyvoreItem:
    """One clothing item within an outfit."""
    item_ref:   str           # "set_id_index", globally unique
    set_id:     str
    index:      int
    name:       str
    category_id: int
    price:      Optional[float] = None
    likes:      int = 0
    image_url:  Optional[str] = None   # CDN URL — not downloadable


@dataclass
class PolyvoreOutfit:
    """One Polyvore outfit (set)."""
    outfit_id:  str           # same as set_id
    name:       str
    split:      str           # "train" | "valid" | "test"
    items:      list[PolyvoreItem] = field(default_factory=list)
    likes:      int = 0
    views:      int = 0
    set_url:    Optional[str] = None

    def item_refs(self) -> list[str]:
        return [it.item_ref for it in self.items]

    def category_ids(self) -> list[int]:
        return [it.category_id for it in self.items]


@dataclass
class CompatibilityEntry:
    label:      int           # 1 = compatible, 0 = incompatible
    item_refs:  list[str]     # resolved item reference strings


@dataclass
class FITBQuestion:
    question_refs:  list[str]   # partial outfit item refs
    answer_refs:    list[str]   # 4 candidates; answer_refs[0] is correct
    blank_position: int
    correct_ref:    str         # = answer_refs[0]


# ─────────────────────────────────────────────────────────────────────────────
# PolyvoreDataset
# ─────────────────────────────────────────────────────────────────────────────

class PolyvoreDataset:
    """
    Loads and indexes the full Polyvore dataset from a directory containing
    the five standard files.

    Parameters
    ----------
    data_dir : path to directory containing the Polyvore JSON/txt files
    """

    TRAIN_FILE  = "train_no_dup.json"
    VALID_FILE  = "valid_no_dup.json"
    TEST_FILE   = "test_no_dup.json"
    COMPAT_FILE = "fashion_compatibility_prediction.txt"
    FITB_FILE   = "fill_in_blank_test.json"

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._check_files()
        self._load_outfits()

    # ── Public interface ────────────────────────────────────────────────────

    @property
    def num_outfits(self) -> int:
        return len(self.outfits)

    @property
    def num_items(self) -> int:
        return len(self.item_registry)

    def get_outfit(self, outfit_id: str) -> Optional[PolyvoreOutfit]:
        return self._outfit_map.get(outfit_id)

    def get_item(self, item_ref: str) -> Optional[PolyvoreItem]:
        return self.item_registry.get(item_ref)

    def train_outfits(self) -> list[PolyvoreOutfit]:
        return [o for o in self.outfits if o.split == "train"]

    def valid_outfits(self) -> list[PolyvoreOutfit]:
        return [o for o in self.outfits if o.split == "valid"]

    def test_outfits(self) -> list[PolyvoreOutfit]:
        return [o for o in self.outfits if o.split == "test"]

    def load_compatibility(self) -> list[CompatibilityEntry]:
        """
        Parse fashion_compatibility_prediction.txt.

        Returns a list of CompatibilityEntry objects.
        Each entry has a binary label (1/0) and a list of item refs.
        Items not found in the registry are reported but not silently dropped.
        """
        path = os.path.join(self.data_dir, self.COMPAT_FILE)
        entries: list[CompatibilityEntry] = []
        missing = 0

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = int(parts[0])
                refs  = parts[1:]
                # Validate that items exist
                valid_refs = []
                for ref in refs:
                    if ref in self.item_registry:
                        valid_refs.append(ref)
                    else:
                        missing += 1
                if len(valid_refs) >= 2:
                    entries.append(CompatibilityEntry(label=label, item_refs=valid_refs))

        if missing > 0:
            print(f"[PolyvoreDataset] Compatibility: {missing} item refs not found in registry (skipped).")
        return entries

    def load_fitb(self) -> list[FITBQuestion]:
        """
        Parse fill_in_blank_test.json.

        Returns a list of FITBQuestion objects.
        answer_refs[0] is always the correct answer (standard Polyvore convention).
        Questions with unresolvable items are skipped.
        """
        path = os.path.join(self.data_dir, self.FITB_FILE)
        with open(path) as f:
            raw = json.load(f)

        questions: list[FITBQuestion] = []
        skipped = 0

        for q in raw:
            q_refs = q["question"]
            a_refs = q["answers"]
            blank  = q["blank_position"]

            # Check all refs resolvable
            all_refs = q_refs + a_refs
            if not all(r in self.item_registry for r in all_refs):
                skipped += 1
                continue

            questions.append(FITBQuestion(
                question_refs=q_refs,
                answer_refs=a_refs,
                blank_position=blank,
                correct_ref=a_refs[0],
            ))

        if skipped:
            print(f"[PolyvoreDataset] FITB: {skipped} questions skipped (unresolvable items).")
        return questions

    def category_id_counts(self) -> dict[int, int]:
        """Return {category_id: count} across all items."""
        counts: dict[int, int] = {}
        for item in self.item_registry.values():
            counts[item.category_id] = counts.get(item.category_id, 0) + 1
        return counts

    # ── Internal ────────────────────────────────────────────────────────────

    def _check_files(self) -> None:
        required = [self.TRAIN_FILE, self.VALID_FILE, self.TEST_FILE,
                    self.COMPAT_FILE, self.FITB_FILE]
        for fname in required:
            p = os.path.join(self.data_dir, fname)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Polyvore file not found: {p}\n"
                    "Extract polyvore.tar.gz into data/polyvore/"
                )

    def _load_outfits(self) -> None:
        """Load all three splits and build the item registry."""
        self.outfits: list[PolyvoreOutfit] = []
        self.item_registry: dict[str, PolyvoreItem] = {}
        self._outfit_map: dict[str, PolyvoreOutfit] = {}

        for fname, split in [
            (self.TRAIN_FILE,  "train"),
            (self.VALID_FILE,  "valid"),
            (self.TEST_FILE,   "test"),
        ]:
            path = os.path.join(self.data_dir, fname)
            with open(path) as f:
                raw_outfits = json.load(f)

            for raw in raw_outfits:
                set_id = str(raw["set_id"])
                items: list[PolyvoreItem] = []

                for raw_item in raw.get("items", []):
                    idx      = int(raw_item["index"])
                    item_ref = f"{set_id}_{idx}"
                    item = PolyvoreItem(
                        item_ref=item_ref,
                        set_id=set_id,
                        index=idx,
                        name=str(raw_item.get("name", "")),
                        category_id=int(raw_item.get("categoryid", 0)),
                        price=raw_item.get("price"),
                        likes=int(raw_item.get("likes", 0)),
                        image_url=raw_item.get("image"),
                    )
                    items.append(item)
                    self.item_registry[item_ref] = item

                outfit = PolyvoreOutfit(
                    outfit_id=set_id,
                    name=str(raw.get("name", "")),
                    split=split,
                    items=items,
                    likes=int(raw.get("likes", 0)),
                    views=int(raw.get("views", 0)),
                    set_url=raw.get("set_url"),
                )
                self.outfits.append(outfit)
                self._outfit_map[set_id] = outfit
