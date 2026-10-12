"""
inspect_dataset.py
==================
Step 1 & 2 — Inspect the Fashion Product Images dataset (archive.zip)
and build the Wardrobe Circle category mapping.

Run:
    python src/data/inspect_dataset.py --zip /path/to/archive.zip

Outputs
-------
data/category_mapping.json
data/class_distribution.csv
"""

import argparse
import json
import os
import zipfile
from collections import Counter

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Wardrobe Circle category mapping
# Maps dataset articleType values → broad wardrobe categories.
# Only articleTypes actually present in the dataset are kept.
# ─────────────────────────────────────────────────────────────────────────────
WARDROBE_MAPPING_RAW: dict[str, list[str]] = {
    "Tops": [
        "Tshirts", "Shirts", "Tops", "Blouses", "Kurtas", "Kurtis",
        "Tunics", "Casual Shirts", "Formal Shirts", "Polo",
        "Sweatshirts", "Tank Tops", "Tube Tops",
        "Lounge Tshirts", "Camisoles", "Shrug",
    ],
    "Bottoms": [
        "Jeans", "Trousers", "Shorts", "Skirts", "Capris",
        "Track Pants", "Leggings", "Jeggings", "Churidar",
        "Dhoti Pants", "Salwar", "Patiala",
        "Lounge Pants", "Lounge Shorts", "Rain Trousers",
        "Tights", "Stockings",
    ],
    "Outerwear": [
        "Jackets", "Sweaters", "Blazers & Coats", "Blazers",
        "Nehru Jackets", "Waistcoat", "Rain Jacket",
        "Windcheater", "Denim Jacket", "Tracksuits",
        "Suits", "Bath Robe", "Robe",
    ],
    "Dresses": [
        "Dresses", "Sarees", "Nightdress", "Night suits",
        "Salwar and Dupatta", "Kurta Sets",
        "Jumpsuit", "Rompers", "Baby Dolls",
        "Lehenga Choli", "Swimwear",
    ],
    "Footwear": [
        "Casual Shoes", "Sports Shoes", "Formal Shoes",
        "Heels", "Sandals", "Flip Flops", "Flats",
        "Sneakers", "Boots", "Mules", "Sports Sandals", "Booties",
    ],
    "Accessories": [
        "Watches", "Handbags", "Sunglasses", "Belts",
        "Backpacks", "Wallets", "Clutches", "Scarves",
        "Caps", "Earrings", "Pendant", "Necklace and Chains",
        "Ties", "Socks", "Bra", "Briefs", "Boxers",
        "Innerwear Vests", "Trunk", "Hair Accessory",
        "Stoles", "Gloves", "Cufflinks", "Key chain",
        "Travel Accessory", "Wristbands", "Suspenders",
        "Ties and Cufflinks", "Mufflers", "Headband",
        "Hat", "Shapewear", "Dupatta",
        "Laptop Bag", "Messenger Bag", "Duffel Bag",
        "Rucksacks", "Trolley Bag", "Waist Pouch", "Mobile Pouch",
        "Jewellery Set", "Bangle", "Bracelet", "Ring",
        "Accessory Gift Set", "Clothing Set",
    ],
}


def build_article_to_wardrobe(wardrobe_map: dict[str, list[str]]) -> dict[str, str]:
    """Flatten the wardrobe mapping into {articleType: wardrobeCategory}."""
    mapping: dict[str, str] = {}
    for wardrobe_cat, article_types in wardrobe_map.items():
        for at in article_types:
            mapping[at] = wardrobe_cat
    return mapping


def inspect(zip_path: str, out_dir: str = "data") -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("WARDROBE CIRCLE — Dataset Inspection Report")
    print("=" * 60)
    print(f"\nZIP file  : {zip_path}")
    print(f"Output dir: {out_dir}\n")

    # ── 1. Read styles.csv from ZIP ──────────────────────────────────────────
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Find styles.csv (may be in root or subdir)
        csv_candidates = [n for n in all_names if n.endswith("styles.csv")]
        if not csv_candidates:
            raise FileNotFoundError("styles.csv not found inside the ZIP.")
        csv_name = csv_candidates[0]
        print(f"Found metadata : {csv_name}")

        with zf.open(csv_name) as f:
            # Some rows have embedded newlines — use python engine
            df = pd.read_csv(f, on_bad_lines="skip", dtype=str, engine="python")

        # Count image files (filenames like images/XXXXX.jpg)
        image_files = [n for n in all_names if n.startswith("images/") and n.endswith(".jpg")]
        image_ids = {os.path.splitext(os.path.basename(n))[0] for n in image_files}

    print(f"Total image files in ZIP : {len(image_files):,}")
    print(f"Total metadata rows      : {len(df):,}")
    print(f"\nColumns : {list(df.columns)}\n")

    # ── 2. Missing values ────────────────────────────────────────────────────
    print("Missing values per column:")
    missing = df.isnull().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            print(f"  {col:30s} {cnt:6,}")
    if missing.sum() == 0:
        print("  (none)")

    # ── 3. Unique counts ─────────────────────────────────────────────────────
    for col in ["masterCategory", "subCategory", "articleType", "gender",
                "baseColour", "season"]:
        if col in df.columns:
            print(f"\nUnique {col}: {df[col].nunique()}")

    # ── 4. Article-type distribution ─────────────────────────────────────────
    print("\nTop 30 articleType counts:")
    at_counts = df["articleType"].value_counts()
    print(at_counts.head(30).to_string())

    # ── 5. Image ↔ metadata match ────────────────────────────────────────────
    df["id"] = df["id"].str.strip()
    matched = df["id"].isin(image_ids).sum()
    unmatched = len(df) - matched
    print(f"\nImages matched with metadata : {matched:,}")
    print(f"Metadata rows with no image  : {unmatched:,}")

    # ── 6. Build category mapping ────────────────────────────────────────────
    article_to_wardrobe = build_article_to_wardrobe(WARDROBE_MAPPING_RAW)

    # Only keep articleTypes actually in the dataset
    dataset_types = set(df["articleType"].dropna().unique())
    filtered_mapping: dict[str, str] = {}
    for at, wc in article_to_wardrobe.items():
        if at in dataset_types:
            filtered_mapping[at] = wc

    # Any dataset articleType not covered → "Other"
    covered = set(filtered_mapping.keys())
    uncovered = dataset_types - covered
    if uncovered:
        print(f"\nArticle types mapped to 'Other' ({len(uncovered)}):")
        for u in sorted(uncovered):
            print(f"  {u}")
        for u in uncovered:
            filtered_mapping[u] = "Other"

    # Build final mapping output
    mapping_output = {
        "description": (
            "Maps Fashion Product Images articleType values to "
            "Wardrobe Circle broad wardrobe categories."
        ),
        "wardrobe_categories": list(
            sorted({v for v in filtered_mapping.values()})
        ),
        "mapping": filtered_mapping,
    }

    mapping_path = os.path.join(out_dir, "category_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_output, f, indent=2)
    print(f"\nSaved category mapping → {mapping_path}")

    # ── 7. Class distribution ────────────────────────────────────────────────
    df["wardrobe_category"] = df["articleType"].map(filtered_mapping)
    dist = (
        df.groupby(["wardrobe_category", "articleType"])
        .size()
        .reset_index(name="count")
        .sort_values(["wardrobe_category", "count"], ascending=[True, False])
    )
    dist_path = os.path.join(out_dir, "class_distribution.csv")
    dist.to_csv(dist_path, index=False)
    print(f"Saved class distribution  → {dist_path}")

    print("\nWardrobe category totals:")
    totals = df["wardrobe_category"].value_counts()
    for cat, cnt in totals.items():
        print(f"  {cat:15s}  {cnt:6,}")

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Fashion Product Images dataset")
    parser.add_argument(
        "--zip",
        default=os.path.join(os.path.dirname(__file__), "../../..", "archive.zip"),
        help="Path to archive.zip (default: auto-detect relative to project)",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="Output directory for mapping and distribution files (default: data/)",
    )
    args = parser.parse_args()

    zip_path = os.path.abspath(args.zip)
    inspect(zip_path, args.out)
