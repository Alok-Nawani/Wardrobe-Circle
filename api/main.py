"""
api/main.py
===========
Wardrobe Circle — FastAPI inference service (Phase 1 + Phase 2 + Phase 3)

Endpoints
---------
GET  /health                  — liveness + all component status
POST /predict                 — Phase 1: clothing category prediction
POST /analyze-clothing        — Phase 2: full clothing profile
POST /similar-clothing        — Phase 2: visually similar items
POST /recommend-outfits       — Phase 3: outfit recommendation from wardrobe
POST /recommend-missing-item  — Phase 3: fill-in-the-blank item recommendation

Start:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

# ── Allow importing from src/ ─────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from predict    import load_model, predict_image          # Phase 1 — unchanged
from color      import extract_colors                     # Phase 2
from attributes import MetadataIndex, get_category_hints  # Phase 2
from embeddings import EmbeddingExtractor                  # Phase 2
from similarity import SimilarityIndex                     # Phase 2
import recommendation as _rec                              # Phase 3
from candidate_generation import WardrobeItem              # Phase 3

# Wardrobe persistence (new for frontend)
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
import wardrobe_db as _wdb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH          = os.path.join(BASE_DIR, "models", "clothing_classifier_best.pth")
CLASSES_PATH        = os.path.join(BASE_DIR, "models", "class_names.json")
PROCESSED_DIR       = os.path.join(BASE_DIR, "data", "processed")
INDEX_PATH          = os.path.join(BASE_DIR, "data", "embeddings", "index.npz")
COMPAT_MODEL_PATH   = os.path.join(BASE_DIR, "models", "compatibility_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────
_model          = None          # Phase 1 classifier (EfficientNet-B0)
_class_names: list[str] = []
_device         = None
_extractor      = None          # Phase 2 embedding extractor
_meta_index     = None          # Phase 2 metadata lookup
_sim_index      = None          # Phase 2 similarity index


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load everything once on startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _class_names, _device, _extractor, _meta_index, _sim_index

    # Phase 1 — classifier
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: model checkpoint not found at {MODEL_PATH}")
        print("Run: python src/train.py --epochs 5")
    else:
        print("Loading Phase 1 classifier …")
        _model, _class_names, _device = load_model(MODEL_PATH, CLASSES_PATH)
        print(f"  Classes: {_class_names}  |  Device: {_device}")

        # Phase 2 — embedding extractor (reuses the same loaded model)
        print("Initialising Phase 2 embedding extractor …")
        _extractor = EmbeddingExtractor(_model, _device)
        print(f"  Embedding dim: 1280")

    # Phase 2 — metadata index
    try:
        print("Loading metadata index …")
        _meta_index = MetadataIndex(PROCESSED_DIR)
        print(f"  Metadata entries: {len(_meta_index):,}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    # Phase 2 — similarity index (optional — graceful degradation if missing)
    if os.path.exists(INDEX_PATH):
        print("Loading similarity index …")
        _sim_index = SimilarityIndex(INDEX_PATH)
    else:
        print(f"NOTE: Similarity index not found at {INDEX_PATH}")
        print("Run: python src/build_embeddings.py  (to enable /similar-clothing)")

    # Phase 3 — compatibility model (optional — graceful degradation if missing)
    if os.path.exists(COMPAT_MODEL_PATH):
        print("Loading Phase 3 compatibility model …")
        try:
            _rec._ensure_loaded()
            info = _rec.get_model_info()
            print(f"  Training outfits: {info['training_outfits']:,}  "
                  f"| Pairs: {info['training_pairs']:,}  "
                  f"| Categories: {info['unique_categories']}")
        except Exception as e:
            print(f"WARNING: Could not load compatibility model: {e}")
    else:
        print(f"NOTE: Compatibility model not found at {COMPAT_MODEL_PATH}")
        print("Run: python src/polyvore/train_compatibility.py")

    print("\n✓ Wardrobe Circle API ready.\n")
    yield
    # Cleanup
    _model = None


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wardrobe Circle — AI API",
    description=(
        "Phase 1 + Phase 2 + Phase 3 AI backend for Wardrobe Circle.\n\n"
        "**Phase 1**: Clothing category classification (EfficientNet-B0)\n\n"
        "**Phase 2**: Full clothing profile — dominant color, attributes, "
        "visual embedding, and similarity search.\n\n"
        "**Phase 3**: Outfit compatibility and recommendation (Polyvore-trained model)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: decode upload → PIL Image + temp file path
# ─────────────────────────────────────────────────────────────────────────────
async def _decode_upload(file: UploadFile) -> tuple[Image.Image, str]:
    """
    Read an uploaded file, validate it is a decodable image, write to a
    temporary JPEG file, and return (PIL Image, tmp_path).

    Caller is responsible for deleting the temp file.
    Raises HTTPException on any error.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image. Make sure you upload a valid JPEG or PNG file. ({type(e).__name__})",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format="JPEG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {type(e).__name__}")

    return image, tmp_path


def _require_model() -> None:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not loaded. Run: python src/train.py --epochs 5",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 1 (unchanged) ──────────────────────────────────────────────────────
class PredictionItem(BaseModel):
    category: str
    confidence: float

class PredictResponse(BaseModel):
    category: str
    confidence: float
    top_predictions: list[PredictionItem]

# ── Phase 2 ──────────────────────────────────────────────────────────────────
class ColorEntry(BaseModel):
    color: str          # hex, e.g. "#1a237e"
    percentage: float

class AttributesResponse(BaseModel):
    gender:        Optional[str] = None
    season:        Optional[str] = None
    base_colour:   Optional[str] = None   # dataset label, may differ from extracted
    article_type:  Optional[str] = None
    product_name:  Optional[str] = None
    source:        str                    # "dataset_metadata" | "not_in_dataset"

class AnalyzeResponse(BaseModel):
    category:           str
    confidence:         float
    top_predictions:    list[PredictionItem]
    dominant_color:     str
    dominant_hex:       str
    color_palette:      list[str]
    color_distribution: list[ColorEntry]
    attributes:         AttributesResponse

class SimilarItem(BaseModel):
    image_path:  str
    image_name:  str            # filename only (basename)
    category:    str
    similarity:  float

class SimilarResponse(BaseModel):
    query_category: str
    query_confidence: float
    results: list[SimilarItem]
    index_size: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    embeddings_loaded: bool
    similarity_index_loaded: bool
    compatibility_model_loaded: bool
    index_size: Optional[int] = None
    num_classes: Optional[int] = None
    classes: Optional[list[str]] = None


# ── Phase 3 ──────────────────────────────────────────────────────────────────────────────
class WardrobeItemIn(BaseModel):
    """Input schema for one wardrobe item in Phase 3 requests."""
    item_id:        str
    category:       str             # Wardrobe Circle category name
    image_path:     Optional[str] = None
    dominant_color: Optional[str] = None
    dominant_hex:   Optional[str] = None
    metadata:       dict           = {}


class OutfitResult(BaseModel):
    items:               list[dict]
    compatibility_score: float
    categories:          list[str]


class RecommendOutfitsRequest(BaseModel):
    items:   list[WardrobeItemIn]
    top_k:   int = 5


class RecommendOutfitsResponse(BaseModel):
    recommendations:    list[OutfitResult]
    model_note:         str


class MissingItemResult(BaseModel):
    item:               dict
    compatibility_score: float


class RecommendMissingRequest(BaseModel):
    outfit_items:    list[WardrobeItemIn]
    candidate_items: list[WardrobeItemIn]
    top_k:           int = 5


class RecommendMissingResponse(BaseModel):
    recommendations: list[MissingItemResult]
    model_note:      str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="API health check")
async def health():
    """Returns the status of all loaded components."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        embeddings_loaded=_extractor is not None,
        similarity_index_loaded=_sim_index is not None,
        compatibility_model_loaded=_rec.is_model_loaded(),
        index_size=len(_sim_index) if _sim_index else None,
        num_classes=len(_class_names) if _class_names else None,
        classes=_class_names if _class_names else None,
    )


@app.post("/predict", response_model=PredictResponse, summary="[Phase 1] Predict clothing category")
async def predict(
    file: UploadFile = File(..., description="Clothing image (JPEG or PNG)"),
    top_k: int = Query(default=3, ge=1, le=10),
):
    """
    **Phase 1 endpoint — unchanged.**

    Upload a clothing image and receive a wardrobe category prediction.
    """
    _require_model()
    image, tmp_path = await _decode_upload(file)
    try:
        result = predict_image(
            image_path=tmp_path,
            model=_model,
            class_names=_class_names,
            device=_device,
            top_k=top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}")
    finally:
        os.unlink(tmp_path)

    return PredictResponse(
        category=result["category"],
        confidence=result["confidence"],
        top_predictions=[
            PredictionItem(category=p["category"], confidence=p["confidence"])
            for p in result["top_predictions"]
        ],
    )


@app.post("/analyze-clothing", response_model=AnalyzeResponse,
          summary="[Phase 2] Full clothing profile")
async def analyze_clothing(
    file: UploadFile = File(..., description="Clothing image (JPEG or PNG)"),
    top_k: int = Query(default=3, ge=1, le=10),
    n_colors: int = Query(default=3, ge=1, le=8),
):
    """
    **Phase 2 endpoint.**

    Returns a full clothing profile:
    - Wardrobe category + confidence (from EfficientNet-B0 classifier)
    - Dominant color + palette (from K-Means color extraction)
    - Attributes (from dataset metadata if image is in the dataset)

    The visual embedding is computed internally but not returned in this
    response — use `/similar-clothing` for embedding-based retrieval.
    """
    _require_model()
    image, tmp_path = await _decode_upload(file)

    try:
        # ── Category prediction (Phase 1) ─────────────────────────────────────
        pred = predict_image(
            image_path=tmp_path,
            model=_model,
            class_names=_class_names,
            device=_device,
            top_k=top_k,
        )

        # ── Color extraction ──────────────────────────────────────────────────
        try:
            color_result = extract_colors(image, n_colors=n_colors)
        except Exception as ce:
            # Non-fatal: return nulls rather than crashing
            color_result = {
                "dominant_color": "unknown",
                "dominant_hex": "#000000",
                "color_palette": [],
                "color_distribution": [],
            }

        # ── Attribute lookup ──────────────────────────────────────────────────
        if _meta_index is not None:
            attr = _meta_index.lookup(tmp_path)
        else:
            attr = {
                "gender": None, "season": None, "base_colour": None,
                "article_type": None, "product_name": None,
                "source": "metadata_index_not_loaded",
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}")
    finally:
        os.unlink(tmp_path)

    return AnalyzeResponse(
        category=pred["category"],
        confidence=pred["confidence"],
        top_predictions=[
            PredictionItem(category=p["category"], confidence=p["confidence"])
            for p in pred["top_predictions"]
        ],
        dominant_color=color_result["dominant_color"],
        dominant_hex=color_result["dominant_hex"],
        color_palette=color_result["color_palette"],
        color_distribution=[
            ColorEntry(color=d["color"], percentage=d["percentage"])
            for d in color_result["color_distribution"]
        ],
        attributes=AttributesResponse(
            gender=attr.get("gender"),
            season=attr.get("season"),
            base_colour=attr.get("base_colour"),
            article_type=attr.get("article_type"),
            product_name=attr.get("product_name"),
            source=attr.get("source", "unknown"),
        ),
    )


@app.post("/similar-clothing", response_model=SimilarResponse,
          summary="[Phase 2] Find visually similar clothing")
async def similar_clothing(
    file: UploadFile = File(..., description="Clothing image (JPEG or PNG)"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """
    **Phase 2 endpoint.**

    Extracts a 1280-dim visual embedding from the uploaded image and returns
    the top-K most visually similar items from the reference index.

    Similarity is measured using cosine similarity of EfficientNet-B0 feature
    vectors. Values range from 0 (unrelated) to 1 (identical).

    **Requires**: run `python src/build_embeddings.py` first.
    """
    _require_model()

    if _sim_index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Similarity index not loaded. "
                "Run: python src/build_embeddings.py  then restart the server."
            ),
        )
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Embedding extractor not initialised.")

    image, tmp_path = await _decode_upload(file)

    try:
        # ── Category (for response context) ───────────────────────────────────
        pred = predict_image(
            image_path=tmp_path,
            model=_model,
            class_names=_class_names,
            device=_device,
            top_k=1,
        )

        # ── Visual embedding ──────────────────────────────────────────────────
        query_emb = _extractor.extract(image)

        # ── Similarity search ─────────────────────────────────────────────────
        results = _sim_index.search(query_emb, top_k=top_k, exclude_path=tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {type(e).__name__}")
    finally:
        os.unlink(tmp_path)

    return SimilarResponse(
        query_category=pred["category"],
        query_confidence=pred["confidence"],
        results=[
            SimilarItem(
                image_path=r["image_path"],
                image_name=os.path.basename(r["image_path"]),
                category=r["category"],
                similarity=r["similarity"],
            )
            for r in results
        ],
        index_size=len(_sim_index),
    )


# ───────────────────────────────────────────────────────────────────────────────
MODEL_NOTE = (
    "Compatibility scores are model-generated ranking signals based on "
    "learned category co-occurrence patterns from the 2017 Polyvore dataset. "
    "They do not represent objective fashion quality or current fashion trends."
)


def _to_wardrobe_item(item_in: WardrobeItemIn) -> WardrobeItem:
    return WardrobeItem(
        item_id=item_in.item_id,
        category=item_in.category,
        image_path=item_in.image_path,
        dominant_color=item_in.dominant_color,
        dominant_hex=item_in.dominant_hex,
        metadata=item_in.metadata,
    )


def _require_compat_model() -> None:
    if not _rec.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "Compatibility model not loaded. "
                "Run: python src/polyvore/train_compatibility.py  then restart the server."
            ),
        )


@app.post(
    "/recommend-outfits",
    response_model=RecommendOutfitsResponse,
    summary="[Phase 3] Recommend outfits from a wardrobe",
)
async def recommend_outfits(request: RecommendOutfitsRequest):
    """
    **Phase 3 endpoint.**

    Given a list of wardrobe items, generate and rank the top-K compatible
    outfit combinations.

    - Compatibility is scored using a Logistic Regression model trained on
      Polyvore category co-occurrence patterns.
    - Category constraints prevent obviously invalid combinations
      (e.g., 3 tops, dress + bottom).
    - Scores are model-generated ranking signals — not objective fashion advice.

    Requires: `python src/polyvore/train_compatibility.py` to be run first.
    """
    _require_compat_model()

    if not request.items:
        raise HTTPException(status_code=400, detail="items list is empty.")

    valid_categories = {"Tops", "Bottoms", "Dresses", "Outerwear", "Footwear", "Accessories"}
    for item in request.items:
        if item.category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category '{item.category}'. Valid: {sorted(valid_categories)}",
            )

    try:
        wardrobe = [_to_wardrobe_item(it) for it in request.items]
        results  = _rec.recommend_outfits(wardrobe, top_k=max(1, request.top_k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {type(e).__name__}")

    return RecommendOutfitsResponse(
        recommendations=[
            OutfitResult(
                items=r["items"],
                compatibility_score=r["compatibility_score"],
                categories=r["categories"],
            )
            for r in results
        ],
        model_note=MODEL_NOTE,
    )


@app.post(
    "/recommend-missing-item",
    response_model=RecommendMissingResponse,
    summary="[Phase 3] Rank candidate items to complete a partial outfit",
)
async def recommend_missing_item(request: RecommendMissingRequest):
    """
    **Phase 3 endpoint.**

    Given a partial outfit and a list of candidate items, rank the candidates
    by how well they would complete the outfit (Fill-In-The-Blank style).

    - Scoring uses the same Polyvore-trained compatibility model.
    - answer[0] is NOT assumed correct — all candidates are scored independently.
    - Scores are model-generated ranking signals — not objective fashion advice.

    Requires: `python src/polyvore/train_compatibility.py` to be run first.
    """
    _require_compat_model()

    if not request.candidate_items:
        raise HTTPException(status_code=400, detail="candidate_items list is empty.")

    try:
        outfit  = [_to_wardrobe_item(it) for it in request.outfit_items]
        cands   = [_to_wardrobe_item(it) for it in request.candidate_items]
        results = _rec.recommend_missing_item(outfit, cands, top_k=max(1, request.top_k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {type(e).__name__}")

    return RecommendMissingResponse(
        recommendations=[
            MissingItemResult(
                item=r["item"],
                compatibility_score=r["compatibility_score"],
            )
            for r in results
        ],
        model_note=MODEL_NOTE,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Wardrobe CRUD  (Phase 4 — persistence for frontend)
# Items are stored as JSON + uploaded images on disk.
# Architecture designed to migrate to PostgreSQL in a future phase.
# ───────────────────────────────────────────────────────────────────────────────

# Mount uploaded images as a static route so the frontend can load them
app.mount(
    "/wardrobe/images",
    StaticFiles(directory=str(_wdb.UPLOADS_DIR)),
    name="wardrobe_images",
)


class WardrobeItemRecord(BaseModel):
    """Full wardrobe item as returned by the API."""
    id:                 str
    category:           str
    confidence:         Optional[float] = None
    dominant_color:     str
    dominant_hex:       str
    color_palette:      list[str]
    color_distribution: list[dict]
    attributes:         dict
    image_url:          str     # e.g. "/wardrobe/images/{filename}"
    created_at:         str


def _record_to_response(record: dict, base_url: str = "") -> dict:
    """Convert a DB record to a frontend-friendly dict."""
    return {
        **record,
        "image_url": f"/wardrobe/images/{record['image_filename']}",
    }


@app.post(
    "/wardrobe/items",
    summary="[Wardrobe] Upload and analyze a clothing image, save to wardrobe",
    status_code=201,
)
async def wardrobe_add_item(
    file: UploadFile = File(..., description="Clothing image (JPEG or PNG)"),
    n_colors: int = Query(default=3, ge=1, le=8),
):
    """
    Analyzes the uploaded clothing image (Phase 1 + Phase 2) and saves the
    result as a persistent wardrobe item.

    Returns the saved item record including image_url.
    """
    _require_model()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image. Upload a valid JPEG or PNG. ({type(e).__name__})",
        )

    # Determine file extension for storage
    ct = (file.content_type or "").lower()
    ext = ".png" if "png" in ct else ".jpg"

    # Write temp file for analysis
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format="JPEG")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {type(e).__name__}")

    try:
        pred = predict_image(
            image_path=tmp_path,
            model=_model,
            class_names=_class_names,
            device=_device,
            top_k=3,
        )

        try:
            color_result = extract_colors(image, n_colors=n_colors)
        except Exception:
            color_result = {
                "dominant_color": "unknown",
                "dominant_hex": "#000000",
                "color_palette": [],
                "color_distribution": [],
            }

        if _meta_index is not None:
            attr = _meta_index.lookup(tmp_path)
        else:
            attr = {
                "gender": None, "season": None, "base_colour": None,
                "article_type": None, "product_name": None,
                "source": "metadata_index_not_loaded",
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}")
    finally:
        os.unlink(tmp_path)

    # Persist
    try:
        record = _wdb.add_item(
            image_bytes=contents,
            image_ext=ext,
            category=pred["category"],
            confidence=pred["confidence"],
            dominant_color=color_result["dominant_color"],
            dominant_hex=color_result["dominant_hex"],
            color_palette=color_result["color_palette"],
            color_distribution=color_result["color_distribution"],
            attributes=attr,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save item: {type(e).__name__}")

    return _record_to_response(record)


@app.get("/wardrobe/items", summary="[Wardrobe] List all wardrobe items")
async def wardrobe_list_items():
    """Returns all wardrobe items, newest first."""
    items = _wdb.list_items()
    return {"items": [_record_to_response(it) for it in items], "count": len(items)}


@app.get("/wardrobe/items/{item_id}", summary="[Wardrobe] Get a single wardrobe item")
async def wardrobe_get_item(item_id: str):
    item = _wdb.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")
    return _record_to_response(item)


@app.delete("/wardrobe/items/{item_id}", summary="[Wardrobe] Delete a wardrobe item")
async def wardrobe_delete_item(item_id: str):
    deleted = _wdb.delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")
    return {"deleted": item_id}
