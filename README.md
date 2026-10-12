# Wardrobe Circle AI

> **B.Tech CSE Major Project** — AI-powered digital wardrobe and outfit recommendation platform.

Wardrobe Circle lets users photograph their clothing, have it automatically catalogued by AI, generate compatible outfit suggestions from their wardrobe, and identify missing items that would complete a look.

**GitHub repository:** [https://github.com/Alok-Nawani/Wardrobe-Circle](https://github.com/Alok-Nawani/Wardrobe-Circle)

---

## Features

The table below distinguishes implemented features from features planned for future development.

| Feature | Phase | Status |
|---|---|---|
| Clothing category classification (EfficientNet-B0, 6 categories) | 1 | ✅ Implemented |
| Dominant colour + palette extraction (K-Means, CIELAB) | 2 | ✅ Implemented |
| Visual attribute lookup from dataset metadata | 2 | ✅ Implemented |
| 1280-dim visual embeddings (EfficientNet-B0 backbone) | 2 | ✅ Implemented |
| Cosine-similarity clothing search | 2 | ✅ Implemented |
| Outfit compatibility scoring (Polyvore-trained Logistic Regression) | 3 | ✅ Implemented |
| Outfit recommendation from wardrobe | 3 | ✅ Implemented |
| Fill-in-the-blank missing-item recommendation | 3 | ✅ Implemented |
| Wardrobe CRUD (upload, list, delete, image serving) | 4 | ✅ Implemented |
| React frontend (Dashboard, Wardrobe, Add Clothing, Generate Outfit, Missing Item) | 4 | ✅ Implemented |
| Community borrowing / renting | — | 🔲 Planned |
| User authentication | — | 🔲 Planned |
| PostgreSQL persistence | — | 🔲 Planned |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React Frontend  (Vite + Tailwind CSS)    :5173                 │
│  pages: Dashboard · MyWardrobe · AddClothing                    │
│         GenerateOutfit · FindMissingItem                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / multipart form
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend                          :8000                  │
│  api/main.py                                                     │
│                                                                  │
│  GET  /health                  liveness + component status       │
│  POST /predict                 Phase 1: clothing classification  │
│  POST /analyze-clothing        Phase 2: full clothing profile    │
│  POST /similar-clothing        Phase 2: visual similarity search │
│  POST /recommend-outfits       Phase 3: outfit recommendation    │
│  POST /recommend-missing-item  Phase 3: fill-in-the-blank       │
│  POST /wardrobe/items          add item to wardrobe             │
│  GET  /wardrobe/items          list all wardrobe items          │
│  GET  /wardrobe/items/{id}     get one item                     │
│  DELETE /wardrobe/items/{id}   delete item                      │
│  GET  /wardrobe/images/{file}  serve uploaded image             │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│  AI Pipeline  (src/)                  │
│  Phase 1: EfficientNet-B0 classifier  │
│  Phase 2: K-Means colour, embeddings, │
│           similarity index            │
│  Phase 3: Polyvore compatibility model│
└───────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, Tailwind CSS v3, React Router DOM v7, Lucide React |
| Backend | FastAPI, Uvicorn, Pydantic |
| AI — classification | EfficientNet-B0 (PyTorch / torchvision, pretrained on ImageNet) |
| AI — colour | Mini-Batch K-Means (scikit-learn), CIELAB colour space |
| AI — embeddings | EfficientNet-B0 backbone features (1280-dim, L2-normalised) |
| AI — similarity | Cosine similarity over NumPy embedding index |
| AI — compatibility | Logistic Regression on Polyvore category co-occurrence (scikit-learn) |
| Wardrobe persistence | JSON file + uploaded images on disk |
| Training dataset | Myntra Fashion Product Images (Kaggle) |
| Compatibility dataset | Polyvore (2017, 17,316 training outfits) |

---

## Repository Structure

```
Wardrobe-Circle/
├── api/
│   ├── main.py              ← FastAPI server (all endpoints, Phases 1–4)
│   └── wardrobe_db.py       ← JSON-based wardrobe persistence layer
│
├── data/
│   ├── category_mapping.json   ← 144 Myntra articleTypes → 6 WC categories
│   ├── class_distribution.csv  ← Dataset class balance statistics
│   ├── embeddings/             ← Embedding index lives here (gitignored)
│   ├── polyvore/               ← Polyvore dataset files (gitignored)
│   ├── processed/              ← Train/val/test CSV splits (gitignored)
│   ├── raw/images/             ← Myntra images used for embedding index (gitignored)
│   └── wardrobe/               ← Per-user wardrobe data (gitignored)
│
├── frontend/
│   ├── src/
│   │   ├── api/             ← Centralised fetch wrappers (client.js + per-feature)
│   │   ├── components/      ← Shared UI components
│   │   ├── context/         ← WardrobeContext (global state)
│   │   ├── hooks/           ← Custom React hooks
│   │   └── pages/           ← Route-level page components
│   ├── .env.example         ← Frontend environment variable template
│   └── package.json
│
├── models/
│   ├── class_names.json          ← 6 category names (committed, required at startup)
│   └── compatibility_model.pkl   ← Polyvore-trained model (committed, 340 KB)
│                                    clothing_classifier_best.pth is NOT in Git
│
├── results/
│   ├── metrics.json              ← Phase 1 test accuracy + per-class metrics
│   ├── confusion_matrix.png      ← Normalised confusion matrix
│   ├── training_history.png      ← Loss/accuracy training curves
│   └── compatibility_metrics.json ← Phase 3 evaluation results
│
├── src/
│   ├── train.py             ← Phase 1 training loop
│   ├── evaluate.py          ← Phase 1 test evaluation
│   ├── predict.py           ← Single-image inference (CLI)
│   ├── model.py             ← EfficientNet-B0 model builder
│   ├── dataset.py           ← PyTorch Dataset + data transforms
│   ├── color.py             ← K-Means colour extraction
│   ├── attributes.py        ← Dataset metadata index + attribute lookup
│   ├── embeddings.py        ← 1280-dim embedding extractor
│   ├── similarity.py        ← Cosine similarity search
│   ├── build_embeddings.py  ← Builds data/embeddings/index.npz
│   ├── recommendation.py    ← Public Phase 3 recommendation API
│   ├── candidate_generation.py ← Outfit candidate generation + filtering
│   ├── data/
│   │   ├── inspect_dataset.py   ← Inspect Myntra dataset, generate category mapping
│   │   └── prepare_dataset.py   ← Create train/val/test CSV splits
│   └── polyvore/
│       ├── loader.py            ← Load the 5 Polyvore JSON/txt files
│       ├── compatibility.py     ← Category co-occurrence compatibility model
│       └── train_compatibility.py ← Train + evaluate the compatibility model
│
├── tests/
│   ├── test_color.py        ← 22 tests — colour extraction (standalone)
│   ├── test_embeddings.py   ← 36 tests — embedding extractor (standalone)
│   ├── test_phase3.py       ← 103 tests — Phase 3 compatibility (standalone)
│   ├── test_api.py          ← 51 tests — Phase 1+2 API (requires running server)
│   └── test_wardrobe_api.py ← Wardrobe CRUD tests (requires running server)
│
├── .env.example             ← Root environment variable template (backend)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** and npm
- **~3 GB disk space** for datasets and model weights (none of this is included in Git)

> The backend requires PyTorch. On Apple Silicon, MPS acceleration is detected automatically. On NVIDIA, CUDA is used. CPU is the fallback.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Alok-Nawani/Wardrobe-Circle.git
cd Wardrobe-Circle
```

### 2. Install backend dependencies

It is strongly recommended to use a virtual environment to avoid conflicts with system packages:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure environment variables

**Frontend** (required):

```bash
cp frontend/.env.example frontend/.env
# Edit frontend/.env if your backend runs on a port other than 8000
```

The only variable needed:

```
VITE_API_BASE_URL=http://localhost:8000
```

**Backend**: No environment variables are required for the current implementation. The root `.env.example` is a placeholder for future configuration (e.g. a database URL or secret key).

---

## Obtaining Required Model Artifacts

Two model files are required. One is committed to Git; the other must be generated locally.

### `models/clothing_classifier_best.pth` — **not in Git** (16 MB)

This is the trained EfficientNet-B0 classifier. It is excluded from Git because of its size. Without it, the server starts but all of the following endpoints return HTTP 503:

- `POST /predict`
- `POST /analyze-clothing`
- `POST /similar-clothing`

**How to generate it** — requires the Myntra dataset (see [Obtaining Required Datasets](#obtaining-required-datasets)):

```bash
# Step 1: Prepare the dataset splits (run once)
python src/data/prepare_dataset.py

# Step 2: Train the classifier
python src/train.py --epochs 5                    # quick run, ~10 min on CPU
python src/train.py --epochs 20 --finetune        # full training, recommended
```

The checkpoint is saved automatically to `models/clothing_classifier_best.pth`.

### `models/compatibility_model.pkl` — **committed** (340 KB)

This file is already in the repository. No action required. It enables:

- `POST /recommend-outfits`
- `POST /recommend-missing-item`

If you ever need to retrain it (requires Polyvore data — see below):

```bash
python src/polyvore/train_compatibility.py
```

### `data/embeddings/index.npz` — **not in Git** (14 MB)

Required for `POST /similar-clothing`. Build it after training the classifier:

```bash
python src/build_embeddings.py
```

---

## Obtaining Required Datasets

### Myntra Fashion Product Images (for classifier training)

1. Download the dataset from Kaggle:  
   [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)  
   The downloaded file is `archive.zip` (approximately 565 MB).

2. Place `archive.zip` one directory above the project root (i.e. at `../archive.zip` relative to the repository).

3. Run the dataset preparation scripts:

```bash
# Inspect the dataset and generate the category mapping
python src/data/inspect_dataset.py

# Extract images and create train/val/test splits
python src/data/prepare_dataset.py
```

Both scripts auto-detect `../archive.zip` by default. Run `python src/data/inspect_dataset.py --help` or `python src/data/prepare_dataset.py --help` for available options.

### Polyvore Dataset (for retraining the compatibility model)

Source: [https://github.com/xthan/polyvore-dataset/](https://github.com/xthan/polyvore-dataset/)

> ⚠️ **Redistribution permissions for the Polyvore JSON files have not been verified.** These files are excluded from this repository. Download them from the official source linked above.

After downloading `polyvore.tar.gz`, extract it into `data/polyvore/`:

```bash
tar -xzf polyvore.tar.gz -C data/polyvore/
```

Expected files after extraction:

```
data/polyvore/train_no_dup.json                      (43.6 MB)
data/polyvore/valid_no_dup.json                       (3.5 MB)
data/polyvore/test_no_dup.json                        (7.3 MB)
data/polyvore/fill_in_blank_test.json                 (1.1 MB)
data/polyvore/fashion_compatibility_prediction.txt  (516 KB)
```

---

## Running the Application

### Start the backend

The server **must be started from the project root** (not from inside `api/`):

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The server loads models at startup and prints status for each component. If a model file is missing, the server still starts but returns HTTP 503 for the affected endpoints. Check `GET /health` for component status.

Interactive API documentation is available at: `http://localhost:8000/docs`

```bash
curl http://localhost:8000/health
```

### Start the frontend

In a separate terminal, from the project root:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Running Tests

Tests must be run from the **project root** directory, not from inside `src/` or `tests/`.

### Standalone tests (no server required)

```bash
# Phase 2 — colour extraction (22 tests)
python tests/test_color.py

# Phase 2 — embedding extractor (36 tests)
python tests/test_embeddings.py

# Phase 3 — compatibility model (103 tests)
python tests/test_phase3.py
```

> **Note:** `test_color.py` and the compatibility-model section of `test_phase3.py` require a working `scikit-learn` / `scipy` installation. See [Troubleshooting](#troubleshooting) if you encounter an import error.

### Tests requiring a running API server

Start the backend first, then in a separate terminal:

```bash
# Phase 1+2 API endpoints (51 tests)
python tests/test_api.py

# Wardrobe CRUD API endpoints
python tests/test_wardrobe_api.py
```

### Frontend production build

```bash
cd frontend
npm run build
```

### Known test status

| Test file | Last known result | Notes |
|---|---|---|
| `test_embeddings.py` | ✅ **36/36 passed** | Runs standalone; requires model + index |
| `test_phase3.py` (dataset loading) | ✅ **15/15 passed** | Polyvore loader tests pass |
| `test_phase3.py` (compatibility model) | ⚠️ **Not run** | Blocked by broken system scipy `.so` — project code is not at fault |
| `test_color.py` | ⚠️ **Not run** | Blocked by same scipy issue |
| `test_api.py` | ⏸ **Not run** | Requires a running server |
| `test_wardrobe_api.py` | ⏸ **Not run** | Requires a running server |
| Frontend `npm run build` | ✅ **Clean** | Built in 1.53 s, zero warnings |

---

## Project Evaluation Results

The figures below are results from evaluating the models on held-out test sets during project development. They reflect model performance on the specific datasets used and do not constitute guarantees of real-world performance.

### Clothing Classifier (Phase 1)

Trained on the Myntra Fashion Product Images dataset — 500 samples per class, 5 epochs, EfficientNet-B0 backbone.

| Metric | Value |
|---|---|
| Test accuracy | **86.0%** |
| Macro F1 | 0.860 |
| Best category | Footwear (F1 = 0.987) |
| Weakest category | Tops (F1 = 0.777) |

Full per-class breakdown: `results/metrics.json`  
Confusion matrix: `results/confusion_matrix.png`

### Compatibility Model (Phase 3)

> **Important:** This model scores outfit compatibility based on **category co-occurrence statistics** learned from 17,316 Polyvore training outfits. It does **not** analyse the visual colour, texture, or style of clothing images — the Polyvore CDN images are no longer accessible. Compatibility scores are ranking signals derived from which clothing categories historically appeared together in curated outfits. They do not represent objective fashion advice.

Evaluated on 7,076 Polyvore compatibility labels and 3,076 fill-in-the-blank questions.

| Metric | Value |
|---|---|
| Accuracy | 76.9% |
| Precision | 69.9% |
| Recall | 82.3% |
| F1 | 75.6% |
| FITB Top-1 accuracy | 56.6% |
| FITB Top-3 accuracy | 95.9% |

Full results: `results/compatibility_metrics.json`

---

## Known Limitations

1. **Compatibility is category-based, not visual.** The compatibility model uses category co-occurrence from 2017 Polyvore data. It does not evaluate visual colour, texture, or style because Polyvore CDN images are no longer available. Scores indicate which clothing categories commonly appear together in historical outfit data.
2. **Historical data.** The compatibility model reflects 2017 Polyvore fashion patterns and does not capture current trends.
3. **Category mapping is approximate.** Polyvore's 363 category IDs are mapped to 6 Wardrobe Circle categories. Fine-grained distinctions (e.g. different types of tops) are lost.
4. **Attributes are only available for Myntra dataset items.** The `POST /analyze-clothing` endpoint returns metadata attributes (gender, season, etc.) by looking up the uploaded image in the Myntra dataset index. Newly uploaded images that are not in the dataset return `null` for these fields. No attributes are fabricated.
5. **File-based wardrobe storage.** The wardrobe is stored as `data/wardrobe/wardrobe.json` with images on disk. This is intentionally simple and designed for future migration to a database.
6. **`data/raw/images/` is not in Git.** This directory holds images used to build the visual embedding index. It must be generated locally from the Myntra dataset using `prepare_dataset.py` (see [Obtaining Required Datasets](#obtaining-required-datasets)).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| API returns 503 for `/predict`, `/analyze-clothing`, `/similar-clothing` | `models/clothing_classifier_best.pth` is missing. Run `python src/train.py --epochs 5` after preparing the dataset |
| API returns 503 for `/similar-clothing` | `data/embeddings/index.npz` is missing. Run `python src/build_embeddings.py` |
| API returns 503 for `/recommend-outfits` or `/recommend-missing-item` | `models/compatibility_model.pkl` is missing. Run `python src/polyvore/train_compatibility.py` (requires Polyvore data) |
| `uvicorn: command not found` | Run `pip install uvicorn` or use `python -m uvicorn api.main:app --reload` |
| Frontend cannot reach backend | Check `frontend/.env` has `VITE_API_BASE_URL=http://localhost:8000` and the backend is running |
| Import errors in tests | Run tests from the project root, not from inside `src/` or `tests/` |
| `sklearn` / `scipy` import error | Install in a virtual environment: `pip install --upgrade scipy scikit-learn`, or reinstall scipy: `pip install --force-reinstall scipy` |
| `ModuleNotFoundError` for `polyvore` or `color` | Ensure you are running from the project root; the test files add `src/` to the path automatically |
