# 💊 Medicine Salt Identification AI

AI-powered system to identify, compare, and find alternatives for Indian medicines by salt composition.

---

## Features
| Feature | Description |
|---|---|
| **Medicine Lookup** | Enter brand name → get salt, manufacturer, price, all equivalents |
| **Compare** | Two medicines → identical / partial / different verdict |
| **Disease Search** | Enter disease → get all relevant medicines by salt group |
| **OCR Scan** | Upload medicine strip photo → auto-extract name → lookup |
| **Voice Query** | Speak naturally → routed to correct endpoint |

## Three-Tier Matching
1. **Exact** — direct brand name match
2. **Fuzzy** — handles typos, misspellings (RapidFuzz, threshold ≥ 0.85)
3. **Semantic/Synonym** — "Acetaminophen" → finds Paracetamol medicines

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (for strip scanning)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract

# Build dataset (68 medicines, 25 diseases — replace with Kaggle full dataset for production)
python data/build_dataset.py

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Open browser
open http://localhost:8000
```

---

## API Endpoints

```
POST /api/lookup    {"medicine": "Crocin"}
POST /api/compare   {"medicine1": "Crocin", "medicine2": "Dolo 650"}
POST /api/disease   {"disease": "fever"}
POST /api/ocr       {"image": "<base64>"}
POST /api/voice     {"transcript": "compare Crocin and Dolo 650"}
GET  /api/brands    → list of all known brands
GET  /health        → health check
GET  /docs          → interactive Swagger UI
```

---

## Production Upgrade (Full Dataset)

Replace `data/medicines.csv` with the full Kaggle dataset:
- [A-Z Medicine Dataset of India (253k)](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india)
- [Drug Prescription to Disease Dataset (14k)](https://www.kaggle.com/datasets/manncodes/drug-prescription-to-disease-dataset)

The engine auto-indexes everything on startup — no code changes needed.

---

## Tech Stack
- **FastAPI** — REST API
- **RapidFuzz** — Fuzzy string matching
- **scikit-learn** — ML classification (for full dataset)
- **Tesseract + OpenCV** — OCR
- **Web Speech API** — Voice input (browser-native)
- **Pandas** — Data processing

---

## Project Structure
```
medicine-ai/
├── main.py                 # FastAPI app
├── requirements.txt
├── engine/
│   ├── salt_engine.py      # Three-tier matching pipeline
│   └── ocr_processor.py    # OCR + image preprocessing
├── data/
│   ├── build_dataset.py    # Dataset builder
│   ├── medicines.csv       # Medicine database
│   ├── disease_map.json    # Disease → salt mapping
│   └── salt_synonyms.json  # Canonical salt forms
└── static/
    └── index.html          # Single-page frontend
```

---

## Research Reference
Based on: *IMRS-SCA — Intelligent Medicine Recommendation System with Salt Composition Analysis*  
IJARCCE Vol. 14, Issue 11, November 2025 · DOI: 10.17148/IJARCCE.2025.141114
