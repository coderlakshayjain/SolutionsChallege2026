"""
Medicine Salt Identification AI — FastAPI Backend
Endpoints:
  POST /api/lookup   — medicine name → salt + alternatives
  POST /api/compare  — compare two medicines
  POST /api/disease  — disease → medicine suggestions
  POST /api/ocr      — image → OCR → lookup
  POST /api/voice    — voice transcript → route to lookup/compare/disease
  GET  /api/brands   — list all known brand names (for autocomplete)
  GET  /health       — health check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from engine.salt_engine import get_engine
from engine.ocr_processor import get_ocr


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build dataset if not present
    from data.build_dataset import build_and_save
    import os
    if not os.path.exists("data/medicines.csv"):
        print("Building dataset...")
        build_and_save()
    # Load engine
    get_engine()
    get_ocr()
    yield

app = FastAPI(
    title="Medicine Salt AI",
    description="Identify, compare and find alternatives for Indian medicines by salt composition",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class LookupRequest(BaseModel):
    medicine: str = Field(..., example="Crocin", description="Brand name of the medicine")

class CompareRequest(BaseModel):
    medicine1: str = Field(..., example="Crocin")
    medicine2: str = Field(..., example="Dolo 650")

class DiseaseRequest(BaseModel):
    disease: str = Field(..., example="fever")

class OCRRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image of medicine strip")

class VoiceRequest(BaseModel):
    transcript: str = Field(..., example="compare Crocin and Dolo 650")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Medicine Salt AI is running", "docs": "/docs"}

@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "status": "ok",
        "medicines_loaded": len(engine.medicines),
        "diseases_loaded": len(engine.disease_map),
    }

@app.get("/api/brands")
async def list_brands():
    """Return all known brand names for autocomplete."""
    engine = get_engine()
    return {"brands": engine.all_brands()}

@app.post("/api/lookup")
async def lookup(req: LookupRequest):
    """Look up a medicine by brand name. Returns salt info and alternatives."""
    if not req.medicine.strip():
        raise HTTPException(400, "Medicine name cannot be empty")
    engine = get_engine()
    result = engine.lookup(req.medicine.strip())
    if not result["found"]:
        raise HTTPException(404, f"Medicine '{req.medicine}' not found in database")
    return result

@app.post("/api/compare")
async def compare(req: CompareRequest):
    """Compare two medicines and return equivalence verdict."""
    if not req.medicine1.strip() or not req.medicine2.strip():
        raise HTTPException(400, "Both medicine names are required")
    engine = get_engine()
    result = engine.compare(req.medicine1.strip(), req.medicine2.strip())
    return result

@app.post("/api/disease")
async def disease_search(req: DiseaseRequest):
    """Find medicines for a given disease or symptom."""
    if not req.disease.strip():
        raise HTTPException(400, "Disease name cannot be empty")
    engine = get_engine()
    result = engine.disease_search(req.disease.strip())
    if not result["found"]:
        raise HTTPException(404, f"No medicines found for '{req.disease}'")
    return result

@app.post("/api/ocr")
async def ocr_scan(req: OCRRequest):
    """Extract medicine name from strip image (base64) and look it up."""
    if not req.image:
        raise HTTPException(400, "Image data is required")
    engine = get_engine()
    ocr = get_ocr()
    result = ocr.process_and_lookup(req.image, engine)
    if not result["success"]:
        raise HTTPException(422, f"OCR failed: {result.get('error', 'Unknown error')}")
    return result

@app.post("/api/voice")
async def voice_query(req: VoiceRequest):
    """
    Parse a natural language voice transcript and route to the correct API.
    Examples:
      "compare Crocin and Dolo 650" → compare
      "what is Augmentin" → lookup
      "medicines for fever" → disease
    """
    t = req.transcript.lower().strip()
    engine = get_engine()

    # Detect intent
    compare_pattern = re.search(
        r"compare\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+)", t
    )
    disease_pattern = re.search(
        r"(?:medicines?|tablets?|drugs?|treatment)\s+(?:for|to treat|for treating)\s+(.+)", t
    )
    lookup_patterns = [
        re.search(r"(?:what is|lookup|find|search for|tell me about|show me)\s+(.+)", t),
        re.search(r"^(.+?)\s+(?:tablet|capsule|syrup|medicine|drug)$", t),
    ]

    if compare_pattern:
        med1 = compare_pattern.group(1).strip().title()
        med2 = compare_pattern.group(2).strip().title()
        result = engine.compare(med1, med2)
        return {"intent": "compare", "params": {"medicine1": med1, "medicine2": med2}, "result": result}

    elif disease_pattern:
        disease = disease_pattern.group(1).strip()
        result = engine.disease_search(disease)
        return {"intent": "disease", "params": {"disease": disease}, "result": result}

    else:
        # Try lookup patterns
        for pattern in lookup_patterns:
            if pattern:
                medicine = pattern.group(1).strip().title()
                result = engine.lookup(medicine)
                if result["found"]:
                    return {"intent": "lookup", "params": {"medicine": medicine}, "result": result}

        # Fallback: treat whole transcript as medicine name
        medicine = req.transcript.strip().title()
        result = engine.lookup(medicine)
        if result["found"]:
            return {"intent": "lookup", "params": {"medicine": medicine}, "result": result}

        raise HTTPException(422, f"Could not understand query: '{req.transcript}'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
