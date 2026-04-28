"""
Three-tier Salt Matching Engine
Tier 1: Exact string match
Tier 2: Fuzzy match (rapidfuzz, threshold ≥ 0.85)
Tier 3: Synonym-aware semantic fallback
"""

import json
import re
import pandas as pd
from rapidfuzz import fuzz, process
from typing import Optional
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class SaltEngine:
    def __init__(self):
        self.medicines: pd.DataFrame = pd.DataFrame()
        self.disease_map: dict = {}
        self.salt_synonyms: dict = {}
        self._salt_index: dict = {}          # salt_normalized → [medicine rows]
        self._brand_index: dict = {}         # brand_lower → row
        self._synonym_reverse: dict = {}     # synonym_lower → canonical_name
        self._loaded = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self):
        med_path = os.path.join(DATA_DIR, "medicines.csv")
        dis_path = os.path.join(DATA_DIR, "disease_map.json")
        syn_path = os.path.join(DATA_DIR, "salt_synonyms.json")

        if not os.path.exists(med_path):
            raise FileNotFoundError(f"Dataset not found at {med_path}. Run data/build_dataset.py first.")

        self.medicines = pd.read_csv(med_path)
        with open(dis_path) as f:
            self.disease_map = json.load(f)
        with open(syn_path) as f:
            self.salt_synonyms = json.load(f)

        self._build_indexes()
        self._loaded = True
        print(f"✓ SaltEngine loaded: {len(self.medicines)} medicines")

    def _build_indexes(self):
        # Brand → row
        for _, row in self.medicines.iterrows():
            self._brand_index[row["brand"].lower()] = row.to_dict()

        # Salt → [rows]
        for _, row in self.medicines.iterrows():
            key = self._normalize_salt(row["salt"])
            self._salt_index.setdefault(key, []).append(row.to_dict())

        # Synonym reverse lookup
        for canonical, synonyms in self.salt_synonyms.items():
            for syn in synonyms:
                self._synonym_reverse[syn.lower()] = canonical

    def _normalize_salt(self, salt: str) -> str:
        """Lower, strip doses, sort components so order doesn't matter."""
        salt = salt.lower()
        # Remove dosage numbers: "500mg", "40 mg", "60000iu", etc.
        salt = re.sub(r"\d+\s*(mg|mcg|iu|g|ml|%)", "", salt)
        # Split on '+', sort components, rejoin
        parts = [p.strip() for p in salt.split("+")]
        parts = sorted(p.strip() for p in parts if p.strip())
        return " + ".join(parts)

    # ------------------------------------------------------------------
    # Tier 1 — Exact
    # ------------------------------------------------------------------
    def _exact_lookup(self, query: str) -> Optional[dict]:
        return self._brand_index.get(query.lower())

    # ------------------------------------------------------------------
    # Tier 2 — Fuzzy
    # ------------------------------------------------------------------
    def _fuzzy_lookup(self, query: str, threshold: int = 85) -> Optional[dict]:
        brands = list(self._brand_index.keys())
        result = process.extractOne(query.lower(), brands, scorer=fuzz.WRatio)
        if result and result[1] >= threshold:
            return self._brand_index[result[0]], result[1]
        return None, 0

    # ------------------------------------------------------------------
    # Tier 3 — Synonym / semantic
    # ------------------------------------------------------------------
    def _synonym_lookup(self, query: str) -> Optional[str]:
        """Return canonical salt name if query matches a known synonym."""
        q = query.lower()
        # Direct synonym match
        if q in self._synonym_reverse:
            return self._synonym_reverse[q]
        # Partial synonym match
        for syn, canonical in self._synonym_reverse.items():
            if syn in q or q in syn:
                return canonical
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def lookup(self, medicine_name: str) -> dict:
        """
        Look up a medicine by brand name.
        Returns: brand info + all equivalent alternatives.
        """
        # Tier 1
        row = self._exact_lookup(medicine_name)
        match_tier = 1
        confidence = 1.0

        # Tier 2
        if not row:
            row, score = self._fuzzy_lookup(medicine_name)
            if row:
                match_tier = 2
                confidence = score / 100
            else:
                # Tier 3 — try synonym → find any medicine with that salt
                canonical = self._synonym_lookup(medicine_name)
                if canonical:
                    for salt_key, rows in self._salt_index.items():
                        if canonical.lower() in salt_key:
                            row = rows[0]
                            match_tier = 3
                            confidence = 0.70
                            break

        if not row:
            return {"found": False, "query": medicine_name}

        # Find all medicines with same normalized salt
        salt_key = self._normalize_salt(row["salt"])
        alternatives = self._salt_index.get(salt_key, [])
        alternatives = [m for m in alternatives if m["brand"] != row["brand"]]

        return {
            "found": True,
            "query": medicine_name,
            "matched_brand": row["brand"],
            "salt": row["salt"],
            "manufacturer": row["manufacturer"],
            "type": row["type"],
            "price": row["price"],
            "match_tier": match_tier,
            "confidence": round(confidence, 2),
            "alternatives": sorted(alternatives, key=lambda x: x["price"]),
            "salt_normalized": salt_key,
        }

    def compare(self, med1: str, med2: str) -> dict:
        """Compare two medicines. Returns equivalence verdict."""
        r1 = self.lookup(med1)
        r2 = self.lookup(med2)

        if not r1["found"] or not r2["found"]:
            return {
                "found": r1["found"] and r2["found"],
                "med1": r1,
                "med2": r2,
                "verdict": "unknown",
            }

        s1 = r1["salt_normalized"]
        s2 = r2["salt_normalized"]

        # Exact salt match
        if s1 == s2:
            verdict = "identical"
            verdict_text = "Same salt — therapeutically equivalent"
            safe_to_substitute = True
        else:
            # Check overlap (multi-salt combos)
            s1_parts = set(s1.split(" + "))
            s2_parts = set(s2.split(" + "))
            overlap = s1_parts & s2_parts
            if overlap:
                verdict = "partial"
                verdict_text = f"Partial overlap: shared salt(s) — {', '.join(overlap)}"
                safe_to_substitute = False
            else:
                # Fuzzy similarity between salt strings
                sim = fuzz.ratio(s1, s2)
                if sim >= 80:
                    verdict = "similar"
                    verdict_text = "Similar salt class — consult pharmacist"
                    safe_to_substitute = False
                else:
                    verdict = "different"
                    verdict_text = "Different salts — not interchangeable"
                    safe_to_substitute = False

        price_diff = r2["price"] - r1["price"]

        return {
            "found": True,
            "med1": r1,
            "med2": r2,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "safe_to_substitute": safe_to_substitute,
            "price_diff": price_diff,
            "cheaper": r1["matched_brand"] if r1["price"] < r2["price"] else r2["matched_brand"],
            "warning": "Always consult a licensed pharmacist or doctor before substituting medicines.",
        }

    def disease_search(self, disease: str) -> dict:
        """Find medicines for a given disease/symptom."""
        d = disease.lower().strip()

        # Direct match
        salts = self.disease_map.get(d)

        # Fuzzy match on disease name
        if not salts:
            result = process.extractOne(d, list(self.disease_map.keys()), scorer=fuzz.WRatio)
            if result and result[1] >= 75:
                salts = self.disease_map[result[0]]
                d = result[0]

        if not salts:
            return {"found": False, "disease": disease, "medicines": []}

        medicines_by_salt = []
        for salt in salts:
            salt_key = self._normalize_salt(salt)
            options = []
            for key, rows in self._salt_index.items():
                if salt_key in key or key in salt_key:
                    options.extend(rows)
            if options:
                medicines_by_salt.append({
                    "salt": salt,
                    "medicines": sorted(options, key=lambda x: x["price"])[:5],
                })

        return {
            "found": True,
            "disease": d,
            "salts": salts,
            "medicines_by_salt": medicines_by_salt,
            "warning": "These are general recommendations only. Always consult a doctor for diagnosis and prescription.",
        }

    def all_brands(self) -> list:
        return sorted(self._brand_index.keys())


# Singleton
_engine: Optional[SaltEngine] = None

def get_engine() -> SaltEngine:
    global _engine
    if _engine is None:
        _engine = SaltEngine()
        _engine.load()
    return _engine
