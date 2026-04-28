"""
OCR Processor — extracts medicine names from strip/packet images.
Uses OpenCV for preprocessing + Tesseract for text extraction.
"""

import re
import base64
import io
from typing import Optional
import cv2
import numpy as np
import pytesseract
from PIL import Image


class OCRProcessor:
    def __init__(self):
        # Tesseract config: treat as single block of text, alphanumeric + common chars
        self.tess_config = r"--oem 3 --psm 6"

    def _decode_image(self, image_data: str) -> np.ndarray:
        """Accept base64 string or raw bytes, return OpenCV image."""
        if isinstance(image_data, str):
            # Strip data URI prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
        else:
            img_bytes = image_data

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline optimised for medicine strip text:
        1. Grayscale
        2. Resize to standard height (helps Tesseract with small fonts)
        3. CLAHE contrast enhancement
        4. Adaptive threshold (handles uneven lighting on foil strips)
        5. Slight denoise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale up if image is small
        h, w = gray.shape
        if h < 400:
            scale = 400 / h
            gray = cv2.resize(gray, (int(w * scale), 400), interpolation=cv2.INTER_CUBIC)

        # CLAHE — improves contrast in local regions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Slight denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        return denoised

    def _extract_text(self, img: np.ndarray) -> str:
        """Run Tesseract on preprocessed image."""
        pil_img = Image.fromarray(img)
        text = pytesseract.image_to_string(pil_img, config=self.tess_config)
        return text.strip()

    def _parse_medicine_names(self, raw_text: str) -> list[str]:
        """
        Heuristic parser to extract likely medicine/brand names from OCR output.
        Looks for:
        - Capitalized words (brand names are usually title case or ALL CAPS)
        - Lines with dosage patterns (e.g. "500mg", "10 mg")
        - Lines that look like medicine names (not pure numbers/symbols)
        """
        candidates = []
        lines = raw_text.split("\n")

        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue

            # Skip lines that are mostly numbers/symbols
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.4:
                continue

            # Skip common non-medicine lines
            skip_keywords = ["batch", "mfg", "exp", "lot", "store", "keep", "manufactured",
                             "marketed", "distributed", "price", "mrp", "gst", "tel", "www"]
            if any(kw in line.lower() for kw in skip_keywords):
                continue

            # Prioritise lines with dosage info (likely medicine + dose)
            has_dose = bool(re.search(r"\d+\s*(mg|mcg|ml|iu|g|%)", line, re.IGNORECASE))

            # Clean up OCR artifacts
            cleaned = re.sub(r"[^\w\s\+\-\.]", " ", line).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)

            if cleaned:
                candidates.append({
                    "text": cleaned,
                    "has_dose": has_dose,
                    "confidence": "high" if has_dose else "medium",
                })

        # Sort: lines with dosage info first
        candidates.sort(key=lambda x: x["has_dose"], reverse=True)
        return candidates[:5]  # Top 5 candidates

    def process(self, image_data: str) -> dict:
        """
        Main entry point. Takes base64 image, returns extracted medicine name candidates.
        """
        try:
            img = self._decode_image(image_data)
            processed = self._preprocess(img)
            raw_text = self._extract_text(processed)
            candidates = self._parse_medicine_names(raw_text)

            return {
                "success": True,
                "raw_text": raw_text,
                "candidates": candidates,
                "top_candidate": candidates[0]["text"] if candidates else None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_text": "",
                "candidates": [],
                "top_candidate": None,
            }

    def process_and_lookup(self, image_data: str, engine) -> dict:
        """OCR → auto-lookup the top candidate in the medicine database."""
        ocr_result = self.process(image_data)
        if not ocr_result["success"] or not ocr_result["top_candidate"]:
            return {**ocr_result, "lookup": None}

        # Try all candidates until we get a match
        lookup_result = None
        for candidate in ocr_result["candidates"]:
            # Extract just the first word/phrase (brand name usually comes first)
            name_guess = candidate["text"].split()[0]
            result = engine.lookup(name_guess)
            if result["found"]:
                lookup_result = result
                break

        return {**ocr_result, "lookup": lookup_result}


# Singleton
_ocr: Optional[OCRProcessor] = None

def get_ocr() -> OCRProcessor:
    global _ocr
    if _ocr is None:
        _ocr = OCRProcessor()
    return _ocr
