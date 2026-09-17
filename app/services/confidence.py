from __future__ import annotations

from app.config import OCR_HIGH_CONFIDENCE_THRESHOLD, OCR_REVIEW_THRESHOLD


class ConfidenceEngine:
    def assess(self, ocr_confidence: float | None, validation: dict, *, source: str = "OCR", risk: str = "medium") -> tuple[str, float]:
        score = float(ocr_confidence or 0.0)
        if validation.get("valid") is True:
            score = min(1.0, score + (0.03 if risk == "high" else 0.05))
        elif validation.get("valid") is False and ocr_confidence is not None:
            score = max(0.0, score - 0.25)

        threshold = OCR_HIGH_CONFIDENCE_THRESHOLD + (0.05 if risk == "high" else 0.0)
        if validation.get("valid") is True and score >= min(0.99, threshold):
            status = "VERIFIED"
        elif score >= OCR_REVIEW_THRESHOLD and validation.get("valid") is not False:
            status = "HIGH_CONFIDENCE"
        else:
            status = "NEEDS_REVIEW"

        if source == "VLM" and validation.get("valid") is True:
            status = "VLM_VERIFIED"

        return status, round(score, 4)
