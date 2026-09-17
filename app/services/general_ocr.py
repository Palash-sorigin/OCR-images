from pathlib import Path

from app.services.paddle_engine import PaddleOCREngine


class GeneralOCRService:
    """General screenshot/form OCR. No VLM is involved here."""

    def __init__(self, engine: PaddleOCREngine) -> None:
        self.engine = engine

    def extract(self, image_path: Path) -> dict:
        raw, items, error = self.engine.predict(str(image_path))
        return {
            "raw": raw,
            "items": items,
            "error": error,
            "text": "\n".join(item["text"] for item in items),
        }
