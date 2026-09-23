from __future__ import annotations

from app.config import ENABLE_VLM, VLM_VERSION


class VLMEngine:
    """Single shared PaddleOCR-VL instance for the process.

    Both the specialized container-OCR fallback and the general field
    verifier previously loaded their own PaddleOCR-VL model independently,
    doubling VLM memory/startup cost. They now share this one instance,
    the same way PaddleOCREngine is shared for standard OCR.
    """

    def __init__(self) -> None:
        self.vlm = None
        self.error: str | None = None
        if ENABLE_VLM:
            self._load()

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCRVL

            self.vlm = PaddleOCRVL(
                pipeline_version=VLM_VERSION,
                use_layout_detection=True,
                use_seal_recognition=True,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_ocr_for_image_block=True,
            )
            print(f"[startup] Optional PaddleOCR-VL {VLM_VERSION} loaded (shared).")
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = str(exc)
            print(f"[startup] Optional PaddleOCR-VL unavailable: {exc}")
