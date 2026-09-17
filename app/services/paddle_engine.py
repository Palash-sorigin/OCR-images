from __future__ import annotations

from typing import Any

from app.config import OCR_CPU_THREADS


class PaddleOCREngine:
    """Single shared Standard PaddleOCR instance for the process."""

    def __init__(self) -> None:
        self.ocr = None
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine_config={
                    "paddle_static": {
                        "run_mode": "paddle",
                        "cpu_threads": OCR_CPU_THREADS,
                        "enable_cinn": False,
                        "mkldnn_cache_capacity": 0,
                    }
                },
            )
            print("[startup] Standard PaddleOCR loaded.")
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = str(exc)
            print(f"[startup] Standard PaddleOCR failed: {exc}")

    @staticmethod
    def _result_json(result: Any) -> dict:
        try:
            value = result.json
            return value if isinstance(value, dict) else {"result": str(value)}
        except Exception:
            return {"result": str(result)}

    def predict(self, image_path: str) -> tuple[list[dict], list[dict], str | None]:
        if self.ocr is None:
            return [], [], self.error or "PaddleOCR is unavailable."

        try:
            raw: list[dict] = []
            items: list[dict] = []
            results = self.ocr.predict(image_path)

            for result in results:
                result_dict = self._result_json(result)
                raw.append(result_dict)
                res = result_dict.get("res", {}) or {}
                texts = res.get("rec_texts", []) or []
                scores = res.get("rec_scores", []) or []
                boxes = res.get("rec_boxes", []) or []

                for text, score, box in zip(texts, scores, boxes):
                    items.append(
                        {
                            "text": str(text),
                            "confidence": float(score),
                            "box": box,
                        }
                    )

            return raw, items, None
        except Exception as exc:  # pragma: no cover - model dependent
            return [], [], str(exc)
