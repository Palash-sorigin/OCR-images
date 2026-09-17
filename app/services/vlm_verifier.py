from __future__ import annotations

from pathlib import Path

from app.config import ENABLE_VLM, VLM_VERSION
from app.utils.candidates import extract_iso_candidates


class VLMVerifier:
    """Optional, safe VLM fallback. It never overrides deterministic validation by itself."""

    def __init__(self) -> None:
        self.vlm = None
        self.error: str | None = None
        if ENABLE_VLM:
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
                print(f"[startup] Optional PaddleOCR-VL {VLM_VERSION} loaded.")
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                print(f"[startup] Optional PaddleOCR-VL unavailable: {exc}")

    def verify_field(self, image_path: Path, field_name: str, expected_value: str | None = None) -> dict:
        if self.vlm is None:
            return {"available": False, "status": "NOT_CONFIGURED", "field": field_name, "reason": self.error or "VLM is disabled."}

        try:
            results = self.vlm.predict(str(image_path))
            raw = []
            text_parts = []
            for result in results:
                try:
                    data = result.json
                except Exception:
                    data = {"result": str(result)}
                raw.append(data)
                res = data.get("res", {}) if isinstance(data, dict) else {}
                for block in (res.get("parsing_res_list", []) or []):
                    content = block.get("block_content", "")
                    if content:
                        text_parts.append(str(content))

            text = "\n".join(text_parts)
            matched = bool(expected_value and expected_value.upper() in text.upper())
            iso_candidates = extract_iso_candidates(text)
            return {
                "available": True,
                "status": "MATCH" if matched else "NO_MATCH",
                "field": field_name,
                "expected_value": expected_value,
                "matched": matched,
                "iso_candidates": iso_candidates,
                "raw": raw,
            }
        except Exception as exc:  # pragma: no cover
            return {"available": True, "status": "ERROR", "field": field_name, "error": str(exc)}
