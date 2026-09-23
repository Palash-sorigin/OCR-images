from __future__ import annotations

from pathlib import Path

from app.services.vlm_engine import VLMEngine
from app.utils.candidates import extract_iso_candidates


class VLMVerifier:
    """Optional, safe VLM fallback. It never overrides deterministic validation by itself.

    extract_text() runs the (expensive) VLM prediction exactly once. Callers
    that need to check several fields against the same image should call it
    once and pass the resulting text into match_field() for each field,
    instead of re-running the VLM per field.
    """

    def __init__(self, engine: VLMEngine | None = None) -> None:
        self.vlm = engine.vlm if engine else None
        self.error: str | None = engine.error if engine else "VLM is disabled."

    def extract_text(self, image_path: Path) -> dict:
        if self.vlm is None:
            return {"available": False, "text": "", "raw": [], "error": self.error or "VLM is disabled."}

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

            return {"available": True, "text": "\n".join(text_parts), "raw": raw, "error": None}
        except Exception as exc:  # pragma: no cover
            return {"available": True, "text": "", "raw": [], "error": str(exc)}

    @staticmethod
    def match_field(text: str, field_name: str, expected_value: str | None = None) -> dict:
        matched = bool(expected_value and expected_value.upper() in text.upper())
        return {
            "available": True,
            "status": "MATCH" if matched else "NO_MATCH",
            "field": field_name,
            "expected_value": expected_value,
            "matched": matched,
            "iso_candidates": extract_iso_candidates(text),
        }
