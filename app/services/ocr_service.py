from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import ENABLE_VLM, OUTPUT_DIR, SAVE_DEBUG_OUTPUT, VLM_VERSION
from app.services.paddle_engine import PaddleOCREngine
from app.utils.candidates import (
    extract_from_json_objects,
    extract_iso_candidates,
    ocr_text_candidates,
    spatial_container_candidates,
)
from app.utils.iso6346 import complete_iso6346_number, is_valid_iso6346, normalize_container_number
from app.utils.preprocessing import preprocess_image


class ContainerOCRService:
    """Specialized container-number OCR: standard PaddleOCR first, optional VLM fallback."""

    def __init__(self, engine: PaddleOCREngine | None = None) -> None:
        self.engine = engine or PaddleOCREngine()
        self.ocr = self.engine.ocr
        self.ocr_error = self.engine.error
        self.vlm = None
        self.vlm_error = None
        if ENABLE_VLM:
            self._load_vlm()

    def _load_vlm(self) -> None:
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
            print(f"[startup] Optional PaddleOCR-VL {VLM_VERSION} loaded for container fallback.")
        except Exception as exc:  # pragma: no cover
            self.vlm_error = str(exc)
            print(f"[startup] Optional PaddleOCR-VL unavailable: {exc}")

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @staticmethod
    def _result_json(result: Any) -> dict:
        try:
            value = result.json
            return value if isinstance(value, dict) else {"result": str(value)}
        except Exception:
            return {"result": str(result)}

    @staticmethod
    def _expand_and_validate_candidates(candidates: list[str]) -> list[str]:
        valid: list[str] = []
        for candidate in candidates:
            candidate = normalize_container_number(candidate)
            if is_valid_iso6346(candidate):
                valid.append(candidate)
                continue
            completed = complete_iso6346_number(candidate)
            if completed and is_valid_iso6346(completed):
                valid.append(completed)
        return list(dict.fromkeys(valid))

    def _ocr_stage(self, processed_path: Path) -> tuple[str | None, list[dict], list[dict], str | None]:
        raw, items, error = self.engine.predict(str(processed_path))
        if error:
            return None, raw, items, error

        rec_texts = [item["text"] for item in items]
        rec_scores = [item["confidence"] for item in items]
        rec_boxes = [item["box"] for item in items]
        candidates: list[str] = []
        candidates.extend(ocr_text_candidates(rec_texts))
        candidates.extend(spatial_container_candidates(rec_texts, rec_scores, rec_boxes))
        candidates.extend(extract_from_json_objects(raw))
        valid = self._expand_and_validate_candidates(list(dict.fromkeys(candidates)))
        best = self._rank_valid_candidates(valid, rec_texts, rec_scores)
        return best, raw, items, None

    @staticmethod
    def _rank_valid_candidates(valid: list[str], rec_texts: list[str], rec_scores: list[float]) -> str | None:
        if not valid:
            return None
        scores = {candidate: 0.0 for candidate in valid}
        normalized = [normalize_container_number(text) for text in rec_texts]
        compact = "".join(normalized)
        for candidate in valid:
            for text, confidence in zip(normalized, rec_scores):
                if text == candidate:
                    scores[candidate] = max(scores[candidate], float(confidence) + 1.0)
            if candidate in compact:
                scores[candidate] += 0.5
        return max(valid, key=lambda candidate: scores[candidate])

    def _vlm_stage(self, processed_path: Path) -> tuple[str | None, list[dict], str | None]:
        if self.vlm is None:
            return None, [], self.vlm_error
        raw: list[dict] = []
        try:
            results = self.vlm.predict(str(processed_path))
            candidates: list[str] = []
            for result in results:
                data = self._result_json(result)
                raw.append(data)
                res = data.get("res", {}) or {}
                for block in res.get("parsing_res_list", []) or []:
                    content = block.get("block_content", "")
                    if content:
                        candidates.extend(extract_iso_candidates(content))
            candidates.extend(extract_from_json_objects(raw))
            valid = self._expand_and_validate_candidates(candidates)
            return (valid[0] if valid else None), raw, None
        except Exception as exc:  # pragma: no cover
            return None, raw, str(exc)

    def process(self, image_path: Path) -> dict:
        processed_path = preprocess_image(image_path, OUTPUT_DIR)

        # Primary path: standard OCR.
        ocr_number, ocr_raw, ocr_items, ocr_error = self._ocr_stage(processed_path)
        all_candidates = self._expand_and_validate_candidates(
            ocr_text_candidates([item["text"] for item in ocr_items])
            + spatial_container_candidates(
                [item["text"] for item in ocr_items],
                [item["confidence"] for item in ocr_items],
                [item["box"] for item in ocr_items],
            )
            + extract_from_json_objects(ocr_raw)
        )

        # Optional fallback: VLM only after OCR fails to produce a valid number.
        vlm_number = None
        vlm_raw: list[dict] = []
        if not ocr_number and self.vlm is not None:
            vlm_number, vlm_raw, _ = self._vlm_stage(processed_path)

        number = ocr_number or vlm_number
        method = "ocr" if ocr_number else ("vlm" if vlm_number else "none")
        if vlm_number and vlm_number not in all_candidates:
            all_candidates.append(vlm_number)

        result = {
            "success": True,
            "containerNumber": number,
            "method": method,
            "confidence": self._estimate_confidence(number, ocr_items),
            "candidates": [
                {"value": candidate, "source": "validated", "iso6346Valid": True}
                for candidate in dict.fromkeys(all_candidates)
            ],
            "ocrUsed": True,
            "vlmUsed": bool(vlm_number),
        }
        if ocr_error:
            result["ocrError"] = ocr_error
        if self.vlm_error:
            result["vlmError"] = self.vlm_error

        if SAVE_DEBUG_OUTPUT:
            self._save_json(OUTPUT_DIR / f"{image_path.stem}_ocr.json", ocr_raw)
            if vlm_raw:
                self._save_json(OUTPUT_DIR / f"{image_path.stem}_vlm.json", vlm_raw)
            self._save_json(OUTPUT_DIR / f"{image_path.stem}_result.json", result)
        return result

    @staticmethod
    def _estimate_confidence(number: str | None, items: list[dict]) -> float | None:
        if not number:
            return None
        normalized = normalize_container_number(number)
        matches = [float(i["confidence"]) for i in items if normalize_container_number(str(i["text"])) == normalized]
        return round(max(matches), 4) if matches else None
