from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import ENABLE_LLM_EXTRACTION, OUTPUT_DIR, SAVE_DEBUG_OUTPUT
from app.config.document_schemas import DOCUMENT_TYPES
from app.schemas.extraction import FieldResult, ImageResult
from app.services.confidence import ConfidenceEngine
from app.services.document_classifier import DocumentClassifier
from app.services.document_cross_validators import cross_validate
from app.services.document_text_extractor import extract_fields
from app.services.general_ocr import GeneralOCRService
from app.services.normalizer import Normalizer
from app.services.validator import FieldValidator
from app.services.vlm_verifier import VLMVerifier
from app.utils.preprocessing import inspect_image_quality, preprocess_image

_BARE_KM_PATTERN = re.compile(r"(?<!\.)\b(\d{3,6})\s*km\b", re.IGNORECASE)
_DECIMAL_KM_PATTERN = re.compile(r"\b(\d{1,5}\.\d)\s*km\b", re.IGNORECASE)


class DocumentExtractionPipeline:
    """VLM + OCR hybrid extraction for a single fixed document schema.

    Mirrors SequentialExtractionPipeline's philosophy (never invent a value;
    prefer null + manual review over a guess) but uses label-anchored regex
    over flattened text instead of bounding-box spatial matching, because
    these are photographed receipts/forms rather than clean web screenshots.
    VLM and OCR are run independently and cross-checked per field -- see
    document_text_extractor.extract_fields for the agreement logic.

    For ``visit_ticket`` documents when ``ENABLE_LLM_EXTRACTION`` is true,
    the VLM is bypassed entirely: raw OCR candidates are built and sent
    (as text, no image) to Gemini Flash, which selects the best value per
    field.  This is dramatically cheaper while preserving accuracy.
    """

    def __init__(
        self,
        document_type: str,
        general_ocr: GeneralOCRService,
        vlm: VLMVerifier,
        *,
        gemini_engine=None,
        candidate_builder=None,
    ) -> None:
        if document_type not in DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type: {document_type}")
        config = DOCUMENT_TYPES[document_type]
        self.document_type = document_type
        self.field_rules = config["fields"]
        self.fields_by_subtype = config.get("fields_by_subtype")
        self.classifier = DocumentClassifier(config["classify_rules"], config["default_subtype"])
        self.general_ocr = general_ocr
        self.vlm = vlm
        self.normalizer = Normalizer()
        self.validator = FieldValidator()
        self.confidence = ConfidenceEngine()

        # OCR + LLM path (visit_ticket only when enabled)
        self.gemini_engine = gemini_engine
        self.candidate_builder = candidate_builder
        self._use_llm = (
            ENABLE_LLM_EXTRACTION
            and document_type == "visit_ticket"
            and gemini_engine is not None
            and getattr(gemini_engine, "available", False)
        )

    def process_images(self, image_paths: list[tuple[str, Path]]) -> list[ImageResult]:
        results: list[ImageResult] = []
        total = len(image_paths)
        for index, (filename, image_path) in enumerate(image_paths, start=1):
            print("=" * 70)
            print(f"[{self.document_type}] Processing image {index}/{total}: {filename}")
            print("=" * 70)
            try:
                result = self.process_one(filename, image_path)
            except Exception as exc:
                result = ImageResult(
                    filename=filename, status="ERROR", document_type=self.document_type, warnings=[str(exc)],
                )
            results.append(result)
        return results

    def process_one(self, filename: str, image_path: Path) -> ImageResult:
        quality = inspect_image_quality(image_path)
        if quality["status"] == "UNREADABLE":
            return ImageResult(
                filename=filename, status="ERROR", document_type=self.document_type,
                quality=quality, warnings=[quality["reason"]],
            )

        processed_path = preprocess_image(image_path, OUTPUT_DIR)
        print("[1/6] Image preprocessing")
        ocr_result = self.general_ocr.extract(processed_path)

        if self._use_llm:
            fields, subtype = self._process_with_llm(ocr_result)
        else:
            fields, subtype = self._process_with_vlm(ocr_result, image_path)

        if self.document_type == "vehicle_display":
            vlm_text = "" if self._use_llm else self._vlm_text(image_path)
            self._resolve_km_readings(fields, f"{vlm_text}\n{ocr_result['text']}")

        print("[5/6] Normalization + validation")
        self._normalize_fields(fields)
        self._validate_fields(fields)
        conflicts = cross_validate(self.document_type, fields)
        print("[6/6] Manual review")
        manual_review = self._build_manual_review(fields, conflicts)

        status = "NEEDS_REVIEW" if manual_review or conflicts else "PROCESSED"
        image_result = ImageResult(
            filename=filename,
            status=status,
            document_type=self.document_type,
            document_subtype=subtype["subtype"],
            screen_confidence=subtype["confidence"],
            quality=quality,
            fields={name: FieldResult(**value) for name, value in fields.items()},
            warnings=list(subtype.get("evidence", [])),
            conflicts=conflicts,
            manual_review=manual_review,
            raw_ocr=ocr_result["items"],
        )
        if SAVE_DEBUG_OUTPUT:
            self._save_debug(filename, image_path, image_result.model_dump())
        return image_result

    # ── OCR + LLM path (visit_ticket) ──────────────────────────────────────

    def _process_with_llm(
        self, ocr_result: dict,
    ) -> tuple[dict[str, dict[str, Any]], dict]:
        """Run the cheap OCR-only + Gemini Flash path."""
        from app.services.ocr_candidate_builder import (
            build_candidates,
            build_llm_context,
        )

        print("[2/6] Skipping VLM (using OCR + LLM path)")
        print("[3/6] Sub-type classification (OCR-only)")
        subtype = self.classifier.classify(ocr_result["text"])
        active_fields = self._fields_for_subtype(subtype["subtype"])

        print("[4/6] Building candidates + LLM extraction")
        candidates = build_candidates(ocr_result["items"], active_fields)
        context = build_llm_context(
            ocr_result, active_fields, subtype["subtype"], candidates,
        )
        llm_result = self.gemini_engine.extract_fields(context, active_fields)

        fields = self._build_fields_from_llm(
            llm_result, candidates, ocr_result["items"],
        )
        return fields, subtype

    # ── VLM + OCR hybrid path (all other document types) ───────────────────

    def _process_with_vlm(
        self, ocr_result: dict, image_path: Path,
    ) -> tuple[dict[str, dict[str, Any]], dict]:
        """Run the existing VLM+OCR cross-check path."""
        print("[2/6] VLM extraction")
        vlm_text = self._vlm_text(image_path)
        print("[3/6] Sub-type classification")
        combined_for_classify = f"{ocr_result['text']}\n{vlm_text}"
        subtype = self.classifier.classify(combined_for_classify)
        print("[4/6] Field extraction (VLM + OCR cross-check)")
        active_fields = self._fields_for_subtype(subtype["subtype"])
        fields = extract_fields(
            active_fields,
            vlm_text=vlm_text,
            ocr_text=ocr_result["text"],
            ocr_items=ocr_result["items"],
        )
        return fields, subtype

    # ── LLM result → field dicts ───────────────────────────────────────────

    @staticmethod
    def _build_fields_from_llm(
        llm_result: dict[str, dict[str, Any]],
        candidates: dict[str, dict[str, Any]],
        ocr_items: list[dict],
    ) -> dict[str, dict[str, Any]]:
        """Convert LLM selections into the field-result shape the rest of
        the pipeline expects (value / confidence / status / source / bbox /
        reason), with OCR-dominant blended confidence scoring.
        """
        fields: dict[str, dict[str, Any]] = {}

        for name, llm_field in llm_result.items():
            value = llm_field.get("value")
            ocr_indices = llm_field.get("ocr_indices", [])
            reason = llm_field.get("reason", "")

            if value is None:
                fields[name] = {
                    "value": None, "confidence": None,
                    "status": "NOT_FOUND", "source": "OCR+LLM",
                    "bbox": None,
                    "reason": reason or "Not found by OCR or LLM.",
                }
                continue

            # Find the OCR confidence of the selected value.
            ocr_conf = _find_ocr_confidence(value, ocr_indices, ocr_items, candidates.get(name, {}))

            # Blended confidence: OCR-dominant.  The LLM agreement gives a
            # small uplift (max 5 %) but never dominates the score.  A low-
            # confidence OCR read stays low even if the LLM selects it.
            llm_bonus = 0.05
            # Hard ceiling: the blended score never exceeds ocr_conf + 5 %
            # and is capped at 0.95 (only full validation lifts to ~1.0).
            blended = min(0.95, ocr_conf + llm_bonus)

            bbox = _find_bbox(ocr_indices, ocr_items)

            fields[name] = {
                "value": value,
                "confidence": round(blended, 4),
                "status": "OCR_CANDIDATE",
                "source": "OCR+LLM",
                "bbox": bbox,
                "reason": reason,
            }

        return fields

    def _fields_for_subtype(self, subtype: str) -> dict[str, dict]:
        if not self.fields_by_subtype:
            return self.field_rules
        names = self.fields_by_subtype.get(subtype)
        if not names:
            # Unrecognized/UNKNOWN subtype: fall back to the full field set
            # rather than extracting nothing, mirroring
            # SCREEN_FIELDS["UNKNOWN"] in the portal pipeline.
            return self.field_rules
        return {name: self.field_rules[name] for name in names}

    def _vlm_text(self, image_path: Path) -> str:
        if self.vlm.vlm is None:
            return ""
        extraction = self.vlm.extract_text(image_path)
        if not extraction.get("available") or extraction.get("error"):
            return ""
        return extraction.get("text", "")

    @staticmethod
    def _resolve_km_readings(fields: dict, text: str) -> None:
        """Disambiguate the dashboard's 2-3 unlabeled "...km" numbers.

        Observed layout: a decimal reading (trip meter) and one-or-two bare
        integer readings (range-to-empty, odometer) appear together with no
        text label distinguishing them. Odometer is cumulative and, across
        every sample observed, larger than the remaining range, so the
        largest bare integer is treated as odometer and the smallest as
        range-to-empty. This is a heuristic, not a guarantee -- both fields
        are kept at "low" risk and the cross-validator flags an implausible
        result (range > odometer) for manual review.
        """
        decimal_match = _DECIMAL_KM_PATTERN.search(text)
        if decimal_match:
            fields["trip_km"] = {
                "value": decimal_match.group(1), "confidence": 0.7, "status": "OCR_CANDIDATE",
                "source": "OCR", "bbox": None, "reason": "Decimal '...km' reading, read as trip distance.",
            }

        bare_values = sorted({int(m.group(1)) for m in _BARE_KM_PATTERN.finditer(text)})
        if len(bare_values) >= 2:
            fields["range_to_empty_km"] = {
                "value": str(bare_values[0]), "confidence": 0.55, "status": "OCR_CANDIDATE",
                "source": "OCR", "bbox": None,
                "reason": "Smallest bare '...km' reading; heuristically read as range-to-empty.",
            }
            fields["odometer_km"] = {
                "value": str(bare_values[-1]), "confidence": 0.55, "status": "OCR_CANDIDATE",
                "source": "OCR", "bbox": None,
                "reason": "Largest bare '...km' reading; heuristically read as odometer.",
            }
        elif len(bare_values) == 1:
            fields["odometer_km"] = {
                "value": str(bare_values[0]), "confidence": 0.5, "status": "OCR_CANDIDATE",
                "source": "OCR", "bbox": None,
                "reason": "Only one bare '...km' reading found; assumed odometer.",
            }

    def _normalize_fields(self, fields: dict) -> None:
        for name, field in fields.items():
            kind = self.field_rules.get(name, {}).get("kind", "text")
            if field.get("value") is not None:
                field["value"] = self.normalizer.normalize(field["value"], kind)

    def _validate_fields(self, fields: dict) -> None:
        for name, field in fields.items():
            value = field.get("value")
            if value is None:
                continue
            kind = self.field_rules.get(name, {}).get("kind", "text")
            validation = self.validator.validate(value, kind)
            field["validation"] = validation
            status, confidence = self.confidence.assess(
                field.get("confidence"), validation, source=field.get("source", "OCR"),
                risk=self.field_rules.get(name, {}).get("risk", "medium"),
            )
            field["confidence"] = confidence
            field["status"] = status if validation.get("valid") else "NEEDS_REVIEW"
            if not validation.get("valid"):
                field["reason"] = validation.get("reason")

    @staticmethod
    def _build_manual_review(fields: dict, conflicts: list[dict]) -> list[dict]:
        review = []
        for name, field in fields.items():
            if field.get("status") in {"NEEDS_REVIEW", "INVALID", "NOT_FOUND"}:
                review.append({
                    "field": name,
                    "current_value": field.get("value"),
                    "status": field.get("status"),
                    "reason": field.get("reason") or "Field needs manual confirmation.",
                })
        for conflict in conflicts:
            review.append({
                "field": conflict.get("field", "cross_field"),
                "status": "CONFLICT",
                "reason": conflict.get("reason", "Cross-field conflict."),
            })
        return review

    def _save_debug(self, filename: str, image_path: Path, data: dict) -> None:
        # image_path.stem (from the request's temp file) is unique per
        # upload; filename alone is not, so it is kept only as a readable
        # prefix rather than the uniqueness key.
        path = OUTPUT_DIR / f"{self.document_type}_{Path(filename).stem}_{image_path.stem}_analysis.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ── module-level helpers for LLM field building ────────────────────────────

def _find_ocr_confidence(
    value: str,
    ocr_indices: list[int],
    ocr_items: list[dict],
    field_candidates: dict,
) -> float:
    """Look up the original OCR confidence for `value`.

    Priority: (1) the OCR item at the indices the LLM cited, (2) the
    candidate list, (3) a conservative fallback.
    """
    import re as _re
    compact = _re.sub(r"[^A-Z0-9]", "", value.upper())

    # Check the OCR items the LLM referenced.
    for idx in ocr_indices:
        if 0 <= idx < len(ocr_items):
            item = ocr_items[idx]
            item_compact = _re.sub(r"[^A-Z0-9]", "", str(item.get("text", "")).upper())
            if compact and (compact in item_compact or item_compact in compact):
                return float(item.get("confidence", 0.5))

    # Check the pre-built candidate lists.
    for bucket in ("candidates", "possible_candidates"):
        for cand in field_candidates.get(bucket, []):
            cand_compact = _re.sub(r"[^A-Z0-9]", "", str(cand.get("value", "")).upper())
            if compact == cand_compact:
                return float(cand.get("confidence", 0.5))

    # Fallback: conservative score.
    return 0.50


def _find_bbox(ocr_indices: list[int], ocr_items: list[dict]) -> list[float] | None:
    """Return the bounding box from the first valid OCR index."""
    for idx in ocr_indices:
        if 0 <= idx < len(ocr_items):
            box = ocr_items[idx].get("box")
            if box and len(box) >= 4:
                try:
                    return [float(v) for v in box[:4]]
                except (TypeError, ValueError):
                    continue
    return None
