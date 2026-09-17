from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import OUTPUT_DIR, SAVE_DEBUG_OUTPUT
from app.config.field_rules import FIELD_RULES
from app.schemas.extraction import FieldResult, ImageResult
from app.services.confidence import ConfidenceEngine
from app.services.container_ocr_adapter import ContainerOCRAdapter
from app.services.cross_field_validator import CrossFieldValidator
from app.services.field_extractor import FieldExtractor
from app.services.general_ocr import GeneralOCRService
from app.services.normalizer import Normalizer
from app.services.validator import FieldValidator
from app.services.screen_classifier import ScreenClassifier
from app.services.status_detector import StatusDetector
from app.services.table_extractor import TableExtractor
from app.services.vlm_verifier import VLMVerifier
from app.utils.iso6346 import is_valid_iso6346
from app.utils.preprocessing import inspect_image_quality, preprocess_image


class SequentialExtractionPipeline:
    """One image completes the whole pipeline before the next image starts."""

    def __init__(self, general_ocr: GeneralOCRService, container_ocr: ContainerOCRAdapter, vlm: VLMVerifier) -> None:
        self.general_ocr = general_ocr
        self.container_ocr = container_ocr
        self.vlm = vlm
        self.classifier = ScreenClassifier()
        self.extractor = FieldExtractor()
        self.normalizer = Normalizer()
        self.validator = FieldValidator()
        self.confidence = ConfidenceEngine()
        self.cross_validator = CrossFieldValidator()
        self.status_detector = StatusDetector()
        self.table_extractor = TableExtractor()

    def process_images(self, image_paths: list[tuple[str, Path]]) -> list[ImageResult]:
        results: list[ImageResult] = []
        total = len(image_paths)
        for index, (filename, image_path) in enumerate(image_paths, start=1):
            print("=" * 70)
            print(f"Processing image {index}/{total}: {filename}")
            print("=" * 70)
            try:
                result = self.process_one(filename, image_path)
            except Exception as exc:
                result = ImageResult(filename=filename, status="ERROR", warnings=[str(exc)])
            results.append(result)
            print(f"Image {index} complete.")
        return results

    def process_one(self, filename: str, image_path: Path) -> ImageResult:
        quality = inspect_image_quality(image_path)
        if quality["status"] == "UNREADABLE":
            return ImageResult(filename=filename, status="ERROR", quality=quality, warnings=[quality["reason"]])

        processed_path = preprocess_image(image_path, OUTPUT_DIR)
        print("[1/8] Image preprocessing")
        ocr_result = self.general_ocr.extract(processed_path)
        print("[2/8] Screen classification")
        screen = self.classifier.classify(ocr_result["text"])
        print("[3/8] General OCR")
        print("[4/8] Field extraction")
        fields = self.extractor.extract(ocr_result["items"], screen["screen_type"])

        print("[5/8] Specialized container OCR")
        container_result = None
        # The specialized service is called once per image. It remains useful for container photos
        # and for screenshots where a container number is a target field.
        if self._container_expected(screen["screen_type"], fields):
            container_result = self.container_ocr.extract(image_path)
            self._merge_container_result(fields, container_result)

        print("[6/8] Normalization")
        self._normalize_fields(fields)
        print("[7/8] Validation")
        self._validate_fields(fields)
        conflicts = self.cross_validator.validate(fields)
        print("[8/8] Confidence assessment")

        # VLM is only a fallback for fields that remain uncertain.
        initial_review = self._build_manual_review(fields, conflicts)
        self._vlm_fallback(image_path, fields, initial_review)
        manual_review = self._build_manual_review(fields, conflicts)

        status = "NEEDS_REVIEW" if manual_review or conflicts else "PROCESSED"
        image_result = ImageResult(
            filename=filename,
            status=status,
            screen_type=screen["screen_type"],
            screen_confidence=screen["confidence"],
            quality=quality,
            fields={name: FieldResult(**value) for name, value in fields.items()},
            warnings=list(screen.get("evidence", [])),
            conflicts=conflicts,
            manual_review=manual_review,
            raw_ocr=ocr_result["items"],
            container_ocr=container_result,
            operation_status=self.status_detector.detect(ocr_result["text"]),
            tables=self.table_extractor.extract(ocr_result["items"]),
        )
        if SAVE_DEBUG_OUTPUT:
            self._save_debug(filename, image_result.model_dump())
        return image_result

    @staticmethod
    def _container_expected(screen_type: str, fields: dict) -> bool:
        if screen_type == "UNKNOWN":
            return True
        return any(
            name.startswith(("export_container", "dpd_container")) and field.get("value")
            for name, field in fields.items()
        ) or screen_type in {"PIN_GENERATION_PSA", "PIN_GENERATION_APM", "PIN_GENERATION_NSFT", "TRUCK_BOOKING_NSFT"}

    @staticmethod
    def _merge_container_result(fields: dict, result: dict | None) -> None:
        """Reconcile specialized OCR evidence without blindly filling a field.

        A container OCR result is only copied into a field when general OCR has
        already associated the same value with a container field. Otherwise it
        remains evidence for review; this prevents a number found anywhere in a
        screenshot from being assigned to the wrong container slot.
        """
        if not result or not result.get("containerNumber"):
            return
        candidate = result["containerNumber"]
        for name, field in fields.items():
            if "container" not in name or not field.get("value"):
                continue
            if str(field["value"]).upper() == str(candidate).upper():
                field["source"] = "OCR+CONTAINER_OCR"
                field["reason"] = "General OCR and specialized container OCR agree."
                return

    def _normalize_fields(self, fields: dict) -> None:
        for name, field in fields.items():
            kind = FIELD_RULES.get(name, {}).get("kind", "text")
            if field.get("value") is not None:
                field["value"] = self.normalizer.normalize(field["value"], kind)

    def _validate_fields(self, fields: dict) -> None:
        for name, field in fields.items():
            value = field.get("value")
            if value is None:
                continue
            kind = FIELD_RULES.get(name, {}).get("kind", "text")
            validation = self.validator.validate(value, kind)
            field["validation"] = validation
            status, confidence = self.confidence.assess(field.get("confidence"), validation, source=field.get("source", "OCR"), risk=FIELD_RULES.get(name, {}).get("risk", "medium"))
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

    def _vlm_fallback(self, image_path: Path, fields: dict, manual_review: list[dict]) -> None:
        if self.vlm.vlm is None:
            return
        # Verify only high-risk unresolved fields. Do not let VLM override a valid deterministic result.
        for item in list(manual_review):
            name = item["field"]
            rule = FIELD_RULES.get(name, {})
            if rule.get("risk") != "high":
                continue
            current = fields.get(name, {}).get("value")
            result = self.vlm.verify_field(image_path, name, current)
            if result.get("status") == "MATCH" and current:
                validation = fields[name].get("validation", {})
                if validation.get("valid"):
                    fields[name]["status"] = "VLM_VERIFIED"
                    fields[name]["source"] = "VLM"

    @staticmethod
    def _save_debug(filename: str, data: dict) -> None:
        path = OUTPUT_DIR / f"{Path(filename).stem}_analysis.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
