"""Generic label-anchored regex extraction over flattened OCR/VLM text.

The portal-screenshot pipeline (field_extractor.py) associates a label to a
value using bounding-box geometry, which works well for clean rectangular
web screenshots. These documents are photographed receipts and multi-column
forms where reading order and skew make spatial heuristics unreliable, so
extraction here works over a flattened text blob instead: find the field's
label, then search a window of text right after it for a value matching the
field's pattern. Fields with no reliable label (e.g. an unlabeled dashboard
reading) are matched by a standalone pattern over the whole text.

Because two independent extraction passes are run -- one over the VLM's
text, one over standard OCR's text -- a field found identically by both is
strong evidence and is scored accordingly; a field found by only one, or
found with disagreeing values by both, is surfaced rather than guessed.
"""

from __future__ import annotations

import re
from typing import Any


def _build_label_pattern(rule: dict) -> re.Pattern | None:
    if rule.get("label_pattern"):
        return re.compile(rule["label_pattern"], re.IGNORECASE)
    aliases = rule.get("aliases") or []
    if not aliases:
        return None
    # Longest alias first so a specific label ("Seal No 1") is tried before
    # a shorter one that could also match a different field's text.
    escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    return re.compile(r"(?:" + "|".join(escaped) + r")\s*[:\.]?\s*", re.IGNORECASE)


def extract_field_from_text(text: str, rule: dict) -> str | None:
    """Return the first value matching `rule` in `text`, or None."""
    if not text:
        return None
    pattern = rule.get("pattern")
    if not pattern:
        return None
    aliases = rule.get("aliases") or []

    if not aliases and not rule.get("label_pattern"):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        value = match.group(1) if match.groups() else match.group(0)
        return value.strip() or None

    label_re = _build_label_pattern(rule)
    if label_re is None:
        return None
    window = rule.get("window", 80)
    value_re = re.compile(pattern, re.IGNORECASE)

    for label_match in label_re.finditer(text):
        snippet = text[label_match.end():label_match.end() + window]
        value_match = value_re.search(snippet)
        if not value_match:
            continue
        value = value_match.group(1) if value_match.groups() else value_match.group(0)
        value = value.strip()
        if value:
            return value
    return None


def _find_bbox_for_value(value: str, ocr_items: list[dict]) -> list[float] | None:
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not compact:
        return None
    for item in ocr_items:
        item_compact = re.sub(r"[^A-Z0-9]", "", str(item.get("text", "")).upper())
        if item_compact and (compact in item_compact or item_compact in compact):
            box = item.get("box")
            if box and len(box) >= 4:
                try:
                    return [float(v) for v in box[:4]]
                except (TypeError, ValueError):
                    return None
    return None


def extract_fields(
    field_rules: dict[str, dict],
    *,
    vlm_text: str,
    ocr_text: str,
    ocr_items: list[dict],
) -> dict[str, dict[str, Any]]:
    """Extract every field in `field_rules`, reconciling the VLM and OCR passes.

    Returns the same field-result shape the rest of the pipeline (Normalizer,
    FieldValidator, ConfidenceEngine) already expects: value/confidence/
    status/source/bbox/reason.
    """
    fields: dict[str, dict[str, Any]] = {}

    for name, rule in field_rules.items():
        vlm_value = extract_field_from_text(vlm_text, rule)
        ocr_value = extract_field_from_text(ocr_text, rule)

        if vlm_value and ocr_value:
            agree = _normalize_loose(vlm_value) == _normalize_loose(ocr_value)
            value = vlm_value
            source = "VLM+OCR" if agree else "VLM"
            reason = "VLM and OCR agree." if agree else (
                f"VLM read {vlm_value!r} but OCR read {ocr_value!r}; using VLM, flagged for review."
            )
            status = "OCR_CANDIDATE"
        elif vlm_value:
            value, source, status = vlm_value, "VLM", "OCR_CANDIDATE"
            reason = "Only found by VLM; OCR did not detect this field."
        elif ocr_value:
            value, source, status = ocr_value, "OCR", "OCR_CANDIDATE"
            reason = "Only found by OCR; VLM did not detect this field."
        else:
            fields[name] = {
                "value": None, "confidence": None, "status": "NOT_FOUND", "source": "SYSTEM",
                "bbox": None, "reason": "Not found by VLM or OCR.",
            }
            continue

        bbox = _find_bbox_for_value(value, ocr_items)
        fields[name] = {
            "value": value,
            "confidence": 0.97 if source == "VLM+OCR" else 0.75,
            "status": status, "source": source, "bbox": bbox, "reason": reason,
        }

    return fields


def _normalize_loose(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())
