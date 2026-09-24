"""Build structured per-field candidate lists from raw OCR items.

This module sits between the PaddleOCR output and the Gemini LLM call.
For every field in the active schema it scans the OCR items, runs the
field's regex pattern, and groups results into ``candidates`` (strong
matches) and ``possible_candidates`` (partial / low-confidence matches).
Bounding-box locations and OCR item indices are preserved so the LLM and
downstream auditing can trace every value back to the source.
"""

from __future__ import annotations

import re
from typing import Any

# Confidence thresholds for bucketing OCR matches.
_HIGH_CONF = 0.80
_LOW_CONF = 0.50


def build_candidates(
    ocr_items: list[dict],
    field_rules: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """Return ``{field_name: {candidates: [...], possible_candidates: [...]}}``."""

    result: dict[str, dict[str, Any]] = {}

    for name, rule in field_rules.items():
        pattern = rule.get("pattern")
        if not pattern:
            # Fields with no regex pattern (e.g. vehicle-display odometer
            # resolved externally) are skipped.
            result[name] = {"candidates": [], "possible_candidates": []}
            continue

        value_re = re.compile(pattern, re.IGNORECASE)
        label_re = _build_label_re(rule)

        candidates: list[dict] = []
        possible: list[dict] = []

        for idx, item in enumerate(ocr_items):
            text = str(item.get("text", ""))
            conf = float(item.get("confidence", 0.0))
            box = item.get("box")

            # --- label-anchored match ---
            if label_re:
                for lm in label_re.finditer(text):
                    window = text[lm.end(): lm.end() + rule.get("window", 80)]
                    vm = value_re.search(window)
                    if vm:
                        val = vm.group(1) if vm.groups() else vm.group(0)
                        entry = _entry(val.strip(), conf, idx, box, anchored=True)
                        (candidates if conf >= _HIGH_CONF else possible).append(entry)

            # --- standalone pattern match (for fields with no aliases) ---
            aliases = rule.get("aliases") or []
            if not aliases and not rule.get("label_pattern"):
                vm = value_re.search(text)
                if vm:
                    val = vm.group(1) if vm.groups() else vm.group(0)
                    entry = _entry(val.strip(), conf, idx, box, anchored=False)
                    (candidates if conf >= _HIGH_CONF else possible).append(entry)

        # Also scan the combined OCR text for label-anchored matches that
        # span two adjacent OCR items (label in one item, value in the next).
        combined = "\n".join(str(it.get("text", "")) for it in ocr_items)
        if label_re:
            for lm in label_re.finditer(combined):
                window = combined[lm.end(): lm.end() + rule.get("window", 80)]
                vm = value_re.search(window)
                if vm:
                    val = vm.group(1) if vm.groups() else vm.group(0)
                    val = val.strip()
                    # Avoid duplicates already found per-item.
                    if not _already_found(val, candidates, possible):
                        entry = _entry(val, 0.70, -1, None, anchored=True)
                        possible.append(entry)

        # De-duplicate by value, keeping the highest-confidence entry.
        candidates = _dedup(candidates)
        possible = _dedup(possible)
        # Remove any possible_candidate that is already a full candidate.
        cand_values = {c["value"] for c in candidates}
        possible = [p for p in possible if p["value"] not in cand_values]

        result[name] = {"candidates": candidates, "possible_candidates": possible}

    return result


def build_llm_context(
    ocr_result: dict,
    field_rules: dict[str, dict],
    subtype: str,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the complete context payload sent to the LLM.

    The payload includes:
    - ``full_ocr_text``: the raw concatenated OCR text for overall context.
    - ``ocr_items``: every OCR-detected text span with confidence + bbox so
      the LLM can reason about spatial layout.
    - ``field_schema``: which fields we expect and what they look like.
    - ``candidate_mapping``: pre-matched candidates per field.
    - ``document_subtype``: the classified terminal format.
    """
    # Build a concise schema description for the LLM.
    schema_desc: dict[str, dict] = {}
    for name, rule in field_rules.items():
        schema_desc[name] = {
            "aliases": rule.get("aliases", []),
            "kind": rule.get("kind", "text"),
            "risk": rule.get("risk", "medium"),
            "description": _kind_description(rule.get("kind", "text")),
        }

    # Slim down the OCR items to only what the LLM needs.
    slim_items = []
    for idx, item in enumerate(ocr_result.get("items", [])):
        slim_items.append({
            "index": idx,
            "text": str(item.get("text", "")),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
            "bbox": item.get("box"),
        })

    return {
        "document_subtype": subtype,
        "full_ocr_text": ocr_result.get("text", ""),
        "ocr_items": slim_items,
        "field_schema": schema_desc,
        "candidate_mapping": candidates,
    }


# ── helpers ────────────────────────────────────────────────────────────────


def _build_label_re(rule: dict) -> re.Pattern | None:
    if rule.get("label_pattern"):
        return re.compile(rule["label_pattern"], re.IGNORECASE)
    aliases = rule.get("aliases") or []
    if not aliases:
        return None
    escaped = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    return re.compile(r"(?:" + "|".join(escaped) + r")\s*[:\.]?\s*", re.IGNORECASE)


def _entry(value: str, confidence: float, ocr_index: int, bbox, *, anchored: bool) -> dict:
    return {
        "value": value,
        "confidence": round(confidence, 4),
        "ocr_index": ocr_index,
        "bbox": bbox,
        "label_anchored": anchored,
    }


def _already_found(value: str, *lists: list[dict]) -> bool:
    upper = re.sub(r"[^A-Z0-9]", "", value.upper())
    for lst in lists:
        for item in lst:
            if re.sub(r"[^A-Z0-9]", "", item["value"].upper()) == upper:
                return True
    return False


def _dedup(entries: list[dict]) -> list[dict]:
    """Keep the highest-confidence entry per normalized value."""
    best: dict[str, dict] = {}
    for entry in entries:
        key = re.sub(r"[^A-Z0-9]", "", entry["value"].upper())
        if key not in best or entry["confidence"] > best[key]["confidence"]:
            best[key] = entry
    return list(best.values())


def _kind_description(kind: str) -> str:
    return {
        "container": "ISO 6346 shipping container number (e.g. TCLU1234567)",
        "truck_registration": "Indian vehicle registration (e.g. MH03FC0372)",
        "seal_number": "Alphanumeric security seal number",
        "datetime": "Date and time stamp",
        "weight": "Numeric weight value",
        "alphanumeric": "Alphanumeric identifier/code",
        "text": "Free-form text",
        "code": "Short code or abbreviation",
        "iso_size_type": "3-4 digit ISO size/type code",
    }.get(kind, kind)
