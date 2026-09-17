from __future__ import annotations

import re
from typing import Any

from app.config.field_rules import FIELD_RULES, SCREEN_FIELDS


def _box(item: dict) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    if not box or len(box) < 4:
        return None
    try:
        return tuple(float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None


def _compact(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class FieldExtractor:
    """Associate form labels with nearby OCR values using layout, not fixed coordinates."""

    def extract(self, items: list[dict], screen_type: str = "UNKNOWN") -> dict[str, dict[str, Any]]:
        field_names = SCREEN_FIELDS.get(screen_type, SCREEN_FIELDS["UNKNOWN"])
        label_indices = {i for i, item in enumerate(items) if self._is_any_label(item.get("text", ""))}
        fields: dict[str, dict[str, Any]] = {}

        for field_name in field_names:
            rule = FIELD_RULES[field_name]
            aliases = {_compact(alias) for alias in rule["aliases"]}
            labels = [
                (i, item) for i, item in enumerate(items)
                if i in label_indices and self._matches_label(item.get("text", ""), aliases)
            ]

            occurrence = 0 if field_name.endswith("_1") else 1 if field_name.endswith("_2") else 0
            label_entry = labels[occurrence] if occurrence < len(labels) else (labels[0] if labels else None)
            if label_entry is None:
                fields[field_name] = {
                    "value": None, "confidence": None, "status": "NOT_FOUND", "source": "SYSTEM",
                    "bbox": None, "reason": "Field label not located by OCR.",
                }
                continue

            label_index, label = label_entry
            label_text = str(label.get("text", ""))
            label_box = _box(label)
            inline = self._inline_value(label_text, aliases)
            if inline:
                fields[field_name] = {
                    "value": inline,
                    "confidence": float(label.get("confidence", 0.0)),
                    "status": "OCR_CANDIDATE", "source": "OCR",
                    "bbox": list(label_box) if label_box else None,
                }
                continue

            candidate = self._nearest_value(label_index, label, items, label_indices)
            if candidate is None:
                fields[field_name] = {
                    "value": None, "confidence": None, "status": "EMPTY", "source": "SYSTEM",
                    "bbox": list(label_box) if label_box else None,
                    "reason": "Field label was found but no nearby value was detected.",
                }
                continue

            fields[field_name] = {
                "value": candidate["text"],
                "confidence": candidate["confidence"],
                "status": "OCR_CANDIDATE", "source": "OCR",
                "bbox": candidate.get("box"),
            }

        return fields

    @classmethod
    def _is_any_label(cls, text: str) -> bool:
        return cls._label_alias(text) is not None

    @classmethod
    def _label_alias(cls, text: str) -> str | None:
        raw = str(text).strip()
        compact = _compact(raw)
        for rule in FIELD_RULES.values():
            for alias in rule["aliases"]:
                a = _compact(alias)
                if compact == a:
                    return a
                if ":" in raw:
                    left = _compact(raw.split(":", 1)[0])
                    if left == a:
                        return a
                # Allow OCR to glue a numeric/alphanumeric value directly to a label,
                # but never treat DRIVER + DRIVERNAME as the same label.
                if len(a) >= 5 and compact.startswith(a):
                    suffix = compact[len(a):]
                    if suffix and (suffix[0].isdigit() or (a.endswith("NO") and suffix[0].isalpha())):
                        return a
        return None

    @classmethod
    def _matches_label(cls, text: str, aliases: set[str]) -> bool:
        alias = cls._label_alias(text)
        return alias in aliases if alias else False

    @staticmethod
    def _inline_value(label_text: str, aliases: set[str]) -> str | None:
        raw = str(label_text).strip()
        if ":" in raw:
            left, suffix = raw.split(":", 1)
            if _compact(left) in aliases and suffix.strip():
                return suffix.strip()
        compact = _compact(raw)
        for alias in sorted(aliases, key=len, reverse=True):
            if len(alias) >= 5 and compact.startswith(alias) and len(compact) > len(alias):
                suffix = compact[len(alias):]
                if suffix and (suffix[0].isdigit() or alias.endswith("NO")):
                    return suffix
        return None

    @staticmethod
    def _nearest_value(label_index: int, label: dict, items: list[dict], label_indices: set[int]) -> dict | None:
        lb = _box(label)
        if not lb:
            return None
        lx1, ly1, lx2, ly2 = lb
        lcy = (ly1 + ly2) / 2
        candidates = []

        for index, item in enumerate(items):
            if index == label_index or index in label_indices:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            ib = _box(item)
            if not ib:
                continue
            x1, y1, x2, y2 = ib
            icy = (y1 + y2) / 2
            vertical_overlap = min(ly2, y2) - max(ly1, y1)
            same_row = vertical_overlap >= -0.25 * max(ly2 - ly1, y2 - y1)
            right_of_label = x1 >= lx2 - 10
            below_label = y1 >= ly2 - 5
            dx = max(0.0, x1 - lx2)
            dy = abs(icy - lcy)

            if same_row and right_of_label and dx <= 900:
                candidates.append((dx + dy * 2, item))
            elif below_label and abs((x1 + x2) / 2 - (lx1 + lx2) / 2) <= 450 and (y1 - ly2) <= 180:
                candidates.append(((y1 - ly2) * 2 + abs((x1 + x2) / 2 - (lx1 + lx2) / 2), item))

        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1] if candidates else None
