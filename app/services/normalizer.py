from __future__ import annotations

import re

from app.utils.iso6346 import normalize_container_number


class Normalizer:
    def normalize(self, value: str | None, kind: str) -> str | None:
        if value is None:
            return None
        value = re.sub(r"\s+", " ", str(value)).strip()
        if not value:
            return None

        if kind in {"numeric", "count"}:
            return re.sub(r"\D", "", value) or None
        if kind == "phone":
            digits = re.sub(r"\D", "", value)
            # Strip common Indian country/trunk prefixes.
            # +91 / 091 / 91 followed by a 10-digit mobile number.
            if digits.startswith("091") and len(digits) == 13:
                digits = digits[3:]
            elif digits.startswith("91") and len(digits) == 12:
                digits = digits[2:]
            elif digits.startswith("0") and len(digits) == 11:
                digits = digits[1:]
            # If OCR captured only the country code ("91", "+91", "091")
            # or a fragment too short to be a phone number, discard it.
            if len(digits) < 10:
                return None
            return digits or None
        if kind == "container":
            return normalize_container_number(value)
        if kind in {"code", "alphanumeric", "license", "truck_registration"}:
            return re.sub(r"[^A-Za-z0-9]", "", value).upper() or None
        if kind == "transaction":
            return value.upper().replace("+", " + ")
        return value.upper()
