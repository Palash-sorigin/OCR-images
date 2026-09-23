from __future__ import annotations

import re
from datetime import datetime

from app.utils.iso6346 import normalize_container_number

# Date/time formats seen across EIR, Form 13, and Visit Ticket documents:
# "11/08/2026 16:22", "11/Aug/2026 22:39:14", "28-Jun-2026 21:22:41", "27/05/2026 12:29".
_DATETIME_FORMATS = (
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%d/%b/%Y %H:%M:%S", "%d/%b/%Y %H:%M",
    "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M",
)
_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d/%b/%Y", "%d-%b-%Y")


def _parse_first(value: str, formats: tuple[str, ...]) -> datetime | None:
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


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
        if kind in {"code", "alphanumeric", "license", "truck_registration", "seal_number"}:
            return re.sub(r"[^A-Za-z0-9]", "", value).upper() or None
        if kind == "transaction":
            return value.upper().replace("+", " + ")
        if kind == "iso_size_type":
            return re.sub(r"\D", "", value) or None
        if kind == "datetime":
            cleaned = re.sub(r"[.,]", "", value).strip()
            parsed = _parse_first(cleaned, _DATETIME_FORMATS)
            return parsed.isoformat() if parsed else value.upper()
        if kind == "date":
            cleaned = re.sub(r"[.,]", "", value).strip()
            parsed = _parse_first(cleaned, _DATE_FORMATS)
            return parsed.date().isoformat() if parsed else value.upper()
        if kind in {"weight", "voltage", "temperature", "distance_km"}:
            # Keep a leading minus (temperature can be sub-zero); strip units/labels.
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            return match.group(0) if match else None
        if kind == "percentage":
            match = re.search(r"\d{1,3}", value)
            if not match:
                return None
            return str(min(100, int(match.group(0))))
        return value.upper()
