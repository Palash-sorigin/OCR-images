from __future__ import annotations

import re
from typing import Any

from app.utils.iso6346 import is_valid_iso6346


class FieldValidator:
    def validate(self, value: Any, kind: str) -> dict:
        if value is None or value == "":
            return {"valid": False, "status": "NOT_FOUND", "reason": "No value"}

        value = str(value)

        if kind == "numeric":
            ok = value.isdigit()
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected digits"}

        if kind == "count":
            ok = value.isdigit() and 0 <= int(value) <= 1000
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a reasonable non-negative count"}

        if kind == "phone":
            ok = bool(re.fullmatch(r"[6-9]\d{9}", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a 10-digit Indian mobile number"}

        if kind == "container":
            ok = is_valid_iso6346(value)
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "iso6346_valid": ok, "reason": None if ok else "Failed ISO 6346 format/check digit validation"}

        if kind == "container_size":
            ok = value in {"20", "40", "45"}
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a configured container size (20/40/45)"}

        if kind == "name":
            ok = bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,99}", value)) and not any(ch.isdigit() for ch in value)
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Unexpected name format"}

        if kind in {"code", "alphanumeric", "license", "truck_registration"}:
            ok = bool(re.fullmatch(r"[A-Z0-9]{2,40}", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Unexpected alphanumeric format"}

        if kind == "time":
            ok = bool(re.search(r"\b\d{1,2}:\d{2}\b", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Time could not be recognized"}

        return {"valid": True, "status": "VALID", "reason": None}
