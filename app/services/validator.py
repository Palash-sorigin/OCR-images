from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.utils.iso6346 import is_valid_iso6346

# Indian vehicle registration: state code + RTO code + series letters + number,
# e.g. MH03FC0372. truck_registration used to share the loose alphanumeric
# check below, which would accept almost anything; every real plate observed
# across the container/EIR/visit-ticket documents matches this stricter form.
_VEHICLE_REG_PATTERN = re.compile(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}")


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

        if kind == "truck_registration":
            ok = bool(_VEHICLE_REG_PATTERN.fullmatch(value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected an Indian vehicle registration (e.g. MH03FC0372)"}

        if kind in {"code", "alphanumeric", "license"}:
            ok = bool(re.fullmatch(r"[A-Z0-9]{2,40}", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Unexpected alphanumeric format"}

        if kind == "seal_number":
            ok = bool(re.fullmatch(r"[A-Z0-9]{4,15}", value)) and any(ch.isdigit() for ch in value)
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected an alphanumeric seal number containing at least one digit"}

        if kind == "iso_size_type":
            ok = bool(re.fullmatch(r"\d{3,4}", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a 3-4 digit ISO 6346 size/type code"}

        if kind == "time":
            ok = bool(re.search(r"\b\d{1,2}:\d{2}\b", value))
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Time could not be recognized"}

        if kind == "date":
            try:
                datetime.fromisoformat(value)
                ok = True
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Date could not be recognized"}

        if kind == "datetime":
            try:
                datetime.fromisoformat(value)
                ok = True
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Date/time could not be recognized"}

        if kind == "weight":
            try:
                ok = 0 < float(value) < 200000
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a plausible positive weight"}

        if kind == "voltage":
            try:
                ok = 0 <= float(value) <= 1000
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a plausible voltage reading"}

        if kind == "temperature":
            try:
                ok = -50 <= float(value) <= 150
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a plausible temperature reading"}

        if kind == "distance_km":
            try:
                ok = 0 <= float(value) < 2_000_000
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a plausible distance in km"}

        if kind == "percentage":
            try:
                ok = 0 <= float(value) <= 100
            except ValueError:
                ok = False
            return {"valid": ok, "status": "VALID" if ok else "INVALID", "reason": None if ok else "Expected a percentage between 0 and 100"}

        return {"valid": True, "status": "VALID", "reason": None}
