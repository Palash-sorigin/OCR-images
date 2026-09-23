from __future__ import annotations

from datetime import datetime


def _value(fields: dict, name: str):
    item = fields.get(name, {})
    return item.get("value") if isinstance(item, dict) else None


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _check_order(fields: dict, earlier_name: str, later_name: str, label: str) -> list[dict]:
    earlier = _parse_iso(_value(fields, earlier_name))
    later = _parse_iso(_value(fields, later_name))
    if earlier and later and earlier > later:
        return [{
            "type": "DATE_ORDER_CONFLICT",
            "field": later_name,
            "reason": f"{label}: {earlier_name} ({earlier.isoformat()}) is after {later_name} ({later.isoformat()}).",
        }]
    return []


def validate_eir(fields: dict) -> list[dict]:
    return _check_order(fields, "in_datetime", "out_datetime", "Gate-in must not be after gate-out")


def validate_form13(fields: dict) -> list[dict]:
    return _check_order(fields, "gate_open_datetime", "vessel_sailing_datetime", "Gate open must not be after vessel sailing")


def validate_visit_ticket(fields: dict) -> list[dict]:
    return []


def validate_vehicle_display(fields: dict) -> list[dict]:
    conflicts = []
    soc = _value(fields, "battery_soc_percent")
    if soc is not None:
        try:
            if not (0 <= float(soc) <= 100):
                conflicts.append({
                    "type": "OUT_OF_RANGE",
                    "field": "battery_soc_percent",
                    "reason": "Battery state-of-charge reading is outside 0-100%.",
                })
        except (TypeError, ValueError):
            pass
    odometer = _value(fields, "odometer_km")
    range_km = _value(fields, "range_to_empty_km")
    try:
        if odometer is not None and range_km is not None and float(range_km) > float(odometer):
            conflicts.append({
                "type": "IMPLAUSIBLE_READING",
                "field": "range_to_empty_km",
                "reason": "Range-to-empty reading is larger than the odometer reading; the two 'km' values may be swapped.",
            })
    except (TypeError, ValueError):
        pass
    return conflicts


def validate_container_seal(fields: dict) -> list[dict]:
    return []


VALIDATORS = {
    "eir": validate_eir,
    "form13": validate_form13,
    "visit_ticket": validate_visit_ticket,
    "vehicle_display": validate_vehicle_display,
    "container_seal": validate_container_seal,
}


def cross_validate(document_type: str, fields: dict) -> list[dict]:
    validator = VALIDATORS.get(document_type)
    return validator(fields) if validator else []
