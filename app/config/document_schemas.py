"""Fixed field schemas for the non-portal document types (EIR, Form 13,
Visit Ticket, vehicle dashboard, container seal).

Unlike the portal-screenshot field rules (app/config/field_rules.py), which
associate labels to values using OCR bounding-box geometry, these documents
are photographed receipts/forms with inconsistent layout, skew, and
rotation. Fields here are extracted with label-anchored regular expressions
over a flattened text blob (see app/services/document_text_extractor.py),
sourced from both the VLM and standard OCR so the two can be cross-checked.

Each field rule:
    aliases:       label text(s) that precede the value in the document.
                    An empty list means the field has no reliable text
                    label and is found purely by its value pattern
                    (e.g. a lone "27.9V" reading with no adjacent word).
    label_pattern: optional raw regex overriding the auto-built label
                    match, needed where a generic alias would also match
                    a different field's label (e.g. "Driver" vs "Driver id").
    pattern:       regex for the *value* portion. When aliases are given,
                    this is searched within a window of text right after
                    the label (group 1 if present, else the whole match).
                    When aliases are empty, this pattern is searched over
                    the full text directly and MUST have exactly one
                    capture group.
    window:        how many characters after the label to search for the
                    value (default 80).
    kind:          drives Normalizer/FieldValidator behaviour.
    risk:          "high" | "medium" | "low" -- feeds ConfidenceEngine and
                    manual-review triage the same way the portal pipeline
                    already does.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# EIR (Equipment Interchange Report / Receipt)
# ---------------------------------------------------------------------------

EIR_FIELDS = {
    "eir_transaction": {
        "aliases": [],
        "pattern": r"(DELIVER(?:ED)? IMPORT CONTAINER|RECEIVED? EXPORT CONTAINER|PICK[- ]?UP)",
        "kind": "text", "risk": "medium",
    },
    "terminal_name": {
        "aliases": [],
        "pattern": r"(Nhava Sheva Freeport Terminal|DP World Nhava Sheva ICT|NSICT|NSFT)",
        "kind": "text", "risk": "low",
    },
    "container_no": {
        "aliases": ["Container No", "Container"],
        "pattern": r"([A-Z]{4}\s?\d{6,7})", "kind": "container", "risk": "high",
    },
    "iso_size_type_code": {
        "aliases": ["ISO", "Iso Code"], "pattern": r"(\d{3,4})",
        "kind": "iso_size_type", "risk": "medium",
    },
    "size": {
        "aliases": ["Size"], "pattern": r"(\d{2,3})",
        "kind": "container_size", "risk": "medium",
    },
    "container_type_code": {
        "aliases": ["Type"], "pattern": r"([A-Za-z]{1,4})",
        "kind": "code", "risk": "low",
    },
    "status": {
        "aliases": ["Status"], "pattern": r"([A-Za-z]{2,10})",
        "kind": "text", "risk": "medium",
    },
    "gross_weight": {
        "aliases": ["Gross Wt", "Weight"], "pattern": r"([\d.]{1,10})",
        "kind": "weight", "risk": "medium",
    },
    "vessel_name": {
        "aliases": ["Vessel Name", "VSL"], "pattern": r"([A-Za-z][A-Za-z0-9 .'-]{2,40})",
        "kind": "text", "risk": "low",
    },
    "voyage_no": {
        "aliases": ["Voyage", "VIA"], "pattern": r"([A-Z0-9]{4,15})",
        "kind": "alphanumeric", "risk": "low",
    },
    "destination": {
        "aliases": ["Dest/POD2", "Destination"], "pattern": r"([A-Za-z][A-Za-z0-9 .,()'-]{2,60})",
        "kind": "text", "risk": "medium", "window": 100,
    },
    "group_code": {
        "aliases": ["Group Code"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "medium",
    },
    "hazardous_class": {
        "aliases": ["Hazardous", "Haz/IMO"], "pattern": r"([0-9A-Za-z().: ]{1,20})",
        "kind": "text", "risk": "medium",
    },
    "seal_no_1": {
        "aliases": ["Seal 1", "Seal1"], "pattern": r"([A-Z0-9]{3,15})",
        "kind": "seal_number", "risk": "high",
    },
    "seal_no_2": {
        "aliases": ["Seal 2", "Seal2"], "pattern": r"([A-Z0-9]{3,15})",
        "kind": "seal_number", "risk": "low",
    },
    "in_datetime": {
        "aliases": ["In Date"], "pattern": r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium",
    },
    "out_datetime": {
        "aliases": ["Out Date"], "pattern": r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium",
    },
    "eir_datetime": {
        # DP World layout has no In/Out pair, just a single "Date:" stamp.
        "aliases": ["Date"], "label_pattern": r"\bDate\s*[:\.]?\s*",
        "pattern": r"(\d{1,2}[/-][A-Za-z]{3}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium",
    },
    "line": {
        "aliases": ["Line"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "low",
    },
    "bat_no": {
        "aliases": ["BAT", "Bat No"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "low",
    },
    "truck_no": {
        "aliases": ["Truck No", "Truck"], "label_pattern": r"\bTruck(?:\s*No)?\s*[:\.]?\s*",
        "pattern": r"([A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{4})",
        "kind": "truck_registration", "risk": "high",
    },
    "trucking_company": {
        "aliases": ["Trucking Company"], "pattern": r"([A-Za-z][A-Za-z0-9 .'-]{2,40})",
        "kind": "text", "risk": "low",
    },
    "driver_name": {
        "aliases": ["Driver"], "label_pattern": r"\bDriver(?!\s*[Ii]d)\s*[:\.]?\s*",
        "pattern": r"([A-Za-z][A-Za-z .'-]{2,40})",
        "kind": "name", "risk": "medium",
    },
    "driver_id": {
        "aliases": ["Driver id", "License"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "alphanumeric", "risk": "medium",
    },
    "remarks": {
        "aliases": ["Remarks"], "pattern": r"([A-Za-z0-9][A-Za-z0-9 .,'-]{1,60})",
        "kind": "text", "risk": "low",
    },
    "transaction_no": {
        "aliases": ["Transaction", "Trans. No", "Barcode"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "alphanumeric", "risk": "low",
    },
    "yard_position": {
        "aliases": ["Yard position"], "pattern": r"([A-Z0-9.]{2,15})",
        "kind": "alphanumeric", "risk": "low",
    },
}

EIR_CLASSIFY_RULES = {
    "NSFT": ["NHAVA SHEVA FREEPORT TERMINAL", "NSFT,MUMBAI", "NSFT CLERK"],
    "DPWORLD": ["DP WORLD NHAVA SHEVA ICT", "NSICT", "POWERED BY I-TEK"],
}

# NSFT prints separate In/Out timestamps; DP World prints one combined
# "Date:" stamp instead. Restricting the field list per subtype (after
# classification) avoids the other layout's exclusive fields showing up as
# spurious NOT_FOUND manual-review noise on every single receipt -- the
# same reasoning behind SCREEN_FIELDS in app/config/field_rules.py.
EIR_FIELDS_BY_SUBTYPE = {
    "NSFT": [name for name in EIR_FIELDS if name not in {"eir_datetime", "yard_position"}],
    "DPWORLD": [name for name in EIR_FIELDS if name not in {"in_datetime", "out_datetime"}],
}

# ---------------------------------------------------------------------------
# Form 13 / E-Gate Pass
# ---------------------------------------------------------------------------

FORM13_FIELDS = {
    "form13_no": {
        "aliases": ["Form 13"], "pattern": r"(\d{6,12})",
        "kind": "alphanumeric", "risk": "high",
    },
    "terminal": {
        "aliases": ["Terminal"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "low",
    },
    "generated_datetime": {
        "aliases": ["Generated Date & Time"],
        "pattern": r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium", "window": 60,
    },
    "trade_type": {
        "aliases": ["Trade Type"], "pattern": r"([A-Za-z][A-Za-z ]{2,20})",
        "kind": "text", "risk": "low",
    },
    "preadvise_type": {
        "aliases": ["Pre-advise Type"], "pattern": r"([A-Za-z][A-Za-z ]{2,20})",
        "kind": "text", "risk": "low",
    },
    "vessel_name": {
        "aliases": ["Vessel Name"], "pattern": r"([A-Za-z][A-Za-z0-9 .'-]{2,40})",
        "kind": "text", "risk": "medium",
    },
    "voyage_no": {
        "aliases": ["Terminal Vessel Visit"], "pattern": r"([A-Z0-9()]{4,20})",
        "kind": "alphanumeric", "risk": "low",
    },
    "gate_open_datetime": {
        "aliases": ["Gate Open"], "pattern": r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium",
    },
    "vessel_sailing_datetime": {
        "aliases": ["Vessel Sailing Time"], "pattern": r"(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "low",
    },
    "shipper_name": {
        "aliases": ["Shipper Name"], "pattern": r"([A-Za-z][A-Za-z0-9 .,'&-]{2,60})",
        "kind": "text", "risk": "medium", "window": 100,
    },
    "cha_name": {
        "aliases": ["CHA Name"], "pattern": r"([A-Za-z][A-Za-z0-9 .,'&-]{2,60})",
        "kind": "text", "risk": "low", "window": 100,
    },
    "liner_name": {
        "aliases": ["Liner Name"], "pattern": r"([A-Za-z][A-Za-z0-9 .,'&-]{2,60})",
        "kind": "text", "risk": "low", "window": 100,
    },
    "liner_booking_no": {
        "aliases": ["Liner Booking No"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "alphanumeric", "risk": "medium",
    },
    "container_no": {
        "aliases": ["Container No"], "label_pattern": r"\bContainer No\.?\s*[:\.]?\s*",
        "pattern": r"([A-Z]{4}\s?\d{6,7})", "kind": "container", "risk": "high",
    },
    "iso_code_size": {
        "aliases": ["ISO Code"], "pattern": r"([0-9]{3,4}\s?\[?\d{0,2}\s?FT\]?)",
        "kind": "text", "risk": "medium",
    },
    "line_seal_no": {
        "aliases": ["Line Seal No"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "seal_number", "risk": "high",
    },
    "custom_seal_no": {
        "aliases": ["Custom Seal No"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "seal_number", "risk": "medium",
    },
    "shipping_bill_no": {
        "aliases": ["Shipping Bill No"], "pattern": r"([A-Z0-9]{4,20})",
        "kind": "alphanumeric", "risk": "high",
    },
    "port_of_discharge": {
        "aliases": ["Port of Discharge"], "pattern": r"([A-Za-z][A-Za-z0-9 .,()'-]{2,60})",
        "kind": "text", "risk": "medium", "window": 100,
    },
    "port_of_final_destination": {
        "aliases": ["Port of Final Destination"], "pattern": r"([A-Za-z][A-Za-z0-9 .,()'-]{2,60})",
        "kind": "text", "risk": "low", "window": 100,
    },
    "commodity_type": {
        "aliases": ["Commodity Type"], "pattern": r"([A-Za-z][A-Za-z ]{2,30})",
        "kind": "text", "risk": "low",
    },
    "vgm_kg": {
        "aliases": ["VGM"], "pattern": r"([\d.]{1,10})",
        "kind": "weight", "risk": "medium",
    },
    "driver_name": {
        "aliases": ["Driver Name"], "pattern": r"([A-Za-z][A-Za-z .'-]{2,40})",
        "kind": "name", "risk": "medium",
    },
    "truck_no": {
        "aliases": ["Truck No"], "label_pattern": r"\bTruck\s*No\.?\s*[:\.]?\s*",
        "pattern": r"([A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{4})",
        "kind": "truck_registration", "risk": "high",
    },
    "bat_no": {
        "aliases": ["Bat No"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "low",
    },
}

# ---------------------------------------------------------------------------
# Visit Ticket (drop-off / pick-up passes, multiple terminal formats)
# ---------------------------------------------------------------------------

VISIT_TICKET_FIELDS = {
    "ticket_type": {
        "aliases": [],
        "pattern": r"(Drop-?Off Ticket-?Export|Pick-?Up Ticket-?Import|Received Export Container|Pick[- ]?Up)",
        "kind": "text", "risk": "medium",
    },
    "terminal_name": {
        "aliases": [],
        "pattern": r"(Gateway Terminals India|DP World Nhava Sheva ICT|PSA Mumbai|NSICT)",
        "kind": "text", "risk": "low",
    },
    "date_time": {
        "aliases": ["Date"], "label_pattern": r"\bDate\s*[:\.]?\s*",
        "pattern": r"(\d{1,2}[/-][A-Za-z0-9]{2,3}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)",
        "kind": "datetime", "risk": "medium",
    },
    "truck_no": {
        "aliases": ["Trailer No", "LIC NO", "Truck"], "label_pattern": r"\b(?:Trailer\s*No|LIC\s*NO|Truck)\s*[:\.]?\s*",
        "pattern": r"([A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{4})",
        "kind": "truck_registration", "risk": "high",
    },
    "bat_id": {
        "aliases": ["Bat Id", "BAT NO", "BAT"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "low",
    },
    "container_no": {
        "aliases": ["Cntr No", "Container"], "label_pattern": r"\b(?:Cntr No|Container)\s*[:\.]?\s*",
        "pattern": r"([A-Z]{4}\s?\d{6,7})", "kind": "container", "risk": "high",
    },
    "iso_code": {
        "aliases": ["ISO Code"], "pattern": r"([0-9]{3,4}[A-Za-z0-9]{0,2})",
        "kind": "text", "risk": "medium",
    },
    "gross_weight": {
        "aliases": ["Gross Wt"], "pattern": r"([\d.]{1,10})",
        "kind": "weight", "risk": "medium",
    },
    "yard_location": {
        "aliases": ["Yard Loc", "Yard position"], "pattern": r"([A-Za-z0-9.]{2,15})",
        "kind": "alphanumeric", "risk": "low",
    },
    "seal_no_1": {
        "aliases": ["Seal No 1", "Seal1"], "pattern": r"([A-Z0-9]{3,15})",
        "kind": "seal_number", "risk": "high",
    },
    "seal_no_2": {
        "aliases": ["Seal No 2", "Seal2"], "pattern": r"([A-Z0-9]{3,15})",
        "kind": "seal_number", "risk": "low",
    },
    "group_code": {
        "aliases": ["Group Code"], "pattern": r"([A-Z0-9]{2,10})",
        "kind": "alphanumeric", "risk": "medium",
    },
    "loc_slip": {
        "aliases": ["LOC SLIP"], "pattern": r"([A-Za-z][A-Za-z ]{2,20})",
        "kind": "text", "risk": "low",
    },
    "status": {
        "aliases": ["Status"], "pattern": r"([A-Za-z]{2,10})",
        "kind": "text", "risk": "low",
    },
    "remark": {
        "aliases": ["Remark"], "pattern": r"([A-Za-z0-9][A-Za-z0-9 .,'-]{1,60})",
        "kind": "text", "risk": "low",
    },
}

VISIT_TICKET_CLASSIFY_RULES = {
    "GTI_DROPOFF_EXPORT": ["DROP-OFF TICKET-EXPORT", "GATEWAY TERMINALS"],
    "GTI_PICKUP_IMPORT": ["PICK-UP TICKET-IMPORT"],
    "DPWORLD_NSICT": ["DP WORLD NHAVA SHEVA ICT", "NSICT"],
    "PSA_BMCT_PICKUP": ["PSA MUMBA", "BMCT-TID"],
}

# Same reasoning as EIR_FIELDS_BY_SUBTYPE: these four terminal formats print
# different field sets (e.g. only GTI's pickup ticket shows seal numbers;
# only DP World shows a loc-slip/status/creator block), so restrict
# extraction per subtype once classified rather than treating every field
# from every format as expected on every ticket. UNKNOWN (classifier
# couldn't decide) intentionally has no entry -- see
# DocumentExtractionPipeline._fields_for_subtype, which falls back to the
# full field set, mirroring SCREEN_FIELDS["UNKNOWN"] in the portal pipeline.
VISIT_TICKET_FIELDS_BY_SUBTYPE = {
    "GTI_DROPOFF_EXPORT": [
        "ticket_type", "terminal_name", "date_time", "truck_no", "bat_id",
        "container_no", "iso_code", "gross_weight", "yard_location", "remark",
    ],
    "GTI_PICKUP_IMPORT": [
        "ticket_type", "terminal_name", "date_time", "truck_no", "bat_id",
        "container_no", "iso_code", "gross_weight", "yard_location",
        "seal_no_1", "seal_no_2", "group_code",
    ],
    "DPWORLD_NSICT": [
        "ticket_type", "terminal_name", "date_time", "bat_id", "loc_slip",
        "container_no", "iso_code", "group_code", "truck_no", "status", "yard_location",
    ],
    "PSA_BMCT_PICKUP": [
        "ticket_type", "terminal_name", "date_time", "truck_no", "bat_id", "container_no",
    ],
}

# ---------------------------------------------------------------------------
# Vehicle dashboard display (KM & Battery percentage)
# ---------------------------------------------------------------------------
#
# These readings are not text-labelled the way a form is (icons, not words,
# mark most gauges), so most fields here are pattern-only and inherently
# best-effort. Treat this schema as lower-confidence telemetry extraction,
# not a structured form; see README for the disambiguation heuristic used
# for the two "...km" readings (range-to-empty vs. odometer).

VEHICLE_DISPLAY_FIELDS = {
    "battery_soc_percent": {
        "aliases": ["SOC"], "pattern": r"(\d{1,3})\s*%",
        "kind": "percentage", "risk": "medium",
    },
    "battery_voltage": {
        "aliases": [], "pattern": r"\b(\d{1,3}\.\d)\s*V\b",
        "kind": "voltage", "risk": "medium",
    },
    "cabin_temp_c": {
        "aliases": [], "pattern": r"\b(-?\d{1,3})\s*°\s*C\b",
        "kind": "temperature", "risk": "low",
    },
    "speed_kmh": {
        "aliases": [], "pattern": r"\b(\d{1,3})\s*\n?\s*km\s*/\s*h\b",
        "kind": "distance_km", "risk": "low",
    },
    "time_of_reading": {
        "aliases": [], "pattern": r"\b(AM|PM)\s*\d{1,2}[:\s.]\d{2}\b",
        "kind": "text", "risk": "low",
    },
    "gear_position": {
        "aliases": [], "pattern": r"\b([DPRN]\d{0,2})\b",
        "kind": "code", "risk": "low",
    },
    "power_percent": {
        "aliases": ["%POWER", "POWER"], "pattern": r"(\d{1,3})\s*%?",
        "kind": "percentage", "risk": "low",
    },
    # These three are resolved together by DocumentExtractionPipeline
    # (see _resolve_km_readings), not by the generic single-field regex
    # pass: the dashboard shows 2-3 unlabeled "...km" numbers at once, and
    # telling odometer/trip/range apart needs the whole set, not one regex.
    # `pattern: None` makes extract_fields() skip them so the resolver's
    # values aren't clobbered by a generic (and wrong) single-field match.
    "odometer_km": {"aliases": [], "pattern": None, "kind": "distance_km", "risk": "low"},
    "trip_km": {"aliases": [], "pattern": None, "kind": "distance_km", "risk": "low"},
    "range_to_empty_km": {"aliases": [], "pattern": None, "kind": "distance_km", "risk": "low"},
}

VEHICLE_DISPLAY_CLASSIFY_RULES = {
    "EV_DASHBOARD": ["SOC", "KM/H", "%POWER"],
}

# ---------------------------------------------------------------------------
# Container seal close-up photo
# ---------------------------------------------------------------------------

CONTAINER_SEAL_FIELDS = {
    "seal_number": {
        "aliases": [], "pattern": r"\b([A-Z]{0,3}\s?\d{4,10})\b",
        "kind": "seal_number", "risk": "high",
    },
    "seal_manufacturer": {
        "aliases": [],
        "pattern": r"(INFINEUM|OSS|CAMBRIDGE|TESA|MECCANO|UNISTRAP|SEALOCK|E-SEAL)",
        "kind": "text", "risk": "low",
    },
    "seal_type_code": {
        "aliases": [], "pattern": r"\[([A-Z])\]",
        "kind": "code", "risk": "low",
    },
    "container_no": {
        "aliases": ["Container", "Cntr"], "pattern": r"([A-Z]{4}\s?\d{6,7})",
        "kind": "container", "risk": "medium",
    },
}

CONTAINER_SEAL_CLASSIFY_RULES = {
    "SEAL_PHOTO": ["SEAL", "SECURITY"],
}

# ---------------------------------------------------------------------------
# Registry consumed by app/services/document_pipeline.py and
# app/routes/documents.py
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = {
    "eir": {
        "fields": EIR_FIELDS,
        "fields_by_subtype": EIR_FIELDS_BY_SUBTYPE,
        "classify_rules": EIR_CLASSIFY_RULES,
        "default_subtype": "NSFT",
    },
    "form13": {
        "fields": FORM13_FIELDS,
        "fields_by_subtype": None,
        "classify_rules": None,
        "default_subtype": "FORM13",
    },
    "visit_ticket": {
        "fields": VISIT_TICKET_FIELDS,
        "fields_by_subtype": VISIT_TICKET_FIELDS_BY_SUBTYPE,
        "classify_rules": VISIT_TICKET_CLASSIFY_RULES,
        "default_subtype": "UNKNOWN",
    },
    "vehicle_display": {
        "fields": VEHICLE_DISPLAY_FIELDS,
        "fields_by_subtype": None,
        "classify_rules": VEHICLE_DISPLAY_CLASSIFY_RULES,
        "default_subtype": "EV_DASHBOARD",
    },
    "container_seal": {
        "fields": CONTAINER_SEAL_FIELDS,
        "fields_by_subtype": None,
        "classify_rules": CONTAINER_SEAL_CLASSIFY_RULES,
        "default_subtype": "SEAL_PHOTO",
    },
}
