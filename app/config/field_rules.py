FIELD_RULES = {
    "pin_reference_no": {"aliases": ["PIN Ref No", "PIN Ref No.", "PIN Reference No", "PIN Reference Number"], "kind": "alphanumeric", "risk": "high"},
    "pin_no": {"aliases": ["PIN No", "PIN No."], "kind": "numeric", "risk": "high"},
    "tt_no": {"aliases": ["TT No", "TT No.", "Truck Licence No", "Truck License No"], "kind": "truck_registration", "risk": "high"},
    "bat_no": {"aliases": ["BAT No", "BAT No."], "kind": "alphanumeric", "risk": "low"},
    "truck_company": {"aliases": ["Truck Company"], "kind": "name", "risk": "low"},
    "container_count": {"aliases": ["No. Of Container", "No Of Container", "No. of Container"], "kind": "count", "risk": "medium"},
    "driver_mobile": {"aliases": ["Driver Mobile No", "Driver Mobile No."], "kind": "phone", "risk": "high"},
    "driver_sms_mobile": {"aliases": ["Driver Mobile No(For SMS)", "Driver Mobile No (For SMS)"], "kind": "phone", "risk": "high"},
    "group_code": {"aliases": ["Group Code"], "kind": "code", "risk": "medium"},
    "size": {"aliases": ["Size"], "kind": "container_size", "risk": "medium"},
    "rf_id": {"aliases": ["RF ID"], "kind": "alphanumeric", "risk": "low"},
    "driving_license_no": {"aliases": ["Driving License No", "Driving License No.", "Driving License No :"], "kind": "license", "risk": "high"},
    "driver_name": {"aliases": ["Driver Name"], "kind": "name", "risk": "medium"},
    "container_type": {"aliases": ["Container Type"], "kind": "code", "risk": "medium"},
    "transaction_type": {"aliases": ["Transaction Type"], "kind": "transaction", "risk": "medium"},
    "export_container_no_1": {"aliases": ["Export Container No 1", "Export Container No. 1"], "kind": "container", "risk": "high"},
    "export_container_no_2": {"aliases": ["Export Container No 2", "Export Container No. 2"], "kind": "container", "risk": "high"},
    "booking_no_1": {"aliases": ["Booking No 1", "Booking No. 1"], "kind": "alphanumeric", "risk": "high"},
    "booking_no_2": {"aliases": ["Booking No 2", "Booking No. 2"], "kind": "alphanumeric", "risk": "high"},
    "booking_no": {"aliases": ["Booking No", "Booking No."], "kind": "alphanumeric", "risk": "high"},
    "window": {"aliases": ["Window"], "kind": "text", "risk": "low"},
    "available_count": {"aliases": ["Available Count"], "kind": "count", "risk": "low"},
    "used_count": {"aliases": ["Used Count"], "kind": "count", "risk": "low"},
    "gate_start_time": {"aliases": ["Gate Start Time"], "kind": "time", "risk": "low"},
    "import_pin_no": {"aliases": ["Import PIN No", "Import PIN No."], "kind": "numeric", "risk": "high"},
    "cfs_code": {"aliases": ["CFS CODE", "CFS Code"], "kind": "code", "risk": "medium"},
    "cont_size": {"aliases": ["Cont Size"], "kind": "container_size", "risk": "medium"},
    "dpd_container_no_1": {"aliases": ["DPD Container No 1", "DPD Container No. 1"], "kind": "container", "risk": "high"},
    "dpd_container_no_2": {"aliases": ["DPD Container No 2", "DPD Container No. 2"], "kind": "container", "risk": "high"},
    "container_origin_1": {"aliases": ["Container Origin"], "kind": "text", "risk": "medium"},
    "container_origin_2": {"aliases": ["Container Origin"], "kind": "text", "risk": "medium"},
    "via_no_1": {"aliases": ["VIA No", "Via No"], "kind": "alphanumeric", "risk": "medium"},
    "via_no_2": {"aliases": ["VIA No", "Via No"], "kind": "alphanumeric", "risk": "medium"},
    "genset_1": {"aliases": ["GenSet", "Genset"], "kind": "text", "risk": "low"},
    "genset_2": {"aliases": ["GenSet", "Genset"], "kind": "text", "risk": "low"},
}

PIN_FIELDS = [
    "pin_reference_no", "pin_no", "tt_no", "bat_no", "truck_company", "container_count",
    "driver_mobile", "group_code", "size", "rf_id", "driving_license_no", "driver_name",
    "container_type", "transaction_type", "export_container_no_1", "export_container_no_2",
    "booking_no_1", "booking_no_2", "window", "available_count", "gate_start_time", "used_count",
]

APM_PIN_FIELDS = [
    "pin_reference_no", "pin_no", "tt_no", "driver_mobile", "container_count", "driving_license_no",
    "driver_sms_mobile", "group_code", "container_type", "size", "bat_no", "driver_name",
    "window", "available_count", "gate_start_time", "used_count",
]

TRUCK_BOOKING_FIELDS = [
    "tt_no", "truck_company", "driving_license_no", "driver_name", "export_container_no_1",
    "container_origin_1", "booking_no_1", "via_no_1", "genset_1", "export_container_no_2",
    "container_origin_2", "booking_no_2", "via_no_2", "genset_2", "import_pin_no", "cfs_code",
    "cont_size", "dpd_container_no_1", "dpd_container_no_2",
]

SCREEN_FIELDS = {
    "PIN_GENERATION_PSA": PIN_FIELDS,
    "PIN_GENERATION_APM": APM_PIN_FIELDS,
    "PIN_GENERATION_NSFT": PIN_FIELDS,
    "TRUCK_BOOKING_NSFT": TRUCK_BOOKING_FIELDS,
    "UNKNOWN": list(FIELD_RULES),
}
