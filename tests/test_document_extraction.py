"""Regex-extraction tests using text transcribed directly from the real
sample documents in docs/docs/{EIR,Form 13,Visit Ticket,KM and Battery
percentage,Container Seal}. paddleocr/PaddleOCR-VL aren't installed in this
environment, so these tests exercise the extraction/normalization/
validation/cross-field logic against ground-truth text instead of live OCR
output -- the same approach used for tests/test_pipeline_fixes.py.
"""

from app.config.document_schemas import (
    CONTAINER_SEAL_FIELDS,
    EIR_CLASSIFY_RULES,
    EIR_FIELDS,
    FORM13_FIELDS,
    VEHICLE_DISPLAY_CLASSIFY_RULES,
    VEHICLE_DISPLAY_FIELDS,
    VISIT_TICKET_CLASSIFY_RULES,
    VISIT_TICKET_FIELDS,
)
from app.services.document_classifier import DocumentClassifier
from app.services.document_cross_validators import cross_validate
from app.services.document_pipeline import DocumentExtractionPipeline
from app.services.document_text_extractor import extract_fields
from app.services.normalizer import Normalizer
from app.services.validator import FieldValidator

# ---------------------------------------------------------------------------
# Ground-truth text transcribed from real sample images
# ---------------------------------------------------------------------------

EIR_NSFT_TEXT = """
Nhava Sheva Freeport Terminal
NSFT,MUMBAI
EQUIPMENT INTERCHANGE REPORT
DELIVER IMPORT CONTAINER
Date Time: 11/08/2026 16:22
Transaction: 1406918
Container No: BLKU2603860
ISO: 2270
Size: 20 Type:CT
Status: FCL
Hazardous: 3(3295) 3(1993) () ()
Vessel Name: APL BARCELONA
Voyage: ABARS1191
Gross Wt: 24.09
Dest/POD2: ULA-MUMBAI INTERNATIONAL CARGO TERMINALS (ULA CFS) PVT LTD
Group Code: ULAD2
In Date: 11/08/2026 12:19
Out Date: 11/08/2026 16:22
Seal 1: 8481395
Line: JMC
BAT: 6256
Lane: GATE 4
Truck No: MH03FC0367
Trucking Company: SORIGIN
Driver id: MH2320090024719
Driver: SITARAM MANE
NSFT Clerk: c220607346
"""

EIR_DPWORLD_TEXT = """
DP World Nhava Sheva ICT
EIR : Deliver Import Container
Container: TXGU8278315
Barcode: 17063828
Date:11/Aug/2026 22:39:14
Truck No: MH03FC0368
VSL:EVER LIBRA
Trans. No.: 26081101099
Bat No.: G512
VIA: EVBS1197
Status: Full
Iso Code: 4510
Seal1:TSF0478034
Size: 40
Destination: CFSULA
Weight: 32525
Driver: GADRV
"""

FORM13_TEXT = """
E - GATE FORM
Form 13/Form 6/SEZ 4
Terminal: NSIGT
Form 13: 16344955
Generated Date & Time: 27/05/2026 12:29
EXPORT PREADVICE DETAILS
Pre-advise Type Export Full
Trade Type Foreign
Vessel Name HUBERT SCHULTE
Terminal Vessel Visit (VOY No.) HLES0575 (IP622A)
Gate Open 27/05/2026 14:00
Vessel Sailing Time 01/06/2026 20:00
Liner Name [Code] (Mumbai) MSC MEDITARRANEAN SHIPPI..
Shipper Name [Code] CEVA LOGISTICS INDIA PVT LTD [null]
Container No. CAAU5845674
ISO Code [Size] 4510 [40 FT]
Line Seal No. LG00755114
Custom Seal No. ESST01130521
Shipping Bill No 3455889
Port of Discharge London Gateway Port
Port of Final Destination London Gateway Port
Commodity Type GENERAL
VGM (KG) 8953
Driver Name RAJENDRA
Truck No. MH03FC0368
Bat No. G512
"""

VISIT_TICKET_GTI_DROPOFF_TEXT = """
GATEWAY TERMINALS INDIA PVT.LTD
Drop-Off Ticket-Export
Date : 04-06-2026 20:08
Trailer No:MH03FC0372
Bat Id :CY27
Cntr No :MRKU7845180
ISO Code :22G1
Gross Wt. :3.13
Yard Loc :Read SMS
Remark :
"""

VISIT_TICKET_GTI_PICKUP_TEXT = """
GATEWAY TERMINALS INDIA PVT.LTD
Pick-Up Ticket-Import
Date : 09-08-2026 19:34
Trailer No:MH03FC0371
Bat Id :CY30
Cntr No :TGBU8932660
ISO Code :4400
Yard Loc :03J24A.2
Gross Wt :28.52
Seal No 1 :40343856
Group Code:ULA
"""

VISIT_TICKET_PSA_TEXT = """
PSA MUMBA
BMCT-TID
10/08/26 10:4
LIC NO: MH03FC0368
BAT NO: G512
Import
Container Size Loc E/F Pod
ULA/SCS/20'/F 20' B0851 F ULA
"""

VEHICLE_DISPLAY_TEXT_1 = """
SOC L 100% H
C 36°C H
27.9V
0
km/h
->223km
PM12:41
834.6km
20834km
"""

VEHICLE_DISPLAY_TEXT_2 = """
100 %POWER
SOC L 51% H
C 31°C H
28.6V
0
km/h
->111km
PM11 23
A D02
6502.3km
17384km
"""

CONTAINER_SEAL_TEXT_1 = """
INFINEUM
OSS Security Seals [H]
2569011
"""

CONTAINER_SEAL_TEXT_2 = """
C7207353
"""


def _extract(field_rules, text):
    return extract_fields(field_rules, vlm_text=text, ocr_text=text, ocr_items=[])


def _finalize(field_rules, fields):
    """Run normalize + validate the same way DocumentExtractionPipeline does."""
    normalizer, validator = Normalizer(), FieldValidator()
    for name, field in fields.items():
        kind = field_rules.get(name, {}).get("kind", "text")
        if field.get("value") is not None:
            field["value"] = normalizer.normalize(field["value"], kind)
            field["validation"] = validator.validate(field["value"], kind)
    return fields


# ---------------------------------------------------------------------------
# EIR
# ---------------------------------------------------------------------------

def test_eir_nsft_extraction():
    fields = _finalize(EIR_FIELDS, _extract(EIR_FIELDS, EIR_NSFT_TEXT))

    assert fields["container_no"]["value"] == "BLKU2603860"
    assert fields["container_no"]["validation"]["valid"] is True
    assert fields["truck_no"]["value"] == "MH03FC0367"
    assert fields["truck_no"]["validation"]["valid"] is True
    assert fields["seal_no_1"]["value"] == "8481395"
    assert fields["size"]["value"] == "20"
    assert fields["gross_weight"]["value"] == "24.09"
    assert fields["in_datetime"]["value"] == "2026-08-11T12:19:00"
    assert fields["out_datetime"]["value"] == "2026-08-11T16:22:00"
    # "Driver:" must not match "Driver id:"'s value.
    assert fields["driver_name"]["value"] == "SITARAM MANE"
    assert fields["driver_id"]["value"] == "MH2320090024719"


def test_eir_dpworld_extraction():
    fields = _finalize(EIR_FIELDS, _extract(EIR_FIELDS, EIR_DPWORLD_TEXT))

    assert fields["container_no"]["value"] == "TXGU8278315"
    assert fields["container_no"]["validation"]["valid"] is True
    assert fields["truck_no"]["value"] == "MH03FC0368"
    assert fields["size"]["value"] == "40"
    assert fields["gross_weight"]["value"] == "32525"
    assert fields["eir_datetime"]["value"] == "2026-08-11T22:39:14"


def test_eir_classifier_distinguishes_terminals():
    classifier = DocumentClassifier(EIR_CLASSIFY_RULES, "NSFT")
    assert classifier.classify(EIR_NSFT_TEXT)["subtype"] == "NSFT"
    assert classifier.classify(EIR_DPWORLD_TEXT)["subtype"] == "DPWORLD"


def test_eir_in_after_out_date_is_flagged():
    fields = {
        "in_datetime": {"value": "2026-08-11T18:00:00"},
        "out_datetime": {"value": "2026-08-11T12:00:00"},
    }
    conflicts = cross_validate("eir", fields)
    assert any(c["type"] == "DATE_ORDER_CONFLICT" for c in conflicts)


def test_eir_normal_date_order_has_no_conflict():
    fields = {
        "in_datetime": {"value": "2026-08-11T12:19:00"},
        "out_datetime": {"value": "2026-08-11T16:22:00"},
    }
    assert cross_validate("eir", fields) == []


# ---------------------------------------------------------------------------
# Form 13
# ---------------------------------------------------------------------------

def test_form13_extraction():
    fields = _finalize(FORM13_FIELDS, _extract(FORM13_FIELDS, FORM13_TEXT))

    assert fields["form13_no"]["value"] == "16344955"
    assert fields["container_no"]["value"] == "CAAU5845674"
    assert fields["container_no"]["validation"]["valid"] is True
    assert fields["truck_no"]["value"] == "MH03FC0368"
    assert fields["shipping_bill_no"]["value"] == "3455889"
    assert fields["vgm_kg"]["value"] == "8953"
    assert fields["generated_datetime"]["value"] == "2026-05-27T12:29:00"


# ---------------------------------------------------------------------------
# Visit Ticket
# ---------------------------------------------------------------------------

def test_visit_ticket_gti_dropoff_extraction():
    fields = _finalize(VISIT_TICKET_FIELDS, _extract(VISIT_TICKET_FIELDS, VISIT_TICKET_GTI_DROPOFF_TEXT))

    assert fields["truck_no"]["value"] == "MH03FC0372"
    assert fields["container_no"]["value"] == "MRKU7845180"
    assert fields["container_no"]["validation"]["valid"] is True
    assert fields["bat_id"]["value"] == "CY27"


def test_visit_ticket_psa_extraction():
    fields = _finalize(VISIT_TICKET_FIELDS, _extract(VISIT_TICKET_FIELDS, VISIT_TICKET_PSA_TEXT))

    assert fields["truck_no"]["value"] == "MH03FC0368"
    assert fields["bat_id"]["value"] == "G512"


def test_visit_ticket_classifier():
    classifier = DocumentClassifier(VISIT_TICKET_CLASSIFY_RULES, "UNKNOWN")
    assert classifier.classify(VISIT_TICKET_GTI_DROPOFF_TEXT)["subtype"] == "GTI_DROPOFF_EXPORT"
    assert classifier.classify(VISIT_TICKET_PSA_TEXT)["subtype"] == "PSA_BMCT_PICKUP"


# ---------------------------------------------------------------------------
# Vehicle display (KM & battery)
# ---------------------------------------------------------------------------

def test_vehicle_display_soc_voltage_temp():
    fields = _extract(VEHICLE_DISPLAY_FIELDS, VEHICLE_DISPLAY_TEXT_1)
    assert fields["battery_soc_percent"]["value"] == "100"
    assert fields["battery_voltage"]["value"] == "27.9"
    assert fields["cabin_temp_c"]["value"] == "36"


def test_vehicle_display_km_disambiguation_sample_1():
    fields = {}
    DocumentExtractionPipeline._resolve_km_readings(fields, VEHICLE_DISPLAY_TEXT_1)
    assert fields["trip_km"]["value"] == "834.6"
    assert fields["odometer_km"]["value"] == "20834"
    assert fields["range_to_empty_km"]["value"] == "223"


def test_vehicle_display_km_disambiguation_sample_2():
    fields = {}
    DocumentExtractionPipeline._resolve_km_readings(fields, VEHICLE_DISPLAY_TEXT_2)
    assert fields["trip_km"]["value"] == "6502.3"
    assert fields["odometer_km"]["value"] == "17384"
    assert fields["range_to_empty_km"]["value"] == "111"


def test_vehicle_display_range_greater_than_odometer_is_flagged():
    fields = {
        "odometer_km": {"value": "500"},
        "range_to_empty_km": {"value": "900"},
        "battery_soc_percent": {"value": "50"},
    }
    conflicts = cross_validate("vehicle_display", fields)
    assert any(c["type"] == "IMPLAUSIBLE_READING" for c in conflicts)


def test_vehicle_display_classifier():
    classifier = DocumentClassifier(VEHICLE_DISPLAY_CLASSIFY_RULES, "EV_DASHBOARD")
    assert classifier.classify(VEHICLE_DISPLAY_TEXT_1)["subtype"] == "EV_DASHBOARD"


# ---------------------------------------------------------------------------
# Container seal
# ---------------------------------------------------------------------------

def test_container_seal_extraction_with_manufacturer():
    fields = _finalize(CONTAINER_SEAL_FIELDS, _extract(CONTAINER_SEAL_FIELDS, CONTAINER_SEAL_TEXT_1))
    assert fields["seal_number"]["value"] == "2569011"
    assert fields["seal_manufacturer"]["value"] == "INFINEUM"
    assert fields["seal_type_code"]["value"] == "H"


def test_container_seal_extraction_bare_code():
    fields = _finalize(CONTAINER_SEAL_FIELDS, _extract(CONTAINER_SEAL_FIELDS, CONTAINER_SEAL_TEXT_2))
    assert fields["seal_number"]["value"] == "C7207353"


# ---------------------------------------------------------------------------
# VLM/OCR agreement and disagreement
# ---------------------------------------------------------------------------

def test_agreement_between_vlm_and_ocr_yields_high_confidence_source():
    rules = {"container_no": EIR_FIELDS["container_no"]}
    fields = extract_fields(
        rules,
        vlm_text="Container No: BLKU2603860",
        ocr_text="Container No: BLKU2603860",
        ocr_items=[],
    )
    assert fields["container_no"]["source"] == "VLM+OCR"
    assert fields["container_no"]["confidence"] == 0.97


def test_disagreement_between_vlm_and_ocr_is_surfaced_not_silently_resolved():
    rules = {"container_no": EIR_FIELDS["container_no"]}
    fields = extract_fields(
        rules,
        vlm_text="Container No: BLKU2603860",
        ocr_text="Container No: BLKU2603861",
        ocr_items=[],
    )
    assert fields["container_no"]["source"] == "VLM"
    assert "flagged for review" in fields["container_no"]["reason"]


# ---------------------------------------------------------------------------
# New normalizer/validator kinds
# ---------------------------------------------------------------------------

def test_truck_registration_strict_validation():
    validator = FieldValidator()
    assert validator.validate("MH03FC0372", "truck_registration")["valid"] is True
    assert validator.validate("ABC123", "truck_registration")["valid"] is False


def test_seal_number_requires_a_digit():
    validator = FieldValidator()
    assert validator.validate("TSF0478034", "seal_number")["valid"] is True
    assert validator.validate("ABCDEFGH", "seal_number")["valid"] is False


def test_datetime_normalization_and_validation():
    normalizer, validator = Normalizer(), FieldValidator()
    value = normalizer.normalize("11/08/2026 16:22", "datetime")
    assert value == "2026-08-11T16:22:00"
    assert validator.validate(value, "datetime")["valid"] is True


def test_percentage_and_weight_kinds():
    normalizer, validator = Normalizer(), FieldValidator()
    pct = normalizer.normalize("100%", "percentage")
    assert pct == "100"
    assert validator.validate(pct, "percentage")["valid"] is True

    weight = normalizer.normalize("24.09", "weight")
    assert weight == "24.09"
    assert validator.validate(weight, "weight")["valid"] is True
