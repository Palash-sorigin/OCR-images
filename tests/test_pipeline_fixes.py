from pathlib import Path

from app.services.field_extractor import FieldExtractor
from app.services.ocr_service import ContainerOCRService
from app.services.paddle_engine import PaddleOCREngine
from app.services.pipeline import SequentialExtractionPipeline
from app.services.vlm_engine import VLMEngine
from app.services.vlm_verifier import VLMVerifier


def _item(text, box, confidence=0.95):
    return {"text": text, "confidence": confidence, "box": box}


# ---------------------------------------------------------------------------
# Container OCR should only run when the screen's schema has a container field
# ---------------------------------------------------------------------------

def test_container_expected_skips_screens_without_container_fields():
    assert SequentialExtractionPipeline._container_expected("PIN_GENERATION_APM") is False


def test_container_expected_runs_for_screens_with_container_fields():
    assert SequentialExtractionPipeline._container_expected("PIN_GENERATION_PSA") is True
    assert SequentialExtractionPipeline._container_expected("PIN_GENERATION_NSFT") is True
    assert SequentialExtractionPipeline._container_expected("TRUCK_BOOKING_NSFT") is True
    assert SequentialExtractionPipeline._container_expected("UNKNOWN") is True


# ---------------------------------------------------------------------------
# Field extraction must not let a blank field steal a neighboring field's
# real value, and the below-label fallback must stay within roughly one row.
# ---------------------------------------------------------------------------

def test_blank_field_does_not_steal_next_rows_value():
    items = [
        _item("Booking No 1", [0, 100, 150, 130]),
        _item("Booking No 2", [0, 160, 150, 190]),
        _item("BKNO002", [200, 160, 280, 190]),
    ]

    fields = FieldExtractor().extract(items, screen_type="TRUCK_BOOKING_NSFT")

    assert fields["booking_no_2"]["value"] == "BKNO002"
    assert fields["booking_no_1"]["value"] is None
    assert fields["booking_no_1"]["status"] == "EMPTY"


def test_below_fallback_does_not_reach_past_one_row():
    items = [
        _item("Genset", [0, 100, 100, 120]),
        _item("UNRELATEDTOKEN", [20, 220, 120, 240]),
    ]

    fields = FieldExtractor().extract(items, screen_type="TRUCK_BOOKING_NSFT")

    assert fields["genset_1"]["value"] is None


def test_same_row_value_still_assigned():
    items = [
        _item("TT No", [0, 100, 100, 130]),
        _item("MH03FC0372", [150, 100, 300, 130]),
    ]
    fields = FieldExtractor().extract(items, screen_type="TRUCK_BOOKING_NSFT")
    assert fields["tt_no"]["value"] == "MH03FC0372"


# ---------------------------------------------------------------------------
# VLM should run once per image, not once per unresolved high-risk field.
# ---------------------------------------------------------------------------

class _FakeVLMResult:
    def __init__(self, content):
        self._content = content

    @property
    def json(self):
        return {"res": {"parsing_res_list": [{"block_content": self._content}]}}


class _FakeVLM:
    def __init__(self, text):
        self.call_count = 0
        self._text = text

    def predict(self, path):
        self.call_count += 1
        return [_FakeVLMResult(self._text)]


def test_vlm_runs_once_per_image_across_multiple_pending_fields():
    fake_vlm = _FakeVLM("TT No 1234567890 PIN No 998877 Mobile 9876543210")
    verifier = VLMVerifier(engine=None)
    verifier.vlm = fake_vlm

    pipeline = SequentialExtractionPipeline(general_ocr=None, container_ocr=None, vlm=verifier)

    fields = {
        "tt_no": {"value": "1234567890", "validation": {"valid": True}},
        "pin_no": {"value": "998877", "validation": {"valid": True}},
        "driver_mobile": {"value": "9876543210", "validation": {"valid": True}},
    }
    manual_review = [
        {"field": "tt_no", "status": "NEEDS_REVIEW"},
        {"field": "pin_no", "status": "NEEDS_REVIEW"},
        {"field": "driver_mobile", "status": "NEEDS_REVIEW"},
    ]

    pipeline._vlm_fallback(Path("fake.jpg"), fields, manual_review)

    assert fake_vlm.call_count == 1
    assert fields["tt_no"]["status"] == "VLM_VERIFIED"
    assert fields["pin_no"]["status"] == "VLM_VERIFIED"
    assert fields["driver_mobile"]["status"] == "VLM_VERIFIED"


def test_container_service_and_verifier_share_one_vlm_instance():
    engine = VLMEngine()
    engine.vlm = object()
    engine.error = None

    container_service = ContainerOCRService(engine=PaddleOCREngine(), vlm_engine=engine)
    verifier = VLMVerifier(engine=engine)

    assert container_service.vlm is engine.vlm
    assert verifier.vlm is engine.vlm
