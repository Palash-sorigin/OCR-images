"""Tests for the OCR + image + Gemini Flash-Lite extraction path (visit
ticket). No real network calls are made -- GeminiLLMEngine's model call is
stubbed via monkeypatching _call_model, matching the FakeVLM approach
already used in tests/test_pipeline_fixes.py.
"""

import time
from pathlib import Path

import app.services.gemini_llm_engine as gemini_module
from app.services.document_pipeline import DocumentExtractionPipeline
from app.services.gemini_llm_engine import GeminiLLMEngine
from app.services.ocr_candidate_builder import build_candidates


def _make_engine(available: bool = True) -> GeminiLLMEngine:
    """Build a GeminiLLMEngine without touching the network or requiring
    a real API key -- __init__ normally short-circuits when GEMINI_API_KEY
    is unset, so fake a loaded client directly."""
    engine = GeminiLLMEngine.__new__(GeminiLLMEngine)
    engine.client = object() if available else None
    engine.model = "gemini-3.1-flash-lite"
    engine.error = None if available else "GEMINI_API_KEY is not set."
    return engine


def _write_temp_image(tmp_path: Path) -> Path:
    from PIL import Image
    path = tmp_path / "fake.jpg"
    Image.new("RGB", (10, 10), color="white").save(path)
    return path


FIELD_RULES = {
    "truck_no": {"aliases": ["Truck No"], "pattern": r"([A-Z]{2}\d{2}[A-Z]{1,3}\d{4})", "kind": "truck_registration", "risk": "high"},
    "container_no": {"aliases": ["Container No"], "pattern": r"([A-Z]{4}\d{7})", "kind": "container", "risk": "high"},
}


# ---------------------------------------------------------------------------
# GeminiLLMEngine: unavailable / timeout / success / hallucination guard
# ---------------------------------------------------------------------------

def test_engine_unavailable_returns_null_result_immediately(tmp_path):
    engine = _make_engine(available=False)
    image = _write_temp_image(tmp_path)
    context = {"ocr_items": [], "candidate_mapping": {}, "full_ocr_text": "", "document_subtype": "X"}

    t0 = time.perf_counter()
    result, meta = engine.extract_fields(image, context, FIELD_RULES)
    elapsed = time.perf_counter() - t0

    assert meta["status"] == "UNAVAILABLE"
    assert elapsed < 1.0  # must not attempt any network call
    assert all(v["value"] is None for v in result.values())


def test_engine_hard_timeout_is_enforced(tmp_path, monkeypatch):
    engine = _make_engine(available=True)
    image = _write_temp_image(tmp_path)
    context = {"ocr_items": [], "candidate_mapping": {}, "full_ocr_text": "", "document_subtype": "X"}

    def slow_call(image_bytes, mime_type, user_prompt):
        time.sleep(2.0)  # much longer than the test's timeout below
        return '{"truck_no": {"value": "MH03FC0372", "ocr_indices": [], "reason": "x"}}'

    monkeypatch.setattr(engine, "_call_model", slow_call)
    monkeypatch.setattr(gemini_module, "GEMINI_TIMEOUT_SECONDS", 0.3)

    t0 = time.perf_counter()
    result, meta = engine.extract_fields(image, context, FIELD_RULES)
    elapsed = time.perf_counter() - t0

    assert meta["status"] == "TIMEOUT"
    assert elapsed < 1.5  # bounded by the timeout, NOT the 2s sleep
    assert all(v["value"] is None for v in result.values())


def test_engine_successful_call_returns_validated_fields(tmp_path, monkeypatch):
    engine = _make_engine(available=True)
    image = _write_temp_image(tmp_path)
    context = {
        "ocr_items": [
            {"index": 0, "text": "Truck No:", "confidence": 0.9, "bbox": None},
            {"index": 1, "text": "MH03FC0372", "confidence": 0.9, "bbox": None},
        ],
        "candidate_mapping": {},
        "full_ocr_text": "Truck No: MH03FC0372",
        "document_subtype": "X",
    }

    def fake_call(image_bytes, mime_type, user_prompt):
        return '{"truck_no": {"value": "MH03FC0372", "ocr_indices": [1], "reason": "matches label"}, "container_no": {"value": null, "ocr_indices": [], "reason": "not present"}}'

    monkeypatch.setattr(engine, "_call_model", fake_call)

    result, meta = engine.extract_fields(image, context, FIELD_RULES)

    assert meta["status"] == "OK"
    assert result["truck_no"]["value"] == "MH03FC0372"
    assert result["container_no"]["value"] is None


def test_engine_discards_hallucinated_value_not_in_ocr_data(tmp_path, monkeypatch):
    engine = _make_engine(available=True)
    image = _write_temp_image(tmp_path)
    context = {
        "ocr_items": [{"index": 0, "text": "MH03FC0372", "confidence": 0.9, "bbox": None}],
        "candidate_mapping": {},
        "full_ocr_text": "MH03FC0372",
        "document_subtype": "X",
    }

    def fabricating_call(image_bytes, mime_type, user_prompt):
        # A value the OCR data never contained -- must be discarded.
        return '{"truck_no": {"value": "KA05ZZ9999", "ocr_indices": [0], "reason": "fabricated"}, "container_no": {"value": null, "ocr_indices": [], "reason": ""}}'

    monkeypatch.setattr(engine, "_call_model", fabricating_call)

    result, meta = engine.extract_fields(image, context, FIELD_RULES)

    assert meta["status"] == "OK"
    assert result["truck_no"]["value"] is None
    assert "discarded" in result["truck_no"]["reason"]


def test_engine_unparseable_response_falls_back_to_null(tmp_path, monkeypatch):
    engine = _make_engine(available=True)
    image = _write_temp_image(tmp_path)
    context = {"ocr_items": [], "candidate_mapping": {}, "full_ocr_text": "", "document_subtype": "X"}

    monkeypatch.setattr(engine, "_call_model", lambda *a, **k: "not json at all")

    result, meta = engine.extract_fields(image, context, FIELD_RULES)

    assert meta["status"] == "ERROR"
    assert all(v["value"] is None for v in result.values())


# ---------------------------------------------------------------------------
# DocumentExtractionPipeline: falls back to fast OCR-only, never to VLM
# ---------------------------------------------------------------------------

class _FailingGeminiEngine:
    """Simulates Gemini being unavailable/erroring for every call."""
    available = True
    model = "gemini-3.1-flash-lite"

    def extract_fields(self, image_path, llm_context, field_rules):
        return (
            {name: {"value": None, "ocr_indices": [], "reason": "forced failure"} for name in field_rules},
            {"status": "ERROR", "elapsed_s": 0.01, "error": "forced failure"},
        )


class _CountingVLM:
    """Tracks whether the VLM path was ever invoked -- it must not be,
    even when the LLM path fails."""
    vlm = None  # VLM disabled; if _process_with_vlm ran, this alone would no-op,
    # so we also assert on call_count below via the wrapped verify method.
    call_count = 0

    def extract_text(self, image_path):
        self.call_count += 1
        return {"available": True, "text": ""}


def test_llm_failure_falls_back_to_fast_ocr_only_not_vlm(tmp_path):
    from app.services.general_ocr import GeneralOCRService

    class _FakePaddleEngine:
        def predict(self, path):
            items = [
                {"text": "Truck No:", "confidence": 0.9, "box": [0, 0, 10, 10]},
                {"text": "MH03FC0372", "confidence": 0.9, "box": [20, 0, 40, 10]},
            ]
            return [], items, None

    general_ocr = GeneralOCRService(_FakePaddleEngine())
    vlm = _CountingVLM()
    gemini = _FailingGeminiEngine()

    pipeline = DocumentExtractionPipeline("visit_ticket", general_ocr, vlm, gemini_engine=gemini)
    assert pipeline._use_llm is True  # confirms the LLM path is actually selected

    image = _write_temp_image(tmp_path)
    result = pipeline.process_one(image.name, image)

    # The LLM failed, so it must have fallen back to OCR-only -- meaning
    # the (much slower) VLM path was never touched.
    assert vlm.call_count == 0
    assert result.fields["truck_no"].value == "MH03FC0372"


# ---------------------------------------------------------------------------
# ocr_candidate_builder: cross-item matches get real confidence, not a flat
# placeholder (regression test for the bug found while testing this path)
# ---------------------------------------------------------------------------

def test_candidate_builder_promotes_cross_item_match_to_real_confidence():
    ocr_items = [
        {"text": "Container No", "confidence": 0.93, "box": [0, 0, 10, 10]},
        {"text": "MRKU7845180", "confidence": 0.89, "box": [20, 0, 40, 10]},
    ]
    rules = {"container_no": {"aliases": ["Container No"], "pattern": r"([A-Z]{4}\d{7})", "kind": "container"}}

    candidates = build_candidates(ocr_items, rules)

    assert candidates["container_no"]["possible_candidates"] == []
    assert len(candidates["container_no"]["candidates"]) == 1
    top = candidates["container_no"]["candidates"][0]
    assert top["value"] == "MRKU7845180"
    assert top["confidence"] == 0.89  # real OCR confidence, not the old flat 0.70
    assert top["ocr_index"] == 1
