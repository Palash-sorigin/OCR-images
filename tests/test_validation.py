from app.services.cross_field_validator import CrossFieldValidator
from app.services.normalizer import Normalizer
from app.services.validator import FieldValidator


def test_phone_normalization_and_validation():
    value = Normalizer().normalize("+91 9594894803", "phone")
    assert value == "9594894803"
    assert FieldValidator().validate(value, "phone")["valid"]


def test_container_count_conflict():
    fields = {
        "container_count": {"value": "1"},
        "export_container_no_1": {"value": "ABCU1234567"},
        "export_container_no_2": {"value": "XYZU1234567"},
    }
    conflicts = CrossFieldValidator().validate(fields)
    assert conflicts[0]["type"] == "CONTAINER_COUNT_MISMATCH"


def test_only_import_export_conflict():
    fields = {
        "transaction_type": {"value": "ONLY IMPORT"},
        "export_container_no_1": {"value": "ABCU1234567"},
    }
    conflicts = CrossFieldValidator().validate(fields)
    assert any(c["type"] == "TRANSACTION_EXPORT_CONFLICT" for c in conflicts)


def test_high_risk_requires_stronger_confidence():
    from app.services.confidence import ConfidenceEngine
    status, _ = ConfidenceEngine().assess(0.91, {"valid": True}, risk="high")
    assert status == "HIGH_CONFIDENCE"
