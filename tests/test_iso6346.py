from app.utils.iso6346 import calculate_check_digit, is_valid_iso6346


def test_known_valid_container_number():
    assert calculate_check_digit("MSCU663987") == 0
    assert is_valid_iso6346("MSCU6639870")
    assert is_valid_iso6346("RLTU3049034")


def test_invalid_check_digit():
    assert not is_valid_iso6346("MSCU6639873")
    assert not is_valid_iso6346("RLTU3049035")


def test_invalid_format():
    assert not is_valid_iso6346("12345678901")
    assert not is_valid_iso6346("ABCU1234567X")
