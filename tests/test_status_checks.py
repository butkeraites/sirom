from sirom.status_checks import has_errors


def test_has_errors_detects_error_messages():
    assert has_errors(["[OK] fine", "[ERROR] broke", "[INFO] note"]) is True


def test_has_errors_false_without_error():
    assert has_errors(["[OK] fine", "[INFO] note"]) is False


def test_has_errors_empty_status():
    assert has_errors([]) is False
