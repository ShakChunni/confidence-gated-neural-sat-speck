"""Deadline summaries retain failures and distinguish observations from reruns."""

import pytest

from scripts.report_reference_encryption_calibration import deadline_row


def rows():
    return [
        {"id": "a", "success": True, "wall": 100.0},
        {"id": "b", "success": True, "wall": 100.1},
        {"id": "c", "success": False, "wall": 75.0},
        {"id": "d", "success": True, "wall": 299.9},
    ]


def test_inclusive_boundaries_and_failure_denominator():
    result = deadline_row("fixture", "m", rows(), 4, "success", "wall", "id")
    assert result == {
        "study": "fixture",
        "method": "m",
        "n": 4,
        "recovered_by_100s": 1,
        "recovered_by_200s": 2,
        "recovered_by_300s": 3,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "b"),
        ("success", "false"),
        ("wall", float("nan")),
        ("wall", True),
        ("wall", 0.0),
        ("wall", 300.1),
    ],
)
def test_invalid_cohort_record_rejected(field, value):
    records = rows()
    records[0][field] = value
    with pytest.raises(ValueError):
        deadline_row("fixture", "m", records, 4, "success", "wall", "id")


def test_missing_record_rejected():
    with pytest.raises(ValueError):
        deadline_row("fixture", "m", rows()[:-1], 4, "success", "wall", "id")
