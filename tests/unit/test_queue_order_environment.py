"""Battery allowance changes only the AC requirement, before any outcomes."""

import pytest

from neural_sat_speck.followups import environment


@pytest.mark.parametrize("ac", [False, True])
def test_author_allows_battery_or_ac_with_observed_sleep_protection(monkeypatch, ac):
    sample = {
        "supported": True,
        "accepted": ac,
        "ac_power_observed": ac,
        "sleep_assertion_observed": True,
        "assertions_returncode": 0,
        "power_returncode": 0,
        "raw_power": f"Now drawing from '{'AC Power' if ac else 'Battery Power'}'\n",
    }
    monkeypatch.setattr(environment, "original_power_sample", lambda: sample)
    observed = environment.power_sample()
    assert observed["accepted"] is True
    assert observed["original_ac_policy_accepted"] is ac
    assert observed["ac_power_observed"] is ac
    assert observed["power_source"] == ("AC Power" if ac else "Battery Power")


@pytest.mark.parametrize(
    "field,value",
    [
        ("supported", False),
        ("sleep_assertion_observed", False),
        ("assertions_returncode", 1),
        ("power_returncode", 1),
    ],
)
def test_battery_allowance_keeps_telemetry_and_sleep_gates(monkeypatch, field, value):
    sample = {
        "supported": True,
        "accepted": False,
        "ac_power_observed": False,
        "sleep_assertion_observed": True,
        "assertions_returncode": 0,
        "power_returncode": 0,
        "raw_power": "Now drawing from 'Battery Power'\n",
        field: value,
    }
    monkeypatch.setattr(environment, "original_power_sample", lambda: sample)
    assert environment.power_sample()["accepted"] is False


@pytest.mark.parametrize(
    "raw,ac",
    [
        ("Now drawing from 'Unknown Power'\n", False),
        ("", False),
        (None, False),
        ("Now drawing from 'Battery Power'\nNow drawing from 'AC Power'\n", True),
        ("Now drawing from 'AC Power'\nNow drawing from 'Unknown Power'\n", True),
        ("Now drawing from 'Battery Power'\n", True),
        ("Now drawing from 'AC Power'\n", False),
    ],
)
def test_unrecognized_ambiguous_or_inconsistent_power_source_rejected(
    monkeypatch, raw, ac
):
    sample = {
        "supported": True,
        "accepted": False,
        "ac_power_observed": ac,
        "sleep_assertion_observed": True,
        "assertions_returncode": 0,
        "power_returncode": 0,
        "raw_power": raw,
    }
    monkeypatch.setattr(environment, "original_power_sample", lambda: sample)
    assert environment.power_sample()["accepted"] is False


def test_interpreter_alias_accepts_same_venv_but_rejects_other_venv_or_base(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    actual = venv / "bin" / "python"
    actual.write_bytes(b"synthetic-executable")
    alias = tmp_path / "alias"
    alias.symlink_to(venv, target_is_directory=True)
    assert environment.same_python_environment(str(actual), alias / "bin" / "python")
    other = tmp_path / "other" / "bin"
    other.mkdir(parents=True)
    (other / "python").symlink_to(actual)
    assert not environment.same_python_environment(str(other / "python"), actual)
    assert not environment.same_python_environment("relative/bin/python", actual)
