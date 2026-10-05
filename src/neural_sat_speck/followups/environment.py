"""The author's pre-outcome battery allowance for opened-data diagnostics."""

import re
from pathlib import Path

from neural_sat_speck.backend.environment import power_sample as original_power_sample


def same_python_environment(reported: str, expected: Path) -> bool:
    """Preserve the venv directory identity as well as its executable inode."""
    actual = Path(reported)
    return (
        actual.is_absolute()
        and actual.parent.resolve() == expected.parent.resolve()
        and actual.samefile(expected)
    )


def power_sample() -> dict:
    sample = original_power_sample()
    raw = sample.get("raw_power")
    sources = (
        re.findall(r"^Now drawing from '([^'\r\n]*)'\s*$", raw, re.M)
        if isinstance(raw, str)
        else []
    )
    source = (
        sources[0]
        if len(sources) == 1 and sources[0] in {"AC Power", "Battery Power"}
        else None
    )
    return {
        **sample,
        "original_ac_policy_accepted": sample.get("accepted"),
        "required_policy": "battery_allowed_author_instruction_20261005",
        "power_source": source,
        "accepted": sample.get("supported") is True
        and source is not None
        and sample.get("ac_power_observed") is (source == "AC Power")
        and sample.get("assertions_returncode") == 0
        and sample.get("power_returncode") == 0
        and sample.get("sleep_assertion_observed") is True,
    }
