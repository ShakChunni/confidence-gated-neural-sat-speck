"""Synthetic-only contracts for interrupted conditioning-ledger sealing."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import (
    THREAD_ENVIRONMENT,
    atomic_json,
    read_checked_json,
    sha256_file,
)
from neural_sat_speck.followups import conditioning_control as control
from neural_sat_speck.followups import conditioning_runner as runner
from neural_sat_speck.followups import conditioning_seal as seal


def _power_sample(monotonic: float) -> dict:
    return {
        "observed_monotonic": monotonic,
        "power": {
            "power_source": "AC Power",
            "supported": True,
            "accepted": True,
            "required_policy": "battery_allowed_author_instruction_20261005",
            "assertions_returncode": 0,
            "power_returncode": 0,
            "sleep_assertion_observed": True,
            "ac_power_observed": True,
            "original_ac_policy_accepted": True,
            "raw_assertions": "synthetic assertion receipt",
            "raw_power": "Now drawing from 'AC Power'\n",
            "utc": "2026-10-06T00:00:00+00:00",
        },
    }


@pytest.fixture
def campaign_fixture(tmp_path, monkeypatch):
    """A tiny, pinned-shape synthetic campaign; no solver or cohort is loaded."""
    campaign = tmp_path / "campaign"
    (campaign / "history").mkdir(parents=True)
    (campaign / "coordinator.lock").touch()
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {"synthetic_manifest": True})

    cells = control.fixed_cells("pilot")
    paths = {"python": Path(sys.executable).absolute()}
    cases = {}
    for cell in cells:
        ident = cell["case_id"]
        if ident in cases:
            continue
        public = tmp_path / f"{ident}_public.json"
        truth = tmp_path / f"{ident}_truth.json"
        atomic_json(public, {"case": ident, "role": "public"})
        atomic_json(truth, {"case": ident, "role": "truth"})
        paths[ident + ":public"] = public
        paths[ident + ":truth"] = truth
        cases[ident] = {
            "public": {"sha256": sha256_file(public)},
            "truth": {"sha256": sha256_file(truth)},
            "formula_sha256": "f" * 64,
        }
    manifest = {
        "mode": "pilot",
        "source_root": str(source_root),
        "source_commit": "a" * 40,
        "cells": cells,
        "cases": cases,
    }

    def load(_path):
        return manifest, paths

    monkeypatch.setattr(seal, "load_manifest", load)
    monkeypatch.setattr(seal, "verify_cases", lambda *_: None)
    monkeypatch.setattr(runner, "load_manifest", load)
    monkeypatch.setattr(runner, "verify_cases", lambda *_: None)
    monkeypatch.setattr(seal, "_guardian_is_running", lambda *_: False)
    monkeypatch.setattr(
        runner, "_validate_tmpdir", lambda telemetry, _root: telemetry["tmpdir"]
    )

    return {
        "campaign": campaign,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "paths": paths,
        "cells": cells,
        "case_root": tmp_path,
    }


def _payload_and_config(fx, cell: dict, token: str, directory: Path, deadline: float):
    ident = cell["case_id"]
    manifest, paths = fx["manifest"], fx["paths"]
    payload = {
        "cell": cell,
        "invocation_id": token,
        "directory": str(directory),
        "repo_root": manifest["source_root"],
        "source_commit": manifest["source_commit"],
        "python_path": str(paths["python"]),
        "public_path": str(paths[ident + ":public"]),
        "truth_path": str(paths[ident + ":truth"]),
        "public_sha256": manifest["cases"][ident]["public"]["sha256"],
        "truth_sha256": manifest["cases"][ident]["truth"]["sha256"],
        "formula_sha256": manifest["cases"][ident]["formula_sha256"],
        "deadline_monotonic": deadline,
    }
    config = {
        "deadline_monotonic": deadline,
        "invocation": token,
        "directory": str(directory),
        "repo_root": manifest["source_root"],
        "coordinator_pid": 777,
        "worker_arguments": [
            str(paths["python"]),
            "-m",
            "neural_sat_speck.followups.conditioning_worker",
            str(directory / "payload.json"),
        ],
        "receipt": str(directory / "guardian_receipt.json"),
    }
    return payload, config


def _write_pending(fx, cell: dict, token: str, *, no_child=False):
    campaign = fx["campaign"]
    directory = campaign / "cells" / cell["cell_id"] / token
    directory.mkdir(parents=True)
    deadline = 1300.0
    payload, config = _payload_and_config(fx, cell, token, directory, deadline)
    atomic_json(directory / "payload.json", payload)
    atomic_json(directory / "guardian.json", config)
    loss = {"coordinator_pid": 777, "observed_monotonic": 1005.0}
    atomic_json(directory / "coordinator_loss.json", loss)
    receipt = {
        "watchdog_pid": 555,
        "invocation": token,
        "deadline_monotonic": deadline,
        "status": "coordinator_stop",
        "observed_monotonic": 1006.0,
        "cleanup_completed_monotonic": 1007.0,
        "process_group_absent": True,
    }
    if no_child:
        receipt["worker_not_started"] = True
    else:
        receipt.update(
            worker_pid=222,
            process_group=222,
            worker_returncode=-9,
        )
        atomic_json(
            directory / "process.json",
            {
                "pid": 222,
                "process_group": 222,
                "invocation": token,
                "supervisor_pid": 555,
            },
        )
        atomic_json(directory / "environment_samples.json", [_power_sample(1001.0)])
        atomic_json(
            directory / "journal.json",
            {
                "completed_seconds": {"input_loading": 0.1},
                "phase": "solving",
                "phase_started_monotonic": 1002.0,
            },
        )
        # A late raw positive stays forensic evidence; it cannot become a result.
        atomic_json(
            directory / "worker_result.json",
            {
                "invocation_id": token,
                "cell_id": cell["cell_id"],
                "formula_sha256": payload["formula_sha256"],
                "outcome": "recovered",
                "completed_at_monotonic": deadline + 0.1,
            },
        )
    atomic_json(directory / "guardian_receipt.json", receipt)
    return directory, receipt


def _write_previous_completed_solver_unknown(fx, cell: dict, token: str):
    campaign = fx["campaign"]
    directory = campaign / "cells" / cell["cell_id"] / token
    directory.mkdir(parents=True)
    start, deadline, observed = 500.0, 800.0, 501.0
    payload, config = _payload_and_config(fx, cell, token, directory, deadline)
    atomic_json(directory / "payload.json", payload)
    atomic_json(directory / "guardian.json", config)
    receipt = {
        "watchdog_pid": 555,
        "worker_pid": 222,
        "process_group": 222,
        "invocation": token,
        "deadline_monotonic": deadline,
        "status": "worker_exit",
        "worker_returncode": 0,
        "observed_monotonic": observed,
        "cleanup_completed_monotonic": observed + 0.1,
        "process_group_absent": True,
    }
    atomic_json(directory / "guardian_receipt.json", receipt)
    atomic_json(
        directory / "process.json",
        {
            "pid": 222,
            "process_group": 222,
            "invocation": token,
            "supervisor_pid": 555,
        },
    )
    samples = [_power_sample(observed)]
    atomic_json(directory / "environment_samples.json", samples)
    tmpdir = Path(fx["manifest"]["source_root"]) / "runtime" / "conditioning_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        directory / "python_worker_receipt.json",
        {
            "is_certified": True,
            "pid": 222,
            "parent_pid": 555,
            "git_commit": fx["manifest"]["source_commit"],
            "qos_class": 33,
            "darwin_priority": 0,
            "darwin_bg_state": 0,
            "io_policy_state": 0,
            "python_executable": str(fx["paths"]["python"]),
            "thread_environment": THREAD_ENVIRONMENT,
            "tmpdir": str(tmpdir),
            "tmpdir_safe": True,
        },
    )
    raw = {
        "invocation_id": token,
        "cell_id": cell["cell_id"],
        "formula_sha256": fx["manifest"]["cases"][cell["case_id"]]["formula_sha256"],
        "outcome": "solver_unknown",
        "solver_status": "unknown",
        "completed_at_monotonic": observed - 0.1,
    }
    atomic_json(directory / "worker_result.json", raw)
    row = {
        "cell": cell,
        "directory": str(directory),
        "outcome": "solver_unknown",
        "started_monotonic": start,
        "deadline_monotonic": deadline,
        "observed_monotonic": observed,
        "parent_wall_seconds": observed - start,
        "guardian_receipt": receipt,
        "cleanup_wall_seconds": 0.1,
        "phase_accounting": None,
        "environment_samples": samples,
        "worker_result": raw,
        "completed_utc": "2026-10-06T00:00:00+00:00",
    }
    atomic_json(directory / "supervisor_result.json", row)
    return row, directory


def _write_running_ledger(fx, starts: dict, records: dict):
    campaign = fx["campaign"]
    path = campaign / "ledger.json"
    manifest_path = fx["manifest_path"]
    ledger = {
        "schema": "conditioning_control_results_v1",
        "mode": "pilot",
        "manifest_path": str(manifest_path.absolute()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": fx["manifest"]["source_commit"],
        "started_utc": "2026-10-06T00:00:00+00:00",
        "status": "RUNNING",
        "starts": starts,
        "records": records,
    }
    atomic_json(campaign / "history" / "0000.json", ledger)
    atomic_json(path, ledger)
    return path, ledger


def test_seals_completed_rows_and_interrupted_late_positive_without_retry(
    campaign_fixture,
):
    fx = campaign_fixture
    first, pending = fx["cells"][:2]
    first_token, pending_token = "1" * 32, "2" * 32
    prior, prior_dir = _write_previous_completed_solver_unknown(fx, first, first_token)
    pending_dir, _ = _write_pending(fx, pending, pending_token)
    starts = {
        first["cell_id"]: {
            "directory": str(prior_dir),
            "invocation_id": first_token,
            "utc": "2026-10-06T00:00:00+00:00",
        },
        pending["cell_id"]: {
            "directory": str(pending_dir),
            "invocation_id": pending_token,
            "utc": "2026-10-06T00:00:01+00:00",
        },
    }
    ledger_path, original = _write_running_ledger(fx, starts, {first["cell_id"]: prior})
    original_bytes = ledger_path.read_bytes()
    prior_bytes = (prior_dir / "supervisor_result.json").read_bytes()

    output = fx["campaign"] / "sealed_ledger.json"
    sealed = seal.seal_running_ledger(ledger_path, fx["manifest_path"], output)

    assert sealed["status"] == "INTERRUPTED"
    assert sealed["records"][first["cell_id"]] == prior
    interrupted = sealed["records"][pending["cell_id"]]
    assert interrupted["outcome"] == "infrastructure_interrupted"
    assert interrupted["dispatched"] is True
    assert interrupted["worker_result"]["outcome"] == "recovered"
    assert interrupted["worker_result"]["completed_at_monotonic"] >= 1300.0
    assert interrupted["outcome"] not in control.SCIENTIFIC_OUTCOMES
    assert all(
        sealed["records"][cell["cell_id"]]["outcome"] == "unavailable"
        for cell in fx["cells"][2:]
    )
    assert read_checked_json(output) == sealed
    assert (
        read_checked_json(pending_dir / "supervisor_sealed_result.json") == interrupted
    )
    assert ledger_path.read_bytes() == original_bytes
    assert (prior_dir / "supervisor_result.json").read_bytes() == prior_bytes
    assert read_checked_json(ledger_path) == original


def test_pre_dispatch_crash_is_sealed_only_from_no_child_terminal_receipt(
    campaign_fixture,
):
    fx = campaign_fixture
    cell = fx["cells"][0]
    token = "3" * 32
    directory, _ = _write_pending(fx, cell, token, no_child=True)
    starts = {
        cell["cell_id"]: {
            "directory": str(directory),
            "invocation_id": token,
            "utc": "2026-10-06T00:00:00+00:00",
        }
    }
    ledger_path, _ = _write_running_ledger(fx, starts, {})
    output = fx["campaign"] / "sealed_ledger.json"

    sealed = seal.seal_running_ledger(ledger_path, fx["manifest_path"], output)

    row = sealed["records"][cell["cell_id"]]
    assert row["outcome"] == "infrastructure_interrupted"
    assert row["dispatched"] is False
    assert row["phase_accounting"] is None
    assert row["environment_samples"] == []


@pytest.mark.parametrize("damage", ["no_receipt", "group_present", "wrong_loss_pid"])
def test_refuses_unproved_or_mismatched_guardian_stop(campaign_fixture, damage):
    fx = campaign_fixture
    cell = fx["cells"][0]
    token = "4" * 32
    directory, receipt = _write_pending(fx, cell, token)
    if damage == "no_receipt":
        (directory / "guardian_receipt.json").unlink()
        (directory / "guardian_receipt.json.sha256").unlink()
    elif damage == "group_present":
        receipt["process_group_absent"] = False
        atomic_json(directory / "guardian_receipt.json", receipt)
    else:
        atomic_json(
            directory / "coordinator_loss.json",
            {"coordinator_pid": 778, "observed_monotonic": 1005.0},
        )
    ledger_path, _ = _write_running_ledger(
        fx,
        {
            cell["cell_id"]: {
                "directory": str(directory),
                "invocation_id": token,
                "utc": "2026-10-06T00:00:00+00:00",
            }
        },
        {},
    )
    output = fx["campaign"] / "sealed_ledger.json"

    with pytest.raises((ValueError, OSError)):
        seal.seal_running_ledger(ledger_path, fx["manifest_path"], output)
    assert not output.exists()


def test_refuses_live_guardian_and_history_ledger_gap(campaign_fixture, monkeypatch):
    fx = campaign_fixture
    cell = fx["cells"][0]
    token = "5" * 32
    directory, _ = _write_pending(fx, cell, token)
    ledger_path, ledger = _write_running_ledger(
        fx,
        {
            cell["cell_id"]: {
                "directory": str(directory),
                "invocation_id": token,
                "utc": "2026-10-06T00:00:00+00:00",
            }
        },
        {},
    )
    output = fx["campaign"] / "sealed_ledger.json"
    monkeypatch.setattr(seal, "_guardian_is_running", lambda *_: True)
    with pytest.raises(ValueError, match="still running"):
        seal.seal_running_ledger(ledger_path, fx["manifest_path"], output)
    assert not output.exists()

    monkeypatch.setattr(seal, "_guardian_is_running", lambda *_: False)
    # A validly checksummed history entry that is ahead of ledger.json is an
    # ambiguous persist() crash window; the sealer must not guess which is current.
    advanced = copy.deepcopy(ledger)
    advanced["history_fixture_marker"] = "newer archived revision"
    atomic_json(fx["campaign"] / "history" / "0000.json", advanced)
    with pytest.raises(ValueError, match="Latest archived ledger differs"):
        seal.seal_running_ledger(ledger_path, fx["manifest_path"], output)
    assert not output.exists()
