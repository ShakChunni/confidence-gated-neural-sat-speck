"""Skip tests whose platform features are missing, and say so in the header.

The experiment runners stop solvers with POSIX process groups (os.killpg),
lock files with fcntl and cap memory with resource. Windows has none of these,
so the tests of those runners cannot import or run there. Some guard tests also
create symbolic links, which Windows allows only with Developer Mode or admin
rights. The watchdogs also watch their child with os.waitid(WNOWAIT), which
CPython on macOS provides only from Python 3.13. Detection is by capability, not by platform name, so a system that has
the feature runs the test.

Nothing here changes what a test asserts. The cipher, SAT model, ranking,
verification and statistics tests run on every platform.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

HAS_POSIX_PROCESS_CONTROL = (
    importlib.util.find_spec("fcntl") is not None
    and importlib.util.find_spec("resource") is not None
    and hasattr(os, "killpg")
)


def _can_symlink() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.symlink(tmp, Path(tmp) / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
    return True


HAS_SYMLINK = _can_symlink()

HAS_WAITID = hasattr(os, "waitid") and hasattr(os, "WNOWAIT")

# Modules that import fcntl or resource (directly or through the runner they
# test), or whose fixtures drive process groups in every test.
POSIX_MODULES = [
    "unit/test_backend_guards.py",
    "unit/test_backend_reviewed_resume.py",
    "unit/test_backend_runner.py",
    "unit/test_backend_watchdog_cleanup.py",
    "unit/test_conditioning_control.py",
    "unit/test_conditioning_seal.py",
    "unit/test_phase6b_h1_protocol.py",
    "unit/test_phase6c_protocol.py",
    "unit/test_prospective_continuation.py",
    "unit/test_prospective_queue_cost_analysis.py",
    "unit/test_prospective_queue_cost_execution.py",
    "unit/test_prospective_queue_cost_grid.py",
    "unit/test_queue_order_guards.py",
    "unit/test_queue_order_pilot.py",
    "unit/test_queue_order_runner.py",
    "unit/test_ranking_depth_analysis.py",
    "unit/test_ranking_depth_audit.py",
    "unit/test_ranking_depth_execution.py",
    "unit/test_ranking_depth_recovery.py",
    "unit/test_ranking_depth_segment_audit.py",
    "unit/test_simon_comparison_input.py",
    "unit/test_simon_confirmation_runner.py",
    "unit/test_simon_development_runner.py",
    "unit/test_simon_method_worker.py",
    "unit/test_simon_operational_supervisor.py",
    "unit/test_simon_sat_only_300s_controls.py",
    "unit/test_simon_stage2_exit_policy.py",
    "unit/test_simon_stage2_runner.py",
    "unit/test_simon_stage2_seal.py",
]

# Single tests in otherwise portable modules that need POSIX process control.
POSIX_TESTS = {
    "tests/unit/test_depth_map_runner.py::test_parent_watchdog_terminates_hanging_worker",
    "tests/unit/test_simon_confirmation_launch.py::test_confirmation_analysis_rejects_equal_partial_grid",
    "tests/unit/test_simon_stage2_watchdog.py::test_accepts_loaded_then_all_eight_calls_and_advances_deadline",
    "tests/unit/test_simon_stage2_watchdog.py::test_waits_briefly_for_event_sidecar_without_rejecting_complete_event",
    "tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls",
    "tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed",
    "tests/unit/test_simon_stage2_watchdog.py::test_campaign_deadline_caps_loading_deadline",
    "tests/unit/test_simon_stage2_watchdog.py::test_configuration_drift_is_rejected_before_worker_dispatch",
    "tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_after_dispatch_stops_worker_group",
}

SYMLINK_TESTS = {
    "tests/unit/test_backend_evidence.py::test_worker_directory_cannot_be_redirected_or_symlinked",
    "tests/unit/test_backend_guards.py::test_keeps_venv_executable_symlink",
    "tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected",
    "tests/unit/test_publication_evidence_package.py::test_formula_symlink_outside_pinned_directory_rejected",
    "tests/unit/test_queue_order_environment.py::test_interpreter_alias_accepts_same_venv_but_rejects_other_venv_or_base",
    "tests/unit/test_simon_stage2_control.py::test_interpreter_file_pin_preserves_lexical_symlink",
    "tests/unit/test_worker_telemetry.py::test_symlink_escape_tmpdir_rejected",
}

# Tests that start a real watchdog, which refuses to run without os.waitid.
WAITID_TESTS = {
    "tests/unit/test_backend_runner.py::test_nonzero_worker_exit_cannot_credit_a_written_result",
    "tests/unit/test_backend_runner.py::test_parent_kills_hanging_phase",
    "tests/unit/test_backend_runner.py::test_pid_persistence_failure_still_cleans_owned_child",
    "tests/unit/test_backend_runner.py::test_supervisor_enforces_deadline_after_coordinator_is_killed",
    "tests/unit/test_backend_watchdog_cleanup.py::test_main_preserves_error_receipt_and_never_signals_after_reaping",
    "tests/unit/test_conditioning_control.py::test_independent_guardian_deadline_and_cleanup",
    "tests/unit/test_protocol_manifest.py::test_prospective_execution_accepted_on_frozen_authorized",
    "tests/unit/test_queue_order_runner.py::test_coordinator_death_stops_child_before_its_original_deadline",
    "tests/unit/test_queue_order_runner.py::test_real_guardian_kills_hanging_followup_worker_in_every_phase",
    "tests/unit/test_simon_stage2_watchdog.py::test_accepts_loaded_then_all_eight_calls_and_advances_deadline",
    "tests/unit/test_simon_stage2_watchdog.py::test_campaign_deadline_caps_loading_deadline",
    "tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_after_dispatch_stops_worker_group",
    "tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_before_dispatch_writes_stop_receipt",
    "tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed",
    "tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls",
    "tests/unit/test_simon_stage2_watchdog.py::test_waits_briefly_for_event_sidecar_without_rejecting_complete_event",
}

collect_ignore = [] if HAS_POSIX_PROCESS_CONTROL else list(POSIX_MODULES)


def pytest_report_header(config: pytest.Config) -> list[str]:
    lines = []
    if not HAS_POSIX_PROCESS_CONTROL:
        lines.append(
            f"no POSIX process control: {len(POSIX_MODULES)} runner test modules "
            f"not collected and {len(POSIX_TESTS)} tests skipped (see tests/conftest.py)"
        )
    if not HAS_WAITID:
        lines.append(
            f"no os.waitid (macOS before Python 3.13): {len(WAITID_TESTS)} watchdog tests skipped"
        )
    if not HAS_SYMLINK:
        lines.append(f"no symlink permission: {len(SYMLINK_TESTS)} tests skipped")
    return lines


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        node = item.nodeid.split("[", 1)[0]
        if not HAS_POSIX_PROCESS_CONTROL and node in POSIX_TESTS:
            item.add_marker(
                pytest.mark.skip(reason="needs POSIX process groups (os.killpg)")
            )
        if not HAS_WAITID and node in WAITID_TESTS:
            item.add_marker(
                pytest.mark.skip(reason="needs os.waitid (macOS: Python 3.13 or newer)")
            )
        if not HAS_SYMLINK and node in SYMLINK_TESTS:
            item.add_marker(
                pytest.mark.skip(reason="needs permission to create symbolic links")
            )
