#!/usr/bin/env bash
# Levels 1 and 2 of docs/REPRODUCE.md, inside the Linux image from the Dockerfile.
set -u
E=docs/paper/evidence
T=$(mktemp -d)
status=0
step() { echo; echo "== $1"; }

step "Known-answer test"
neural-sat-speck kat || status=1

step "Recorded file hashes"
python - <<'PY' || status=1
import hashlib, re, subprocess, sys
from pathlib import Path
ok = bad = absent = other = 0
for line in subprocess.run(["git", "ls-files", "*.sha256"], capture_output=True, text=True, check=True).stdout.split("\n"):
    if not line:
        continue
    side = Path(line)
    m = re.match(r"^([0-9a-f]{64}) [ *]?(\S.*?)\s*$", side.read_text().split("\n")[0])
    if not m:
        other += 1
        continue
    name = Path(m[2])
    candidates = [side.parent / name.name, name]
    target = next((c for c in candidates if c.is_file()), None)
    if target is None:
        absent += 1
    elif hashlib.sha256(target.read_bytes()).hexdigest() == m[1]:
        ok += 1
    else:
        bad += 1
        print("MISMATCH", target)
print(f"matched {ok}, mismatched {bad}, target not in repo {absent}, other format {other}")
sys.exit(1 if bad else 0)
PY

step "Saved analyses against saved summaries"
python scripts/analyze_d063.py "$E/d063_campaign_20261011/ledger_k00-31.jsonl" > "$T/d063.json" || status=1
python scripts/analyze_d064.py "$E/d064_campaign_20261011/ledger_k00-31.jsonl" > "$T/d064.json" || status=1
python scripts/analyze_d065.py "$E/d065_predictions/predictions.json" "$T/d065.json" "$E/d065_campaign_20261012/ledger_k00-31.jsonl" > /dev/null || status=1
python scripts/analyze_d066.py "$E/d066_campaign_20261027/ledger_k00-63.jsonl" "$E/d064_campaign_20261011/ledger_k00-31.jsonl" > "$T/d066.json" || status=1
python - "$T" "$E" <<'PY' || status=1
import json, sys
t, e = sys.argv[1:]
folders = {"d063": "d063_campaign_20261011", "d064": "d064_campaign_20261011",
           "d065": "d065_campaign_20261012", "d066": "d066_campaign_20261027"}
bad = 0
for name, folder in folders.items():
    same = json.load(open(f"{t}/{name}.json")) == json.load(open(f"{e}/{folder}/summary.json"))
    print(name, "identical" if same else "DIFFERENT")
    bad += not same
sys.exit(bad)
PY

step "Figure counts and input hashes"
python scripts/generate_report_figures.py --profile report --force > /dev/null 2>&1 && echo "report profile passed" || status=1

step "Test suite"
# These cannot pass reliably in this image, and they do not touch the saved results:
#   six need macOS caffeinate; one requires the frozen Python 3.14.7;
#   the rest depend on the container's process tree or clock precision.
S=tests/unit/test_simon_operational_supervisor.py
W=tests/unit/test_simon_stage2_watchdog.py
python -m pytest -q -p no:cacheprovider \
  --deselect tests/unit/test_protocol_manifest.py::test_prospective_execution_accepted_on_frozen_authorized \
  --deselect $S::test_integrity_exit_holds_without_retry \
  --deselect $S::test_os_kill_restarts_once_then_integrity_exit_holds \
  --deselect $S::test_completed_grid_invokes_analysis_without_another_runner \
  --deselect $S::test_baseline_corruption_holds_before_dispatch \
  --deselect $S::test_caffeinate_assertion_includes_full_power_and_user_flags \
  --deselect $S::test_guarantee_ordinary_scheduling_behavior \
  --deselect $W::test_accepts_loaded_then_all_eight_calls_and_advances_deadline \
  --deselect $W::test_deadline_stops_hangs_and_retains_only_completed_calls \
  --deselect $W::test_coordinator_loss_before_dispatch_writes_stop_receipt \
  2>&1 | tail -1
test "${PIPESTATUS[0]}" -eq 0 || status=1

echo
if [ "$status" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit "$status"
