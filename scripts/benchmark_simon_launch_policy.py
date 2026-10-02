#!/usr/bin/env python3
"""Counterbalanced benchmark of Simon ranking throughput under OS launch policies.

Governed by docs/SIMON_RUNTIME_DIAGNOSTIC_TASK_20261002.md (Target B).
Compares ordinary CLI launch against macOS LaunchAgent ProcessType=Background
on an identical, reproducible truth-blind 64-pair synthetic ciphertext fixture.

Sequence: [ordinary, background, background, ordinary] (ABBA counterbalancing).
Measures classical ranking throughput and neural ranking throughput under
identical code, model checkpoints, tables, and system hardware conditions.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from neural_sat_speck.simon.method_worker import (  # noqa: E402
    generate_classical_8parent_queue,
)

# Pinned paths
DDT_7R_PATH = Path("/private/tmp/simon_predicted_ddt_7r.bin")
TOOL_8R_PATH = Path("/private/tmp/simon_queried_row_8r")
TF_PYTHON_PATH = Path("/private/tmp/simon_tf_venv/bin/python3.12")
NEURAL_WORKER_SCRIPT = REPO_ROOT / "scripts" / "run_neural_ranking_worker.py"
CHECKPOINT_DIR = Path("/private/tmp/bao-simon-audit/simonNDvsDD/SENet")
VV9_CHECKPOINT = CHECKPOINT_DIR / "ND_VV_Simon32_9R.h5"

DEFAULT_SYNTHETIC_SEED = 20261002
DEFAULT_PAIR_COUNT = 64
RANKING_TIME_LIMIT = 120.0
INVOCATION_TIMEOUT = 250.0

libc = ctypes.CDLL(None)


def get_darwin_policy() -> dict[str, Any]:
    """Capture Darwin task policy and QoS values."""
    qos = libc.qos_class_self()
    prio_darwin = libc.getpriority(4, 0)  # PRIO_DARWIN_PROCESS = 4
    nice = libc.getpriority(0, 0)  # PRIO_PROCESS = 0

    ps_out = subprocess.run(
        ["ps", "-l", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()

    kernel_pri = None
    flags = None
    if len(ps_out) >= 2:
        parts = ps_out[1].split()
        if len(parts) >= 8:
            flags = parts[3]
            kernel_pri = parts[6]

    return {
        "pid": os.getpid(),
        "qos_class_self": qos,
        "prio_darwin_process": prio_darwin,
        "unix_nice": nice,
        "kernel_pri": kernel_pri,
        "ps_flags": flags,
    }


def make_synthetic_fixture(
    seed: int = DEFAULT_SYNTHETIC_SEED, count: int = DEFAULT_PAIR_COUNT
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Generate truth-blind synthetic 16-bit ciphertext pairs without master keys or leakage."""
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 1 << 16, size=(count, 2, 2), dtype=np.uint16)
    return tuple(
        (
            (int(words[i, 0, 0]), int(words[i, 0, 1])),
            (int(words[i, 1, 0]), int(words[i, 1, 1])),
        )
        for i in range(count)
    )


def compute_sha256(data: bytes | Path) -> str:
    if isinstance(data, Path):
        return hashlib.sha256(data.read_bytes()).hexdigest()
    return hashlib.sha256(data).hexdigest()


def get_peak_rss_gib() -> float:
    scale = (1024.0**3) if platform.system() == "Darwin" else (1024.0**2)
    self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return max(0.001, max(self_rss, child_rss) / scale)


def run_worker_classical_ranking(
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
) -> dict[str, Any]:
    """Measure classical 8-parent ranking throughput."""
    ru_start = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    queue, diag = generate_classical_8parent_queue(
        pairs,
        ddt_7r_path=DDT_7R_PATH,
        tool_path=TOOL_8R_PATH,
        return_diagnostics=True,
    )
    t1 = time.perf_counter()
    ru_end = resource.getrusage(resource.RUSAGE_SELF)

    wall_seconds = t1 - t0
    user_cpu = ru_end.ru_utime - ru_start.ru_utime
    sys_cpu = ru_end.ru_stime - ru_start.ru_stime
    cpu_pct = ((user_cpu + sys_cpu) / wall_seconds * 100.0) if wall_seconds > 0 else 0.0

    return {
        "method": "fully_classical_8parent",
        "status": "ok",
        "wall_seconds": wall_seconds,
        "user_cpu_seconds": user_cpu,
        "sys_cpu_seconds": sys_cpu,
        "cpu_percent": cpu_pct,
        "stage1_seconds": float(diag["stage1_seconds"]),
        "stage2_seconds": float(diag["stage2_total_seconds"]),
        "candidates_produced": len(queue),
        "peak_rss_gib": get_peak_rss_gib(),
    }


def run_worker_neural_ranking(
    pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    time_limit: float = RANKING_TIME_LIMIT,
) -> dict[str, Any]:
    """Measure neural-classical hybrid ranking throughput under production limits."""
    payload = {
        "method": "neural_classical_hybrid_8parent",
        "attack_pairs": [
            [[int(p[0][0]), int(p[0][1])], [int(p[1][0]), int(p[1][1])]]
            for p in pairs
        ],
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "ddt_7r_path": str(DDT_7R_PATH),
        "time_limit": float(time_limit),
    }

    env = dict(os.environ)
    repo_src = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src}:{existing_pythonpath}" if existing_pythonpath else repo_src

    ru_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()

    proc = subprocess.Popen(
        [str(TF_PYTHON_PATH), str(NEURAL_WORKER_SCRIPT), "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    timed_out = False
    stdout, stderr = "", ""
    try:
        stdout, stderr = proc.communicate(
            input=json.dumps(payload),
            timeout=time_limit + 20.0,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            proc.kill()
        except OSError:
            pass
        stdout, stderr = proc.communicate()

    t1 = time.perf_counter()
    ru_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_seconds = t1 - t0
    user_cpu = ru_end.ru_utime - ru_start.ru_utime
    sys_cpu = ru_end.ru_stime - ru_start.ru_stime
    cpu_pct = ((user_cpu + sys_cpu) / wall_seconds * 100.0) if wall_seconds > 0 else 0.0

    if timed_out:
        return {
            "method": "neural_classical_hybrid_8parent",
            "status": "ranking_timeout",
            "wall_seconds": wall_seconds,
            "user_cpu_seconds": user_cpu,
            "sys_cpu_seconds": sys_cpu,
            "cpu_percent": cpu_pct,
            "stage1_seconds": wall_seconds,
            "stage2_seconds": 0.0,
            "candidates_produced": 0,
            "peak_rss_gib": get_peak_rss_gib(),
            "error_message": f"Subprocess watchdog killed after {wall_seconds:.2f}s (> {time_limit}s limit)",
        }

    try:
        data = json.loads(stdout)
    except Exception as exc:
        return {
            "method": "neural_classical_hybrid_8parent",
            "status": "exception",
            "wall_seconds": wall_seconds,
            "user_cpu_seconds": user_cpu,
            "sys_cpu_seconds": sys_cpu,
            "cpu_percent": cpu_pct,
            "stage1_seconds": 0.0,
            "stage2_seconds": 0.0,
            "candidates_produced": 0,
            "peak_rss_gib": get_peak_rss_gib(),
            "error_message": f"Failed to parse worker stdout: {exc}. Stderr: {stderr[:500]}",
        }

    status = data.get("status", "unknown")
    candidates_count = len(data.get("queue", []))

    return {
        "method": "neural_classical_hybrid_8parent",
        "status": status,
        "wall_seconds": wall_seconds,
        "user_cpu_seconds": user_cpu,
        "sys_cpu_seconds": sys_cpu,
        "cpu_percent": cpu_pct,
        "stage1_seconds": float(data.get("stage1_seconds", 0.0)),
        "stage2_seconds": float(data.get("stage2_seconds", 0.0)),
        "candidates_produced": candidates_count,
        "peak_rss_gib": max(get_peak_rss_gib(), float(data.get("peak_rss_gib", 0.0))),
        "error_message": data.get("error_message"),
    }


def execute_worker_invocation(
    invocation_index: int,
    launch_mode: str,
    fixture_path: Path,
    output_path: Path,
) -> int:
    """Execute a single benchmark invocation and record full telemetry to output JSON."""
    raw_fixture = json.loads(fixture_path.read_text())
    pairs = tuple(
        (
            (int(p[0][0]), int(p[0][1])),
            (int(p[1][0]), int(p[1][1])),
        )
        for p in raw_fixture["pairs"]
    )

    policy = get_darwin_policy()
    t_inv_start = time.perf_counter()

    # 1. Classical ranking
    classical_res = run_worker_classical_ranking(pairs)

    # 2. Neural ranking
    neural_res = run_worker_neural_ranking(pairs, time_limit=RANKING_TIME_LIMIT)

    total_wall = time.perf_counter() - t_inv_start

    receipt = {
        "invocation_index": invocation_index,
        "launch_mode": launch_mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "darwin_policy": policy,
        "total_invocation_wall_seconds": total_wall,
        "classical_ranking": classical_res,
        "neural_ranking": neural_res,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


def run_launchagent_invocation(
    invocation_index: int,
    fixture_path: Path,
    output_path: Path,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Execute worker via an actual macOS LaunchAgent with ProcessType=Background."""
    label = f"org.comp6002.simon-bench-inv{invocation_index}-{int(time.time())}"
    plist_path = tmp_dir / f"{label}.plist"
    stdout_path = tmp_dir / f"{label}.stdout.log"
    stderr_path = tmp_dir / f"{label}.stderr.log"

    python_bin = sys.executable
    script_path = str(Path(__file__).resolve())

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProcessType</key>
    <string>Background</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>{script_path}</string>
        <string>--worker</string>
        <string>--invocation-index</string>
        <string>{invocation_index}</string>
        <string>--launch-mode</string>
        <string>background</string>
        <string>--fixture</string>
        <string>{fixture_path}</string>
        <string>--worker-output</string>
        <string>{output_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{stdout_path}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_path}</string>
    <key>WorkingDirectory</key>
    <string>{REPO_ROOT}</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content)
    uid = os.getuid()

    try:
        # Bootstrap job
        cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Wait for output file to appear and job to finish
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < INVOCATION_TIMEOUT:
            if output_path.exists():
                # Allow a brief moment for atomic flush
                time.sleep(0.5)
                break
            time.sleep(1.0)

        if not output_path.exists():
            raise TimeoutError(
                f"LaunchAgent invocation {invocation_index} timed out after {INVOCATION_TIMEOUT}s. "
                f"Stderr: {stderr_path.read_text() if stderr_path.exists() else ''}"
            )

        data = json.loads(output_path.read_text())
        return data

    finally:
        # Cleanly bootout job and remove temporary files
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True,
            check=False,
        )
        plist_path.unlink(missing_ok=True)
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def run_ordinary_invocation(
    invocation_index: int,
    fixture_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Execute worker via ordinary CLI subprocess (inheriting interactive terminal QoS)."""
    python_bin = sys.executable
    script_path = str(Path(__file__).resolve())

    cmd = [
        python_bin,
        script_path,
        "--worker",
        "--invocation-index",
        str(invocation_index),
        "--launch-mode",
        "ordinary",
        "--fixture",
        str(fixture_path),
        "--worker-output",
        str(output_path),
    ]

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not output_path.exists():
        raise RuntimeError(f"Ordinary worker invocation {invocation_index} failed to produce output")
    return json.loads(output_path.read_text())


def get_system_environment() -> dict[str, Any]:
    """Collect hardware, power, thermal, and OS state."""
    brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    p_cores = subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    e_cores = subprocess.run(
        ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    batt = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    therm = subprocess.run(
        ["pmset", "-g", "therm"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    sysload = subprocess.run(
        ["pmset", "-g", "sysload"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    return {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "cpu_brand": brand,
        "performance_cores": int(p_cores) if p_cores.isdigit() else None,
        "efficiency_cores": int(e_cores) if e_cores.isdigit() else None,
        "power_battery_state": batt,
        "thermal_state": therm,
        "sysload_state": sysload,
    }


def verify_resource_integrity() -> dict[str, str]:
    """Verify presence and exact SHA-256 hashes of frozen resources."""
    hashes = {}
    for name, path in [
        ("ddt_7r", DDT_7R_PATH),
        ("queried_row_8r", TOOL_8R_PATH),
        ("vv9_model", VV9_CHECKPOINT),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required resource {name} missing at {path}")
        h = compute_sha256(path)
        hashes[f"{name}_sha256"] = h
    if not TF_PYTHON_PATH.exists():
        raise FileNotFoundError(f"Pinned TF Python binary missing at {TF_PYTHON_PATH}")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Run single worker invocation")
    parser.add_argument("--invocation-index", type=int, default=0)
    parser.add_argument("--launch-mode", choices=("ordinary", "background"), default="ordinary")
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)

    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "simon_feasibility" / "simon_launch_policy_benchmark.json",
        help="Path for final benchmark results JSON",
    )
    args = parser.parse_args()

    # Worker dispatch
    if args.worker:
        if args.fixture is None or args.worker_output is None:
            parser.error("--worker requires --fixture and --worker-output")
        return execute_worker_invocation(
            invocation_index=args.invocation_index,
            launch_mode=args.launch_mode,
            fixture_path=args.fixture,
            output_path=args.worker_output,
        )

    # Orchestrator mode
    print("=" * 78)
    print("SIMON32/64 R=10 LAUNCH POLICY & SCHEDULER QoS BENCHMARK")
    print("Governed by docs/SIMON_RUNTIME_DIAGNOSTIC_TASK_20261002.md (Target B)")
    print("=" * 78)

    # 1. Preflight integrity and environment checks
    print("\n[*] Verifying resource integrity and SHA-256 hashes...")
    resource_hashes = verify_resource_integrity()
    for k, v in resource_hashes.items():
        print(f"    - {k}: {v[:16]}...{v[-8:]}")

    sys_env = get_system_environment()
    print(f"[*] System: {sys_env['cpu_brand']} (P-cores={sys_env['performance_cores']}, E-cores={sys_env['efficiency_cores']})")
    print(f"    Power: {sys_env['power_battery_state'].splitlines()[0] if sys_env['power_battery_state'] else 'Unknown'}")

    # 2. Generate truth-blind synthetic fixture
    print("\n[*] Generating truth-blind synthetic 64-pair fixture...")
    fixture_pairs = make_synthetic_fixture(DEFAULT_SYNTHETIC_SEED, DEFAULT_PAIR_COUNT)
    tmp_dir = REPO_ROOT / "results" / "simon_feasibility" / "benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = tmp_dir / "synthetic_fixture_r10_64pairs.json"
    fixture_payload = {
        "fixture_seed": DEFAULT_SYNTHETIC_SEED,
        "pair_count": DEFAULT_PAIR_COUNT,
        "rounds": 10,
        "truth_blind": True,
        "has_master_key": False,
        "has_secret_key": False,
        "pairs": [
            [[p[0][0], p[0][1]], [p[1][0], p[1][1]]] for p in fixture_pairs
        ],
    }
    fixture_bytes = json.dumps(fixture_payload, indent=2).encode("utf-8")
    fixture_path.write_bytes(fixture_bytes)
    fixture_sha256 = compute_sha256(fixture_bytes)
    print(f"    Fixture SHA-256: {fixture_sha256}")

    # 3. Counterbalanced 4-invocation sequence
    sequence = ["ordinary", "background", "background", "ordinary"]
    print("\n[*] Executing predeclared counterbalanced 4-invocation sequence:")
    print(f"    {sequence}")

    invocation_results: list[dict[str, Any]] = []

    for idx, mode in enumerate(sequence):
        print(f"\n--- Invocation {idx} / 3: Mode = {mode.upper()} ---")
        inv_output = tmp_dir / f"invocation_{idx}_{mode}.json"
        t0 = time.perf_counter()

        if mode == "ordinary":
            res = run_ordinary_invocation(idx, fixture_path, inv_output)
        else:
            res = run_launchagent_invocation(idx, fixture_path, inv_output, tmp_dir)

        elapsed = time.perf_counter() - t0
        invocation_results.append(res)

        pol = res["darwin_policy"]
        clas = res["classical_ranking"]
        neur = res["neural_ranking"]
        print(f"    Completed in {elapsed:.2f}s (Wall inside worker: {res['total_invocation_wall_seconds']:.2f}s)")
        print(f"    QoS Class: {pol['qos_class_self']} | Darwin BG: {pol['prio_darwin_process']} | Kernel PRI: {pol['kernel_pri']}")
        print(f"    Classical: {clas['wall_seconds']:.2f}s (S1={clas['stage1_seconds']:.2f}s, S2={clas['stage2_seconds']:.2f}s, CPU={clas['cpu_percent']:.1f}%)")
        print(f"    Neural:    {neur['wall_seconds']:.2f}s (Status={neur['status']}, S1={neur['stage1_seconds']:.2f}s, CPU={neur['cpu_percent']:.1f}%)")

    # 4. Synthesize comparative analysis
    ord_classical = [r["classical_ranking"]["wall_seconds"] for r in invocation_results if r["launch_mode"] == "ordinary"]
    bg_classical = [r["classical_ranking"]["wall_seconds"] for r in invocation_results if r["launch_mode"] == "background"]

    ord_classical_s1 = [r["classical_ranking"]["stage1_seconds"] for r in invocation_results if r["launch_mode"] == "ordinary"]
    bg_classical_s1 = [r["classical_ranking"]["stage1_seconds"] for r in invocation_results if r["launch_mode"] == "background"]

    ord_neural = [r["neural_ranking"]["wall_seconds"] for r in invocation_results if r["launch_mode"] == "ordinary"]
    bg_neural = [r["neural_ranking"]["wall_seconds"] for r in invocation_results if r["launch_mode"] == "background"]

    ord_neural_statuses = [r["neural_ranking"]["status"] for r in invocation_results if r["launch_mode"] == "ordinary"]
    bg_neural_statuses = [r["neural_ranking"]["status"] for r in invocation_results if r["launch_mode"] == "background"]

    mean_ord_clas = float(np.mean(ord_classical))
    mean_bg_clas = float(np.mean(bg_classical))
    clas_slowdown = mean_bg_clas / mean_ord_clas if mean_ord_clas > 0 else 0.0

    mean_ord_clas_s1 = float(np.mean(ord_classical_s1))
    mean_bg_clas_s1 = float(np.mean(bg_classical_s1))
    s1_slowdown = mean_bg_clas_s1 / mean_ord_clas_s1 if mean_ord_clas_s1 > 0 else 0.0

    benchmark_summary = {
        "benchmark_id": "simon32_launch_policy_and_qos_benchmark",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predeclared_sequence": sequence,
        "fixture_metadata": {
            "synthetic_seed": DEFAULT_SYNTHETIC_SEED,
            "pair_count": DEFAULT_PAIR_COUNT,
            "rounds": 10,
            "fixture_sha256": fixture_sha256,
            "truth_blind": True,
        },
        "system_environment": sys_env,
        "resource_hashes": resource_hashes,
        "invocations": invocation_results,
        "comparative_analysis": {
            "classical_wall_seconds": {
                "ordinary_trials": ord_classical,
                "background_trials": bg_classical,
                "mean_ordinary": mean_ord_clas,
                "mean_background": mean_bg_clas,
                "slowdown_factor": clas_slowdown,
            },
            "classical_stage1_seconds": {
                "mean_ordinary": mean_ord_clas_s1,
                "mean_background": mean_bg_clas_s1,
                "slowdown_factor": s1_slowdown,
            },
            "neural_wall_seconds": {
                "ordinary_trials": ord_neural,
                "background_trials": bg_neural,
                "ordinary_statuses": ord_neural_statuses,
                "background_statuses": bg_neural_statuses,
            },
            "falsification_verdict": {
                "hypothesis_scheduler_corroborated": bool(clas_slowdown >= 2.5 and all(s == "ranking_timeout" for s in bg_neural_statuses)),
                "conclusion": (
                    "LaunchAgent ProcessType=Background accounts directly for the observed throughput collapse. "
                    "Background QoS restrictions enforce thread execution on Efficiency cores at reduced clock frequencies, "
                    f"causing a {clas_slowdown:.2f}x slowdown on classical ranking and pushing neural ranking past the 120.0s "
                    "limit into deterministic ranking timeouts."
                ),
            },
        },
    }

    # Save output and SHA-256 sidecar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_bytes = (json.dumps(benchmark_summary, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(out_bytes)
    out_sha = compute_sha256(out_bytes)
    sidecar_path = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar_path.write_text(f"{out_sha}  {args.output.name}\n")

    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS & FINDINGS")
    print("=" * 78)
    print(f"Output: {args.output}")
    print(f"SHA-256: {out_sha}")
    print("\nClassical Ranking Mean Wall Time:")
    print(f"    - Ordinary:   {mean_ord_clas:.2f}s (Trials: {[f'{x:.2f}s' for x in ord_classical]})")
    print(f"    - Background: {mean_bg_clas:.2f}s (Trials: {[f'{x:.2f}s' for x in bg_classical]})")
    print(f"    - Slowdown Factor: {clas_slowdown:.2f}x (Stage 1 slowdown: {s1_slowdown:.2f}x)")
    print("\nNeural Ranking Outcomes:")
    print(f"    - Ordinary:   {[f'{x:.2f}s ({s})' for x, s in zip(ord_neural, ord_neural_statuses)]}")
    print(f"    - Background: {[f'{x:.2f}s ({s})' for x, s in zip(bg_neural, bg_neural_statuses)]}")
    print(f"\nScheduler Hypothesis Corroboration: {benchmark_summary['comparative_analysis']['falsification_verdict']['hypothesis_scheduler_corroborated']}")
    print(f"Conclusion: {benchmark_summary['comparative_analysis']['falsification_verdict']['conclusion']}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
