"""Prepare, freeze, then measure the D-058 single-thread encryption reference."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.analysis.encryption_calibration import (  # noqa: E402
    PROFILES,
    sha256,
    summarize_samples,
    validate_manifest,
    verify_native,
    write_evidence,
)

MANIFEST = ROOT / "docs/encryption_cost_calibration_manifest_20261009.json"
PINNED = [
    "scripts/calibrate_reference_encryptions.py",
    "src/neural_sat_speck/analysis/encryption_calibration.py",
    "docs/ENCRYPTION_COST_CALIBRATION_PROTOCOL_20261009.md",
    "docs/encryption_cost_calibration_manifest_20261009.json",
    "tools/reference_encryption.c",
    "tools/reference_encryption.h",
    "tools/encryption_calibration.c",
    *[
        f"src/neural_sat_speck/{c}/{f}.py"
        for c in ("speck", "simon")
        for f in ("cipher", "key_schedule", "primitives")
    ],
    "tests/fixtures/simon_reference_oracle.py",
    "tests/fixtures/speck32_64_vectors.py",
    "tests/fixtures/simon32_64_vectors.py",
]


def observe(argv: list[str], deadline: float | None = None) -> dict:
    remaining = 2.0 if deadline is None else min(2.0, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("observation exceeded timing campaign ceiling")
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=remaining)
        return {
            "argv": argv,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "unavailable": str(error)}


def observations(deadline: float | None = None) -> dict:
    return {
        "utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "hardware": observe(
            [
                "sysctl",
                "hw.model",
                "hw.machine",
                "hw.physicalcpu",
                "hw.logicalcpu",
                "hw.memsize",
                "machdep.cpu.brand_string",
            ],
            deadline,
        ),
        "power": observe(["pmset", "-g", "batt"], deadline),
        "core_groups": observe(
            ["sysctl", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"],
            deadline,
        ),
        "power_configuration": observe(["pmset", "-g", "custom"], deadline),
        "thermal": observe(["pmset", "-g", "therm"], deadline),
        "load_average": list(os.getloadavg()),
    }


def no_attack_processes(deadline: float | None = None) -> dict:
    receipt = observe(["ps", "-axo", "pid=,command="], deadline)
    if receipt.get("returncode") != 0:
        raise RuntimeError("cannot check concurrent attack processes")
    scripts = {
        p.name
        for p in (ROOT / "scripts").glob("*.py")
        if p.name.startswith(("run_", "evaluate_", "benchmark_"))
    }
    workers = {
        "neural_sat_speck.simon.method_worker",
        "neural_sat_speck.followups.ranking_depth_worker",
        "neural_sat_speck.followups.depth_worker",
    }
    matches = []
    for line in receipt["stdout"].splitlines():
        pid, command = line.strip().split(maxsplit=1)
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if (
                Path(token).name in scripts | {"cryptominisat5", "cadical", "kissat"}
                or token in workers
            ):
                matches.append({"pid": pid, "signature": token})
    if matches:
        raise RuntimeError(f"concurrent attack processes: {matches}")
    return {"checked_at": datetime.now(UTC).isoformat(), "matching_processes": []}


def check_imported_paths() -> None:
    for name in PINNED:
        if name.startswith("src/") and name.endswith(".py"):
            module_name = name[4:-3].replace("/", ".")
        elif name.startswith("tests/fixtures/"):
            module_name = name[:-3].replace("/", ".")
        else:
            continue
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError(
                f"executed module differs from pinned source: {module_name}"
            )


def check_identity(identity: dict) -> None:
    check_imported_paths()
    for name, expected in identity["pinned_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"pinned source changed: {name}")
    for name, expected in identity["build_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"compiled identity changed: {name}")
    if sha256(Path(identity["compiler_path"])) != identity["compiler_sha256"]:
        raise ValueError("compiler identity changed")


def prepare(output: Path, build: Path) -> None:
    if sys.platform != "darwin":
        raise ValueError(
            "this frozen profile requires Darwin; register a new host profile elsewhere"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("preparation output already exists; do not overwrite evidence")
    if build.exists() and any(build.iterdir()):
        raise ValueError("build directory already contains artifacts; do not overwrite")
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    compiler_name = shutil.which(manifest["compiler"])
    if compiler_name is None:
        raise ValueError("declared clang compiler unavailable")
    compiler = Path(compiler_name).resolve()
    check_imported_paths()
    pinned = {name: sha256(ROOT / name) for name in PINNED}
    compiler_sha256 = sha256(compiler)
    compiler_version = observe([str(compiler), "--version"])
    if compiler_version.get("returncode") != 0:
        raise ValueError("compiler version unavailable")
    commands, files = [], []
    for name in ("reference_encryption", "encryption_calibration"):
        target = build / (name + ".o")
        command = [
            str(compiler),
            *manifest["compiler_flags"],
            "-c",
            str(ROOT / "tools" / (name + ".c")),
            "-o",
            str(target),
        ]
        subprocess.run(command, check=True, timeout=60)
        commands.append(command)
        files.append(target)
    binary = build / "reference_encryption_calibration"
    command = [str(compiler), "-fno-lto", *map(str, files), "-o", str(binary)]
    subprocess.run(command, check=True, timeout=60)
    commands.append(command)
    files.append(binary)
    driver_symbols = observe(["nm", "-u", str(build / "encryption_calibration.o")])
    if (
        driver_symbols.get("returncode") != 0
        or "reference_batch" not in driver_symbols["stdout"]
    ):
        raise ValueError("driver lacks opaque external reference_batch call")
    policy = json.loads(
        subprocess.run(
            [str(binary), "--policy-check"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout
    )
    if policy != {
        "qos_class": 25 if sys.platform == "darwin" else 0,
        "nice": 0,
        "threads": 1,
    }:
        raise ValueError("declared effective CPU policy unavailable")
    identity = {
        "schema": "encryption-calibration-execution-v1",
        "status": "PREPARED_FROZEN",
        "prepared_at": datetime.now(UTC).isoformat(),
        "manifest": manifest,
        "pinned_sha256": pinned,
        "build_sha256": {str(p): sha256(p) for p in files},
        "compiler_path": str(compiler),
        "compiler_sha256": compiler_sha256,
        "compiler_version": compiler_version,
        "compile_commands": commands,
        "binary": str(binary),
        "effective_policy_preflight": policy,
        "opaque_batch_call_receipt": driver_symbols,
        "git_commit": observe(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "python": {"path": sys.executable, "version": sys.version},
        "host": observations(),
        "no_attack_check": no_attack_processes(),
    }
    correctness = verify_native(binary, progress=lambda s: print(s, flush=True))
    write_evidence(output / "correctness.json", correctness)
    identity["correctness_sha256"] = sha256(output / "correctness.json")
    check_identity(identity)
    write_evidence(output / "execution_identity.json", identity)
    print(
        "Correctness gates passed; preparation identity frozen. No timings collected.",
        flush=True,
    )


def measure(output: Path) -> None:
    if sys.platform != "darwin":
        raise ValueError("this frozen profile requires Darwin")
    identity_path = output / "execution_identity.json"
    identity = json.loads(identity_path.read_text())
    if (
        identity.get("status") != "PREPARED_FROZEN"
        or identity.get("schema") != "encryption-calibration-execution-v1"
    ):
        raise ValueError("unprepared calibration identity")
    if identity_path.with_name(identity_path.name + ".sha256").read_text().split()[
        0
    ] != sha256(identity_path):
        raise ValueError("execution identity sidecar mismatch")
    manifest = identity["manifest"]
    validate_manifest(manifest)
    check_identity(identity)
    correctness_path = output / "correctness.json"
    if sha256(correctness_path) != identity["correctness_sha256"]:
        raise ValueError("correctness receipt changed")
    correctness = json.loads(correctness_path.read_text())
    if (
        correctness["status"] != "PASS"
        or correctness["fixture_blocks_per_profile"] != 65536
    ):
        raise ValueError("incomplete correctness gate")
    raw_path = output / "samples.json"
    if any(
        (output / name).exists()
        for name in ("samples.json", "summary.json", "summary.csv")
    ):
        raise ValueError("existing timing evidence; no overwrite or selective resume")
    rows = []
    started = time.monotonic()
    deadline = started + manifest["timing_campaign_ceiling_seconds"]
    raw = {
        "status": "RUNNING",
        "identity_sha256": sha256(identity_path),
        "started_at": datetime.now(UTC).isoformat(),
        "no_attack_checks": [],
        "repetition_observations": [],
        "samples": rows,
    }
    write_evidence(raw_path, raw)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    try:
        raw["observations_before"] = observations(deadline)
        for rep in range(manifest["repetitions"]):
            raw["repetition_observations"].append(observations(deadline))
            order = PROFILES[rep:] + PROFILES[:rep]
            for cipher, rounds, scope in order:
                raw["no_attack_checks"].append(no_attack_processes(deadline))
                check_identity(identity)
                remaining = manifest["timing_campaign_ceiling_seconds"] - (
                    time.monotonic() - started
                )
                if remaining <= 0:
                    raise TimeoutError("timing campaign ceiling exhausted")
                command = [
                    identity["binary"],
                    "--measure",
                    cipher,
                    str(rounds),
                    scope,
                    str(manifest["batch_blocks"]),
                    str(manifest["warmup_seconds"]),
                    str(manifest["minimum_measurement_seconds"]),
                ]
                raw["active_sample"] = {
                    "repetition": rep,
                    "ordinal": len(rows) + 1,
                    "cipher": cipher,
                    "rounds": rounds,
                    "scope": scope,
                    "command": command,
                }
                write_evidence(raw_path, raw)
                p = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=min(remaining, manifest["invocation_timeout_seconds"]),
                    env=environment,
                )
                row = json.loads(p.stdout)
                row.update(
                    {
                        "repetition": rep,
                        "ordinal": len(rows) + 1,
                        "recorded_at": datetime.now(UTC).isoformat(),
                        "command": command,
                        "identity_sha256": raw["identity_sha256"],
                    }
                )
                rows.append(row)
                write_evidence(raw_path, raw)
                if (
                    row["threads"] != 1
                    or row["nice"] != 0
                    or (sys.platform == "darwin" and row["qos_class"] != 25)
                ):
                    raise ValueError("effective sample CPU profile drift")
                if (
                    row["checksum"]
                    != correctness["expected_digests"][f"{cipher}:r{rounds}:{scope}"]
                ):
                    raise ValueError("timed checksum failure")
                print(
                    f"Measured {len(rows)}/56: {cipher} R{rounds} {scope}", flush=True
                )
        summary = summarize_samples(
            rows,
            manifest,
            correctness["expected_digests"],
            darwin=sys.platform == "darwin",
            identity_sha256=raw["identity_sha256"],
        )
        check_identity(identity)
        raw["no_attack_checks"].append(no_attack_processes(deadline))
        raw["observations_after"] = observations(deadline)
        raw.pop("active_sample", None)
        raw.update(
            status="COMPLETE",
            completed_at=datetime.now(UTC).isoformat(),
            total_campaign_seconds=time.monotonic() - started,
        )
        if raw["total_campaign_seconds"] > manifest["timing_campaign_ceiling_seconds"]:
            raise TimeoutError("whole campaign exceeded ceiling")
        write_evidence(raw_path, raw)
        write_evidence(
            output / "summary.json",
            {
                "evidence_class": "ENGINEERING",
                "raw_sha256": sha256(raw_path),
                "profiles": summary,
                "interpretation": "local elapsed-time reference; not algorithmic complexity",
            },
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
        csv_path = output / "summary.csv"
        csv_path.write_text(stream.getvalue())
        csv_path.with_name(csv_path.name + ".sha256").write_text(
            f"{sha256(csv_path)}  {csv_path.name}\n"
        )
    except BaseException as error:
        if isinstance(
            error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)
        ):

            def decode(value):
                return (
                    value.decode(errors="replace")
                    if isinstance(value, bytes)
                    else value
                )

            raw["failed_invocation"] = {
                "command": error.cmd,
                "stdout": decode(error.stdout),
                "stderr": decode(error.stderr),
            }
        raw.update(
            status="FAILED",
            failure_type=type(error).__name__,
            failure=str(error),
            stopped_at=datetime.now(UTC).isoformat(),
            total_campaign_seconds=time.monotonic() - started,
        )
        write_evidence(raw_path, raw)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "runtime/encryption_calibration_20261009",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--measure-frozen", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output.resolve(), args.build_dir.resolve())
    else:
        measure(args.output.resolve())


if __name__ == "__main__":
    main()
