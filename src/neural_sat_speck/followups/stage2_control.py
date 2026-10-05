"""Contracts and post-grid descriptive analysis for the SIMON Stage 2 control.

This module deliberately has no cipher, SAT, ranking, TensorFlow, or dataset
imports.  The worker boundary uses only the strict public projection below;
the analyzer performs the truth join only after it has validated closure of the
entire full worker grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROFILE: dict[str, Any] = {
    "schema": "simon_matched_stage2_control_v3",
    "role": "opened_demo_rank_only",
    "seed": 999999,
    "cipher": "simon32_64",
    "rounds": 10,
    "key_indices": list(range(8)),
    "direct_pair_count": 64,
    "verification_block_count": 2,
    "stage1_scorer": "vv9",
    "stage2_scorers": ["vv8", "ddt7"],
    "parent_widths": [1, 8],
    "parent_count": 8,
    "domain_size": 65536,
    "screen_pairs": 16,
    "full_pairs": 64,
    "rescore_top": 512,
    "children_per_parent": 128,
    "candidate_batch_size": 1024,
    "neural_inference_batch_size": 16384,
    "worker_count": 24,
    "ranking_call_count": 136,
    "startup_limit_seconds": 120.0,
    "call_limit_seconds": 180.0,
    "campaign_limit_seconds": 28800.0,
    "process_group_rss_limit_bytes": 2684354560,
    "rss_sample_interval_seconds": 10.0,
    "tensorflow_device": "CPU",
    "tensorflow_intra_op_threads": 8,
    "tensorflow_inter_op_threads": 1,
    "numeric_threads": 1,
    "numeric_thread_env": {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    },
    "power_policy": "AC_or_battery_author_instruction_20261005",
    "retries_allowed": False,
    "resume_allowed": False,
}

_EXPECTED_ASSET_HASHES = {
    "vv9": "2363b843592529fe9409dc25dbc0f50c20565973a4a23e54e5478a03113a7070",
    "vv8": "f4e780e484f998ac7e3f05ff78adf47e9dcc1a7a405c9aec92bb06a2761a65fc",
    "ddt": "52b3f3d4df3e20d64942f95ea2837f55b6d7591f1989e92dffd6212938e25c46",
}
_SOURCE_SCRIPTS = {
    "scripts/prepare_simon_matched_stage2_control.py",
    "scripts/run_simon_matched_stage2_worker.py",
    "scripts/evaluate_simon_matched_stage2_control.py",
    "scripts/analyze_simon_matched_stage2_control.py",
    "scripts/audit_simon_matched_stage2_pilot.py",
}

_JOB_FIELDS = frozenset({"job_id", "key_index", "operation", "scorer"})
_PUBLIC_CASE_FIELDS = frozenset(
    {"cipher", "rounds", "key_index", "instance_id", "ciphertext_pairs"}
)
_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "job",
        "public_case",
        "parents",
        "invocation",
        "directory",
        "source_root",
        "source_commit",
        "source_files",
        "assets",
        "runtime",
        "profile",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "invocation",
        "job_id",
        "ordinal",
        "scorer",
        "parent",
        "ordered_words",
        "component_times_seconds",
        "call_started_monotonic",
        "call_completed_monotonic",
        "phase_wall_seconds",
    }
)
_UNAVAILABLE_OUTCOMES = frozenset(
    {
        "rank_timeout",
        "load_timeout",
        "rss_stop",
        "unavailable",
        "dependency_unavailable",
        "environment_rejected",
        "environment_alarm",
        "integrity_alarm",
        "campaign_timeout",
        "worker_exception",
        "interrupted",
        "not_started",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Value is not canonical JSON data") from error


def ordered_hash(value: object) -> str:
    """Hash a value using compact, sorted-key UTF-8 JSON serialization."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def streaming_sha256(path: str | os.PathLike[str]) -> str:
    """Hash a file with bounded memory, including the large memory-mapped DDT."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def asset_identity(path: str | os.PathLike[str]) -> dict[str, int]:
    """Return the pinned stat identity for an asset, following a path symlink."""
    stat = Path(path).stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def fixed_jobs() -> list[dict[str, Any]]:
    """Return the fixed 24-worker grid; the first three jobs are the key-0 pilot."""
    jobs: list[dict[str, Any]] = []
    for key_index in range(8):
        jobs.append(
            {
                "job_id": f"simon_r10_k{key_index:02d}_stage1_vv9",
                "key_index": key_index,
                "operation": "stage1",
                "scorer": "vv9",
            }
        )
        scorers = ("vv8", "ddt7") if key_index % 2 == 0 else ("ddt7", "vv8")
        for scorer in scorers:
            jobs.append(
                {
                    "job_id": f"simon_r10_k{key_index:02d}_stage2_{scorer}",
                    "key_index": key_index,
                    "operation": "stage2",
                    "scorer": scorer,
                }
            )
    return jobs


def _validate_job(job: object) -> dict[str, Any]:
    if not isinstance(job, Mapping) or set(job) != _JOB_FIELDS:
        raise ValueError(
            "Job must contain exactly job_id, key_index, operation, scorer"
        )
    key_index = job["key_index"]
    if type(key_index) is not int or not 0 <= key_index < 8:
        raise ValueError("Invalid key_index")
    if not isinstance(job["job_id"], str) or not job["job_id"]:
        raise ValueError("Invalid job_id")
    if job["operation"] not in {"stage1", "stage2"}:
        raise ValueError("Invalid operation")
    expected_scorer = "vv9" if job["operation"] == "stage1" else None
    if expected_scorer is not None and job["scorer"] != expected_scorer:
        raise ValueError("Stage 1 must use VV9")
    if job["operation"] == "stage2" and job["scorer"] not in {"vv8", "ddt7"}:
        raise ValueError("Stage 2 scorer must be VV8 or DDT7")
    if dict(job) not in fixed_jobs():
        raise ValueError("Job is not part of the frozen grid")
    return dict(job)


def _word(value: object, name: str = "word") -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must be a genuine unsigned 16-bit integer")
    return value


def _block(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two words")
    return (_word(value[0], name), _word(value[1], name))


def validate_public_case(value: object) -> None:
    """Validate the exact ciphertext-only ranking input; raises on any extra field."""
    if not isinstance(value, Mapping) or set(value) != _PUBLIC_CASE_FIELDS:
        raise ValueError("Public case must contain exactly the five allowed fields")
    if value["cipher"] != "simon32_64":
        raise ValueError("Public case cipher must be SIMON32/64")
    if type(value["rounds"]) is not int or value["rounds"] != 10:
        raise ValueError("Public case rounds must be exactly 10")
    if type(value["key_index"]) is not int or not 0 <= value["key_index"] < 8:
        raise ValueError("Public case key_index must be in 0..7")
    if not isinstance(value["instance_id"], str) or not value["instance_id"].strip():
        raise ValueError("Public case instance_id must be non-empty")
    pairs = value["ciphertext_pairs"]
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 64:
        raise ValueError("Public case must contain exactly 64 ciphertext pairs")
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"ciphertext_pairs[{pair_index}] must contain two blocks")
        _block(pair[0], f"ciphertext_pairs[{pair_index}][0]")
        _block(pair[1], f"ciphertext_pairs[{pair_index}][1]")


def public_case(dataset: object, job: object) -> dict[str, Any]:
    """Project an opened dataset row to ciphertext-only ranking input.

    Only the dataset role metadata, ordered key index, instance identifier, and
    R10 attack ciphertexts are read. No key, plaintext, round-key, label, or
    fresh-verification value is copied into the returned object.
    """
    checked_job = _validate_job(job)
    if not isinstance(dataset, Mapping):
        raise ValueError("Dataset must be a mapping")
    if (
        type(dataset.get("seed")) is not int
        or dataset.get("seed") != 999999
        or dataset.get("split_role") != "demo"
    ):
        raise ValueError("Only the registered opened demo cohort is eligible")
    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("key_count_per_cipher"),
        metadata.get("max_differential_pairs"),
        metadata.get("verification_blocks_count"),
    ) != (8, 64, 2):
        raise ValueError("Unexpected demo cohort dimensions")
    ciphers = dataset.get("ciphers")
    if not isinstance(ciphers, Mapping) or "simon32_64" not in ciphers:
        raise ValueError("Missing SIMON demo cohort")
    simon = ciphers["simon32_64"]
    if not isinstance(simon, Mapping):
        raise ValueError("Malformed SIMON demo cohort")
    instances = simon.get("instances")
    if not isinstance(instances, (list, tuple)) or len(instances) != 8:
        raise ValueError("SIMON demo cohort must have exactly eight instances")
    indices = [
        row.get("instance_index") for row in instances if isinstance(row, Mapping)
    ]
    if len(indices) != 8 or any(type(index) is not int for index in indices):
        raise ValueError("Malformed or missing SIMON key indices")
    if indices != list(range(8)):
        raise ValueError("SIMON key indices must be unique and ordered 0..7")
    row = instances[checked_job["key_index"]]
    if row.get("instance_id") != f"simon32_64_demo_key_{checked_job['key_index']}":
        raise ValueError(
            "SIMON demo instance identifier differs from its fixed ordered key"
        )
    rounds_data = row.get("rounds_data")
    if not isinstance(rounds_data, Mapping) or "10" not in rounds_data:
        raise ValueError("Missing R10 ciphertext data")
    round_data = rounds_data["10"]
    if not isinstance(round_data, Mapping):
        raise ValueError("Malformed R10 data")
    attack_ciphertexts = round_data.get("attack_ciphertexts")
    if (
        not isinstance(attack_ciphertexts, (list, tuple))
        or len(attack_ciphertexts) != 64
    ):
        raise ValueError("R10 cohort must contain exactly 64 attack ciphertext pairs")
    attack_plaintexts = row.get("attack_plaintexts")
    verification_plaintexts = row.get("verification_plaintexts")
    verification_ciphertexts = round_data.get("verification_ciphertexts")
    if not isinstance(attack_plaintexts, (list, tuple)) or len(attack_plaintexts) != 64:
        raise ValueError("R10 cohort must contain exactly 64 attack plaintext pairs")
    if (
        not isinstance(verification_plaintexts, (list, tuple))
        or len(verification_plaintexts) != 2
    ):
        raise ValueError(
            "Demo cohort must contain exactly two fresh verification blocks"
        )
    if (
        not isinstance(verification_ciphertexts, (list, tuple))
        or len(verification_ciphertexts) != 2
    ):
        raise ValueError(
            "Demo cohort must contain exactly two fresh verification ciphertexts"
        )
    for label, collection in (
        ("verification plaintext", verification_plaintexts),
        ("verification ciphertext", verification_ciphertexts),
    ):
        for index, block in enumerate(collection):
            _block(block, f"{label}[{index}]")
    for label, collection in (
        ("attack plaintext", attack_plaintexts),
        ("attack ciphertext", attack_ciphertexts),
    ):
        for index, pair in enumerate(collection):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{label} pair {index} must contain two blocks")
            _block(pair[0], f"{label} pair {index} block 0")
            _block(pair[1], f"{label} pair {index} block 1")
    instance_id = row.get("instance_id")
    result = {
        "cipher": "simon32_64",
        "rounds": 10,
        "key_index": checked_job["key_index"],
        "instance_id": instance_id,
        "ciphertext_pairs": [
            [
                list(_block(pair[0], "ciphertext block")),
                list(_block(pair[1], "ciphertext block")),
            ]
            if isinstance(pair, (list, tuple)) and len(pair) == 2
            else _raise_bad_pair(index)
            for index, pair in enumerate(attack_ciphertexts)
        ],
    }
    validate_public_case(result)
    return result


def _raise_bad_pair(index: int) -> Any:
    raise ValueError(f"R10 attack ciphertext pair {index} must contain two blocks")


def validate_order(order: object) -> tuple[int, ...]:
    """Validate one fixed-fidelity rank result containing 512 unique words."""
    if not isinstance(order, (list, tuple)) or len(order) != 512:
        raise ValueError("A rank result must contain exactly 512 ordered words")
    result = tuple(
        _word(value, f"ordered_words[{index}]") for index, value in enumerate(order)
    )
    if len(set(result)) != 512:
        raise ValueError("A rank result contains duplicate candidate words")
    return result


def assemble_queue(
    parents: object,
    results: object,
    width: int,
) -> tuple[tuple[int, int], ...]:
    """Build one width-1/8 chronological ``(k8, k9)`` candidate queue.

    ``results`` maps zero-based parent ordinals to each parent's complete
    512-word rescore order. The width-1 queue uses only ordinal zero. Rank-sum
    ordering uses zero-based parent/child indices and the frozen tie tuple
    ``(r1+r2, r1, r2, k9, k8)``.
    """
    if type(width) is not int or width not in (1, 8):
        raise ValueError("Queue width must be exactly 1 or 8")
    if not isinstance(parents, (list, tuple)) or len(parents) != 8:
        raise ValueError("Exactly eight ordered parent words are required")
    checked_parents = tuple(_word(value, "parent") for value in parents)
    if len(set(checked_parents)) != 8:
        raise ValueError("Duplicate Stage 1 parents are invalid")
    if not isinstance(results, Mapping):
        raise ValueError("Stage 2 results must map parent ordinal to rank order")
    ordinals = set(results)
    if any(type(value) is not int for value in ordinals) or ordinals != set(
        range(width)
    ):
        raise ValueError("Stage 2 result ordinals must exactly match selected width")

    ranked_paths: list[tuple[int, int, int, int, int]] = []
    for parent_rank in range(width):
        children = validate_order(results[parent_rank])[:128]
        for child_rank, child in enumerate(children):
            parent = checked_parents[parent_rank]
            ranked_paths.append(
                (parent_rank + child_rank, parent_rank, child_rank, parent, child)
            )
    ranked_paths.sort()
    queue = tuple((child, parent) for _, _, _, parent, child in ranked_paths)
    expected_size = width * 128
    if len(queue) != expected_size or len(set(queue)) != expected_size:
        raise ValueError("Queue contains duplicate or missing chronological key tuples")
    return queue


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _validate_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _file_path(record: object, root: Path, name: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"File record {name} must contain exactly path and sha256")
    filename, digest = record["path"], record["sha256"]
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Invalid path for {name}")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"Invalid SHA-256 for {name}")
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    # Keep interpreter venv symlinks lexical: resolving bin/python can change
    # sys.prefix and therefore the environment actually used at launch.
    path = path.absolute()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Pinned file is not a regular file: {name}")
    if streaming_sha256(path) != digest:
        raise ValueError(f"Pinned file SHA-256 mismatch: {name}")
    return path


def _validate_assets(assets: object, root: Path) -> dict[str, Path]:
    expected = {"vv9", "vv8", "ddt"}
    if not isinstance(assets, Mapping) or set(assets) != expected:
        raise ValueError("Manifest assets must pin exactly VV9, VV8, and DDT7")
    result: dict[str, Path] = {}
    for name in sorted(expected):
        record = assets[name]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "identity",
        }:
            raise ValueError(f"Malformed asset record {name}")
        filename = record["path"]
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Invalid asset path: {name}")
        expected_hash = record["sha256"]
        if (
            not isinstance(expected_hash, str)
            or _SHA256_RE.fullmatch(expected_hash) is None
        ):
            raise ValueError(f"Invalid asset SHA-256: {name}")
        expected_identity = record["identity"]
        if not isinstance(expected_identity, Mapping) or set(expected_identity) != {
            "device",
            "inode",
            "size",
            "mtime_ns",
        }:
            raise ValueError(f"Malformed asset stat identity: {name}")
        if any(
            type(expected_identity[field]) is not int or expected_identity[field] < 0
            for field in expected_identity
        ):
            raise ValueError(f"Invalid asset stat identity values: {name}")
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        path = path.absolute()
        if not path.exists() or not path.is_file():
            raise ValueError(f"Asset is not a regular file: {name}")
        if expected_hash != _EXPECTED_ASSET_HASHES[name]:
            raise ValueError(
                f"Asset differs from the frozen published identity: {name}"
            )
        before = asset_identity(path)
        if before != dict(expected_identity):
            raise ValueError(f"Asset identity changed before hashing: {name}")
        if streaming_sha256(path) != expected_hash:
            raise ValueError(f"Asset SHA-256 mismatch: {name}")
        after = asset_identity(path)
        if after != before:
            raise ValueError(f"Asset identity changed during hashing: {name}")
        result[name] = path
    return result


def _validate_runtime(runtime: object, files: Mapping[str, Path]) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != {"python", "tensorflow"}:
        raise ValueError("Runtime pin must contain python and tensorflow records")
    python, tensorflow = runtime["python"], runtime["tensorflow"]
    required_smoke_fields = {
        "executable",
        "python_version",
        "packages",
        "origins",
        "devices",
        "threads",
        "import_boundary",
    }
    if not isinstance(python, Mapping) or set(python) != required_smoke_fields:
        raise ValueError("Malformed base Python runtime pin")
    if not isinstance(tensorflow, Mapping) or set(tensorflow) != required_smoke_fields:
        raise ValueError("Malformed TensorFlow runtime pin")
    version_re = re.compile(r"3\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)\Z")
    if (
        not isinstance(python["python_version"], str)
        or version_re.fullmatch(python["python_version"]) is None
    ):
        raise ValueError("The base runtime must pin a concrete Python 3 version")
    if not isinstance(tensorflow["python_version"], str) or not tensorflow[
        "python_version"
    ].startswith("3.12."):
        raise ValueError("The TensorFlow runtime must use Python 3.12")
    tf_packages = tensorflow["packages"]
    if not isinstance(tf_packages, Mapping) or set(tf_packages) != {
        "numpy",
        "tensorflow",
        "keras",
        "h5py",
    }:
        raise ValueError(
            "TensorFlow runtime must pin NumPy, TensorFlow, Keras, and h5py"
        )
    if tf_packages.get("tensorflow") != "2.16.2":
        raise ValueError("The frozen TensorFlow runtime must be version 2.16.2")
    tf_devices = tensorflow["devices"]
    physical_devices = (
        tf_devices.get("tensorflow_physical")
        if isinstance(tf_devices, Mapping)
        else None
    )
    if (
        not isinstance(physical_devices, list)
        or not physical_devices
        or any(
            not isinstance(device, Mapping)
            or set(device) != {"name", "type"}
            or device.get("type") != "CPU"
            for device in physical_devices
        )
    ):
        raise ValueError("The frozen TensorFlow device must be CPU only")
    python_packages = python["packages"]
    if not isinstance(python_packages, Mapping) or set(python_packages) != {
        "numpy",
        "tensorflow",
        "keras",
        "h5py",
    }:
        raise ValueError("Base Python runtime must pin NumPy")
    for runtime_record, label in ((python, "base"), (tensorflow, "TensorFlow")):
        packages, origins = runtime_record["packages"], runtime_record["origins"]
        if not isinstance(origins, Mapping) or set(origins) != set(packages):
            raise ValueError(f"{label} package origins must cover every pinned package")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in origins.values()
        ):
            raise ValueError(f"{label} package origins must be paths or explicit nulls")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in packages.values()
        ):
            raise ValueError(
                f"{label} package versions must be strings or explicit nulls"
            )
    if not isinstance(python_packages.get("numpy"), str) or not isinstance(
        python["origins"].get("numpy"), str
    ):
        raise ValueError("Base Python must pin its NumPy package and origin")
    for name in ("tensorflow", "keras", "h5py"):
        if python_packages[name] is not None or python["origins"][name] is not None:
            raise ValueError(
                "Base runtime smoke must not import TensorFlow-side packages"
            )
    if any(
        not isinstance(tf_packages[name], str)
        or not isinstance(tensorflow["origins"][name], str)
        for name in tf_packages
    ):
        raise ValueError(
            "TensorFlow runtime must pin every imported package and origin"
        )
    if not all(
        isinstance(value, str) and value
        for value in (python["executable"], tensorflow["executable"])
    ):
        raise ValueError("Runtime executable paths must be non-empty strings")
    base_executable = Path(python["executable"]).absolute()
    tf_executable = Path(tensorflow["executable"]).absolute()
    if not base_executable.samefile(files["python"]):
        raise ValueError("Base Python executable differs from the pinned file")
    if not tf_executable.samefile(files["tf_python"]):
        raise ValueError("TensorFlow executable differs from the pinned file")
    if (
        tf_executable.parent.name != "bin"
        or not files["tf_python"].is_symlink()
        or not tf_executable.parent.samefile(files["tf_python"].parent)
    ):
        raise ValueError("TensorFlow executable is outside the canonical venv bin")
    if not files["tf_python"].resolve(strict=True).is_file():
        raise ValueError(
            "TensorFlow interpreter symlink does not target a regular file"
        )
    for record in (python, tensorflow):
        devices = record["devices"]
        if not isinstance(devices, Mapping) or set(devices) != {
            "machine",
            "processor",
            "tensorflow_physical",
        }:
            raise ValueError("Runtime smoke must pin host and device identity")
        if not all(isinstance(devices[name], str) for name in ("machine", "processor")):
            raise ValueError("Runtime host identity must be textual")
    if tensorflow["devices"].get("tensorflow_physical") != physical_devices:
        raise ValueError("TensorFlow physical device identity is malformed")
    if python["devices"].get("tensorflow_physical") != []:
        raise ValueError("Base runtime smoke must not initialize TensorFlow devices")
    for record, label in ((python, "base Python"), (tensorflow, "TensorFlow")):
        threads = record.get("threads")
        boundary = record.get("import_boundary")
        if not isinstance(threads, Mapping) or not isinstance(boundary, Mapping):
            raise ValueError(f"Malformed {label} smoke metadata")
        if set(boundary) != {
            "minimal_namespace_bootstrap",
            "simon_initializer_loaded",
            "protocol_package_loaded",
            "forbidden_modules_loaded",
            "forbidden_import_attempts",
            "ranking_modules_loaded",
        }:
            raise ValueError(f"{label} import-boundary evidence schema differs")
        numeric = threads.get("numeric_environment")
        if (
            not isinstance(numeric, Mapping)
            or set(numeric) != set(PROFILE["numeric_thread_env"])
            or dict(numeric) != PROFILE["numeric_thread_env"]
        ):
            raise ValueError(f"{label} thread environment differs from the frozen profile")
        if (
            boundary.get("minimal_namespace_bootstrap") is not True
            or boundary.get("simon_initializer_loaded") is not False
        ):
            raise ValueError(
                f"{label} smoke did not preserve the worker import boundary"
            )
        if (
            boundary.get("protocol_package_loaded") is not False
            or boundary.get("forbidden_modules_loaded") != []
            or boundary.get("forbidden_import_attempts") != []
        ):
            raise ValueError(f"{label} smoke imported forbidden protocol/SAT modules")
        loaded_modules = boundary.get("ranking_modules_loaded")
        required_modules = {
            "neural_sat_speck.simon.cipher",
            "neural_sat_speck.simon.ranking_features",
            "neural_sat_speck.simon.published_vv",
            "neural_sat_speck.simon.published_vv_ranking",
            "neural_sat_speck.simon.classical_ranking",
        }
        allowed_modules = required_modules | {
            "neural_sat_speck.simon.key_schedule",
            "neural_sat_speck.simon.primitives",
        }
        if (
            not isinstance(loaded_modules, list)
            or loaded_modules != sorted(set(loaded_modules))
            or not required_modules <= set(loaded_modules)
            or not set(loaded_modules) <= allowed_modules
        ):
            raise ValueError(f"{label} smoke did not load the complete rank-only stack")
    tf_threads = tensorflow["threads"].get("tensorflow")
    if tf_threads != {"intra_op": 8, "inter_op": 1}:
        raise ValueError("TensorFlow must be pinned to intra-op eight/inter-op one")
    for record, label in ((python, "base"), (tensorflow, "TensorFlow")):
        threads = record["threads"]
        expected_threads = {
            "numeric_environment",
            "tensorflow",
            "feature_batch_size",
            "inference_batch_size",
            "candidate_batch_size",
        }
        if set(threads) != expected_threads:
            raise ValueError(f"{label} smoke thread/batch metadata fields differ")
        if (
            threads["feature_batch_size"],
            threads["inference_batch_size"],
            threads["candidate_batch_size"],
        ) != (1024, 16384, 1024):
            raise ValueError(f"{label} smoke batch sizes differ from PROFILE")
    if python["threads"]["tensorflow"] is not None:
        raise ValueError("Base runtime smoke must not initialize TensorFlow threading")


def validate_payload(payload: object) -> None:
    """Validate the exact truth-blind worker payload before any model imports."""
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("Worker payload fields differ from the fixed schema")
    if payload["schema"] != "stage2_rank_payload_v1":
        raise ValueError("Unknown worker payload schema")
    job = _validate_job(payload["job"])
    validate_public_case(payload["public_case"])
    if payload["public_case"]["key_index"] != job["key_index"]:
        raise ValueError("Payload case key index differs from its worker job")
    parents = payload["parents"]
    if job["operation"] == "stage1":
        if parents != []:
            raise ValueError("Stage 1 payload must have an empty parent list")
    else:
        if not isinstance(parents, list) or len(parents) != 8:
            raise ValueError("Stage 2 payload must contain eight ordered parents")
        checked = [_word(value, "parent") for value in parents]
        if len(set(checked)) != 8:
            raise ValueError("Stage 2 parents must be distinct")
    if (
        not isinstance(payload["invocation"], str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["invocation"]) is None
    ):
        raise ValueError(
            "Payload invocation must be a 32-character lowercase hex token"
        )
    if (
        not isinstance(payload["directory"], str)
        or not Path(payload["directory"]).is_absolute()
    ):
        raise ValueError("Payload worker directory must be absolute")
    root = payload["source_root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Payload source_root must be absolute")
    if (
        not isinstance(payload["source_commit"], str)
        or _COMMIT_RE.fullmatch(payload["source_commit"]) is None
    ):
        raise ValueError("Payload source_commit must be a full Git hash")
    source_files = payload["source_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError(
            "Payload source_files must be a nonempty relative-path mapping"
        )
    for relative, record in source_files.items():
        rel = Path(relative) if isinstance(relative, str) else None
        if rel is None or not relative or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Worker source file keys must be safe relative paths")
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError("Worker source file records must contain path and sha256")
        path = record["path"]
        digest = record["sha256"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("Worker source file path/hash pin is malformed")
    required_source_files = {
        "scripts/run_simon_matched_stage2_worker.py",
        "src/neural_sat_speck/followups/stage2_control.py",
        "src/neural_sat_speck/protocol/worker_telemetry.py",
        "src/neural_sat_speck/simon/cipher.py",
        "src/neural_sat_speck/simon/primitives.py",
        "src/neural_sat_speck/simon/ranking_features.py",
        "src/neural_sat_speck/simon/published_vv.py",
        "src/neural_sat_speck/simon/published_vv_ranking.py",
        "src/neural_sat_speck/simon/classical_ranking.py",
    }
    if not required_source_files <= set(source_files):
        raise ValueError("Worker source pins omit part of the rank-only implementation")
    assets = payload["assets"]
    if not isinstance(assets, Mapping) or set(assets) != set(_EXPECTED_ASSET_HASHES):
        raise ValueError("Worker payload must pin exactly VV9, VV8, and DDT7 assets")
    for name, expected in _EXPECTED_ASSET_HASHES.items():
        asset = assets[name]
        if (
            not isinstance(asset, Mapping)
            or set(asset) != {"path", "sha256", "identity"}
            or asset["sha256"] != expected
        ):
            raise ValueError(
                f"Worker payload asset pin differs from fixed identity: {name}"
            )
        if not isinstance(asset["path"], str) or not Path(asset["path"]).is_absolute():
            raise ValueError(f"Worker payload asset path must be absolute: {name}")
        identity = asset["identity"]
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"device", "inode", "size", "mtime_ns"}
            or any(type(value) is not int or value < 0 for value in identity.values())
        ):
            raise ValueError(f"Worker payload asset stat identity is malformed: {name}")
    _validate_runtime(
        payload["runtime"],
        {
            "python": Path(payload["runtime"]["python"]["executable"]),
            "tf_python": Path(payload["runtime"]["tensorflow"]["executable"]),
        },
    )
    if _canonical_json(payload["profile"]) != _canonical_json(PROFILE):
        raise ValueError("Worker payload PROFILE differs from the frozen profile")


def validate_manifest(
    manifest: object, source_root: str | os.PathLike[str]
) -> dict[str, dict[str, Path]]:
    """Validate every pin before a coordinator loads the dataset or any model.

    Returns ``{"files": {name: Path}, "assets": {name: Path}}``. Full-run
    manifests must bind a mechanically passing, source/profile-matched key-0
    pilot report. This routine never loads the dataset or a model.
    """
    root = Path(source_root).resolve(strict=True)
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be an object")
    mode = manifest.get("mode")
    expected_keys = {
        "status",
        "execution_authorized",
        "mode",
        "authorization_decision",
        "source_root",
        "source_commit",
        "profile",
        "jobs",
        "files",
        "assets",
        "runtime",
        "limits",
    }
    if mode == "full":
        expected_keys.add("pilot")
    if set(manifest) != expected_keys:
        raise ValueError("Manifest fields do not match the frozen schema")
    if (
        manifest["status"] != "APPROVED"
        or manifest["execution_authorized"] is not True
        or manifest["authorization_decision"] != "D-053"
    ):
        raise ValueError("Manifest is not author-approved for this execution")
    if mode not in {"pilot", "full"}:
        raise ValueError("Manifest mode must be pilot or full")
    manifest_root = manifest["source_root"]
    if not isinstance(manifest_root, str) or Path(manifest_root).resolve() != root:
        raise ValueError(
            "Manifest source_root does not match the validated source tree"
        )
    commit = manifest["source_commit"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("source_commit must be a full 40-character Git hash")
    try:
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Could not verify clean committed source snapshot") from error
    if actual_commit != commit or status:
        raise ValueError("Source tree is not the pinned clean Git commit")
    if _canonical_json(manifest["profile"]) != _canonical_json(PROFILE):
        raise ValueError("Manifest PROFILE differs from the frozen control profile")
    expected_jobs = fixed_jobs()[:3] if mode == "pilot" else fixed_jobs()
    if _canonical_json(manifest["jobs"]) != _canonical_json(expected_jobs):
        raise ValueError("Manifest jobs differ from the frozen mode grid")
    limits = manifest["limits"]
    expected_limits = [
        "Opened N8; conditional on shared Bao VV9 parents; descriptive only",
        "No solver, recovery, verifier, new dataset or protected seed",
        "Costs are shared-work projections; no autonomous attack timing",
        "TF intra-op eight matches prior autonomous inference; not equal CPU threads",
        "AC or battery allowed before outcomes; raw samples retained",
        "RSS sampled every ten seconds; no instantaneous/all-thread certification",
        "Real key0 pilot excluded; it can warm filesystem caches",
    ]
    if limits != expected_limits:
        raise ValueError(
            "Manifest scientific-limit disclosures differ from the frozen profile"
        )

    raw_files = manifest["files"]
    if not isinstance(raw_files, Mapping):
        raise ValueError("Manifest files must be an object")
    required_files = {"cohort", "python", "tf_python", "protocol", "execution_note"}
    expected_source_paths = {
        "src/" + path.relative_to(root / "src").as_posix()
        for path in (root / "src").rglob("*.py")
        if path.is_file()
    } | _SOURCE_SCRIPTS
    expected_source_keys = {"source:" + path for path in expected_source_paths}
    if set(raw_files) != required_files | expected_source_keys:
        raise ValueError(
            "Manifest is missing required file pins or has unknown file keys"
        )
    validated_files: dict[str, Path] = {}
    for name, record in raw_files.items():
        if name.startswith("source:"):
            relative = name[len("source:") :]
            rel_path = Path(relative)
            if not relative or rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(
                    "Source file keys must be safe repository-relative paths"
                )
            expected_path = (root / rel_path).absolute()
            try:
                expected_path.resolve(strict=True).relative_to(root)
            except ValueError as error:
                raise ValueError("Pinned source path escapes source_root") from error
            path = _file_path(record, root, name)
            if path != expected_path:
                raise ValueError(f"Source file path does not match its key: {name}")
        else:
            path = _file_path(record, root, name)
        validated_files[name] = path
    for required_source in (
        "source:src/neural_sat_speck/followups/stage2_control.py",
        "source:scripts/run_simon_matched_stage2_worker.py",
    ):
        if required_source not in validated_files:
            raise ValueError(f"Missing frozen source file pin: {required_source}")

    validated_assets = _validate_assets(manifest["assets"], root)
    _validate_runtime(manifest["runtime"], validated_files)

    if mode == "full":
        _validate_pilot_binding(manifest["pilot"], root, manifest, commit)
    return {"files": validated_files, "assets": validated_assets}


def _validate_pilot_binding(
    pilot: object,
    root: Path,
    full_manifest: Mapping[str, Any],
    commit: str,
) -> None:
    if not isinstance(pilot, Mapping) or set(pilot) != {"manifest", "report"}:
        raise ValueError("Full run must pin its pilot manifest and mechanical report")
    pilot_manifest_path = _file_path(pilot["manifest"], root, "pilot manifest")
    pilot_report_path = _file_path(pilot["report"], root, "pilot report")
    prior_manifest = _read_checked_artifact(pilot_manifest_path, "pilot manifest")
    report = _read_checked_artifact(pilot_report_path, "pilot report")
    if not isinstance(prior_manifest, Mapping) or not isinstance(report, Mapping):
        raise ValueError("Pinned pilot artifacts must be JSON objects")
    if (
        prior_manifest.get("mode") != "pilot"
        or prior_manifest.get("status") != "APPROVED"
        or prior_manifest.get("execution_authorized") is not True
        or prior_manifest.get("authorization_decision") != "D-053"
        or prior_manifest.get("source_commit") != commit
        or _canonical_json(prior_manifest.get("profile")) != _canonical_json(PROFILE)
        or _canonical_json(prior_manifest.get("jobs"))
        != _canonical_json(fixed_jobs()[:3])
        or prior_manifest.get("source_root") != full_manifest.get("source_root")
        or _canonical_json(prior_manifest.get("files"))
        != _canonical_json(full_manifest.get("files"))
        or _canonical_json(prior_manifest.get("limits"))
        != _canonical_json(full_manifest.get("limits"))
    ):
        raise ValueError("Pilot manifest does not match the full-run source/profile")
    expected_report_fields = {
        "schema",
        "status",
        "mechanical_pass",
        "source_commit",
        "profile",
        "profile_sha256",
        "manifest_sha256",
        "ledger_sha256",
        "jobs",
        "completed_calls",
        "completed_workers",
        "all_groups_absent",
        "truth_join_performed",
        "raw_sidecars_checked",
        "physical_worker_seconds",
    }
    if set(report) != expected_report_fields:
        raise ValueError("Pilot report fields differ from the frozen mechanical schema")
    if (
        report.get("schema") != "stage2_control_pilot_report_v1"
        or report.get("status") != "PASS"
        or report.get("mechanical_pass") is not True
        or report.get("source_commit") != commit
        or _canonical_json(report.get("profile")) != _canonical_json(PROFILE)
        or report.get("profile_sha256") != ordered_hash(PROFILE)
        or report.get("manifest_sha256") != streaming_sha256(pilot_manifest_path)
        or _SHA256_RE.fullmatch(str(report.get("ledger_sha256", ""))) is None
        or report.get("jobs") != [job["job_id"] for job in fixed_jobs()[:3]]
        or type(report.get("completed_calls")) is not int
        or report.get("completed_calls") != 17
        or type(report.get("completed_workers")) is not int
        or report.get("completed_workers") != 3
        or report.get("all_groups_absent") is not True
        or report.get("truth_join_performed") is not False
        or type(report.get("raw_sidecars_checked")) is not int
        or report.get("raw_sidecars_checked") < 1
    ):
        raise ValueError(
            "Pilot mechanical report is not a passing matched 17-call pilot"
        )
    _finite_nonnegative(
        report.get("physical_worker_seconds"), "pilot physical_worker_seconds"
    )
    if _canonical_json(prior_manifest.get("assets")) != _canonical_json(
        full_manifest.get("assets")
    ):
        raise ValueError("Pilot and full manifests do not pin the same ranking assets")
    if _canonical_json(prior_manifest.get("runtime")) != _canonical_json(
        full_manifest.get("runtime")
    ):
        raise ValueError("Pilot and full manifests do not pin the same runtime")


def _read_checked_artifact(path: Path, label: str) -> Any:
    """Read JSON only when its standard atomic_json checksum sidecar matches."""
    try:
        raw = path.read_bytes()
        sidecar = path.with_name(path.name + ".sha256").read_text(encoding="ascii")
        expected = f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n"
        if sidecar != expected:
            raise ValueError(f"{label} checksum sidecar mismatch")
        return json.loads(
            raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"forbidden JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Pinned {label} is not checked JSON") from error


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _validate_result(
    result: object, job: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    if not isinstance(result, Mapping) or set(result) != _RESULT_FIELDS:
        raise ValueError("Call result has an unexpected or incomplete schema")
    if result["job_id"] != job["job_id"] or result["scorer"] != job["scorer"]:
        raise ValueError("Call result identity differs from its worker job")
    if type(result["ordinal"]) is not int or result["ordinal"] != ordinal:
        raise ValueError("Call result ordinal is invalid")
    invocation = result["invocation"]
    if (
        not isinstance(invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
    ):
        raise ValueError("Call result invocation is missing")
    if result["parent"] is not None:
        _word(result["parent"], "result parent")
    if job["operation"] == "stage1":
        if ordinal != 0 or result["parent"] is not None:
            raise ValueError("Stage 1 result must have ordinal 0 and null parent")
    elif result["parent"] is None:
        raise ValueError("Stage 2 result must name the conditioned parent")
    words = validate_order(result["ordered_words"])
    timings = result["component_times_seconds"]
    if not isinstance(timings, Mapping) or not timings:
        raise ValueError("Call result must include component timings")
    for name, seconds in timings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Call component timing names must be non-empty strings")
        _finite_nonnegative(seconds, f"component_times_seconds.{name}")
    start = result["call_started_monotonic"]
    complete = result["call_completed_monotonic"]
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(complete, bool)
        or not isinstance(complete, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(complete))
        or complete <= start
    ):
        raise ValueError("Call monotonic timestamps must be finite and increasing")
    phase = _finite_nonnegative(result["phase_wall_seconds"], "phase_wall_seconds")
    if (
        phase <= 0
        or phase > PROFILE["call_limit_seconds"]
        or phase + 1e-9 < float(complete) - float(start)
    ):
        raise ValueError(
            "Call phase duration is outside the frozen bound or below measured call time"
        )
    checked = dict(result)
    checked["ordered_words"] = words
    return checked


def _closed_jobs_input(closed_jobs: object) -> tuple[str, list[Mapping[str, Any]]]:
    if isinstance(closed_jobs, Mapping):
        if set(closed_jobs) != {"mode", "jobs"}:
            raise ValueError("Closed-grid wrapper must contain exactly mode and jobs")
        mode, rows = closed_jobs["mode"], closed_jobs["jobs"]
    else:
        rows = closed_jobs
        mode = None
    if mode not in {None, "pilot", "full"}:
        raise ValueError("Closed-grid mode must be pilot or full")
    if not isinstance(rows, (list, tuple)) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        raise ValueError("closed_jobs must be a sequence of worker closure records")
    if mode is None:
        modes = {row.get("mode") for row in rows if "mode" in row}
        if len(modes) == 1:
            mode = modes.pop()
        elif len(rows) == 24:
            mode = "full"
        else:
            raise ValueError(
                "Incomplete closure list needs an explicit pilot/full mode"
            )
    if mode == "pilot":
        return "pilot", list(rows)
    return "full", list(rows)


def _validate_closure_row(
    row: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = set(job) | {
        "mode",
        "outcome",
        "dispatched",
        "directory",
        "invocation",
        "calls",
        "load_wall_seconds",
        "worker_wall_seconds",
        "queue_measurements",
        "accepted_events",
        "guardian_receipt",
        "completed_utc",
        "error",
    }
    if set(row) - allowed or not set(job) <= set(row):
        raise ValueError("Worker closure record has unknown or missing fields")
    if any(row[field] != job[field] for field in job):
        raise ValueError("Worker closure record does not match its fixed-grid job")
    if row.get("mode", "full") != "full":
        raise ValueError("Scientific analysis accepts only full-grid worker rows")
    outcome = row.get("outcome")
    known_outcomes = (
        outcome in _UNAVAILABLE_OUTCOMES
        or outcome == "unavailable_after_coordinator_interrupted"
        or (
        isinstance(outcome, str)
        and outcome.startswith("unavailable_after_")
        and outcome.removeprefix("unavailable_after_")
        in _UNAVAILABLE_OUTCOMES
        | {"campaign_timeout", "integrity_alarm", "environment_alarm"}
        )
    )
    if outcome != "completed" and not known_outcomes:
        raise ValueError("Worker outcome is not a recognized completion/unavailability")
    dispatched = row.get("dispatched")
    if type(dispatched) is not bool:
        raise ValueError("Worker closure must say whether a worker was dispatched")
    calls = row.get("calls")
    if not isinstance(calls, (list, tuple)):
        raise ValueError(
            "Worker calls must be a list, including an empty list for tombstones"
        )
    expected_count = 1 if job["operation"] == "stage1" else 8
    checked_calls: dict[int, dict[str, Any]] = {}
    for raw in calls:
        if not isinstance(raw, Mapping) or type(raw.get("ordinal")) is not int:
            raise ValueError("Worker call record is missing a genuine ordinal")
        ordinal = raw["ordinal"]
        if not 0 <= ordinal < expected_count or ordinal in checked_calls:
            raise ValueError(
                "Worker call ordinal is duplicate or outside its assigned range"
            )
        checked_calls[ordinal] = _validate_result(raw, job, ordinal)
    if outcome == "completed" and set(checked_calls) != set(range(expected_count)):
        raise ValueError("Completed worker is missing an assigned rank call")
    if len(checked_calls) != len(calls):
        raise ValueError("Worker call results contain duplicate ordinals")
    if [call["ordinal"] for call in calls] != list(range(len(calls))):
        raise ValueError("Worker calls are not preserved in sequential ordinal order")
    if checked_calls:
        invocations = {call["invocation"] for call in checked_calls.values()}
        if len(invocations) != 1:
            raise ValueError("Call results crossed worker invocation boundaries")

    load_seconds = row.get("load_wall_seconds")
    worker_seconds = row.get("worker_wall_seconds")
    if load_seconds is not None:
        _finite_nonnegative(load_seconds, "load_wall_seconds")
        if load_seconds > PROFILE["startup_limit_seconds"]:
            raise ValueError("Worker load exceeded its hard phase bound")
    if worker_seconds is not None:
        _finite_nonnegative(worker_seconds, "worker_wall_seconds")
    if load_seconds is None and outcome == "completed":
        raise ValueError("Completed worker must report its load time")
    events = row.get("accepted_events")
    receipt = row.get("guardian_receipt")
    if not dispatched:
        if (
            outcome == "completed"
            or calls
            or load_seconds is not None
            or worker_seconds is not None
            or row.get("queue_measurements") != {}
        ):
            raise ValueError(
                "Undispatched unavailable tombstone contains worker output"
            )
        if (
            outcome == "unavailable_after_coordinator_interrupted"
            and receipt is None
            and events is None
            and row.get("queue_measurements") == {}
        ):
            invocation = row.get("invocation")
            directory = row.get("directory")
            if (
                not isinstance(invocation, str)
                or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
                or not isinstance(directory, str)
                or not Path(directory).is_absolute()
                or row.get("error")
                != "Checked start was written before its ledger snapshot; dispatch was not reached"
            ):
                raise ValueError(
                    "Coordinator-interruption orphan lacks its exact checked-start tombstone"
                )
            return {
                "job": dict(job),
                "outcome": outcome,
                "calls": {},
                "load_wall_seconds": None,
                "worker_wall_seconds": None,
                "physical_over_phase_budget_seconds": None,
                "worker_tail_seconds_after_accepted_phases": None,
                "queue_measurements": {},
                "receipt": None,
                "accepted_events": [],
                "dispatch_state": "not_started_orphan",
            }
        if receipt is None and events is None:
            if "directory" in row or "invocation" in row:
                raise ValueError(
                    "Unproven undispatched tombstone cannot claim an invocation"
                )
            dispatch_state = "not_dispatched"
        else:
            invocation = row.get("invocation")
            if (
                not isinstance(events, list)
                or events
                or not isinstance(receipt, Mapping)
                or not isinstance(invocation, str)
                or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
                or receipt.get("schema") != "simon_stage2_guardian_receipt_v1"
                or receipt.get("invocation") != invocation
                or receipt.get("job_id") != job["job_id"]
                or receipt.get("worker_not_started") is not True
                or receipt.get("worker_pid") is not None
                or receipt.get("process_group") is not None
                or receipt.get("process_group_absent") is not True
                or receipt.get("accepted_events") != []
            ):
                raise ValueError(
                    "No-worker receipt does not prove a clean undispatched close"
                )
            _finite_nonnegative(
                receipt.get("cleanup_completed_monotonic"),
                "no-worker guardian cleanup completion",
            )
            if row.get("directory") is not None and (
                not isinstance(row["directory"], str)
                or not Path(row["directory"]).is_absolute()
            ):
                raise ValueError("No-worker invocation directory must be absolute")
            dispatch_state = "not_started_proven"
        return {
            "job": dict(job),
            "outcome": outcome,
            "calls": checked_calls,
            "load_wall_seconds": None,
            "worker_wall_seconds": None,
            "physical_over_phase_budget_seconds": None,
            "worker_tail_seconds_after_accepted_phases": None,
            "queue_measurements": {},
            "receipt": receipt,
            "accepted_events": events or [],
            "dispatch_state": dispatch_state,
        }
    if (
        outcome == "interrupted"
        and calls == []
        and (events is None or events == [])
        and receipt is None
        and load_seconds is None
        and worker_seconds is None
        and row.get("queue_measurements") == {}
    ):
        invocation = row.get("invocation")
        if (
            not isinstance(invocation, str)
            or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
            or not isinstance(row.get("directory"), str)
            or not Path(row["directory"]).is_absolute()
        ):
            raise ValueError(
                "Unknown interrupted dispatch must preserve its invocation directory"
            )
        return {
            "job": dict(job),
            "outcome": outcome,
            "calls": {},
            "load_wall_seconds": None,
            "worker_wall_seconds": None,
            "physical_over_phase_budget_seconds": None,
            "worker_tail_seconds_after_accepted_phases": None,
            "queue_measurements": {},
            "receipt": None,
            "accepted_events": [],
            "dispatch_state": "dispatch_unknown",
        }
    if not isinstance(events, list) or not isinstance(receipt, Mapping):
        raise ValueError(
            "Dispatched worker must bind accepted events to a guardian receipt"
        )
    invocation = row.get("invocation")
    if (
        not isinstance(invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
    ):
        raise ValueError("Dispatched worker invocation is malformed")
    if row.get("directory") is not None and (
        not isinstance(row["directory"], str)
        or not Path(row["directory"]).is_absolute()
    ):
        raise ValueError("Dispatched worker directory must be absolute")
    if (
        receipt.get("schema") != "simon_stage2_guardian_receipt_v1"
        or receipt.get("invocation") != invocation
        or receipt.get("job_id") != job["job_id"]
        or receipt.get("process_group_absent") is not True
        or receipt.get("accepted_events") != events
    ):
        raise ValueError(
            "Guardian receipt identity/events/group absence do not bind the row"
        )
    if any(call["invocation"] != invocation for call in checked_calls.values()):
        raise ValueError("Call result invocation differs from its worker closure")
    worker_pid, process_group = receipt.get("worker_pid"), receipt.get("process_group")
    if worker_pid is not None and (
        type(worker_pid) is not int or worker_pid <= 0 or process_group != worker_pid
    ):
        raise ValueError("Guardian worker process-group identity is malformed")
    if (worker_pid is None) != (process_group is None):
        raise ValueError("Guardian PID and process-group presence disagree")
    if not isinstance(events, list) or len(events) > expected_count + 1:
        raise ValueError("Guardian accepted event list exceeds its fixed call schedule")
    loaded_count = 0
    previous: float | None = None
    if events:
        loaded = events[0]
        if (
            not isinstance(loaded, Mapping)
            or loaded.get("schema") != "stage2_rank_worker_event_v1"
            or loaded.get("event") != "loaded"
            or loaded.get("event_index") != 0
            or loaded.get("invocation") != invocation
            or loaded.get("job_id") != job["job_id"]
            or loaded.get("scorer") != job["scorer"]
            or loaded.get("event_path") != "events/0000.json"
        ):
            raise ValueError(
                "First accepted guardian event is not the matching loaded record"
            )
        _finite_nonnegative(loaded.get("load_wall_seconds"), "loaded event load time")
        if loaded["load_wall_seconds"] > PROFILE["startup_limit_seconds"]:
            raise ValueError("Loaded event exceeded the startup hard bound")
        previous = _finite_nonnegative(
            loaded.get("completed_monotonic"), "loaded event completion"
        )
        _validate_sha(loaded.get("event_sha256"), "loaded event hash")
        loaded_count = 1
    if load_seconds is not None and loaded_count != 1:
        raise ValueError(
            "Coordinator reports load completion without an accepted loaded event"
        )
    if loaded_count and load_seconds is None:
        raise ValueError("Accepted loaded event is missing coordinator load timing")
    for ordinal, event in enumerate(events[1:]):
        if not isinstance(event, Mapping):
            raise ValueError("Accepted guardian event must be an object")
        call = checked_calls.get(ordinal)
        if call is None:
            raise ValueError(
                "Guardian accepted a call event without a preserved call result"
            )
        if (
            event.get("schema") != "stage2_rank_worker_event_v1"
            or event.get("event") != "call_complete"
            or event.get("event_index") != ordinal + 1
            or event.get("ordinal") != ordinal
            or event.get("invocation") != invocation
            or event.get("job_id") != job["job_id"]
            or event.get("scorer") != job["scorer"]
            or event.get("parent") != call["parent"]
            or event.get("result_path") != f"calls/{ordinal:04d}_result.json"
            or event.get("start_path") != f"calls/{ordinal:04d}_start.json"
            or event.get("call_started_monotonic") != call["call_started_monotonic"]
            or event.get("call_completed_monotonic") != call["call_completed_monotonic"]
        ):
            raise ValueError(
                "Accepted completion event differs from its closed rank result"
            )
        _validate_sha(event.get("event_sha256"), "call event hash")
        _validate_sha(event.get("result_sha256"), "call result hash")
        _validate_sha(event.get("start_sha256"), "call start hash")
        completed = _finite_nonnegative(
            event.get("completed_monotonic"), "call event completion"
        )
        if (
            call["call_started_monotonic"] < previous
            or call["call_completed_monotonic"] > completed
        ):
            raise ValueError("Rank call clocks fall outside their accepted phase")
        if (
            previous is None
            or completed < previous
            or call["phase_wall_seconds"] != completed - previous
        ):
            raise ValueError("Accepted phase duration/event clocks are inconsistent")
        previous = completed
    if len(checked_calls) != len(events) - loaded_count:
        raise ValueError(
            "Closed rank results and accepted call events are not one-to-one"
        )
    if outcome == "completed":
        if (
            loaded_count != 1
            or set(checked_calls) != set(range(expected_count))
            or receipt.get("status") != "worker_exit"
            or receipt.get("worker_returncode") != 0
        ):
            raise ValueError(
                "Completed worker does not have a complete successful guardian close"
            )
    elif (
        receipt.get("status") == "worker_exit"
        and receipt.get("worker_returncode") == 0
        and outcome
        not in {
            "interrupted",
            "campaign_timeout",
            "unavailable_after_interrupted",
            "unavailable_after_campaign_timeout",
        }
    ):
        raise ValueError(
            "Unavailable outcome conflicts with a successful worker receipt"
        )
    cleanup = _finite_nonnegative(
        receipt.get("cleanup_completed_monotonic"), "guardian cleanup completion"
    )
    if previous is not None and cleanup < previous:
        raise ValueError("Guardian cleanup completed before its last accepted event")
    if worker_seconds is None:
        raise ValueError("Dispatched worker must report physical wall time")
    accounted_worker_phases = (
        float(load_seconds) if load_seconds is not None else 0.0
    ) + sum(float(call["phase_wall_seconds"]) for call in checked_calls.values())
    if accounted_worker_phases > float(worker_seconds) + 1e-6:
        raise ValueError("Physical worker wall time is below accepted phase durations")
    scheduled_phase_budget = (
        PROFILE["startup_limit_seconds"]
        + PROFILE["call_limit_seconds"] * expected_count
    )
    physical_over_phase_budget = max(0.0, float(worker_seconds) - scheduled_phase_budget)
    worker_tail = max(0.0, float(worker_seconds) - accounted_worker_phases)

    measurements = row.get("queue_measurements")
    if not isinstance(measurements, Mapping) or not set(measurements) <= {"1", "8"}:
        raise ValueError("queue_measurements must be keyed only by width '1' or '8'")
    checked_measurements: dict[str, dict[str, Any]] = {}
    for width_key, value in measurements.items():
        if not isinstance(value, Mapping) or set(value) != {"seconds", "queue_sha256"}:
            raise ValueError("Queue measurement must contain seconds and queue_sha256")
        width = int(width_key)
        if width not in (1, 8):
            raise ValueError("Invalid queue width key")
        _finite_nonnegative(value["seconds"], f"queue_measurements.{width_key}.seconds")
        if (
            not isinstance(value["queue_sha256"], str)
            or _SHA256_RE.fullmatch(value["queue_sha256"]) is None
        ):
            raise ValueError("Invalid queue measurement hash")
        checked_measurements[width_key] = dict(value)
    if job["operation"] == "stage1" and checked_measurements:
        raise ValueError("Stage 1 worker cannot report a Stage 2 queue measurement")
    if job["operation"] == "stage2":
        for width_key in checked_measurements:
            width = int(width_key)
            if not set(range(width)) <= set(checked_calls):
                raise ValueError(
                    "Queue persistence was measured without all required parent calls"
                )
    return {
        "job": dict(job),
        "outcome": outcome,
        "calls": checked_calls,
        "load_wall_seconds": float(load_seconds) if load_seconds is not None else None,
        "worker_wall_seconds": float(worker_seconds),
        "physical_over_phase_budget_seconds": physical_over_phase_budget,
        "worker_tail_seconds_after_accepted_phases": worker_tail,
        "queue_measurements": checked_measurements,
        "receipt": receipt,
        "accepted_events": events,
        "dispatch_state": "worker_started",
    }


def analyze_cells(dataset: object, closed_jobs: object) -> dict[str, Any]:
    """Produce 32 finite-cohort descriptive cells after strict full-grid closure.

    Pilot closures return an explicit non-scientific status without inspecting
    dataset truth. No significance tests or inferential intervals are computed.
    """
    mode, rows = _closed_jobs_input(closed_jobs)
    if mode == "pilot":
        if (
            len(rows) != 3
            or [{key: row.get(key) for key in _JOB_FIELDS} for row in rows]
            != fixed_jobs()[:3]
        ):
            raise ValueError(
                "Pilot closure does not match the fixed key-0 three-worker grid"
            )
        return {
            "schema": "simon_stage2_control_analysis_v1",
            "status": "pilot_not_scientific",
            "evidence_class": "engineering_pilot",
            "cells": [],
            "aggregates": [],
        }

    expected = fixed_jobs()
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        job = {field: row.get(field) for field in _JOB_FIELDS}
        if job not in expected:
            raise ValueError("Closure contains a job outside the fixed grid")
        job_id = job["job_id"]
        if job_id in by_id:
            raise ValueError("Duplicate worker closure record")
        by_id[job_id] = row
    if set(by_id) != {job["job_id"] for job in expected}:
        raise ValueError(
            "Truth join requires all 24 worker jobs, including unavailable tombstones"
        )

    validated: dict[str, dict[str, Any]] = {}
    for job in expected:
        validated[job["job_id"]] = _validate_closure_row(by_id[job["job_id"]], job)

    invocation_ids = [
        row["invocation"]
        for row in by_id.values()
        if isinstance(row.get("invocation"), str)
    ]
    directories = [
        row["directory"]
        for row in by_id.values()
        if isinstance(row.get("directory"), str)
    ]
    if len(invocation_ids) != len(set(invocation_ids)) or len(directories) != len(
        set(directories)
    ):
        raise ValueError(
            "Worker invocation/directory was reused across fixed job slots"
        )

    for key_index in range(8):
        stage1_job = next(
            job
            for job in expected
            if job["key_index"] == key_index and job["operation"] == "stage1"
        )
        stage1_result = validated[stage1_job["job_id"]]["calls"].get(0)
        for stage2_job in (
            job
            for job in expected
            if job["key_index"] == key_index and job["operation"] == "stage2"
        ):
            stage2_row = validated[stage2_job["job_id"]]
            original_row = by_id[stage2_job["job_id"]]
            if stage1_result is None:
                if stage2_row["calls"] or original_row.get("dispatched") is True:
                    raise ValueError(
                        "Stage 2 ran without its shared Stage 1 parent order"
                    )
                continue
            parents = stage1_result["ordered_words"][:8]
            for ordinal, call in stage2_row["calls"].items():
                if call["parent"] != parents[ordinal]:
                    raise ValueError(
                        "Conditional result parent differs from the common Stage 1 order"
                    )

    if not isinstance(dataset, Mapping):
        raise ValueError("Dataset must be provided as a mapping after worker closure")
    # The truth-bearing dataset is not touched until every fixed worker record,
    # call result, event count, receipt, and unavailable tombstone has validated.
    cohort = _analysis_cohort(dataset)

    cells: list[dict[str, Any]] = []
    for key_index in range(8):
        stage1_job = next(
            job
            for job in expected
            if job["key_index"] == key_index and job["operation"] == "stage1"
        )
        stage1 = validated[stage1_job["job_id"]]
        truth = cohort[key_index]
        round_keys = truth["round_keys"]
        true_child, true_parent = round_keys[-2], round_keys[-1]
        instance_id = truth["instance_id"]
        stage1_result = stage1["calls"].get(0)
        parents: tuple[int, ...] | None = None
        true_parent_rank: int | None = None
        parent_status = "stage1_unavailable"
        if stage1_result is not None:
            stage1_order = stage1_result["ordered_words"]
            parents = stage1_order[:8]
            parent_status = (
                "ranked_in_fixed_shortlist"
                if true_parent in stage1_order
                else "not_in_fixed_shortlist"
            )
            if true_parent in stage1_order:
                true_parent_rank = stage1_order.index(true_parent) + 1

        for scorer in ("vv8", "ddt7"):
            stage2_job = next(
                job
                for job in expected
                if job["key_index"] == key_index
                and job["operation"] == "stage2"
                and job["scorer"] == scorer
            )
            stage2 = validated[stage2_job["job_id"]]
            calls: dict[int, dict[str, Any]] = stage2["calls"]
            for width in (1, 8):
                width_key = str(width)
                queue: tuple[tuple[int, int], ...] | None = None
                queue_measurement = stage2["queue_measurements"].get(width_key)
                if parents is not None and set(calls) >= set(range(width)):
                    if queue_measurement is not None:
                        queue = assemble_queue(
                            parents,
                            {
                                ordinal: calls[ordinal]["ordered_words"]
                                for ordinal in range(width)
                            },
                            width,
                        )
                        if queue_measurement["queue_sha256"] != ordered_hash(queue):
                            raise ValueError(
                                "Measured queue hash differs from independent reconstruction"
                            )
                    elif stage2["outcome"] == "completed":
                        raise ValueError(
                            "Completed worker is missing its measured queue persistence record"
                        )
                elif queue_measurement is not None:
                    raise ValueError(
                        "Incomplete queue has a claimed queue persistence measurement"
                    )

                parent_in_width: bool | None
                if true_parent_rank is None:
                    parent_in_width = None if parents is None else False
                else:
                    parent_in_width = true_parent_rank <= width
                child_status: str
                true_child_rank: int | None = None
                child_in_top128: bool | None = None
                if parents is None:
                    child_status = "stage1_unavailable"
                elif parent_in_width is False:
                    child_status = "parent_not_selected"
                elif true_parent_rank is None:
                    # Stage 1 completed, but the true parent is outside its
                    # exact 512-candidate rescore set.
                    child_status = "true_parent_not_in_fixed_shortlist"
                else:
                    ordinal = true_parent_rank - 1
                    call = calls.get(ordinal)
                    if call is None:
                        child_status = "conditional_call_unavailable"
                    else:
                        child_order = call["ordered_words"]
                        if true_child in child_order:
                            true_child_rank = child_order.index(true_child) + 1
                            child_status = "ranked_in_fixed_shortlist"
                            child_in_top128 = true_child_rank <= 128
                        else:
                            child_status = "not_in_fixed_shortlist"
                            child_in_top128 = False

                tuple_contained: bool | None = None
                tuple_position: int | None = None
                if queue is not None:
                    target = (true_child, true_parent)
                    tuple_contained = target in queue
                    if tuple_contained:
                        tuple_position = queue.index(target) + 1
                    if parent_in_width is False and tuple_contained:
                        raise ValueError(
                            "Tuple containment contradicts parent non-containment"
                        )
                    if child_in_top128 is False and tuple_contained:
                        raise ValueError(
                            "Tuple containment contradicts conditional child miss"
                        )
                    if child_in_top128 is True and not tuple_contained:
                        raise ValueError(
                            "Conditional child inclusion contradicts tuple queue"
                        )

                stage1_wall = (
                    stage1["worker_wall_seconds"] if stage1_result is not None else None
                )
                stage2_load = stage2["load_wall_seconds"]
                call_seconds: float | None = None
                if set(calls) >= set(range(width)):
                    call_seconds = sum(
                        float(calls[index]["phase_wall_seconds"])
                        for index in range(width)
                    )
                queue_seconds = (
                    float(queue_measurement["seconds"])
                    if queue_measurement is not None
                    else None
                )
                unallocated_physical = (
                    max(
                        0.0,
                        float(stage2["worker_wall_seconds"])
                        - float(stage2_load)
                        - float(call_seconds),
                    )
                    if stage2["worker_wall_seconds"] is not None
                    and stage2_load is not None
                    and call_seconds is not None
                    else None
                )
                projected_cost = (
                    float(stage1_wall)
                    + float(stage2_load)
                    + float(call_seconds)
                    + float(queue_seconds)
                    if queue is not None
                    and stage1_wall is not None
                    and stage2_load is not None
                    and call_seconds is not None
                    and queue_seconds is not None
                    else None
                )
                cells.append(
                    {
                        "key_index": key_index,
                        "instance_id": instance_id,
                        "scorer": scorer,
                        "parent_width": width,
                        "assigned_key_denominator": 8,
                        "stage1_outcome": stage1["outcome"],
                        "stage2_outcome": stage2["outcome"],
                        "stage1_dispatch_state": stage1["dispatch_state"],
                        "stage2_dispatch_state": stage2["dispatch_state"],
                        "stage1_physical_worker_seconds_over_phase_budget": stage1[
                            "physical_over_phase_budget_seconds"
                        ],
                        "stage1_true_parent_rank_within_512": true_parent_rank,
                        "stage1_true_parent_status": parent_status,
                        "true_parent_in_width": parent_in_width,
                        "queue_complete": queue is not None,
                        "queue_size": len(queue) if queue is not None else None,
                        "queue_sha256": queue_measurement["queue_sha256"]
                        if queue_measurement
                        else None,
                        "true_child_status": child_status,
                        "true_child_rank_within_512": true_child_rank,
                        "true_child_in_top128_given_parent": child_in_top128,
                        "true_tuple_contained": tuple_contained,
                        "true_tuple_position": tuple_position,
                        "stage1_worker_wall_seconds_charged": stage1_wall,
                        "stage2_load_wall_seconds_charged": stage2_load
                        if queue is not None
                        else None,
                        "selected_stage2_call_phase_seconds": call_seconds,
                        "stage2_worker_physical_wall_seconds": stage2[
                            "worker_wall_seconds"
                        ],
                        "stage2_physical_worker_seconds_over_phase_budget": stage2[
                            "physical_over_phase_budget_seconds"
                        ],
                        "stage2_worker_tail_seconds_after_accepted_phases": stage2[
                            "worker_tail_seconds_after_accepted_phases"
                        ],
                        "stage2_unallocated_physical_seconds_for_width": unallocated_physical,
                        "queue_assembly_persistence_seconds": queue_seconds,
                        "projected_ranking_cost_seconds": projected_cost,
                    }
                )

    aggregates = _aggregate_cells(cells)
    return {
        "schema": "simon_stage2_control_analysis_v1",
        "status": "descriptive_opened_demo",
        "evidence_class": "opened_demo_diagnostic",
        "assigned_keys": 8,
        "cell_count": len(cells),
        "significance_tests": "none",
        "cells": cells,
        "aggregates": aggregates,
    }


def _analysis_cohort(dataset: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if (
        type(dataset.get("seed")) is not int
        or dataset.get("seed") != 999999
        or dataset.get("split_role") != "demo"
    ):
        raise ValueError("Analyzer only accepts the registered opened demo cohort")
    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("key_count_per_cipher"),
        metadata.get("max_differential_pairs"),
        metadata.get("verification_blocks_count"),
    ) != (8, 64, 2):
        raise ValueError(
            "Analyzer dataset dimensions differ from the frozen demo cohort"
        )
    ciphers = dataset.get("ciphers")
    if not isinstance(ciphers, Mapping) or "simon32_64" not in ciphers:
        raise ValueError("Analyzer dataset is missing SIMON32/64")
    simon = ciphers["simon32_64"]
    instances = simon.get("instances") if isinstance(simon, Mapping) else None
    if not isinstance(instances, (list, tuple)) or len(instances) != 8:
        raise ValueError("Analyzer requires exactly eight ordered SIMON keys")
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(instances):
        if (
            not isinstance(row, Mapping)
            or type(row.get("instance_index")) is not int
            or row["instance_index"] != index
        ):
            raise ValueError("Analyzer dataset key indexes are malformed or reordered")
        instance_id = row.get("instance_id")
        if instance_id != f"simon32_64_demo_key_{index}":
            raise ValueError(
                "Analyzer dataset instance_id differs from its fixed key ordering"
            )
        rounds_data = row.get("rounds_data")
        if not isinstance(rounds_data, Mapping) or not isinstance(
            rounds_data.get("10"), Mapping
        ):
            raise ValueError("Analyzer dataset lacks R10 truth metadata")
        round_keys = rounds_data["10"].get("round_keys")
        if not isinstance(round_keys, (list, tuple)) or len(round_keys) != 10:
            raise ValueError("Analyzer R10 round-key schedule is malformed")
        if any(type(key) is not int or not 0 <= key <= 0xFFFF for key in round_keys):
            raise ValueError("Analyzer R10 round-key values must be 16-bit words")
        result[index] = {
            "instance_id": instance_id,
            "round_keys": tuple(round_keys),
        }
    return result


def _aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for width in (1, 8):
        for scorer in ("vv8", "ddt7"):
            group = [
                cell
                for cell in cells
                if cell["parent_width"] == width and cell["scorer"] == scorer
            ]
            if len(group) != 8:
                raise ValueError(
                    "Each descriptive arm must retain all eight assigned keys"
                )
            complete = [cell for cell in group if cell["queue_complete"]]
            tuple_hits = sum(cell["true_tuple_contained"] is True for cell in complete)
            operational_hits = sum(
                cell["queue_complete"] and cell["true_tuple_contained"] is True
                for cell in group
            )
            costs = [cell["projected_ranking_cost_seconds"] for cell in group]
            observed_costs = [float(value) for value in costs if value is not None]
            parent_observed = [
                cell
                for cell in group
                if cell["stage1_true_parent_rank_within_512"] is not None
                or cell["stage1_outcome"] == "completed"
            ]
            parent_hits = sum(
                cell["true_parent_in_width"] is True for cell in parent_observed
            )
            conditional_observed = [
                cell
                for cell in group
                if cell["true_parent_in_width"] is True
                and cell["true_child_status"]
                in {"ranked_in_fixed_shortlist", "not_in_fixed_shortlist"}
            ]
            child_hits = sum(
                cell["true_child_in_top128_given_parent"] is True
                for cell in conditional_observed
            )
            aggregates.append(
                {
                    "scorer": scorer,
                    "parent_width": width,
                    "assigned_key_denominator": 8,
                    "stage1_parent_order_available_count": sum(
                        cell["stage1_outcome"] == "completed" for cell in group
                    ),
                    "true_parent_in_width_numerator": parent_hits,
                    "true_parent_in_width_denominator": 8,
                    "true_parent_in_width_given_stage1_available": (
                        parent_hits / len(parent_observed) if parent_observed else None
                    ),
                    "stage1_parent_unavailable_count": 8 - len(parent_observed),
                    "structural_parent_miss_count": len(parent_observed) - parent_hits,
                    "conditional_child_rank_available_count": len(conditional_observed),
                    "conditional_child_top128_numerator": child_hits,
                    "conditional_child_top128_denominator": len(conditional_observed),
                    "conditional_child_top128_given_parent": (
                        child_hits / len(conditional_observed)
                        if conditional_observed
                        else None
                    ),
                    "conditional_child_rank_unavailable_count": sum(
                        cell["true_parent_in_width"] is True
                        and cell["true_child_status"] == "conditional_call_unavailable"
                        for cell in group
                    ),
                    "conditional_child_censored_outside_512_count": sum(
                        cell["true_parent_in_width"] is True
                        and cell["true_child_status"] == "not_in_fixed_shortlist"
                        for cell in group
                    ),
                    "complete_queue_count": len(complete),
                    "unavailable_or_incomplete_queue_count": 8 - len(complete),
                    "true_tuple_containment_numerator": tuple_hits,
                    "true_tuple_containment_denominator": len(complete),
                    "true_tuple_containment_given_complete": (
                        tuple_hits / len(complete) if complete else None
                    ),
                    "usable_true_tuple_queue_yield_numerator": operational_hits,
                    "usable_true_tuple_queue_yield_denominator": 8,
                    "usable_true_tuple_queue_yield": operational_hits / 8,
                    "projected_cost_available_count": len(observed_costs),
                    "projected_cost_mean_seconds": (
                        statistics.fmean(observed_costs) if observed_costs else None
                    ),
                    "projected_cost_median_seconds": (
                        statistics.median(observed_costs) if observed_costs else None
                    ),
                    "projected_cost_by_key_seconds": costs,
                }
            )
    return aggregates


__all__ = [
    "PROFILE",
    "fixed_jobs",
    "public_case",
    "validate_public_case",
    "validate_payload",
    "validate_order",
    "assemble_queue",
    "ordered_hash",
    "streaming_sha256",
    "asset_identity",
    "validate_manifest",
    "analyze_cells",
]
