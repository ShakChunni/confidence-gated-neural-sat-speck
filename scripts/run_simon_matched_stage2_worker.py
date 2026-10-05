#!/usr/bin/env python3
"""Truth-blind, rank-only worker for the frozen SIMON Stage 2 control.

The worker accepts one checksummed-by-parent JSON payload path.  It never
receives a key, plaintext, label, verifier, solver, or dataset path.  Imports of
the SIMON ranking modules are made only after payload, source, and asset
preflight; a minimal package namespace prevents the normal SIMON initializer
from importing SAT and verification code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from typing import Any

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
)
_FORBIDDEN_IMPORTS = (
    "pycryptosat",
    "neural_sat_speck.simon.sat_model",
    "neural_sat_speck.simon.verification",
    "neural_sat_speck.protocol",
)
_REQUIRED_SOURCE_FILES = frozenset(
    {
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
)
_EXPECTED_ASSETS = ("vv9", "vv8", "ddt")
_FEATURE_BATCH_SIZE = 1024
_INFERENCE_BATCH_SIZE = 16384
_DOMAIN_SIZE = 1 << 16
_SCREEN_PAIRS = 16
_FULL_PAIRS = 64
_RESCORE_TOP = 512
_CHILDREN_PER_PARENT = 128


class _ImportBlocker:
    """Fail closed if ranking code reaches solver or verification modules."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del path, target
        if any(
            fullname == item or fullname.startswith(item + ".")
            for item in _FORBIDDEN_IMPORTS
        ):
            self.attempts.append(fullname)
            raise ImportError(f"rank-only import boundary blocked {fullname}")
        return None


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number is forbidden: {value}")


def _read_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("worker payload is not a regular file")
    raw = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise ValueError("payload checksum sidecar is missing or invalid")
    value = json.loads(
        raw,
        object_pairs_hook=_json_no_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("worker payload must be a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _validate_source_files(payload: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    source_root_raw = payload.get("source_root")
    if not isinstance(source_root_raw, str) or not source_root_raw:
        raise ValueError("source_root must be a nonempty absolute path")
    source_root = Path(source_root_raw)
    if not source_root.is_absolute():
        raise ValueError("source_root must be absolute")
    source_root = source_root.resolve(strict=True)
    records = payload.get("source_files")
    if not isinstance(records, dict):
        raise ValueError("source_files must be an object")

    normalized: dict[str, Path] = {}
    for name, record in records.items():
        if not isinstance(name, str):
            raise ValueError("source_files keys must be relative paths")
        rel = name
        relative_path = Path(rel)
        if relative_path.is_absolute() or ".." in relative_path.parts or not rel:
            raise ValueError(f"invalid source relative path: {rel!r}")
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"malformed source record for {rel}")
        recorded_path = record["path"]
        expected_hash = record["sha256"]
        if not isinstance(recorded_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"malformed source path or hash for {rel}")
        source_path = Path(recorded_path)
        if not source_path.is_absolute():
            raise ValueError(f"source path must be absolute: {rel}")
        source_path = source_path.resolve(strict=True)
        if (
            source_root not in source_path.parents
            or source_path.relative_to(source_root).as_posix() != rel
        ):
            raise ValueError(
                f"source path is outside or misaligned with source_root: {rel}"
            )
        if len(expected_hash) != 64 or any(
            c not in "0123456789abcdef" for c in expected_hash
        ):
            raise ValueError(f"invalid source SHA-256 for {rel}")
        if _sha256_file(source_path) != expected_hash:
            raise ValueError(f"source SHA-256 mismatch for {rel}")
        normalized[rel] = source_path

    missing = _REQUIRED_SOURCE_FILES - normalized.keys()
    if missing:
        raise ValueError(
            f"source manifest omits required worker code: {sorted(missing)}"
        )
    return source_root, normalized


def _validate_assets(
    payload: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, int]]]:
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(_EXPECTED_ASSETS):
        raise ValueError(f"assets must contain exactly {list(_EXPECTED_ASSETS)}")
    paths: dict[str, Path] = {}
    identities: dict[str, dict[str, int]] = {}
    for name in _EXPECTED_ASSETS:
        record = assets[name]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "identity",
        }:
            raise ValueError(f"malformed asset record for {name}")
        raw_path = record["path"]
        digest = record["sha256"]
        expected_identity = record["identity"]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError(f"asset path must be absolute for {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"asset SHA-256 is malformed for {name}")
        if not isinstance(expected_identity, dict) or set(expected_identity) != {
            "device",
            "inode",
            "size",
            "mtime_ns",
        }:
            raise ValueError(f"asset stat identity is malformed for {name}")
        if any(
            type(value) is not int or value < 0 for value in expected_identity.values()
        ):
            raise ValueError(f"asset stat identity contains a non-integer for {name}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"asset is not a regular file: {name}")
        current = _identity(path)
        if current != expected_identity:
            raise ValueError(f"asset stat identity mismatch for {name}")
        paths[name] = path
        identities[name] = current
    return paths, identities


def _bootstrap_minimal_namespace(source_root: Path) -> None:
    """Install namespace packages without executing either eager initializer."""
    src_root = source_root / "src"
    package_path = src_root / "neural_sat_speck"
    simon_path = package_path / "simon"
    if not package_path.is_dir() or not simon_path.is_dir():
        raise FileNotFoundError("source snapshot lacks neural_sat_speck/simon")

    for module_name in ("neural_sat_speck", "neural_sat_speck.simon"):
        existing = sys.modules.get(module_name)
        if existing is not None and getattr(existing, "__file__", None) is not None:
            raise RuntimeError(f"eager package initializer already ran: {module_name}")

    package = types.ModuleType("neural_sat_speck")
    package.__package__ = "neural_sat_speck"
    package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    package.__spec__ = importlib.util.spec_from_loader(
        "neural_sat_speck", loader=None, is_package=True
    )
    simon = types.ModuleType("neural_sat_speck.simon")
    simon.__package__ = "neural_sat_speck.simon"
    simon.__path__ = [str(simon_path)]  # type: ignore[attr-defined]
    simon.__spec__ = importlib.util.spec_from_loader(
        "neural_sat_speck.simon", loader=None, is_package=True
    )
    sys.modules["neural_sat_speck"] = package
    sys.modules["neural_sat_speck.simon"] = simon


def _set_numeric_thread_environment() -> dict[str, str]:
    for variable in _THREAD_ENV_VARS:
        os.environ[variable] = "8" if variable == "TF_NUM_INTRAOP_THREADS" else "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return {variable: os.environ[variable] for variable in _THREAD_ENV_VARS}


def _module_origin(module: Any) -> str:
    origin = getattr(module, "__file__", None)
    if not origin:
        raise RuntimeError(f"module has no file origin: {module!r}")
    return str(Path(origin).resolve())


def _remove_import_blocker(blocker: Any) -> None:
    try:
        sys.meta_path.remove(blocker)
    except ValueError:
        pass


def _import_ranking_modules(
    source_root: Path,
    scorer: str,
    *,
    retain_import_blocker: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scorer not in ("vv9", "ddt7"):
        raise ValueError("smoke scorer must be 'vv9' or 'ddt7'")
    numeric_environment = _set_numeric_thread_environment()
    _bootstrap_minimal_namespace(source_root)
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)
    succeeded = False
    try:
        np = importlib.import_module("numpy")
        tf = None
        h5py = None
        keras = None
        if scorer == "vv9":
            tf = importlib.import_module("tensorflow")
            tf.config.threading.set_intra_op_parallelism_threads(8)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            h5py = importlib.import_module("h5py")
            keras = importlib.import_module("keras")

        modules = {
            "cipher": importlib.import_module("neural_sat_speck.simon.cipher"),
            "ranking_features": importlib.import_module(
                "neural_sat_speck.simon.ranking_features"
            ),
            "published_vv": importlib.import_module(
                "neural_sat_speck.simon.published_vv"
            ),
            "published_vv_ranking": importlib.import_module(
                "neural_sat_speck.simon.published_vv_ranking"
            ),
            "classical_ranking": importlib.import_module(
                "neural_sat_speck.simon.classical_ranking"
            ),
        }
        loaded_forbidden = [
            name
            for name in sys.modules
            if any(
                name == item or name.startswith(item + ".")
                for item in _FORBIDDEN_IMPORTS
            )
        ]
        simon_package = sys.modules["neural_sat_speck.simon"]
        if getattr(simon_package, "__file__", None) is not None:
            raise RuntimeError(
                "SIMON package initializer ran despite namespace bootstrap"
            )
        if blocker.attempts or loaded_forbidden:
            raise RuntimeError(
                f"forbidden rank-worker imports: attempts={blocker.attempts}, loaded={loaded_forbidden}"
            )

        package_versions = {
            "numpy": str(np.__version__),
            "tensorflow": None,
            "keras": None,
            "h5py": None,
        }
        origins = {
            "numpy": _module_origin(np),
            "tensorflow": None,
            "keras": None,
            "h5py": None,
        }
        devices: dict[str, Any] = {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "tensorflow_physical": [],
        }
        tf_threads = None
        if tf is not None and h5py is not None and keras is not None:
            package_versions.update(
                tensorflow=str(tf.__version__),
                keras=str(keras.__version__),
                h5py=str(h5py.__version__),
            )
            origins.update(
                tensorflow=_module_origin(tf),
                keras=_module_origin(keras),
                h5py=_module_origin(h5py),
            )
            devices["tensorflow_physical"] = [
                {"name": device.name, "type": device.device_type}
                for device in tf.config.list_physical_devices()
            ]
            tf_threads = {
                "intra_op": int(tf.config.threading.get_intra_op_parallelism_threads()),
                "inter_op": int(tf.config.threading.get_inter_op_parallelism_threads()),
            }

        imported_rank_modules = sorted(
            name for name in sys.modules if name.startswith("neural_sat_speck.simon.")
        )
        runtime = {
            # Preserve the interpreter's configured lexical venv path. Resolving
            # this symlink changes sys.prefix identity on the pinned TF runtime.
            "executable": str(Path(sys.executable).absolute()),
            "python_version": platform.python_version(),
            "packages": package_versions,
            "origins": origins,
            "devices": devices,
            "threads": {
                "numeric_environment": numeric_environment,
                "tensorflow": tf_threads,
                "feature_batch_size": _FEATURE_BATCH_SIZE,
                "inference_batch_size": _INFERENCE_BATCH_SIZE,
                "candidate_batch_size": _FEATURE_BATCH_SIZE,
            },
            "import_boundary": {
                "minimal_namespace_bootstrap": True,
                "simon_initializer_loaded": False,
                "protocol_package_loaded": "neural_sat_speck.protocol" in sys.modules,
                "forbidden_modules_loaded": loaded_forbidden,
                "forbidden_import_attempts": list(blocker.attempts),
                "ranking_modules_loaded": imported_rank_modules,
            },
        }
        if runtime["import_boundary"]["protocol_package_loaded"]:
            raise RuntimeError("protocol package initializer was imported")
        succeeded = True
        return runtime, {
            **modules,
            "numpy": np,
            "tensorflow": tf,
            "h5py": h5py,
            "keras": keras,
            "import_blocker": blocker,
        }
    finally:
        if not (retain_import_blocker and succeeded):
            _remove_import_blocker(blocker)


def smoke(scorer: str = "vv9") -> dict[str, Any]:
    """Import the real rank-only worker stack without loading data or a model.

    ``vv9`` checks the pinned TensorFlow/Keras/HDF5 runtime. ``ddt7`` checks
    the base Python/NumPy path and intentionally does not import TensorFlow.
    """
    source_root = Path(__file__).resolve().parents[1]
    runtime, _ = _import_ranking_modules(source_root, scorer)
    return runtime


def _load_control_module(source_root: Path, source_files: dict[str, Path]) -> Any:
    relative = "src/neural_sat_speck/followups/stage2_control.py"
    path = source_files[relative]
    spec = importlib.util.spec_from_file_location("_verified_stage2_control", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load verified Stage2 control helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_fixed_job(control: Any, payload: dict[str, Any]) -> tuple[str, str, int]:
    job = payload["job"]
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    if not hasattr(control, "fixed_jobs"):
        raise RuntimeError("verified Stage2 control helper has no fixed_jobs API")
    matches = [candidate for candidate in control.fixed_jobs() if candidate == job]
    if len(matches) != 1:
        raise ValueError("payload job is not one of the fixed Stage2 worker jobs")
    job_id = job["job_id"]
    scorer = job["scorer"]
    operation = job["operation"]
    key_index = job["key_index"]
    if (
        isinstance(key_index, bool)
        or not isinstance(key_index, int)
        or not 0 <= key_index < 8
    ):
        raise ValueError("job key_index must be in 0..7")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a nonempty string")
    if operation == "stage1" and scorer == "vv9":
        if payload["parents"] != []:
            raise ValueError("Stage1 payload must not have parents")
    elif operation == "stage2" and scorer in ("vv8", "ddt7"):
        parents = payload["parents"]
        if not isinstance(parents, list) or len(parents) != 8:
            raise ValueError("Stage2 payload must contain eight ordered parents")
        if any(
            isinstance(word, bool)
            or not isinstance(word, int)
            or not 0 <= word <= 0xFFFF
            for word in parents
        ):
            raise ValueError("Stage2 parent words must be strict 16-bit integers")
        if len(set(parents)) != 8:
            raise ValueError("Stage2 parent words must be distinct")
    else:
        raise ValueError("unsupported Stage2 operation/scorer route")
    public_case = payload["public_case"]
    if public_case["key_index"] != key_index:
        raise ValueError("job key index and public case differ")
    return operation, scorer, key_index


def _load_telemetry_module(source_files: dict[str, Path]) -> Any:
    path = source_files["src/neural_sat_speck/protocol/worker_telemetry.py"]
    spec = importlib.util.spec_from_file_location("_verified_worker_telemetry", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load verified stdlib worker telemetry")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_runtime(
    payload: dict[str, Any],
    scorer: str,
    source_root: Path,
    *,
    retain_import_blocker: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_key = "tensorflow" if scorer in ("vv9", "vv8") else "python"
    expected_runtime = payload["runtime"].get(runtime_key)
    if not isinstance(expected_runtime, dict):
        raise ValueError(f"payload runtime lacks {runtime_key} record")
    runtime, modules = _import_ranking_modules(
        source_root,
        "vv9" if scorer in ("vv9", "vv8") else "ddt7",
        retain_import_blocker=retain_import_blocker,
    )
    if runtime != expected_runtime:
        _remove_import_blocker(modules["import_blocker"])
        raise ValueError(f"runtime metadata differs from preflight {runtime_key} smoke")
    return runtime, modules


def _assert_import_boundary(modules: dict[str, Any]) -> None:
    blocker = modules.get("import_blocker")
    if not isinstance(blocker, _ImportBlocker):
        raise RuntimeError("whole-worker import blocker is unavailable")
    loaded_forbidden = [
        name
        for name in sys.modules
        if any(
            name == item or name.startswith(item + ".") for item in _FORBIDDEN_IMPORTS
        )
    ]
    simon_package = sys.modules.get("neural_sat_speck.simon")
    if blocker.attempts or loaded_forbidden:
        raise RuntimeError(
            f"forbidden rank-worker imports: attempts={blocker.attempts}, loaded={loaded_forbidden}"
        )
    if simon_package is None or getattr(simon_package, "__file__", None) is not None:
        raise RuntimeError("SIMON package initializer ran despite namespace bootstrap")


def _check_no_prior_outputs(directory: Path) -> None:
    if not directory.is_dir():
        raise ValueError("payload directory must already exist")
    for name in ("events", "calls"):
        child = directory / name
        if child.exists() and (not child.is_dir() or any(child.iterdir())):
            raise FileExistsError(f"worker output directory is not empty: {child}")
    for name in (
        "worker_result.json",
        "worker_result.json.sha256",
        "worker_exception.json",
    ):
        if (directory / name).exists():
            raise FileExistsError(f"worker output already exists: {name}")


def _write_checked_json(path: Path, value: Any, *, create_only: bool = True) -> str:
    """Atomically persist JSON and its SHA-256 sidecar; return the data hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and (path.exists() or Path(str(path) + ".sha256").exists()):
        raise FileExistsError(f"refusing to overwrite checked artifact {path}")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(data).hexdigest()
    for target, contents in (
        (path, data),
        (
            Path(str(path) + ".sha256"),
            (f"{digest}  {path.name}\n").encode("ascii"),
        ),
    ):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            if create_only and target.exists():
                raise FileExistsError(
                    f"refusing to overwrite checked artifact {target}"
                )
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest


def _stat_all_assets(
    paths: dict[str, Path], expected: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    actual = {name: _identity(path) for name, path in paths.items()}
    if actual != expected:
        raise RuntimeError(
            "one or more asset identities changed during worker execution"
        )
    return actual


def _load_scorer(
    scorer: str,
    paths: dict[str, Path],
    assets: dict[str, Any],
    modules: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if scorer in ("vv9", "vv8"):
        rounds = 9 if scorer == "vv9" else 8
        asset_name = scorer
        model, metadata = modules["published_vv"].load_published_vv_checkpoint(
            paths[asset_name], rounds
        )
        if (
            metadata.get("rounds") != rounds
            or metadata.get("sha256") != assets[asset_name]["sha256"]
        ):
            raise ValueError("loaded VV model depth or asset hash differs from payload")
        if (
            metadata.get("loaded_layers") != 616
            or metadata.get("weight_arrays_exactly_matched") != 1080
            or metadata.get("scalar_parameters_exactly_matched") != 449457
        ):
            raise ValueError("loaded VV checkpoint structure differs from pinned audit")
        return model, {
            "kind": "published_vv_checkpoint",
            "asset": asset_name,
            **metadata,
        }
    if scorer == "ddt7":
        table = modules["classical_ranking"].FullPredictedDdt(
            paths["ddt"], expected_rounds=7
        )
        if getattr(table, "rounds", None) != 7:
            raise ValueError("loaded DDT is not the complete predicted 7R table")
        return table, {
            "kind": "complete_predicted_ddt",
            "asset": "ddt",
            "expected_rounds": 7,
            "loaded_rounds": int(table.rounds),
            "sha256_preflight": assets["ddt"]["sha256"],
            "loaded_bytes": _identity(paths["ddt"])["size"],
            "loading": "read-only memory map; digest verified by campaign preflight",
        }
    raise ValueError(f"unsupported scorer: {scorer}")


def _component_times(result: Any) -> dict[str, float]:
    names = (
        "feature_seconds",
        "inference_seconds",
        "aggregation_seconds",
        "screen_seconds",
        "row_load_seconds",
        "rescore_seconds",
        "sort_seconds",
        "total_seconds",
    )
    values = {
        name: float(getattr(result, name)) for name in names if hasattr(result, name)
    }
    if "total_seconds" not in values or not values:
        raise ValueError("ranker result lacks required component timing fields")
    if any(not (value >= 0.0 and value < float("inf")) for value in values.values()):
        raise ValueError("ranker returned an invalid component timing")
    return values


def _ciphertext_pairs(public_case: dict[str, Any]) -> list[Any]:
    """Return the only ranking input carried by a public-case payload."""
    pairs = public_case.get("ciphertext_pairs")
    if not isinstance(pairs, list) or len(pairs) != _FULL_PAIRS:
        raise ValueError("public case must contain exactly 64 ciphertext pairs")
    return pairs


def _rank_one(
    scorer: str,
    pairs: list[Any],
    parent: int | None,
    model_or_table: Any,
    modules: dict[str, Any],
) -> tuple[tuple[int, ...], dict[str, float]]:
    if scorer in ("vv9", "vv8"):
        result = modules["published_vv_ranking"].rank_neural_domain(
            pairs,
            model_or_table,
            final_key_candidate=parent,
            screen_pairs=_SCREEN_PAIRS,
            full_pairs=_FULL_PAIRS,
            rescore_top=_RESCORE_TOP,
            feature_batch_size=_FEATURE_BATCH_SIZE,
            inference_batch_size=_INFERENCE_BATCH_SIZE,
        )
    elif scorer == "ddt7":
        if parent is None:
            raise ValueError("DDT7 Stage2 call requires a parent candidate")
        result = modules["classical_ranking"].rank_classical_domain(
            pairs,
            model_or_table,
            final_key_candidate=parent,
            expected_rounds=7,
            screen_pairs=_SCREEN_PAIRS,
            full_pairs=_FULL_PAIRS,
            rescore_top=_RESCORE_TOP,
            batch_size=_FEATURE_BATCH_SIZE,
        )
    else:
        raise ValueError(f"unsupported scorer: {scorer}")
    # NumPy's uint16 rank arrays cross a JSON/stdlib boundary here. tolist()
    # preserves scalar types: floating or boolean orders still fail validation.
    native_order = (
        result.order.tolist() if hasattr(result.order, "tolist") else result.order
    )
    ordered_words = modules["control"].validate_order(native_order)
    if len(ordered_words) != _RESCORE_TOP:
        raise ValueError("ranker must return exactly 512 full-rescored words")
    return ordered_words, _component_times(result)


def _call_schedule(
    operation: str, scorer: str, parents: list[int]
) -> tuple[tuple[int, int | None], ...]:
    if operation == "stage1" and scorer == "vv9" and parents == []:
        return ((0, None),)
    if operation == "stage2" and scorer in ("vv8", "ddt7") and len(parents) == 8:
        return tuple((ordinal, parents[ordinal]) for ordinal in range(8))
    raise ValueError("invalid fixed rank-call schedule")


def _event_path(directory: Path, event_index: int) -> Path:
    return directory / "events" / f"{event_index:04d}.json"


def _write_loaded_event(
    directory: Path,
    *,
    invocation: str,
    job_id: str,
    scorer: str,
    completed: float,
    load_wall_seconds: float,
    telemetry: dict[str, Any],
    runtime: dict[str, Any],
    loader_metadata: dict[str, Any],
    asset_identity: dict[str, dict[str, int]],
) -> None:
    event = {
        "schema": "stage2_rank_worker_event_v1",
        "event": "loaded",
        "event_index": 0,
        "invocation": invocation,
        "job_id": job_id,
        "scorer": scorer,
        "completed_monotonic": completed,
        "load_wall_seconds": load_wall_seconds,
        "telemetry": telemetry,
        "runtime": runtime,
        "loader_metadata": loader_metadata,
        "asset_identity": asset_identity,
    }
    _write_checked_json(_event_path(directory, 0), event)


def _run_worker(payload_path: Path) -> dict[str, Any]:
    worker_started = time.perf_counter()
    payload = _read_payload(payload_path)

    # All source and asset checks happen before project, NumPy, TensorFlow, or
    # model imports. The control helper itself is then loaded from verified
    # source and validates the fixed payload/profile contract.
    source_root, source_files = _validate_source_files(payload)
    asset_paths, expected_asset_identity = _validate_assets(payload)
    control = _load_control_module(source_root, source_files)
    control.validate_payload(payload)
    operation, scorer, key_index = _validate_fixed_job(control, payload)
    invocation = payload["invocation"]
    job_id = payload["job"]["job_id"]
    output_directory = Path(payload["directory"])
    _check_no_prior_outputs(output_directory)

    # Recheck physical identities immediately before importing/loading the
    # scorer. The 7R table digest is intentionally owned by root preflight.
    _stat_all_assets(asset_paths, expected_asset_identity)
    runtime, modules = _check_runtime(
        payload, scorer, source_root, retain_import_blocker=True
    )
    modules["control"] = control

    blocker = modules["import_blocker"]
    try:
        return _execute_loaded_job(
            payload=payload,
            source_root=source_root,
            source_files=source_files,
            asset_paths=asset_paths,
            expected_asset_identity=expected_asset_identity,
            runtime=runtime,
            modules=modules,
            worker_started=worker_started,
            operation=operation,
            scorer=scorer,
            key_index=key_index,
            invocation=invocation,
            job_id=job_id,
            output_directory=output_directory,
        )
    finally:
        _remove_import_blocker(blocker)


def _execute_loaded_job(
    *,
    payload: dict[str, Any],
    source_root: Path,
    source_files: dict[str, Path],
    asset_paths: dict[str, Path],
    expected_asset_identity: dict[str, dict[str, int]],
    runtime: dict[str, Any],
    modules: dict[str, Any],
    worker_started: float,
    operation: str,
    scorer: str,
    key_index: int,
    invocation: str,
    job_id: str,
    output_directory: Path,
) -> dict[str, Any]:
    _assert_import_boundary(modules)

    telemetry_module = _load_telemetry_module(source_files)
    telemetry_record = telemetry_module.certify_worker_or_raise(source_root)
    if telemetry_record.git_commit != payload["source_commit"]:
        raise RuntimeError(
            "worker telemetry git commit differs from payload source_commit"
        )
    telemetry = telemetry_record.to_dict()

    load_started = worker_started
    model_or_table, loader_metadata = _load_scorer(
        scorer, asset_paths, payload["assets"], modules
    )
    _assert_import_boundary(modules)
    loaded_asset_identity = _stat_all_assets(asset_paths, expected_asset_identity)
    loaded_completed = time.perf_counter()
    _write_loaded_event(
        output_directory,
        invocation=invocation,
        job_id=job_id,
        scorer=scorer,
        completed=loaded_completed,
        load_wall_seconds=loaded_completed - load_started,
        telemetry=telemetry,
        runtime=runtime,
        loader_metadata=loader_metadata,
        asset_identity=loaded_asset_identity,
    )

    pairs = _ciphertext_pairs(payload["public_case"])
    parents = payload["parents"]
    schedule = _call_schedule(operation, scorer, parents)
    completed_calls: list[dict[str, Any]] = []
    for ordinal, parent in schedule:
        call_started = time.perf_counter()
        start_record = {
            "schema": "stage2_rank_worker_start_v1",
            "invocation": invocation,
            "job_id": job_id,
            "scorer": scorer,
            "ordinal": ordinal,
            "parent": parent,
            "call_started_monotonic": call_started,
        }
        _write_checked_json(
            output_directory / "calls" / f"{ordinal:04d}_start.json", start_record
        )

        ordered_words, component_times = _rank_one(
            scorer,
            pairs,
            parent,
            model_or_table,
            modules,
        )
        _assert_import_boundary(modules)
        call_completed = time.perf_counter()
        result = {
            "invocation": invocation,
            "job_id": job_id,
            "ordinal": ordinal,
            "scorer": scorer,
            "parent": parent,
            "ordered_words": list(ordered_words),
            "component_times_seconds": component_times,
            "call_started_monotonic": call_started,
            "call_completed_monotonic": call_completed,
        }
        result_path = output_directory / "calls" / f"{ordinal:04d}_result.json"
        result_sha256 = _write_checked_json(result_path, result)
        call_ref = {
            "ordinal": ordinal,
            "parent": parent,
            "result": f"calls/{ordinal:04d}_result.json",
            "result_sha256": result_sha256,
        }
        completed_calls.append(call_ref)

        is_final_call = ordinal == schedule[-1][0]
        if is_final_call:
            exit_asset_identity = _stat_all_assets(asset_paths, expected_asset_identity)
            worker_result = {
                "schema": "stage2_rank_worker_result_v1",
                "status": "complete",
                "invocation": invocation,
                "job_id": job_id,
                "job": payload["job"],
                "operation": operation,
                "scorer": scorer,
                "key_index": key_index,
                "parents": parents,
                "completed_call_count": len(completed_calls),
                "calls": completed_calls,
                "runtime": runtime,
                "telemetry": telemetry,
                "loader_metadata": loader_metadata,
                "asset_identity_at_load": loaded_asset_identity,
                "asset_identity_at_exit": exit_asset_identity,
                "worker_started_monotonic": worker_started,
                "loaded_completed_monotonic": loaded_completed,
                "worker_completed_monotonic": time.perf_counter(),
            }
            _write_checked_json(output_directory / "worker_result.json", worker_result)

        event = {
            "schema": "stage2_rank_worker_event_v1",
            "event": "call_complete",
            "event_index": ordinal + 1,
            "invocation": invocation,
            "job_id": job_id,
            "scorer": scorer,
            "ordinal": ordinal,
            "parent": parent,
            "completed_monotonic": time.perf_counter(),
            "result": f"calls/{ordinal:04d}_result.json",
            "result_path": f"calls/{ordinal:04d}_result.json",
            "result_sha256": result_sha256,
        }
        _write_checked_json(_event_path(output_directory, ordinal + 1), event)
    return {
        "job_id": job_id,
        "invocation": invocation,
        "completed_call_count": len(completed_calls),
        "worker_result": str(output_directory / "worker_result.json"),
    }


def _write_exception_if_possible(payload: Any, error: Exception) -> None:
    if not isinstance(payload, dict):
        return
    raw_directory = payload.get("directory")
    invocation = payload.get("invocation")
    job = payload.get("job")
    if (
        not isinstance(raw_directory, str)
        or not isinstance(invocation, str)
        or not isinstance(job, dict)
    ):
        return
    directory = Path(raw_directory)
    if not directory.is_absolute() or not directory.is_dir():
        return
    path = directory / "worker_exception.json"
    if path.exists() or Path(str(path) + ".sha256").exists():
        return
    try:
        _write_checked_json(
            path,
            {
                "schema": "stage2_rank_worker_exception_v1",
                "invocation": invocation,
                "job_id": job.get("job_id"),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "failed_monotonic": time.perf_counter(),
            },
        )
    except Exception:
        # Preserve the original failure for the guardian/operator. Partial call
        # records and prior valid results remain available for audit.
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "payload", type=Path, help="one stage2_rank_payload_v1 JSON file"
    )
    args = parser.parse_args(argv)
    payload: Any = None
    try:
        payload = _read_payload(args.payload)
        summary = _run_worker(args.payload)
        sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as error:
        _write_exception_if_possible(payload, error)
        sys.stderr.write(f"rank-only worker failed: {type(error).__name__}: {error}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
