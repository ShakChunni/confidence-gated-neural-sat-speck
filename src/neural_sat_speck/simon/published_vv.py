"""Audited, safe reconstruction of Bao's published SIMON VV checkpoints.

The original HDF5 files contain a legacy Python-bytecode Lambda. This loader
pins each complete file hash and that Lambda's bytecode hash, replaces only
the audited ``K.clip(x, -10, 10)`` operation in a temporary copy, then checks
every loaded weight against the unmodified source file. No training occurs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_CHECKPOINTS = {
    7: (
        "438a2e4be57370fdcadc85b24970f24670b480434b0efee88c93b401ee2d92f1",
        "35d6a193d376444200fbd071c5c3e387576535105d75718788b03fa75c50714a",
    ),
    8: (
        "f4e780e484f998ac7e3f05ff78adf47e9dcc1a7a405c9aec92bb06a2761a65fc",
        "7287301b72b3ac941b7f49a83343bd2afac6bbc36c84dbb6305902677794b7d3",
    ),
    9: (
        "2363b843592529fe9409dc25dbc0f50c20565973a4a23e54e5478a03113a7070",
        "35d6a193d376444200fbd071c5c3e387576535105d75718788b03fa75c50714a",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def load_published_vv_checkpoint(path: Path | str, rounds: int) -> tuple[Any, dict[str, Any]]:
    """Load VV7/VV8/VV9 safely in a compatible TensorFlow/Keras environment.

    ``tensorflow`` and ``h5py`` are optional dependencies, imported only here.
    The source HDF5 is never edited. Fail closed on provenance or weight drift.
    """
    if rounds not in _CHECKPOINTS:
        raise ValueError("only published VV7, VV8, and VV9 checkpoints are supported")
    try:
        import h5py
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow and h5py are required for VV checkpoint inference") from error

    source = Path(path)
    expected_file_hash, expected_lambda_hash = _CHECKPOINTS[rounds]
    actual_file_hash = _sha256(source)
    if actual_file_hash != expected_file_hash:
        raise ValueError("published VV checkpoint SHA-256 mismatch")
    with h5py.File(source, "r") as handle:
        config = json.loads(handle.attrs["model_config"])
        layers = config["config"]["layers"]
        if len(layers) != 616 or layers[0]["config"].get("batch_input_shape") != [None, 64]:
            raise ValueError("published VV architecture/input shape mismatch")
        if layers[-1]["class_name"] != "Activation" or layers[-1]["config"].get("activation") != "sigmoid":
            raise ValueError("published VV output polarity/activation mismatch")
        lambdas = [layer for layer in layers if layer["class_name"] == "Lambda"]
        if len(lambdas) != 1 or lambdas[0]["name"] != "lambda_1":
            raise ValueError("unexpected published VV Lambda topology")
        original_lambda = lambdas[0]
        serialized = original_lambda["config"]["function"]
        if original_lambda["config"].get("function_type") != "lambda" or not isinstance(serialized, list):
            raise ValueError("unexpected published VV Lambda encoding")
        lambda_hash = hashlib.sha256(base64.b64decode(serialized[0])).hexdigest()
        if lambda_hash != expected_lambda_hash:
            raise ValueError("published VV Lambda bytecode hash mismatch")
        if original_lambda["inbound_nodes"] != [[["dense_23", 0, 0, {}]]]:
            raise ValueError("published VV clipping-layer input changed")
        original_lambda["class_name"] = "Activation"
        original_lambda["config"] = {
            "name": "lambda_1",
            "trainable": True,
            "activation": "safe_clip",
        }

    def safe_clip(values: Any) -> Any:
        return tf.clip_by_value(values, -10.0, 10.0)

    with tempfile.TemporaryDirectory(prefix="simon_vv_safe_") as directory:
        temporary = Path(directory) / source.name
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r+") as handle:
            handle.attrs.modify("model_config", json.dumps(config))
        model = tf.keras.models.load_model(
            temporary,
            compile=False,
            safe_mode=True,
            custom_objects={"safe_clip": safe_clip},
        )

    if model.input_shape != (None, 64) or model.output_shape != (None, 1):
        raise ValueError("loaded VV checkpoint input/output shape mismatch")
    weight_arrays = scalar_parameters = 0
    with h5py.File(source, "r") as handle:
        weights_group = handle["model_weights"]
        for layer in model.layers:
            names = [
                item.decode() if isinstance(item, bytes) else item
                for item in weights_group[layer.name].attrs["weight_names"]
            ]
            loaded_weights = layer.get_weights()
            if len(names) != len(loaded_weights):
                raise ValueError(f"VV layer {layer.name} weight-count mismatch")
            for name, loaded in zip(names, loaded_weights, strict=True):
                if not np.array_equal(weights_group[layer.name][name][()], loaded):
                    raise ValueError(f"VV layer {layer.name} weight-value mismatch")
                weight_arrays += 1
                scalar_parameters += loaded.size
    if weight_arrays != 1080 or scalar_parameters != 449457:
        raise ValueError("VV checkpoint total weight count mismatch")
    return model, {
        "rounds": rounds,
        "sha256": actual_file_hash,
        "lambda_bytecode_sha256": lambda_hash,
        "loaded_layers": len(model.layers),
        "weight_arrays_exactly_matched": weight_arrays,
        "scalar_parameters_exactly_matched": scalar_parameters,
        "safe_lambda_replacement": "tf.clip_by_value(x, -10.0, 10.0)",
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
    }
