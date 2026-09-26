"""Truth-blind numerical/provenance audit of Bao's VV7 and VV8 HDF5 files.

Run under a temporary TensorFlow/Keras environment. No training or attack
instance is loaded. The original HDF5 files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf

from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_directory", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(260926)
    bits = rng.integers(0, 2, size=(8, 64), dtype=np.uint8).astype(np.float32)
    audits = []
    for rounds in (7, 8):
        path = args.checkpoint_directory / f"ND_VV_Simon32_{rounds}R.h5"
        model, audit = load_published_vv_checkpoint(path, rounds)
        first = model(bits, training=False).numpy()
        repeated = model(bits, training=False).numpy()
        predicted = model.predict(bits, batch_size=4, verbose=0)
        if not np.array_equal(first, repeated):
            raise AssertionError("published VV inference is not deterministic on fixed inputs")
        if not np.allclose(first, predicted, atol=5e-7, rtol=0):
            raise AssertionError("published VV predict order/value differs from direct call")
        if not np.array_equal(np.argsort(first[:, 0]), np.argsort(predicted[:, 0])):
            raise AssertionError("published VV fixed-batch order differs by inference path")
        if not np.all(np.isfinite(first)) or not np.all((first > 0) & (first < 1)):
            raise AssertionError("published VV sigmoid outputs are nonfinite/outside (0,1)")
        preclip = tf.keras.Model(model.inputs, model.get_layer("dense_23").output)
        logits = preclip(bits, training=False).numpy()
        expected = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
        max_clip_error = float(np.max(np.abs(first - expected)))
        if max_clip_error > 1e-7:
            raise AssertionError("safe clipping layer differs from audited formula")
        batch_bits = np.random.default_rng(260926 + rounds).integers(
            0, 2, size=(16384, 64), dtype=np.uint8
        ).astype(np.float32)
        small_batches = model.predict(batch_bits, batch_size=1024, verbose=0)
        large_batches = model.predict(batch_bits, batch_size=16384, verbose=0)
        batch_max_error = float(np.max(np.abs(small_batches - large_batches)))
        if batch_max_error > 5e-7:
            raise AssertionError("VV output changes materially with inference batch size")

        def aggregate(predictions: np.ndarray) -> np.ndarray:
            p = predictions.astype(np.float64)
            return np.log2(p / (1 - p + 1e-64) + 1e-64).reshape(-1, 16).sum(axis=1)

        score_small, score_large = aggregate(small_batches), aggregate(large_batches)
        if set(np.argsort(-score_small)[:16]) != set(np.argsort(-score_large)[:16]):
            raise AssertionError("VV synthetic top-16 changes with inference batch size")
        audit.update(
            {
                "fixed_synthetic_batch_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                "fixed_output_float32_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                "max_clip_formula_error": max_clip_error,
                "predict_direct_max_abs_error": float(np.max(np.abs(first - predicted))),
                "output_min": float(first.min()),
                "output_max": float(first.max()),
                "batch_1024_vs_16384_max_output_abs_error": batch_max_error,
                "batch_1024_vs_16384_max_16_pair_score_abs_error": float(
                    np.max(np.abs(score_small - score_large))
                ),
                "batch_1024_vs_16384_top16_set_same": True,
            }
        )
        audits.append(audit)
    print(
        json.dumps(
            {
                "classification": "ENGINEERING_TRUTH_BLIND_PUBLISHED_CHECKPOINT_AUDIT",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "checkpoints": audits,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
