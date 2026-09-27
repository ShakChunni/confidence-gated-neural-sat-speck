"""Optional pinned-checkpoint checks in a separate TensorFlow 3.12 environment."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.simon.published_vv import load_published_vv_checkpoint


def test_published_vv_loader_rejects_unapproved_round() -> None:
    with pytest.raises(ValueError, match="VV7, VV8, and VV9"):
        load_published_vv_checkpoint("unused.h5", 10)


@pytest.fixture(scope="module")
def checkpoint_directory() -> Path:
    pytest.importorskip("tensorflow")
    pytest.importorskip("h5py")
    raw = os.environ.get("SIMON_BAO_CHECKPOINT_DIR")
    if not raw:
        pytest.skip("set SIMON_BAO_CHECKPOINT_DIR for pinned external checkpoints")
    return Path(raw)


@pytest.mark.parametrize("rounds", [7, 8, 9])
def test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula(
    checkpoint_directory: Path, rounds: int
) -> None:
    import tensorflow as tf

    path = checkpoint_directory / f"ND_VV_Simon32_{rounds}R.h5"
    model, provenance = load_published_vv_checkpoint(path, rounds)
    assert provenance["weight_arrays_exactly_matched"] == 1080
    assert provenance["scalar_parameters_exactly_matched"] == 449457
    bits = np.random.default_rng(260926).integers(0, 2, size=(8, 64), dtype=np.uint8)
    bits = bits.astype(np.float32)
    actual = model(bits, training=False).numpy()
    assert np.array_equal(actual, model(bits, training=False).numpy())
    assert np.all(np.isfinite(actual)) and np.all((actual > 0) & (actual < 1))
    preclip = tf.keras.Model(model.inputs, model.get_layer("dense_23").output)
    logits = preclip(bits, training=False).numpy()
    expected = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_published_vv_loader_rejects_tampered_file(
    checkpoint_directory: Path, tmp_path: Path
) -> None:
    original = checkpoint_directory / "ND_VV_Simon32_7R.h5"
    tampered = tmp_path / original.name
    shutil.copy2(original, tampered)
    with tampered.open("r+b") as handle:
        handle.seek(1024)
        byte = handle.read(1)
        handle.seek(1024)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_published_vv_checkpoint(tampered, 7)
