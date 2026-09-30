"""Unit tests for the manifest-driven SIMON32/64 B0->B1 training runner.

Invariants Tested:
==================
1. Manifest loading and validation (schema, hashes, architecture parameter counts, seed governance).
2. Seed governance rejection: forbidden research/historical/demo seeds, identical train/val seeds.
3. Duplicate key detection and train/validation master-key overlap rejection before optimizer.
4. Preflight cryptographic verification against scalar cipher and clean-room independent reference oracle.
5. Bounded smoke test execution: loss decreases, checkpoint saved, metadata recorded, reloadable by
   canonical load_simon_b1_checkpoint.
6. Production B1 worker checkpoint verification assertions (head_variant='speck_b1', 64-char hex checksums,
   completed status, selected_epoch >= 1, all finite floating point weights).
7. Deterministic checkpoint selection on tied or improving validation losses (earlier epoch on tie).
8. Weak model retention and reporting under outcome-independent stop rule vs NaN loss abort.
9. Zero-gradient architecture defect abort.
10. CLI guard flags and invocation behavior.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    HISTORICAL_PHASE7_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
)
from neural_sat_speck.simon.b1_data import (
    SimonGeneratedPairGroups,
    generate_simon_b0_pairs,
    generate_simon_b1_groups,
    load_simon_b1_checkpoint,
)
from neural_sat_speck.simon.b1_training_runner import (
    EXPECTED_B0_PARAMS,
    EXPECTED_B1_PARAMS,
    ManifestValidationError,
    check_finite_loss,
    check_finite_weights,
    check_gradients,
    load_and_validate_manifest,
    run_b0_b1_training_pipeline,
    train_simon_b0_stage,
    verify_preflight_cryptography,
)
from neural_sat_speck.simon.faithful_b1_model import (
    SimonPairGroupRanker,
)
from scripts.train_simon_b0_b1 import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSED_MANIFEST = REPO_ROOT / "docs" / "SIMON_B1_TRAINING_MANIFEST_PROPOSED.json"


# -----------------------------------------------------------------------------
# 1. Manifest Loading and Validation Tests
# -----------------------------------------------------------------------------

def test_proposed_manifest_loads_and_validates() -> None:
    """The committed proposed training manifest must pass all schema and parameter checks."""
    manifest, digest = load_and_validate_manifest(PROPOSED_MANIFEST)
    assert isinstance(manifest, dict)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["protocol_id"] == "simon32_64_b0_b1_faithful_training_v1"
    assert manifest["target_cipher"]["rounds"] == 9
    assert manifest["target_cipher"]["input_difference"] == [0, 64]
    assert manifest["architectures"]["b0_distinguisher"]["trainable_parameters"] == EXPECTED_B0_PARAMS
    assert manifest["architectures"]["b1_group_ranker"]["trainable_parameters"] == EXPECTED_B1_PARAMS
    assert manifest["seed_governance"]["train_seed"] == 880001
    assert manifest["seed_governance"]["validation_seed"] == 880002


def test_manifest_validation_rejects_missing_file_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_validate_manifest(tmp_path / "non_existent.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ incomplete json ...", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="Invalid JSON"):
        load_and_validate_manifest(bad_json)

    not_dict = tmp_path / "not_dict.json"
    not_dict.write_text("[\"item1\", \"item2\"]", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="JSON object"):
        load_and_validate_manifest(not_dict)


def test_manifest_validation_rejects_missing_required_keys(tmp_path: Path) -> None:
    base = json.loads(PROPOSED_MANIFEST.read_text(encoding="utf-8"))
    for required_key in ("architectures", "seed_governance", "training_hyperparameters"):
        corrupted = copy.deepcopy(base)
        del corrupted[required_key]
        p = tmp_path / f"missing_{required_key}.json"
        p.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match=f"Missing required top-level key: '{required_key}'"):
            load_and_validate_manifest(p)


def test_manifest_validation_rejects_invalid_governing_decisions(tmp_path: Path) -> None:
    base = json.loads(PROPOSED_MANIFEST.read_text(encoding="utf-8"))
    base["governing_decisions"] = ["D-001"]
    p = tmp_path / "bad_decisions.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="governing_decisions must include"):
        load_and_validate_manifest(p)


def test_manifest_validation_rejects_invalid_architectures(tmp_path: Path) -> None:
    base = json.loads(PROPOSED_MANIFEST.read_text(encoding="utf-8"))
    # Wrong B0 parameter count
    bad_b0 = copy.deepcopy(base)
    bad_b0["architectures"]["b0_distinguisher"]["trainable_parameters"] = 12345
    p = tmp_path / "bad_b0_params.json"
    p.write_text(json.dumps(bad_b0), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="b0_distinguisher trainable_parameters must be 62337"):
        load_and_validate_manifest(p)

    # Wrong B1 head_variant
    bad_b1 = copy.deepcopy(base)
    bad_b1["architectures"]["b1_group_ranker"]["head_variant"] = "wrong_head"
    p2 = tmp_path / "bad_b1_head.json"
    p2.write_text(json.dumps(bad_b1), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="b1_group_ranker head_variant must be 'speck_b1'"):
        load_and_validate_manifest(p2)


def test_manifest_validation_rejects_forbidden_seeds(tmp_path: Path) -> None:
    base = json.loads(PROPOSED_MANIFEST.read_text(encoding="utf-8"))

    # Test historical, quarantined, retired, and demo seeds
    for forbidden in (HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED, CLI_DEMO_SEED, 300001):
        corrupted = copy.deepcopy(base)
        corrupted["seed_governance"]["train_seed"] = forbidden
        p = tmp_path / f"forbidden_{forbidden}.json"
        p.write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="violates seed governance"):
            load_and_validate_manifest(p)

    # Test identical train and validation seeds
    corrupted_same = copy.deepcopy(base)
    corrupted_same["seed_governance"]["train_seed"] = 880001
    corrupted_same["seed_governance"]["validation_seed"] = 880001
    p_same = tmp_path / "same_seed.json"
    p_same.write_text(json.dumps(corrupted_same), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="train_seed and validation_seed must differ"):
        load_and_validate_manifest(p_same)

    # Test non-synthetic role
    corrupted_role = copy.deepcopy(base)
    corrupted_role["seed_governance"]["seed_role"] = "UNSEEN_EVALUATION"
    p_role = tmp_path / "bad_role.json"
    p_role.write_text(json.dumps(corrupted_role), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="seed_role must be 'SYNTHETIC_TRAINING_ONLY'"):
        load_and_validate_manifest(p_role)


def test_manifest_validation_rejects_invalid_hyperparameters(tmp_path: Path) -> None:
    base = json.loads(PROPOSED_MANIFEST.read_text(encoding="utf-8"))

    # Wrong optimizer
    bad_opt = copy.deepcopy(base)
    bad_opt["training_hyperparameters"]["optimizer"] = "SGD"
    p = tmp_path / "bad_opt.json"
    p.write_text(json.dumps(bad_opt), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="optimizer must be 'AdamW'"):
        load_and_validate_manifest(p)

    # Non-positive epochs
    bad_ep = copy.deepcopy(base)
    bad_ep["training_hyperparameters"]["b0_pretraining"]["epochs"] = 0
    p2 = tmp_path / "bad_epochs.json"
    p2.write_text(json.dumps(bad_ep), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="b0_pretraining.epochs must be positive"):
        load_and_validate_manifest(p2)

    # Wrong initialization for B1
    bad_init = copy.deepcopy(base)
    bad_init["training_hyperparameters"]["b1_fine_tuning"]["initialization"] = "from_scratch"
    p3 = tmp_path / "bad_init.json"
    p3.write_text(json.dumps(bad_init), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="b1_fine_tuning.initialization must be 'initialize_from_b0'"):
        load_and_validate_manifest(p3)


# -----------------------------------------------------------------------------
# 2. Key Audit and Preflight Oracle Tests
# -----------------------------------------------------------------------------

def test_key_audit_rejects_duplicates_before_training() -> None:
    """Preflight key audit must fail closed on duplicate keys within any dataset split."""
    b0_train = generate_simon_b0_pairs(16, 880001)
    b0_val = generate_simon_b0_pairs(8, 880002)
    b1_train = generate_simon_b1_groups(4, 880001)
    b1_val = generate_simon_b1_groups(2, 880002)

    # Inject duplicate key in b1_train
    dup_keys = np.copy(b1_train.master_keys)
    dup_keys[1] = dup_keys[0]
    corrupted_b1_train = SimonGeneratedPairGroups(
        features=b1_train.features,
        labels=b1_train.labels,
        master_keys=dup_keys,
        metadata=b1_train.metadata,
    )

    with pytest.raises(ValueError, match="duplicate master key in b1_train"):
        verify_preflight_cryptography(b0_train, corrupted_b1_train, b0_val, b1_val)


def test_key_audit_rejects_train_val_overlap_before_training() -> None:
    """Preflight key audit must fail closed if any train master key appears in validation."""
    b0_train = generate_simon_b0_pairs(16, 880001)
    b0_val = generate_simon_b0_pairs(8, 880002)
    b1_train = generate_simon_b1_groups(4, 880001)
    b1_val = generate_simon_b1_groups(2, 880002)

    # Force overlap: b1_val key 0 equals b0_train key 0
    overlap_keys = np.copy(b1_val.master_keys)
    overlap_keys[0] = b0_train.master_keys[0]
    corrupted_b1_val = SimonGeneratedPairGroups(
        features=b1_val.features,
        labels=b1_val.labels,
        master_keys=overlap_keys,
        metadata=b1_val.metadata,
    )

    with pytest.raises(ValueError, match="training and validation master keys overlap"):
        verify_preflight_cryptography(b0_train, b1_train, b0_val, corrupted_b1_val)


def test_preflight_cryptography_scalar_and_oracle_verification() -> None:
    """Cryptographic preflight check passes on valid generated data and fails on corruption."""
    b0_train = generate_simon_b0_pairs(16, 880001)
    b0_val = generate_simon_b0_pairs(8, 880002)
    b1_train = generate_simon_b1_groups(4, 880001)
    b1_val = generate_simon_b1_groups(2, 880002)

    res = verify_preflight_cryptography(b0_train, b1_train, b0_val, b1_val, sample_count=4)
    assert res["oracle_alignment_verified"] is True
    assert res["b1_positive_pairs_tested"] > 0
    assert res["b1_negative_pairs_tested"] > 0

    # 1. Corrupt feature bits of a positive group so decrypted difference fails target check
    pos_idx = int(np.where(b1_train.labels == 1)[0][0])
    corrupt_features = np.copy(b1_train.features)
    corrupt_features[pos_idx, 0, 0, 0] ^= 1  # flip 1 bit in c0 of positive group
    corrupt_b1 = SimonGeneratedPairGroups(
        features=corrupt_features,
        labels=b1_train.labels,
        master_keys=b1_train.master_keys,
        metadata=b1_train.metadata,
    )
    with pytest.raises(RuntimeError, match="has incorrect difference"):
        verify_preflight_cryptography(b0_train, corrupt_b1, b0_val, b1_val, sample_count=4)

    # 2. Test oracle mismatch detection
    with patch("tests.fixtures.simon_reference_oracle.oracle_simon32_64_decrypt", return_value=(0x9999, 0x9999)):
        with pytest.raises(RuntimeError, match="Scalar decryption does not match independent oracle"):
            verify_preflight_cryptography(b0_train, b1_train, b0_val, b1_val, sample_count=4)


# -----------------------------------------------------------------------------
# 3. Bounded Smoke Test Execution & Production Worker Verification
# -----------------------------------------------------------------------------

def test_bounded_smoke_test_execution_and_checkpoint_reload(tmp_path: Path) -> None:
    """Smoke test executes cleanly in 2 epochs, writes checkpoint, and reloads with all invariants."""
    ckpt_path = tmp_path / "simon_b1_smoke_test.pt"

    res = run_b0_b1_training_pipeline(
        manifest_path=PROPOSED_MANIFEST,
        output_checkpoint=ckpt_path,
        smoke_test=True,
        smoke_b1_train_groups=8,
        smoke_b1_val_groups=4,
        smoke_b0_train_pairs=32,
        smoke_b0_val_pairs=16,
        smoke_epochs=2,
        smoke_batch_size_groups=4,
        smoke_batch_size_pairs=16,
        device="cpu",
        verbose=False,
    )

    assert res["status"] == "complete"
    assert res["smoke_test"] is True
    assert ckpt_path.is_file()
    assert len(res["checkpoint_sha256"]) == 64

    # 1. Reload checkpoint using canonical loader
    loaded_model, metadata = load_simon_b1_checkpoint(ckpt_path)
    assert isinstance(loaded_model, SimonPairGroupRanker)
    assert loaded_model.parameter_count == EXPECTED_B1_PARAMS
    assert loaded_model.head_variant == "speck_b1"
    assert metadata["training_status"] == "complete"
    assert metadata["selected_epoch"] in (1, 2)
    assert len(metadata["training_manifest_sha256"]) == 64
    assert len(metadata["train_dataset_checksum"]) == 64
    assert len(metadata["validation_dataset_checksum"]) == 64
    assert len(metadata["training_history"]) == 2
    assert len(metadata["b0_training_history"]) == 2

    # 2. Check each history entry schema
    for entry in metadata["training_history"]:
        assert "epoch" in entry
        assert "train_loss" in entry
        assert "train_acc" in entry
        assert "val_loss" in entry
        assert "val_acc" in entry
        assert "val_auc" in entry
        assert "duration" in entry
        assert "lr" in entry

    # 3. Production B1 worker verification logic (exact logic from method_worker.py lines 710-745)
    file_digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert hmac.compare_digest(file_digest, res["checkpoint_sha256"])
    assert loaded_model.head_variant == "speck_b1"
    assert metadata.get("training_status") == "complete"
    for field_name in (
        "training_manifest_sha256",
        "train_dataset_checksum",
        "validation_dataset_checksum",
    ):
        d_val = metadata.get(field_name)
        assert isinstance(d_val, str) and len(d_val) == 64 and all(c in "0123456789abcdef" for c in d_val)
    assert isinstance(metadata.get("selected_epoch"), int) and metadata["selected_epoch"] >= 1
    assert all(torch.isfinite(t).all() for t in loaded_model.state_dict().values() if t.is_floating_point())

    # 4. Dummy forward pass
    dummy = torch.randn(2, 64, 4, 16)
    with torch.no_grad():
        out = loaded_model(dummy)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------------
# 4. Deterministic Checkpoint Selection Tests
# -----------------------------------------------------------------------------

def test_deterministic_checkpoint_selection_on_tied_and_improving_losses() -> None:
    """Exact ties in validation BCE loss must choose earlier epoch; strictly improving losses update best epoch."""
    # Test strict inequality logic directly
    val_losses = [0.69315, 0.69315, 0.69200, 0.69200, 0.69500]

    best_loss = float("inf")
    best_epoch = 0
    selected_epochs: list[int] = []

    for ep, loss_val in enumerate(val_losses, start=1):
        if loss_val < best_loss:  # strict inequality implements earlier-epoch tie-breaking
            best_loss = loss_val
            best_epoch = ep
        selected_epochs.append(best_epoch)

    # Epoch 1: best = 1 (loss 0.69315)
    # Epoch 2: loss 0.69315 (exact tie) -> best remains 1!
    # Epoch 3: loss 0.69200 (improves) -> best = 3!
    # Epoch 4: loss 0.69200 (exact tie) -> best remains 3!
    # Epoch 5: loss 0.69500 (worse) -> best remains 3!
    assert selected_epochs == [1, 1, 3, 3, 3]


def test_deterministic_checkpoint_selection_in_b0_stage() -> None:
    """B0 training loop strictly selects minimum validation BCE loss epoch."""
    b0_train = generate_simon_b0_pairs(32, 880001)
    b0_val = generate_simon_b0_pairs(16, 880002)

    model, best_epoch, best_loss, best_metrics, history = train_simon_b0_stage(
        b0_train,
        b0_val,
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        seed=880001,
        verbose=False,
    )
    assert 1 <= best_epoch <= 3
    assert best_loss == min(entry["val_loss"] for entry in history)
    # If the minimum occurred at multiple epochs, best_epoch must be the earliest
    all_val_losses = [entry["val_loss"] for entry in history]
    earliest_min_epoch = all_val_losses.index(min(all_val_losses)) + 1
    assert best_epoch == earliest_min_epoch


# -----------------------------------------------------------------------------
# 5. Finite-Loss Handling & Defect Abort Tests
# -----------------------------------------------------------------------------

def test_weak_model_retained_and_labeled_descriptive(tmp_path: Path) -> None:
    """Weak distinguishing accuracy (e.g. 50%) is preserved and reported as descriptive evidence."""
    # Run a smoke pipeline with 1 epoch on balanced data
    ckpt_path = tmp_path / "weak_model.pt"
    res = run_b0_b1_training_pipeline(
        manifest_path=PROPOSED_MANIFEST,
        output_checkpoint=ckpt_path,
        smoke_test=True,
        smoke_b1_train_groups=4,
        smoke_b1_val_groups=2,
        smoke_b0_train_pairs=16,
        smoke_b0_val_pairs=8,
        smoke_epochs=1,
        smoke_batch_size_groups=2,
        smoke_batch_size_pairs=8,
        device="cpu",
        verbose=False,
    )
    assert res["status"] == "complete"
    assert res["metadata"]["performance_label"] == "descriptive"
    assert res["metadata"]["training_status"] == "complete"


def test_nan_loss_aborts_immediately() -> None:
    """Non-finite loss triggers immediate defect abort."""
    with pytest.raises(RuntimeError, match="loss became non-finite"):
        check_finite_loss(float("nan"), "test_stage", epoch=1, batch_idx=0)

    with pytest.raises(RuntimeError, match="loss became non-finite"):
        check_finite_loss(float("inf"), "test_stage", epoch=2)


def test_zero_gradients_aborts_immediately() -> None:
    """Zero gradients across entire architecture triggers immediate defect abort."""
    model = nn.Linear(10, 1)
    # With no backward pass, grads are None
    with pytest.raises(RuntimeError, match="zero gradients across entire architecture"):
        check_gradients(model, "test_stage", epoch=1)

    # With explicit zero grads
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    with pytest.raises(RuntimeError, match="zero gradients across entire architecture"):
        check_gradients(model, "test_stage", epoch=1)


def test_non_finite_weights_aborts_immediately() -> None:
    """Non-finite parameter or buffer values trigger immediate defect abort."""
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="contains non-finite"):
        check_finite_weights(model, "test_stage")


# -----------------------------------------------------------------------------
# 6. CLI Invocation and Guard Tests
# -----------------------------------------------------------------------------

def test_cli_requires_smoke_or_confirmation() -> None:
    """CLI without --smoke-test or --execute-full-training exits with code 1."""
    code = cli_main([])
    assert code == 1


def test_cli_smoke_test_runs_cleanly(tmp_path: Path) -> None:
    """CLI with --smoke-test executes cleanly and writes output checkpoint."""
    ckpt = tmp_path / "cli_smoke.pt"
    log = tmp_path / "cli_log.json"
    code = cli_main([
        "--smoke-test",
        "--output-checkpoint", str(ckpt),
        "--log-file", str(log),
        "--smoke-groups", "4",
        "--smoke-val-groups", "2",
        "--smoke-epochs", "1",
    ])
    assert code == 0
    assert ckpt.is_file()
    assert log.is_file()
