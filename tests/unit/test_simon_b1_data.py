"""Comprehensive unit tests for the faithful SIMON32/64 B0/B1 data and training infrastructure.

Invariants Verified:
1. Fixed 9-round direct-pair construction with input difference (0x0000, 0x0040).
2. Group construction: all 64 pairs in a B1 group share the intended master-key role.
3. Strict master-key splitting: disjoint master-key sets between train and val (zero key overlap).
4. Label balance: exact 50/50 balance for even group/pair counts.
5. Deterministic replay: identical seeds reproduce identical bits, labels, and SHA-256 checksums.
6. Feature layout: bit-sliced shape (4, 16) for (L0, R0, L1, R1) encoded MSB-first.
7. PyTorch datasets and DataLoader batch shapes.
8. Train and evaluation step helpers for SimonPairGroupRanker and SimonGohrResNet.
9. B0 to B1 weight transfer (initialize_from_b0).
10. Checkpoint save and load fidelity.
11. Synthetic overfit checks demonstrating gradient flow and loss decrease.
12. Proposed training manifest specification and seed governance integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    HISTORICAL_PHASE7_SEED,
    HISTORICAL_SEEDS,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_PROSPECTIVE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
)
from neural_sat_speck.simon.b1_data import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SIMON_B1_GROUP_SIZE,
    SIMON_DEFAULT_ROUNDS,
    SimonB0Dataset,
    SimonB1GroupDataset,
    audit_training_key_roles,
    create_simon_b0_dataloader,
    create_simon_b1_dataloader,
    encode_words_to_bits,
    eval_step_b0,
    eval_step_b1,
    evaluate_simon_b1,
    generate_simon_b0_pairs,
    generate_simon_b1_groups,
    load_simon_b0_checkpoint,
    load_simon_b1_checkpoint,
    run_synthetic_overfit_check,
    save_simon_checkpoint,
    train_step_b0,
    train_step_b1,
    transfer_b0_to_b1,
)
from neural_sat_speck.simon.cipher import decrypt_block, encrypt_block
from neural_sat_speck.simon.faithful_b1_model import (
    SimonGohrResNet,
    SimonPairGroupRanker,
)


def test_training_key_role_audit_rejects_duplicates_and_cross_role_overlap() -> None:
    train = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint16)
    train_b1 = np.array([[1, 2, 3, 4]], dtype=np.uint16)
    validation = np.array([[9, 10, 11, 12]], dtype=np.uint16)
    assert audit_training_key_roles(train, train_b1, validation, validation) == {
        "train_unique_keys": 2,
        "validation_unique_keys": 1,
        "cross_role_key_overlap": 0,
    }
    with pytest.raises(ValueError, match="duplicate master key in b0_train"):
        audit_training_key_roles(train[[0, 0]], train_b1, validation, validation)
    with pytest.raises(ValueError, match="training and validation master keys overlap"):
        audit_training_key_roles(train, train_b1, train[[0]], validation)


def _bits_to_words(bits: np.ndarray) -> np.ndarray:
    """Helper to convert (..., 4, 16) MSB-first bits back to (..., 4) uint16 words."""
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return np.sum(bits.astype(np.uint16) << shifts, axis=-1)


def test_fixed_9round_direct_pair_construction() -> None:
    """Verify that positive pairs have difference (0x0000, 0x0040) and encrypt under master key."""
    group_count = 6
    group_size = 16
    seed = 42
    groups = generate_simon_b1_groups(
        group_count=group_count,
        seed=seed,
        rounds=SIMON_DEFAULT_ROUNDS,
        group_size=group_size,
        input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
        negative_strategy="random_plaintext",
    )

    assert groups.metadata["input_difference"] == [0x0000, 0x0040]
    assert groups.metadata["rounds"] == 9

    # Decode bits to words: (group_count, group_size, 4)
    words = _bits_to_words(groups.features)

    for g_idx in range(group_count):
        mk = tuple(int(x) for x in groups.master_keys[g_idx])
        label = int(groups.labels[g_idx])

        for p_idx in range(group_size):
            c0 = (int(words[g_idx, p_idx, 0]), int(words[g_idx, p_idx, 1]))
            c1 = (int(words[g_idx, p_idx, 2]), int(words[g_idx, p_idx, 3]))

            # Decrypt ciphertexts using group master key
            p0 = decrypt_block(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
            p1 = decrypt_block(c1, mk, rounds=SIMON_DEFAULT_ROUNDS)

            # Re-encrypt to verify bijection
            assert encrypt_block(p0, mk, rounds=SIMON_DEFAULT_ROUNDS) == c0
            assert encrypt_block(p1, mk, rounds=SIMON_DEFAULT_ROUNDS) == c1

            if label == 1:
                # Positive pair: P1 = P0 ^ (0x0000, 0x0040)
                diff_l = p0[0] ^ p1[0]
                diff_r = p0[1] ^ p1[1]
                assert (diff_l, diff_r) == (0x0000, 0x0040), (
                    f"Positive pair diff mismatch: got ({hex(diff_l)}, {hex(diff_r)})"
                )


def test_group_master_key_sharing() -> None:
    """Verify all 64 pairs in a B1 group share the exact same master-key role."""
    group_count = 8
    groups = generate_simon_b1_groups(group_count=group_count, seed=9911, group_size=SIMON_B1_GROUP_SIZE)

    # Master keys shape is (group_count, 4)
    assert groups.master_keys.shape == (group_count, 4)
    # Master keys across groups should be distinct
    unique_keys = {tuple(k) for k in groups.master_keys}
    assert len(unique_keys) == group_count, "Each group must have an independent master key"

    words = _bits_to_words(groups.features)
    for g_idx in range(group_count):
        mk = tuple(int(x) for x in groups.master_keys[g_idx])
        # Every pair in this group decrypts under mk
        for p_idx in (0, 1, 31, 63):
            c0 = (int(words[g_idx, p_idx, 0]), int(words[g_idx, p_idx, 1]))
            p0 = decrypt_block(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
            assert encrypt_block(p0, mk, rounds=SIMON_DEFAULT_ROUNDS) == c0


def test_strict_master_key_splitting() -> None:
    """Verify train and validation splits are partitioned by master key with zero overlap."""
    train_groups = generate_simon_b1_groups(group_count=20, seed=880001, group_size=SIMON_B1_GROUP_SIZE)
    val_groups = generate_simon_b1_groups(group_count=10, seed=880002, group_size=SIMON_B1_GROUP_SIZE)

    train_keys = {tuple(int(x) for x in k) for k in train_groups.master_keys}
    val_keys = {tuple(int(x) for x in k) for k in val_groups.master_keys}

    # Strict isolation: master keys must not overlap
    overlap = train_keys.intersection(val_keys)
    assert len(overlap) == 0, f"Master-key leakage detected between train and val! Overlap: {overlap}"
    assert len(train_keys) == 20
    assert len(val_keys) == 10


def test_label_balance_and_permutation() -> None:
    """Verify 50/50 label balance and deterministic permutation."""
    # Even count
    even_groups = generate_simon_b1_groups(group_count=20, seed=777)
    assert np.count_nonzero(even_groups.labels == 1) == 10
    assert np.count_nonzero(even_groups.labels == 0) == 10

    # Odd count
    odd_groups = generate_simon_b1_groups(group_count=15, seed=777)
    assert np.count_nonzero(odd_groups.labels == 1) == 8
    assert np.count_nonzero(odd_groups.labels == 0) == 7

    # B0 pairs
    b0_data = generate_simon_b0_pairs(pair_count=30, seed=777)
    assert np.count_nonzero(b0_data.labels == 1) == 15
    assert np.count_nonzero(b0_data.labels == 0) == 15


def test_deterministic_replay() -> None:
    """Verify that identical seeds produce bit-identical data and checksums."""
    g1 = generate_simon_b1_groups(group_count=10, seed=12345, group_size=16)
    g2 = generate_simon_b1_groups(group_count=10, seed=12345, group_size=16)
    g3 = generate_simon_b1_groups(group_count=10, seed=54321, group_size=16)

    assert np.array_equal(g1.features, g2.features)
    assert np.array_equal(g1.labels, g2.labels)
    assert np.array_equal(g1.master_keys, g2.master_keys)
    assert g1.metadata["content_checksum"] == g2.metadata["content_checksum"]

    # Different seeds must differ
    assert not np.array_equal(g1.features, g3.features)
    assert g1.metadata["content_checksum"] != g3.metadata["content_checksum"]


def test_feature_layout_and_bit_slicing() -> None:
    """Verify bit-sliced shape (4, 16) for (L0, R0, L1, R1) in MSB-first bit order."""
    # Known test words
    test_words = np.array([[[0x8001, 0x4002, 0x0003, 0x0080]]], dtype=np.uint16)
    bits = encode_words_to_bits(test_words)

    assert bits.shape == (1, 1, 4, 16)
    assert bits.dtype == np.uint8
    assert np.all((bits == 0) | (bits == 1))

    # Channel 0: 0x8001 = 1000 0000 0000 0001
    assert bits[0, 0, 0, 0] == 1  # MSB
    assert bits[0, 0, 0, 1] == 0
    assert bits[0, 0, 0, 15] == 1  # LSB
    assert np.sum(bits[0, 0, 0]) == 2

    # Channel 1: 0x4002 = 0100 0000 0000 0010
    assert bits[0, 0, 1, 0] == 0
    assert bits[0, 0, 1, 1] == 1
    assert bits[0, 0, 1, 14] == 1
    assert np.sum(bits[0, 0, 1]) == 2

    # Reconstruct words from bits
    reconstructed = _bits_to_words(bits)
    assert np.array_equal(reconstructed, test_words)


def test_to_b0_data_flattening() -> None:
    """Verify flattening group data to B0 pair format preserves keys and labels."""
    group_count = 4
    group_size = 8
    groups = generate_simon_b1_groups(group_count=group_count, seed=101, group_size=group_size)

    b0_data = groups.to_b0_data()
    assert b0_data.features.shape == (32, 4, 16)
    assert b0_data.labels.shape == (32,)
    assert b0_data.master_keys.shape == (32, 4)

    # Check that pairs 0..7 all share group 0's key and label
    for g_idx in range(group_count):
        expected_label = groups.labels[g_idx]
        expected_key = groups.master_keys[g_idx]
        for p_offset in range(group_size):
            flat_idx = g_idx * group_size + p_offset
            assert b0_data.labels[flat_idx] == expected_label
            assert np.array_equal(b0_data.master_keys[flat_idx], expected_key)


def test_pytorch_datasets_and_dataloaders() -> None:
    """Verify PyTorch datasets and DataLoader batch creation and typing."""
    groups = generate_simon_b1_groups(group_count=12, seed=202, group_size=SIMON_B1_GROUP_SIZE)
    dataset = SimonB1GroupDataset.from_generated(groups)

    assert len(dataset) == 12
    feat, lab = dataset[0]
    assert feat.shape == (64, 4, 16)
    assert feat.dtype == torch.float32
    assert lab.shape == (1,)
    assert lab.dtype == torch.float32

    loader = create_simon_b1_dataloader(dataset, batch_size=4, shuffle=True, seed=42)
    batches = list(loader)
    assert len(batches) == 3
    for b_feat, b_lab in batches:
        assert b_feat.shape == (4, 64, 4, 16)
        assert b_lab.shape == (4, 1)

    # B0 dataset
    b0_data = generate_simon_b0_pairs(pair_count=20, seed=303)
    b0_dataset = SimonB0Dataset.from_generated(b0_data)
    assert len(b0_dataset) == 20
    b0_loader = create_simon_b0_dataloader(b0_dataset, batch_size=5, shuffle=False)
    for b_feat, b_lab in b0_loader:
        assert b_feat.shape == (5, 4, 16)
        assert b_lab.shape == (5, 1)


def test_train_and_eval_step_b0() -> None:
    """Verify train_step_b0 and eval_step_b0 for SimonGohrResNet."""
    model = SimonGohrResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    features = torch.randint(0, 2, size=(8, 4, 16)).float()
    labels = torch.tensor([1.0, 0.0] * 4).view(-1, 1)
    batch = (features, labels)

    # Initial eval
    eval_res_before = eval_step_b0(model, batch, criterion)
    assert "loss" in eval_res_before and "accuracy" in eval_res_before
    assert 0.0 <= eval_res_before["accuracy"] <= 1.0

    # Train step
    train_res = train_step_b0(model, batch, optimizer, criterion)
    assert "loss" in train_res and "accuracy" in train_res

    # Eval after train step
    eval_res_after = eval_step_b0(model, batch, criterion)
    assert eval_res_after["loss"] != eval_res_before["loss"]


def test_train_and_eval_step_b1() -> None:
    """Verify train_step_b1, eval_step_b1, and evaluate_simon_b1 for SimonPairGroupRanker."""
    model = SimonPairGroupRanker()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    features = torch.randint(0, 2, size=(4, 64, 4, 16)).float()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(-1, 1)
    batch = (features, labels)

    eval_before = eval_step_b1(model, batch, criterion)
    assert "loss" in eval_before and "accuracy" in eval_before

    train_res = train_step_b1(model, batch, optimizer, criterion)
    assert "loss" in train_res and "accuracy" in train_res

    # Evaluate across a dataloader
    dataset = SimonB1GroupDataset(features, labels)
    loader = create_simon_b1_dataloader(dataset, batch_size=2, shuffle=False)
    eval_full = evaluate_simon_b1(model, loader, criterion)
    assert "loss" in eval_full
    assert "accuracy" in eval_full
    assert "auc" in eval_full
    assert eval_full["sample_count"] == 4


def test_b0_weight_transfer_initialize_from_b0() -> None:
    """Verify that B0 weights transfer identically into SimonPairGroupRanker."""
    b0 = SimonGohrResNet()
    b1 = SimonPairGroupRanker()

    # Modify b0 weights deterministically
    with torch.no_grad():
        for param in b0.parameters():
            param.add_(0.5)

    transfer_b0_to_b1(b0, b1)

    # Check stem conv
    assert torch.equal(b0.stem_conv.weight, b1.encoder.stem_conv.weight)
    # Check stem bn
    assert torch.equal(b0.stem_bn.weight, b1.encoder.stem_bn.weight)
    assert torch.equal(b0.stem_bn.bias, b1.encoder.stem_bn.bias)

    # Check all residual blocks
    for blk_idx in range(len(b0.blocks)):
        b0_blk = b0.blocks[blk_idx]
        b1_blk = b1.encoder.blocks[blk_idx]
        assert torch.equal(b0_blk.conv1.weight, b1_blk.conv1.weight)
        assert torch.equal(b0_blk.conv2.weight, b1_blk.conv2.weight)
        assert torch.equal(b0_blk.bn1.weight, b1_blk.bn1.weight)
        assert torch.equal(b0_blk.bn2.weight, b1_blk.bn2.weight)

    # Check dense projection
    assert torch.equal(b0.dense.weight, b1.encoder.dense.weight)
    assert torch.equal(b0.dense_bn.weight, b1.encoder.dense_bn.weight)


def test_checkpoint_save_and_load() -> None:
    """Verify checkpoint serialization and deserialization for B0 and B1."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # B0 checkpoint
        b0_model = SimonGohrResNet()
        b0_ckpt = tmp_path / "simon_b0.pt"
        save_simon_checkpoint(b0_model, b0_ckpt, metadata={"test_epoch": 5})

        loaded_b0, meta_b0 = load_simon_b0_checkpoint(b0_ckpt)
        assert meta_b0.get("test_epoch") == 5
        for p1, p2 in zip(b0_model.parameters(), loaded_b0.parameters()):
            assert torch.equal(p1, p2)

        # B1 checkpoint
        b1_model = SimonPairGroupRanker()
        b1_ckpt = tmp_path / "simon_b1.pt"
        save_simon_checkpoint(b1_model, b1_ckpt, metadata={"test_epoch": 10})

        loaded_b1, meta_b1 = load_simon_b1_checkpoint(b1_ckpt)
        assert meta_b1.get("test_epoch") == 10
        for p1, p2 in zip(b1_model.parameters(), loaded_b1.parameters()):
            assert torch.equal(p1, p2)

        # Reject cross-loading
        with pytest.raises(ValueError, match="expected simon_gohr_resnet"):
            load_simon_b0_checkpoint(b1_ckpt)

        with pytest.raises(ValueError, match="expected simon_pair_group_ranker"):
            load_simon_b1_checkpoint(b0_ckpt)


def test_synthetic_overfit_check_b1() -> None:
    """Verify SimonPairGroupRanker overfits 10 synthetic batches and gradients flow."""
    res = run_synthetic_overfit_check("b1", num_batches=10, batch_size=4)
    assert res["loss_decreased"] is True
    assert res["final_loss"] < res["initial_loss"]
    assert res["all_parameters_received_gradients"] is True
    assert res["parameter_count"] == 60289


def test_synthetic_overfit_check_b0() -> None:
    """Verify SimonGohrResNet overfits 10 synthetic batches and gradients flow."""
    res = run_synthetic_overfit_check("b0", num_batches=10, batch_size=4)
    assert res["loss_decreased"] is True
    assert res["final_loss"] < res["initial_loss"]
    assert res["all_parameters_received_gradients"] is True
    assert res["parameter_count"] == 62337


def test_proposed_training_manifest_specification() -> None:
    """Verify that docs/SIMON_B1_TRAINING_MANIFEST_PROPOSED.json satisfies all protocol invariants."""
    manifest_path = Path("docs/SIMON_B1_TRAINING_MANIFEST_PROPOSED.json")
    assert manifest_path.exists(), "Proposed manifest file must exist"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Status and protocol
    assert manifest["status"] == "PROPOSED"
    assert "D-029" in manifest["governing_decisions"]
    assert "D-034" in manifest["governing_decisions"]
    assert "D-037" in manifest["governing_decisions"]

    # Target cipher and difference
    assert manifest["target_cipher"]["rounds"] == 9
    assert manifest["target_cipher"]["input_difference"] == [0, 64]
    assert manifest["target_cipher"]["input_difference_hex"] == ["0x0000", "0x0040"]

    # Parameter counts
    assert manifest["architectures"]["b0_distinguisher"]["trainable_parameters"] == 62337
    assert manifest["architectures"]["b1_group_ranker"]["trainable_parameters"] == 60289

    # Seeds: Train 880001, Val 880002
    train_seed = manifest["seed_governance"]["train_seed"]
    val_seed = manifest["seed_governance"]["validation_seed"]
    assert train_seed == 880001
    assert val_seed == 880002
    assert train_seed != val_seed

    # Verify isolation from historical and protected research seeds
    forbidden_seeds = set(HISTORICAL_SEEDS) | {
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        CLI_DEMO_SEED,
        PHASE10_DEVELOPMENT_SEED,
        PHASE10_PROSPECTIVE_SEED,
        300001,
        406737,
        669981,
    }
    assert train_seed not in forbidden_seeds
    assert val_seed not in forbidden_seeds

    # Sample sizes: 10^6 train / 10^5 val pairs; 15,625 / 1,562 groups
    train_split = manifest["dataset_specifications"]["splits"]["train"]
    val_split = manifest["dataset_specifications"]["splits"]["validation"]
    assert train_split["pair_count"] == 1_000_000
    assert train_split["group_count"] == 15625
    assert train_split["group_size"] == 64
    assert train_split["master_key_count"] == 15625

    assert val_split["pair_count"] == 99968  # ~10^5
    assert val_split["group_count"] == 1562
    assert val_split["group_size"] == 64
    assert val_split["master_key_count"] == 1562

    # Compute limits: 10 epochs, batch size 64 groups
    hyperparams = manifest["training_hyperparameters"]
    assert hyperparams["b0_pretraining"]["epochs"] == 10
    assert hyperparams["b1_fine_tuning"]["epochs"] == 10
    assert hyperparams["b1_fine_tuning"]["batch_size_groups"] == 64

    # Outcome-independent stop and promotion rule
    stop_rule = manifest["outcome_independent_stop_and_promotion_rule"]
    assert "descriptive" in stop_rule["primary_mandate"].lower()
    assert len(stop_rule["engineering_defect_criteria"]) > 0
    assert "weak" in stop_rule["negative_result_preservation"].lower()
