"""Manifest-driven B0->B1 training runner for SPECK-head-aligned SIMON32/64.

Governing Rules and Invariants (AGENTS.md, D-029, D-034, D-035, D-037):
========================================================================
1. Architecture Alignment:
   - SimonGohrResNet (B0 distinguisher): 62,337 parameters, head_type='speck'.
   - SimonPairGroupRanker (B1 group ranker): 60,289 parameters, variant='b1_mean',
     head_variant='speck_b1', 64 pairs per group.
   - Weight transfer: exact copying of stem, ResBlocks, dense projection, and dense_bn
     from trained B0 into B1 shared pair encoder before B1 fine-tuning.
2. Seed Governance & Strict Isolation:
   - Train seed 880001, Validation seed 880002 (role: SYNTHETIC_TRAINING_ONLY).
   - Rejects all historical, quarantined, retired, and evaluation seeds.
   - Before ANY optimizer step:
     * Generates B0 and B1 train/val splits.
     * Executes audit_training_key_roles: verifies exact 64-bit master-key uniqueness
       within each split and zero train/validation key intersection.
     * Verifies pair construction and 9-round features against scalar cipher and
       independent reference oracle.
3. Deterministic Checkpoint Selection:
   - Separately for B0 and B1, selects the epoch with minimum validation BCE loss.
   - Exact ties choose the earlier epoch (strict inequality `<`).
4. Finite-Loss & Defect Handling:
   - Aborts on NaN/Inf loss, zero gradients across entire architecture, or non-finite weights.
   - Weak but finite validation performance (e.g. 50.8% accuracy) is retained and reported
     as descriptive evidence (outcome-independent promotion rule).
5. Production Worker Compatibility:
   - Saves checkpoint with all required metadata (training_status="complete",
     training_manifest_sha256, train_dataset_checksum, validation_dataset_checksum,
     selected_epoch >= 1, head_variant="speck_b1").
   - Verified loadable by load_simon_b1_checkpoint and generate_faithful_b1_hybrid_queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED, PROTECTED_RESEARCH_SEEDS
from neural_sat_speck.neural.metrics import binary_accuracy, roc_auc_score
from neural_sat_speck.simon.b1_data import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SIMON_DEFAULT_ROUNDS,
    SimonB0Dataset,
    SimonB1GroupDataset,
    SimonGeneratedB0Data,
    SimonGeneratedPairGroups,
    audit_training_key_roles,
    create_simon_b0_dataloader,
    create_simon_b1_dataloader,
    generate_simon_b0_pairs,
    generate_simon_b1_groups,
    load_simon_b1_checkpoint,
    save_simon_checkpoint,
)
from neural_sat_speck.simon.cipher import decrypt_block, encrypt_block
from neural_sat_speck.simon.faithful_b1_model import (
    SimonGohrResNet,
    SimonPairGroupRanker,
)

# Forbidden seeds: protected research seeds, historical seeds, and demo seeds
FORBIDDEN_SIMON_TRAINING_SEEDS: Final[frozenset[int]] = frozenset(
    PROTECTED_RESEARCH_SEEDS
    | {
        CLI_DEMO_SEED,
        669981,
    }
)

EXPECTED_B0_PARAMS: Final[int] = 62337
EXPECTED_B1_PARAMS: Final[int] = 60289


class ManifestValidationError(ValueError):
    """Raised when training manifest violates schema, policy, or invariants."""


def _sha256_file(path: Path) -> str:
    """Compute lowercase 64-character hex SHA-256 digest of file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_bits_to_words(bits: np.ndarray) -> np.ndarray:
    """Convert (..., 4, 16) MSB-first bits to (..., 4) uint16 words."""
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return np.sum(bits.astype(np.uint16) << shifts, axis=-1)


def load_and_validate_manifest(
    manifest_path: Path | str,
) -> tuple[dict[str, Any], str]:
    """Load and validate the SIMON B0->B1 training manifest.

    Validates schema, parameter counts, seed governance, and hyperparameters.
    Returns (manifest_dict, manifest_sha256).
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    manifest_sha256 = _sha256_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ManifestValidationError(f"Invalid JSON in manifest {path}: {e}") from e

    if not isinstance(manifest, dict):
        raise ManifestValidationError(f"Manifest must be a JSON object, got {type(manifest)}")

    # 1. Required top-level keys
    required_keys = (
        "manifest_version",
        "protocol_id",
        "status",
        "governing_decisions",
        "target_cipher",
        "architectures",
        "seed_governance",
        "dataset_specifications",
        "training_hyperparameters",
        "outcome_independent_stop_and_promotion_rule",
    )
    for key in required_keys:
        if key not in manifest:
            raise ManifestValidationError(f"Missing required top-level key: '{key}'")

    # 2. Governing decisions
    decisions = manifest.get("governing_decisions", [])
    if not isinstance(decisions, list):
        raise ManifestValidationError("governing_decisions must be a list")
    for req_d in ("D-029", "D-034", "D-037"):
        if req_d not in decisions:
            raise ManifestValidationError(
                f"governing_decisions must include '{req_d}', got {decisions}"
            )

    # 3. Target cipher
    target_cipher = manifest.get("target_cipher", {})
    if target_cipher.get("cipher_name") != "SIMON32/64":
        raise ManifestValidationError(
            f"target_cipher.cipher_name must be 'SIMON32/64', got {target_cipher.get('cipher_name')}"
        )
    if target_cipher.get("rounds") != 9:
        raise ManifestValidationError(
            f"target_cipher.rounds must be 9, got {target_cipher.get('rounds')}"
        )
    diff = target_cipher.get("input_difference")
    if diff not in ([0, 64], (0, 64), [0x0000, 0x0040], (0x0000, 0x0040)):
        raise ManifestValidationError(
            f"target_cipher.input_difference must be [0, 64] / (0x0000, 0x0040), got {diff}"
        )

    # 4. Architectures
    archs = manifest.get("architectures", {})
    b0_arch = archs.get("b0_distinguisher", {})
    if b0_arch.get("model_class") != "SimonGohrResNet":
        raise ManifestValidationError(
            f"b0_distinguisher model_class must be 'SimonGohrResNet', got {b0_arch.get('model_class')}"
        )
    if b0_arch.get("trainable_parameters") != EXPECTED_B0_PARAMS:
        raise ManifestValidationError(
            f"b0_distinguisher trainable_parameters must be {EXPECTED_B0_PARAMS}, "
            f"got {b0_arch.get('trainable_parameters')}"
        )
    if b0_arch.get("head_type") != "speck":
        raise ManifestValidationError(
            f"b0_distinguisher head_type must be 'speck', got {b0_arch.get('head_type')}"
        )

    b1_arch = archs.get("b1_group_ranker", {})
    if b1_arch.get("model_class") != "SimonPairGroupRanker":
        raise ManifestValidationError(
            f"b1_group_ranker model_class must be 'SimonPairGroupRanker', got {b1_arch.get('model_class')}"
        )
    if b1_arch.get("variant") != "b1_mean":
        raise ManifestValidationError(
            f"b1_group_ranker variant must be 'b1_mean', got {b1_arch.get('variant')}"
        )
    if b1_arch.get("trainable_parameters") != EXPECTED_B1_PARAMS:
        raise ManifestValidationError(
            f"b1_group_ranker trainable_parameters must be {EXPECTED_B1_PARAMS}, "
            f"got {b1_arch.get('trainable_parameters')}"
        )
    if b1_arch.get("head_variant") != "speck_b1":
        raise ManifestValidationError(
            f"b1_group_ranker head_variant must be 'speck_b1', got {b1_arch.get('head_variant')}"
        )
    if b1_arch.get("group_size") != 64:
        raise ManifestValidationError(
            f"b1_group_ranker group_size must be 64, got {b1_arch.get('group_size')}"
        )

    # 5. Seed governance
    seed_gov = manifest.get("seed_governance", {})
    train_seed = seed_gov.get("train_seed")
    val_seed = seed_gov.get("validation_seed")
    if not isinstance(train_seed, int) or train_seed <= 0:
        raise ManifestValidationError(f"Invalid train_seed: {train_seed}")
    if not isinstance(val_seed, int) or val_seed <= 0:
        raise ManifestValidationError(f"Invalid validation_seed: {val_seed}")
    if train_seed == val_seed:
        raise ManifestValidationError(f"train_seed and validation_seed must differ: got {train_seed}")
    if train_seed in FORBIDDEN_SIMON_TRAINING_SEEDS:
        raise ManifestValidationError(
            f"train_seed {train_seed} violates seed governance (in forbidden/protected seeds)"
        )
    if val_seed in FORBIDDEN_SIMON_TRAINING_SEEDS:
        raise ManifestValidationError(
            f"validation_seed {val_seed} violates seed governance (in forbidden/protected seeds)"
        )
    if seed_gov.get("seed_role") != "SYNTHETIC_TRAINING_ONLY":
        raise ManifestValidationError(
            f"seed_role must be 'SYNTHETIC_TRAINING_ONLY', got {seed_gov.get('seed_role')}"
        )

    # 6. Dataset specifications
    ds_specs = manifest.get("dataset_specifications", {})
    splits = ds_specs.get("splits", {})
    if "train" not in splits or "validation" not in splits:
        raise ManifestValidationError("dataset_specifications.splits must contain 'train' and 'validation'")
    for split_name, split_info in splits.items():
        if not isinstance(split_info, dict):
            raise ManifestValidationError(f"split '{split_name}' must be a dictionary")
        if split_info.get("group_size") != 64:
            raise ManifestValidationError(f"split '{split_name}' group_size must be 64")
        if split_info.get("group_count", 0) <= 0:
            raise ManifestValidationError(f"split '{split_name}' group_count must be positive")
        if split_info.get("pair_count", 0) <= 0:
            raise ManifestValidationError(f"split '{split_name}' pair_count must be positive")

    # 7. Training hyperparameters
    hparams = manifest.get("training_hyperparameters", {})
    if hparams.get("optimizer") != "AdamW":
        raise ManifestValidationError(f"optimizer must be 'AdamW', got {hparams.get('optimizer')}")
    if hparams.get("loss_function") != "BCEWithLogitsLoss":
        raise ManifestValidationError(
            f"loss_function must be 'BCEWithLogitsLoss', got {hparams.get('loss_function')}"
        )

    b0_pre = hparams.get("b0_pretraining", {})
    if b0_pre.get("epochs", 0) <= 0:
        raise ManifestValidationError("b0_pretraining.epochs must be positive")
    if b0_pre.get("batch_size_pairs", 0) <= 0:
        raise ManifestValidationError("b0_pretraining.batch_size_pairs must be positive")
    if b0_pre.get("learning_rate", 0.0) <= 0.0:
        raise ManifestValidationError("b0_pretraining.learning_rate must be positive")

    b1_ft = hparams.get("b1_fine_tuning", {})
    if b1_ft.get("initialization") != "initialize_from_b0":
        raise ManifestValidationError(
            f"b1_fine_tuning.initialization must be 'initialize_from_b0', got {b1_ft.get('initialization')}"
        )
    if b1_ft.get("epochs", 0) <= 0:
        raise ManifestValidationError("b1_fine_tuning.epochs must be positive")
    if b1_ft.get("batch_size_groups", 0) <= 0:
        raise ManifestValidationError("b1_fine_tuning.batch_size_groups must be positive")
    if b1_ft.get("learning_rate", 0.0) <= 0.0:
        raise ManifestValidationError("b1_fine_tuning.learning_rate must be positive")

    # 8. Outcome-independent stop and promotion rule
    stop_rule = manifest.get("outcome_independent_stop_and_promotion_rule", {})
    if "primary_mandate" not in stop_rule:
        raise ManifestValidationError("Missing outcome_independent_stop_and_promotion_rule.primary_mandate")
    if not stop_rule.get("engineering_defect_criteria"):
        raise ManifestValidationError("Missing outcome_independent_stop_and_promotion_rule.engineering_defect_criteria")

    return manifest, manifest_sha256


def verify_preflight_cryptography(
    b0_train: SimonGeneratedB0Data,
    b1_train: SimonGeneratedPairGroups,
    b0_val: SimonGeneratedB0Data,
    b1_val: SimonGeneratedPairGroups,
    *,
    sample_count: int = 16,
) -> dict[str, Any]:
    """Execute preflight key audit and cryptographic oracle checks before any optimizer step.

    1. Executes audit_training_key_roles:
       - Fails closed on duplicate master keys within b0_train, b1_train, b0_val, b1_val.
       - Fails closed if train keys intersect validation keys.
    2. Tests pair construction and 9-round features against:
       - Scalar cipher reference (encrypt_block, decrypt_block)
       - Clean-room independent reference oracle (oracle_simon32_64_encrypt, oracle_simon32_64_decrypt)
       - Expected differential trail (0x0000, 0x0040) on positive pairs.
    """
    # 1. Audit key roles
    audit_result = audit_training_key_roles(
        b0_train.master_keys,
        b1_train.master_keys,
        b0_val.master_keys,
        b1_val.master_keys,
    )

    # 2. Import independent reference oracle
    try:
        from tests.fixtures.simon_reference_oracle import (
            oracle_simon32_64_decrypt,
            oracle_simon32_64_encrypt,
        )
    except ImportError:
        # Resolve via path relative to repo root
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tests.fixtures.simon_reference_oracle import (
            oracle_simon32_64_decrypt,
            oracle_simon32_64_encrypt,
        )

    # 3. Test B1 group pairs
    words_b1 = _decode_bits_to_words(b1_train.features)
    check_groups = min(len(b1_train.labels), max(2, sample_count))
    pos_tested = 0
    neg_tested = 0

    for g_idx in range(check_groups):
        mk = tuple(int(x) for x in b1_train.master_keys[g_idx])
        label = int(b1_train.labels[g_idx])
        pairs_to_check = (0, 1, 31, 63) if b1_train.group_size >= 64 else range(b1_train.group_size)

        for p_idx in pairs_to_check:
            c0 = (int(words_b1[g_idx, p_idx, 0]), int(words_b1[g_idx, p_idx, 1]))
            c1 = (int(words_b1[g_idx, p_idx, 2]), int(words_b1[g_idx, p_idx, 3]))

            # Decrypt with scalar reference and oracle
            p0_scalar = decrypt_block(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
            p1_scalar = decrypt_block(c1, mk, rounds=SIMON_DEFAULT_ROUNDS)
            p0_oracle = oracle_simon32_64_decrypt(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
            p1_oracle = oracle_simon32_64_decrypt(c1, mk, rounds=SIMON_DEFAULT_ROUNDS)

            if p0_scalar != p0_oracle or p1_scalar != p1_oracle:
                raise RuntimeError(
                    f"Scalar decryption does not match independent oracle for B1 group {g_idx}, pair {p_idx}"
                )

            # Re-encrypt with scalar reference and oracle
            c0_scalar = encrypt_block(p0_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS)
            c1_scalar = encrypt_block(p1_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS)
            c0_oracle = oracle_simon32_64_encrypt(p0_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS)
            c1_oracle = oracle_simon32_64_encrypt(p1_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS)

            if c0_scalar != c0 or c1_scalar != c1:
                raise RuntimeError(
                    f"Scalar encryption failed roundtrip for B1 group {g_idx}, pair {p_idx}"
                )
            if c0_oracle != c0 or c1_oracle != c1:
                raise RuntimeError(
                    f"Oracle encryption does not match B1 ciphertext for group {g_idx}, pair {p_idx}"
                )

            # Check differential
            diff = (p0_scalar[0] ^ p1_scalar[0], p0_scalar[1] ^ p1_scalar[1])
            if label == 1:
                if diff != DEFAULT_SIMON_INPUT_DIFFERENCE:
                    raise RuntimeError(
                        f"B1 positive group {g_idx} pair {p_idx} has incorrect difference {diff}, "
                        f"expected {DEFAULT_SIMON_INPUT_DIFFERENCE}"
                    )
                pos_tested += 1
            else:
                if diff == DEFAULT_SIMON_INPUT_DIFFERENCE:
                    raise RuntimeError(
                        f"B1 negative group {g_idx} pair {p_idx} unexpectedly matched target difference!"
                    )
                neg_tested += 1

    # 4. Test B0 single pairs
    words_b0 = _decode_bits_to_words(b0_train.features)
    check_b0_pairs = min(len(b0_train.labels), max(4, sample_count))
    for p_idx in range(check_b0_pairs):
        mk = tuple(int(x) for x in b0_train.master_keys[p_idx])
        label = int(b0_train.labels[p_idx])
        c0 = (int(words_b0[p_idx, 0]), int(words_b0[p_idx, 1]))
        c1 = (int(words_b0[p_idx, 2]), int(words_b0[p_idx, 3]))

        p0_scalar = decrypt_block(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
        p1_scalar = decrypt_block(c1, mk, rounds=SIMON_DEFAULT_ROUNDS)
        p0_oracle = oracle_simon32_64_decrypt(c0, mk, rounds=SIMON_DEFAULT_ROUNDS)
        p1_oracle = oracle_simon32_64_decrypt(c1, mk, rounds=SIMON_DEFAULT_ROUNDS)

        if p0_scalar != p0_oracle or p1_scalar != p1_oracle:
            raise RuntimeError(
                f"Scalar decryption does not match independent oracle for B0 pair {p_idx}"
            )
        if encrypt_block(p0_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS) != c0:
            raise RuntimeError(f"Scalar encryption failed roundtrip for B0 pair {p_idx}")
        if oracle_simon32_64_encrypt(p0_scalar, mk, rounds=SIMON_DEFAULT_ROUNDS) != c0:
            raise RuntimeError(f"Oracle encryption failed roundtrip for B0 pair {p_idx}")

        diff = (p0_scalar[0] ^ p1_scalar[0], p0_scalar[1] ^ p1_scalar[1])
        if label == 1 and diff != DEFAULT_SIMON_INPUT_DIFFERENCE:
            raise RuntimeError(f"B0 positive pair {p_idx} has incorrect difference {diff}")

    return {
        "key_audit": audit_result,
        "oracle_alignment_verified": True,
        "b1_positive_pairs_tested": pos_tested,
        "b1_negative_pairs_tested": neg_tested,
        "b0_pairs_tested": check_b0_pairs,
    }


def check_finite_loss(
    loss_val: float,
    stage: str,
    epoch: int,
    batch_idx: int | None = None,
) -> None:
    """Abort on NaN or infinite loss values."""
    if not math.isfinite(loss_val):
        batch_msg = f", batch {batch_idx}" if batch_idx is not None else ""
        raise RuntimeError(
            f"Engineering defect in {stage} (epoch {epoch}{batch_msg}): "
            f"loss became non-finite ({loss_val})"
        )


def check_gradients(model: nn.Module, stage: str, epoch: int) -> None:
    """Abort on zero gradients across entire architecture."""
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not torch.all(param.grad == 0):
                has_grad = True
                break
    if not has_grad:
        raise RuntimeError(
            f"Engineering defect in {stage} (epoch {epoch}): "
            "zero gradients across entire architecture"
        )


def check_finite_weights(model: nn.Module, stage: str) -> None:
    """Verify all parameters and buffers contain finite floating-point values."""
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Engineering defect in {stage}: parameter or buffer '{name}' "
                "contains non-finite (NaN/Inf) values"
            )


def train_simon_b0_stage(
    b0_train_data: SimonGeneratedB0Data,
    b0_val_data: SimonGeneratedB0Data,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    gamma: float = 0.95,
    seed: int = 880001,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[SimonGohrResNet, int, float, dict[str, Any], list[dict[str, Any]]]:
    """Train SimonGohrResNet (B0 pretraining stage).

    Checkpoint selection: epoch with minimum validation BCE loss; exact ties choose earlier epoch.
    Returns (best_model, best_epoch, best_val_loss, best_metrics, history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimonGohrResNet(head_type="speck")
    if model.parameter_count != EXPECTED_B0_PARAMS:
        raise ValueError(
            f"SimonGohrResNet parameter count mismatch: expected {EXPECTED_B0_PARAMS}, "
            f"got {model.parameter_count}"
        )
    model.to(device)

    train_ds = SimonB0Dataset.from_generated(b0_train_data)
    val_ds = SimonB0Dataset.from_generated(b0_val_data)
    train_loader = create_simon_b0_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = create_simon_b0_dataloader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        train_correct = 0

        for b_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)
            bs = features.size(0)

            optimizer.zero_grad()
            logits = model(features).view(-1, 1)
            loss = criterion(logits, targets)
            loss_val = float(loss.item())
            check_finite_loss(loss_val, "b0_pretraining", epoch, b_idx)

            loss.backward()
            check_gradients(model, "b0_pretraining", epoch)
            optimizer.step()

            train_loss_total += loss_val * bs
            train_samples += bs
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((preds == targets).sum().item())

        train_loss = train_loss_total / max(1, train_samples)
        train_acc = train_correct / max(1, train_samples)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        all_probs: list[float] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device).view(-1, 1)
                bs = features.size(0)
                logits = model(features).view(-1, 1)
                loss = criterion(logits, targets)
                loss_val = float(loss.item())
                check_finite_loss(loss_val, "b0_pretraining_val", epoch)

                val_loss_total += loss_val * bs
                val_samples += bs
                probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                targs = targets.view(-1).int().cpu().tolist()
                all_probs.extend(probs)
                all_targets.extend(targs)

        val_loss = val_loss_total / max(1, val_samples)
        probs_t = torch.tensor(all_probs, dtype=torch.float32)
        targs_t = torch.tensor(all_targets, dtype=torch.int64)
        val_acc = binary_accuracy(probs_t, targs_t)
        val_auc = roc_auc_score(all_probs, all_targets) if len(set(all_targets)) > 1 else 0.5
        duration = time.perf_counter() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "duration": duration,
            "lr": current_lr,
        }
        history.append(log_entry)

        if verbose:
            print(
                f"[B0 Pretraining Epoch {epoch:2d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
                f"lr: {current_lr:.6f} ({duration:.2f}s)"
            )

        # Strict inequality chooses earlier epoch in case of exact tie
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }

    if best_weights is not None:
        model.load_state_dict(best_weights)
    check_finite_weights(model, "b0_pretraining_final")
    model.eval()

    return model, best_epoch, best_val_loss, best_metrics, history


def train_simon_b1_stage(
    b0_model: SimonGohrResNet,
    b1_train_data: SimonGeneratedPairGroups,
    b1_val_data: SimonGeneratedPairGroups,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.0001,
    gamma: float = 0.95,
    seed: int = 880001,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[SimonPairGroupRanker, int, float, dict[str, Any], list[dict[str, Any]]]:
    """Fine-tune SimonPairGroupRanker initialized from trained B0 model.

    Checkpoint selection: epoch with minimum validation BCE loss; exact ties choose earlier epoch.
    Weak but finite validation performance is retained and reported as descriptive.
    Returns (best_model, best_epoch, best_val_loss, best_metrics, history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimonPairGroupRanker(variant="b1_mean", head_variant="speck_b1")
    if model.parameter_count != EXPECTED_B1_PARAMS:
        raise ValueError(
            f"SimonPairGroupRanker parameter count mismatch: expected {EXPECTED_B1_PARAMS}, "
            f"got {model.parameter_count}"
        )

    # Initialize pair encoder from trained B0 model
    model.initialize_from_b0(b0_model)
    model.to(device)

    train_ds = SimonB1GroupDataset.from_generated(b1_train_data)
    val_ds = SimonB1GroupDataset.from_generated(b1_val_data)
    train_loader = create_simon_b1_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = create_simon_b1_dataloader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        train_correct = 0

        for b_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)
            bs = features.size(0)

            optimizer.zero_grad()
            logits = model(features).view(-1, 1)
            loss = criterion(logits, targets)
            loss_val = float(loss.item())
            check_finite_loss(loss_val, "b1_fine_tuning", epoch, b_idx)

            loss.backward()
            check_gradients(model, "b1_fine_tuning", epoch)
            optimizer.step()

            train_loss_total += loss_val * bs
            train_samples += bs
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((preds == targets).sum().item())

        train_loss = train_loss_total / max(1, train_samples)
        train_acc = train_correct / max(1, train_samples)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        all_probs: list[float] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device).view(-1, 1)
                bs = features.size(0)
                logits = model(features).view(-1, 1)
                loss = criterion(logits, targets)
                loss_val = float(loss.item())
                check_finite_loss(loss_val, "b1_fine_tuning_val", epoch)

                val_loss_total += loss_val * bs
                val_samples += bs
                probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                targs = targets.view(-1).int().cpu().tolist()
                all_probs.extend(probs)
                all_targets.extend(targs)

        val_loss = val_loss_total / max(1, val_samples)
        probs_t = torch.tensor(all_probs, dtype=torch.float32)
        targs_t = torch.tensor(all_targets, dtype=torch.int64)
        val_acc = binary_accuracy(probs_t, targs_t)
        val_auc = roc_auc_score(all_probs, all_targets) if len(set(all_targets)) > 1 else 0.5
        duration = time.perf_counter() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "duration": duration,
            "lr": current_lr,
        }
        history.append(log_entry)

        if verbose:
            print(
                f"[B1 Fine-Tuning Epoch {epoch:2d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
                f"lr: {current_lr:.6f} ({duration:.2f}s)"
            )

        # Strict inequality chooses earlier epoch on exact tie
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }

    if best_weights is not None:
        model.load_state_dict(best_weights)
    check_finite_weights(model, "b1_fine_tuning_final")
    model.eval()

    return model, best_epoch, best_val_loss, best_metrics, history


def save_and_verify_b1_checkpoint(
    model: SimonPairGroupRanker,
    output_path: Path | str,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    b1_train_checksum: str,
    b1_val_checksum: str,
    b0_train_checksum: str,
    b0_val_checksum: str,
    b1_selected_epoch: int,
    b1_best_val_loss: float,
    b1_val_accuracy: float,
    b1_val_auc: float,
    b0_selected_epoch: int,
    b0_best_val_loss: float,
    b0_val_accuracy: float,
    b0_val_auc: float,
    b1_history: list[dict[str, Any]],
    b0_history: list[dict[str, Any]],
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Save SimonPairGroupRanker checkpoint and verify production B1 worker compatibility."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "training_status": "complete",
        "performance_label": "descriptive",
        "selected_epoch": int(b1_selected_epoch),
        "best_val_loss": float(b1_best_val_loss),
        "val_accuracy": float(b1_val_accuracy),
        "val_auc": float(b1_val_auc),
        "training_manifest_sha256": str(manifest_sha256).lower(),
        "train_dataset_checksum": str(b1_train_checksum).lower(),
        "validation_dataset_checksum": str(b1_val_checksum).lower(),
        "b0_train_dataset_checksum": str(b0_train_checksum).lower(),
        "b0_validation_dataset_checksum": str(b0_val_checksum).lower(),
        "b0_selected_epoch": int(b0_selected_epoch),
        "b0_best_val_loss": float(b0_best_val_loss),
        "b0_val_accuracy": float(b0_val_accuracy),
        "b0_val_auc": float(b0_val_auc),
        "training_history": b1_history,
        "b0_training_history": b0_history,
        "protocol_id": manifest.get("protocol_id", "simon32_64_b0_b1_faithful_training_v1"),
        "manifest_version": manifest.get("manifest_version", "1.0.0"),
        "governing_decisions": manifest.get("governing_decisions", []),
        "seed_governance": manifest.get("seed_governance", {}),
        "smoke_test": bool(smoke_test),
        "parameter_count": int(model.parameter_count),
        "head_variant": "speck_b1",
        "variant": "b1_mean",
    }

    # Save using canonical save_simon_checkpoint
    save_simon_checkpoint(model, path, metadata=metadata)
    file_sha256 = _sha256_file(path)

    # Reload checkpoint using canonical load_simon_b1_checkpoint
    loaded_model, loaded_meta = load_simon_b1_checkpoint(path)

    # Execute production B1 worker validation assertions (matching method_worker.py)
    if loaded_model.head_variant != "speck_b1":
        raise ValueError("Production B1 checkpoint has the wrong head architecture")
    if not isinstance(loaded_meta, dict) or loaded_meta.get("training_status") != "complete":
        raise ValueError("Production B1 checkpoint lacks completed-training provenance")

    for field_name in (
        "training_manifest_sha256",
        "train_dataset_checksum",
        "validation_dataset_checksum",
    ):
        digest_val = loaded_meta.get(field_name)
        if not isinstance(digest_val, str) or len(digest_val) != 64 or any(
            c not in "0123456789abcdef" for c in digest_val
        ):
            raise ValueError(f"Production B1 checkpoint has invalid {field_name}")

    if type(loaded_meta.get("selected_epoch")) is not int or loaded_meta["selected_epoch"] < 1:
        raise ValueError("Production B1 checkpoint has invalid selected_epoch")

    if any(
        not torch.isfinite(t).all()
        for t in loaded_model.state_dict().values()
        if t.is_floating_point()
    ):
        raise ValueError("Production B1 checkpoint contains non-finite weights or buffers")

    # Run dummy forward pass
    dummy_input = torch.zeros((1, 16, 4, 16), dtype=torch.float32)
    with torch.no_grad():
        out = loaded_model(dummy_input)
    if not torch.isfinite(out).all():
        raise ValueError("Loaded model produced non-finite output on dummy input")

    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256,
        "metadata": loaded_meta,
    }


def run_b0_b1_training_pipeline(
    manifest_path: Path | str,
    *,
    output_checkpoint: Path | str | None = None,
    output_b0_checkpoint: Path | str | None = None,
    smoke_test: bool = False,
    smoke_b1_train_groups: int = 16,
    smoke_b1_val_groups: int = 8,
    smoke_b0_train_pairs: int = 64,
    smoke_b0_val_pairs: int = 32,
    smoke_epochs: int = 2,
    smoke_batch_size_groups: int = 4,
    smoke_batch_size_pairs: int = 16,
    device: str = "cpu",
    preflight_only: bool = False,
    execute_full_training: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full manifest-driven B0 pretraining and B1 fine-tuning pipeline.

    Guards:
    - Rejects full training unless execute_full_training is explicitly True or smoke_test is True.
    - Aborts on non-finite loss or zero gradients.
    - Preserves and reports weak distinguishing accuracy as descriptive evidence.
    """
    manifest, manifest_sha256 = load_and_validate_manifest(manifest_path)
    train_seed = manifest["seed_governance"]["train_seed"]
    val_seed = manifest["seed_governance"]["validation_seed"]

    if not smoke_test and not execute_full_training:
        raise ValueError(
            "Full B0/B1 dataset generation/training is guarded under D-037 / AGENTS.md. "
            "Pass --smoke-test to run smoke test pipeline verification, "
            "or --execute-full-training to confirm a full 1,000,000-pair generation/training run."
        )

    if smoke_test:
        b0_tr_count = smoke_b0_train_pairs
        b0_val_count = smoke_b0_val_pairs
        b1_tr_count = smoke_b1_train_groups
        b1_val_count = smoke_b1_val_groups
        b0_epochs = smoke_epochs
        b1_epochs = smoke_epochs
        b0_bs = smoke_batch_size_pairs
        b1_bs = smoke_batch_size_groups
        default_ckpt = Path("models/simon/simon_b1_smoke.pt")
    else:
        train_split = manifest["dataset_specifications"]["splits"]["train"]
        val_split = manifest["dataset_specifications"]["splits"]["validation"]
        b0_tr_count = train_split["pair_count"]
        b0_val_count = val_split["pair_count"]
        b1_tr_count = train_split["group_count"]
        b1_val_count = val_split["group_count"]
        hparams = manifest["training_hyperparameters"]
        b0_epochs = hparams["b0_pretraining"]["epochs"]
        b1_epochs = hparams["b1_fine_tuning"]["epochs"]
        b0_bs = hparams["b0_pretraining"]["batch_size_pairs"]
        b1_bs = hparams["b1_fine_tuning"]["batch_size_groups"]
        default_ckpt = Path("models/simon/simon_b1_mean_speckhead.pt")

    target_ckpt = Path(output_checkpoint) if output_checkpoint else default_ckpt

    if verbose:
        print("=" * 70)
        print("SIMON32/64 B0->B1 Faithful Training Pipeline")
        print(f"Manifest: {manifest_path} (SHA-256: {manifest_sha256[:16]}...)")
        print(f"Mode: {'SMOKE TEST' if smoke_test else ('PREFLIGHT ONLY' if preflight_only else 'FULL TRAINING')}")
        print(f"B0 pairs: {b0_tr_count} train, {b0_val_count} val")
        print(f"B1 groups: {b1_tr_count} train, {b1_val_count} val (group_size 64)")
        print(f"Seeds: train={train_seed}, val={val_seed}")
        print("=" * 70)

    # Generate data
    if verbose:
        print("[1/4] Generating B0 and B1 datasets...")
    b0_train = generate_simon_b0_pairs(b0_tr_count, train_seed, rounds=SIMON_DEFAULT_ROUNDS)
    b0_val = generate_simon_b0_pairs(b0_val_count, val_seed, rounds=SIMON_DEFAULT_ROUNDS)
    b1_train = generate_simon_b1_groups(b1_tr_count, train_seed, rounds=SIMON_DEFAULT_ROUNDS)
    b1_val = generate_simon_b1_groups(b1_val_count, val_seed, rounds=SIMON_DEFAULT_ROUNDS)

    # Preflight verification before ANY optimizer step
    if verbose:
        print("[2/4] Executing preflight cryptographic audit and independent oracle checks...")
    preflight_result = verify_preflight_cryptography(b0_train, b1_train, b0_val, b1_val)
    if verbose:
        print(f"      Key audit passed: {preflight_result['key_audit']}")
        print(f"      Independent oracle verified: {preflight_result['oracle_alignment_verified']}")

    if preflight_only:
        return {
            "status": "preflight_complete",
            "manifest_sha256": manifest_sha256,
            "preflight_result": preflight_result,
            "b0_train_checksum": b0_train.metadata["content_checksum"],
            "b1_train_checksum": b1_train.metadata["content_checksum"],
        }

    # Stage 1: B0 Pretraining
    if verbose:
        print(f"[3/4] B0 Pretraining ({b0_epochs} epochs, lr={0.001})...")
    b0_lr = manifest["training_hyperparameters"]["b0_pretraining"]["learning_rate"]
    b0_wd = manifest["training_hyperparameters"]["b0_pretraining"]["weight_decay"]
    b0_model, b0_best_ep, b0_best_loss, b0_best_metrics, b0_history = train_simon_b0_stage(
        b0_train,
        b0_val,
        epochs=b0_epochs,
        batch_size=b0_bs,
        learning_rate=b0_lr,
        weight_decay=b0_wd,
        seed=train_seed,
        device=device,
        verbose=verbose,
    )
    if output_b0_checkpoint:
        save_simon_checkpoint(b0_model, output_b0_checkpoint, metadata={"selected_epoch": b0_best_ep})

    # Stage 2: B1 Fine-Tuning
    if verbose:
        print(f"[4/4] B1 Fine-Tuning ({b1_epochs} epochs, lr={0.0005}, transferred from B0)...")
    b1_lr = manifest["training_hyperparameters"]["b1_fine_tuning"]["learning_rate"]
    b1_wd = manifest["training_hyperparameters"]["b1_fine_tuning"]["weight_decay"]
    b1_model, b1_best_ep, b1_best_loss, b1_best_metrics, b1_history = train_simon_b1_stage(
        b0_model,
        b1_train,
        b1_val,
        epochs=b1_epochs,
        batch_size=b1_bs,
        learning_rate=b1_lr,
        weight_decay=b1_wd,
        seed=train_seed,
        device=device,
        verbose=verbose,
    )

    # Save and verify B1 checkpoint against production worker requirements
    if verbose:
        print(f"Saving and verifying B1 checkpoint at {target_ckpt}...")
    saved_info = save_and_verify_b1_checkpoint(
        b1_model,
        target_ckpt,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        b1_train_checksum=b1_train.metadata["content_checksum"],
        b1_val_checksum=b1_val.metadata["content_checksum"],
        b0_train_checksum=b0_train.metadata["content_checksum"],
        b0_val_checksum=b0_val.metadata["content_checksum"],
        b1_selected_epoch=b1_best_ep,
        b1_best_val_loss=b1_best_loss,
        b1_val_accuracy=b1_best_metrics.get("val_acc", 0.5),
        b1_val_auc=b1_best_metrics.get("val_auc", 0.5),
        b0_selected_epoch=b0_best_ep,
        b0_best_val_loss=b0_best_loss,
        b0_val_accuracy=b0_best_metrics.get("val_acc", 0.5),
        b0_val_auc=b0_best_metrics.get("val_auc", 0.5),
        b1_history=b1_history,
        b0_history=b0_history,
        smoke_test=smoke_test,
    )

    if verbose:
        print("=" * 70)
        print("Training Complete!")
        print(f"B0 Best Epoch: {b0_best_ep} (Val BCE: {b0_best_loss:.4f})")
        print(f"B1 Best Epoch: {b1_best_ep} (Val BCE: {b1_best_loss:.4f}, Acc: {b1_best_metrics.get('val_acc', 0.5):.4f}, AUC: {b1_best_metrics.get('val_auc', 0.5):.4f})")
        print(f"Checkpoint SHA-256: {saved_info['checkpoint_sha256']}")
        print("=" * 70)

    return {
        "status": "complete",
        "smoke_test": smoke_test,
        "checkpoint_path": str(target_ckpt),
        "checkpoint_sha256": saved_info["checkpoint_sha256"],
        "manifest_sha256": manifest_sha256,
        "b0_best_epoch": b0_best_ep,
        "b0_best_val_loss": b0_best_loss,
        "b1_best_epoch": b1_best_ep,
        "b1_best_val_loss": b1_best_loss,
        "b1_val_accuracy": b1_best_metrics.get("val_acc", 0.5),
        "b1_val_auc": b1_best_metrics.get("val_auc", 0.5),
        "preflight_result": preflight_result,
        "metadata": saved_info["metadata"],
    }


__all__ = [
    "EXPECTED_B0_PARAMS",
    "EXPECTED_B1_PARAMS",
    "FORBIDDEN_SIMON_TRAINING_SEEDS",
    "ManifestValidationError",
    "check_finite_loss",
    "check_finite_weights",
    "check_gradients",
    "load_and_validate_manifest",
    "run_b0_b1_training_pipeline",
    "save_and_verify_b1_checkpoint",
    "train_simon_b0_stage",
    "train_simon_b1_stage",
    "verify_preflight_cryptography",
]
