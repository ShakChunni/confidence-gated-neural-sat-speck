"""Scalar SIMON32/64 reference components."""

from .attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    SimonAttackPair,
    generate_simon_attack_instances,
    load_simon_attack_instances,
    save_simon_attack_instances,
)
from .cipher import (
    SIMON32_64_BLOCK_WORD_COUNT,
    SIMON32_64_KAT_CIPHERTEXT,
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
    Block,
    decrypt_block,
    decrypt_round,
    encrypt_block,
    encrypt_round,
)
from .key_schedule import (
    SIMON32_64_CONSTANT_C,
    SIMON32_64_KEY_WORD_COUNT,
    SIMON32_64_ROUND_COUNT,
    SIMON32_64_Z0,
    SIMON32_64_Z0_PERIOD,
    expand_key,
)
from .partial import partial_decrypt_final_round
from .primitives import (
    WORD_BITS,
    WORD_MASK,
    WORD_MODULUS,
    mask_word,
    rotate_left,
    rotate_right,
    simon_f,
)
from .sat_model import SimonSatModel
from .verification import SimonVerificationResult, verify_simon_master_key

__all__ = [
    "Block",
    "DEFAULT_SIMON_INPUT_DIFFERENCE",
    "SIMON32_64_BLOCK_WORD_COUNT",
    "SIMON32_64_CONSTANT_C",
    "SIMON32_64_KAT_CIPHERTEXT",
    "SIMON32_64_KAT_KEY",
    "SIMON32_64_KAT_PLAINTEXT",
    "SIMON32_64_KEY_WORD_COUNT",
    "SIMON32_64_ROUND_COUNT",
    "SIMON32_64_Z0",
    "SIMON32_64_Z0_PERIOD",
    "SimonAttackInstance",
    "SimonAttackInstanceConfig",
    "SimonAttackPair",
    "SimonSatModel",
    "SimonVerificationResult",
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "decrypt_block",
    "decrypt_round",
    "encrypt_block",
    "encrypt_round",
    "expand_key",
    "generate_simon_attack_instances",
    "load_simon_attack_instances",
    "mask_word",
    "partial_decrypt_final_round",
    "rotate_left",
    "rotate_right",
    "save_simon_attack_instances",
    "simon_f",
    "verify_simon_master_key",
]
