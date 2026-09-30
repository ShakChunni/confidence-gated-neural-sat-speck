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
from .comparison_input import (
    SimonComparisonInput,
    derive_random_control_seed,
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
    "SimonComparisonInput",
    "SimonSatModel",
    "SimonVerificationResult",
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "decrypt_block",
    "decrypt_round",
    "derive_random_control_seed",
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

try:
    from .b1_data import (
        SimonB0Dataset as SimonB0Dataset,
    )
    from .b1_data import (
        SimonB1GroupDataset as SimonB1GroupDataset,
    )
    from .b1_data import (
        SimonGeneratedB0Data as SimonGeneratedB0Data,
    )
    from .b1_data import (
        SimonGeneratedPairGroups as SimonGeneratedPairGroups,
    )
    from .b1_data import (
        generate_simon_b0_pairs as generate_simon_b0_pairs,
    )
    from .b1_data import (
        generate_simon_b1_groups as generate_simon_b1_groups,
    )
    from .b1_data import (
        load_simon_b0_checkpoint as load_simon_b0_checkpoint,
    )
    from .b1_data import (
        load_simon_b1_checkpoint as load_simon_b1_checkpoint,
    )
    from .b1_data import (
        save_simon_checkpoint as save_simon_checkpoint,
    )
    from .b1_training_runner import (
        load_and_validate_manifest as load_and_validate_manifest,
    )
    from .b1_training_runner import (
        run_b0_b1_training_pipeline as run_b0_b1_training_pipeline,
    )
    from .b1_training_runner import (
        verify_preflight_cryptography as verify_preflight_cryptography,
    )
    from .faithful_b1_model import (
        SimonGohrResNet as SimonGohrResNet,
    )
    from .faithful_b1_model import (
        SimonGohrResNetConfig as SimonGohrResNetConfig,
    )
    from .faithful_b1_model import (
        SimonPairGroupRanker as SimonPairGroupRanker,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend([
        "SimonB0Dataset", "SimonB1GroupDataset", "SimonGeneratedB0Data",
        "SimonGeneratedPairGroups", "SimonGohrResNet", "SimonGohrResNetConfig",
        "SimonPairGroupRanker", "generate_simon_b0_pairs", "generate_simon_b1_groups",
        "load_and_validate_manifest", "load_simon_b0_checkpoint", "load_simon_b1_checkpoint",
        "run_b0_b1_training_pipeline", "save_simon_checkpoint", "verify_preflight_cryptography",
    ])
