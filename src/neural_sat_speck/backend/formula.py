"""Build deterministic, solver-free native-XOR and pure-CNF artifacts.

The builder reuses the repository's Boolean encoders. It records formulas only;
it does not load a SAT solver, inspect master-key truth, or read verification data.
For conditioned diagnostics, callers provide the chronological trailing round-key
values explicitly in ``instance["trailing_round_keys"]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from neural_sat_speck.sat.encoding import (
    VariablePool,
    WordBits,
    add_fixed_word,
    add_modular_addition,
    add_xor_bits,
    add_xor_constant,
    rotate_bits_left,
    rotate_bits_right,
)
from neural_sat_speck.sat.model import _encode_round
from neural_sat_speck.sat.pure_cnf import (
    convert_native_xor_dimacs_to_pure_cnf,
)
from neural_sat_speck.simon.key_schedule import SIMON32_64_ROUND_COUNT
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.speck.key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
    SPECK32_64_ROUND_COUNT,
)
from neural_sat_speck.speck.primitives import WORD_BITS, WORD_MASK

Block: TypeAlias = tuple[int, int]
Observation: TypeAlias = tuple[Block, Block]
XorRecord: TypeAlias = dict[str, object]


class _NoSolveSink:
    """Accept model additions without importing or calling any solver."""

    def add_clause(self, clause: Sequence[int]) -> None:
        del clause

    def add_xor_clause(self, literals: Sequence[int], rhs: bool) -> None:
        del literals, rhs


class _Recorder:
    """Minimal clause recorder used with the existing SPECK bit encoders."""

    def __init__(self) -> None:
        self.cnf_clauses: list[list[int]] = []
        self.xor_clauses: list[tuple[list[int], bool]] = []

    def add_clause(self, clause: Sequence[int]) -> None:
        self.cnf_clauses.append(list(clause))

    def add_xor_clause(self, literals: Sequence[int], rhs: bool) -> None:
        self.xor_clauses.append((list(literals), bool(rhs)))


def _checked_int(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _checked_word(value: object, name: str) -> int:
    return _checked_int(value, name, 0, WORD_MASK)


def _checked_block(value: object, name: str) -> Block:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a two-word sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two words")
    return (
        _checked_word(value[0], f"{name}[0]"),
        _checked_word(value[1], f"{name}[1]"),
    )


def _normalise_inputs(
    cipher: str,
    instance: Mapping[str, object],
    observation_count: int,
    trailing_key_count: int,
) -> tuple[int, list[Observation], tuple[int, ...]]:
    if not isinstance(cipher, str):
        raise TypeError("cipher must be a string")
    if cipher not in {"speck32_64", "simon32_64"}:
        raise ValueError("cipher must be 'speck32_64' or 'simon32_64'")
    if not isinstance(instance, Mapping):
        raise TypeError("instance must be a mapping")

    maximum_rounds = (
        SPECK32_64_ROUND_COUNT if cipher == "speck32_64" else SIMON32_64_ROUND_COUNT
    )
    rounds = _checked_int(instance["rounds"], "rounds", 1, maximum_rounds)
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise TypeError("observation_count must be an integer")
    if observation_count <= 0 or observation_count % 2:
        raise ValueError("observation_count must be a positive even integer")
    if isinstance(trailing_key_count, bool) or not isinstance(trailing_key_count, int):
        raise TypeError("trailing_key_count must be an integer")
    if not 0 <= trailing_key_count <= rounds:
        raise ValueError("trailing_key_count must be between zero and rounds")

    raw_pairs = instance["pairs"]
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise TypeError("instance pairs must be a sequence")
    if observation_count > 2 * len(raw_pairs):
        raise ValueError("observation_count exceeds the available pair prefix")

    observations: list[Observation] = []
    # Read only pairs needed for the requested prefix. Later plaintext/ciphertext
    # records may represent unavailable observations in the measured treatment.
    for pair_index in range(observation_count // 2):
        pair = raw_pairs[pair_index]
        if not isinstance(pair, Mapping):
            raise TypeError(f"pairs[{pair_index}] must be a mapping")
        pt0 = _checked_block(pair["plaintext0"], f"pairs[{pair_index}].plaintext0")
        ct0 = _checked_block(pair["ciphertext0"], f"pairs[{pair_index}].ciphertext0")
        pt1 = _checked_block(pair["plaintext1"], f"pairs[{pair_index}].plaintext1")
        ct1 = _checked_block(pair["ciphertext1"], f"pairs[{pair_index}].ciphertext1")
        observations.extend(((pt0, ct0), (pt1, ct1)))

    if trailing_key_count == 0:
        trailing_keys: tuple[int, ...] = ()
    else:
        # This field is the only key-derived input used by conditioned formulas.
        # In particular, the builder never reads master_key/final_round_subkey.
        raw_trailing_keys = instance["trailing_round_keys"]
        if not isinstance(raw_trailing_keys, Sequence) or isinstance(
            raw_trailing_keys, (str, bytes)
        ):
            raise TypeError("trailing_round_keys must be a sequence")
        if len(raw_trailing_keys) != trailing_key_count:
            raise ValueError("trailing_round_keys length must equal trailing_key_count")
        trailing_keys = tuple(
            _checked_word(value, f"trailing_round_keys[{index}]")
            for index, value in enumerate(raw_trailing_keys)
        )

    return rounds, observations, trailing_keys


def _assumption_literals(
    round_keys: Sequence[WordBits], trailing_values: Sequence[int]
) -> list[int]:
    if not trailing_values:
        return []
    if len(trailing_values) > len(round_keys):
        raise ValueError("too many trailing keys for the round-key schedule")
    literals: list[int] = []
    for bits, value in zip(
        round_keys[-len(trailing_values) :], trailing_values, strict=True
    ):
        literals.extend(
            variable if (value >> bit_index) & 1 else -variable
            for bit_index, variable in enumerate(bits)
        )
    return literals


def _build_speck(
    known_observations: Sequence[Observation],
    rounds: int,
    trailing_keys: Sequence[int],
) -> tuple[
    int, list[list[int]], list[tuple[list[int], bool]], dict[str, object], list[int]
]:
    """Compose the validated SPECK bit-level encoders without a solver."""
    recorder = _Recorder()
    pool = VariablePool()
    master_key_bits = tuple(pool.new_bits(WORD_BITS) for _ in range(4))

    l_words = [master_key_bits[2], master_key_bits[1], master_key_bits[0]]
    round_keys: list[WordBits] = [master_key_bits[3]]
    for index in range(rounds - 1):
        rotated_l = rotate_bits_right(l_words[index], SPECK32_64_ALPHA)
        added_l = add_modular_addition(recorder, pool, round_keys[index], rotated_l)
        next_l = add_xor_constant(recorder, pool, added_l, index)
        next_key = add_xor_bits(
            recorder,
            pool,
            rotate_bits_left(round_keys[index], SPECK32_64_BETA),
            next_l,
        )
        l_words.append(next_l)
        round_keys.append(next_key)

    state_maps: list[dict[str, object]] = []
    for observation_index, (plaintext, ciphertext) in enumerate(known_observations):
        initial_state = (pool.new_bits(WORD_BITS), pool.new_bits(WORD_BITS))
        add_fixed_word(recorder, initial_state[0], plaintext[0])
        add_fixed_word(recorder, initial_state[1], plaintext[1])
        state = initial_state
        round_states: list[dict[str, object]] = [
            {"round": 0, "x": list(state[0]), "y": list(state[1])}
        ]
        for round_index, round_key in enumerate(round_keys, 1):
            state = _encode_round(recorder, pool, state, round_key)
            round_states.append(
                {
                    "round": round_index,
                    "x": list(state[0]),
                    "y": list(state[1]),
                }
            )
        add_fixed_word(recorder, state[0], ciphertext[0])
        add_fixed_word(recorder, state[1], ciphertext[1])
        state_maps.append(
            {"observation_index": observation_index, "round_states": round_states}
        )

    assumptions = _assumption_literals(round_keys, trailing_keys)
    varmap: dict[str, object] = {
        "master_key": [list(bits) for bits in master_key_bits],
        "round_keys": [list(bits) for bits in round_keys],
        "states": state_maps,
    }
    return (
        pool.variable_count,
        recorder.cnf_clauses,
        recorder.xor_clauses,
        varmap,
        assumptions,
    )


def _build_simon(
    known_observations: Sequence[Observation],
    rounds: int,
    trailing_keys: Sequence[int],
) -> tuple[
    int, list[list[int]], list[tuple[list[int], bool]], dict[str, object], list[int]
]:
    """Use the existing SIMON model recorder with a solver-free sink."""
    model = SimonSatModel.from_known_pairs(
        known_observations,
        rounds=rounds,
        solver=_NoSolveSink(),
    )
    assumptions = (
        list(model.assumptions_for_trailing_round_subkeys(trailing_keys))
        if trailing_keys
        else []
    )
    state_maps = [
        {
            "observation_index": observation_index,
            "round_states": [
                {
                    "round": round_index,
                    "left": list(state[0]),
                    "right": list(state[1]),
                }
                for round_index, state in enumerate(states)
            ],
        }
        for observation_index, states in enumerate(model.state_bits)
    ]
    varmap: dict[str, object] = {
        "master_key": [list(bits) for bits in model.master_key_bits],
        "round_keys": [list(bits) for bits in model.round_keys],
        "states": state_maps,
    }
    return (
        model.variable_count,
        [list(clause) for clause in model.recorder.cnf_clauses],
        [(list(literals), rhs) for literals, rhs in model.recorder.xor_clauses],
        varmap,
        assumptions,
    )


def _checked_literal(value: object, num_vars: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(f"{name} must be a nonzero integer literal")
    if abs(value) > num_vars:
        raise ValueError(f"{name} exceeds num_vars")
    return value


def render_native_dimacs(
    num_vars: int,
    cnf_clauses: Sequence[Sequence[int]],
    xor_clauses: Sequence[Mapping[str, object]],
    assumptions: Sequence[int] = (),
) -> str:
    """Render the canonical CryptoMiniSat native-XOR DIMACS representation.

    A nonempty XOR row is normalized to odd parity by toggling the first signed
    literal when ``rhs`` is false. An empty odd-parity row is emitted as a CNF
    empty clause (contradiction); an empty even-parity row is omitted.
    Assumptions are represented as unit clauses and included in the header count.
    """
    _checked_int(num_vars, "num_vars", 0, 2**31 - 1)
    records: list[str] = []
    for clause_index, clause in enumerate(cnf_clauses):
        if not isinstance(clause, Sequence) or isinstance(clause, (str, bytes)):
            raise TypeError(f"cnf_clauses[{clause_index}] must be a sequence")
        literals = [
            _checked_literal(literal, num_vars, f"cnf_clauses[{clause_index}]")
            for literal in clause
        ]
        records.append(" ".join(map(str, literals)) + (" " if literals else "") + "0")

    for index, literal in enumerate(assumptions):
        checked = _checked_literal(literal, num_vars, f"assumptions[{index}]")
        records.append(f"{checked} 0")

    for xor_index, record in enumerate(xor_clauses):
        if not isinstance(record, Mapping):
            raise TypeError(f"xor_clauses[{xor_index}] must be a mapping")
        raw_literals = record["literals"]
        rhs = record["rhs"]
        if not isinstance(raw_literals, Sequence) or isinstance(
            raw_literals, (str, bytes)
        ):
            raise TypeError(f"xor_clauses[{xor_index}].literals must be a sequence")
        if type(rhs) is not bool:
            raise TypeError(f"xor_clauses[{xor_index}].rhs must be bool")
        literals = [
            _checked_literal(literal, num_vars, f"xor_clauses[{xor_index}]")
            for literal in raw_literals
        ]
        if not literals:
            if rhs:
                records.append("0")
            continue
        if not rhs:
            literals[0] = -literals[0]
        records.append("x " + " ".join(map(str, literals)) + " 0")

    lines = [f"p cnf {num_vars} {len(records)}", *records]
    return "\n".join(lines) + "\n"


def build_formula(
    cipher: str,
    instance: Mapping[str, object],
    observation_count: int,
    trailing_key_count: int,
    *,
    include_cnf: bool = True,
) -> dict[str, object]:
    """Build a JSON-serializable SAT artifact without truth or solver access.

    ``instance`` must contain ``rounds`` and pair dictionaries. For a conditioned
    diagnostic it must additionally contain exactly ``trailing_key_count`` values
    in ``trailing_round_keys``, ordered from the earliest assumed round to the
    latest. ``master_key`` and ``verification_pairs`` are deliberately ignored.
    """
    if type(include_cnf) is not bool:
        raise TypeError("include_cnf must be bool")
    rounds, observations, trailing_keys = _normalise_inputs(
        cipher, instance, observation_count, trailing_key_count
    )

    if cipher == "speck32_64":
        num_vars, cnf_clauses, xor_pairs, varmap, assumptions = _build_speck(
            observations, rounds, trailing_keys
        )
        word_names = ["x", "y"]
        master_order = "(l[2], l[1], l[0], k[0])"
    else:
        num_vars, cnf_clauses, xor_pairs, varmap, assumptions = _build_simon(
            observations, rounds, trailing_keys
        )
        word_names = ["left", "right"]
        master_order = "(k[3], k[2], k[1], k[0])"

    xor_clauses: list[XorRecord] = [
        {"literals": list(literals), "rhs": rhs} for literals, rhs in xor_pairs
    ]
    native_dimacs = render_native_dimacs(
        num_vars, cnf_clauses, xor_clauses, assumptions
    )

    pure_cnf: str | None = None
    pure_cnf_num_vars: int | None = None
    pure_clause_count: int | None = None
    auxiliary_start: int | None = None
    auxiliary_end: int | None = None
    if include_cnf:
        pure_cnf, pure_cnf_num_vars = convert_native_xor_dimacs_to_pure_cnf(
            native_dimacs, num_vars
        )
        pure_clause_count = int(pure_cnf.splitlines()[0].split()[3])
        if pure_cnf_num_vars > num_vars:
            auxiliary_start = num_vars + 1
            auxiliary_end = pure_cnf_num_vars

    serialized_observations = [
        {"plaintext": list(plaintext), "ciphertext": list(ciphertext)}
        for plaintext, ciphertext in observations
    ]
    native_record_count = int(native_dimacs.splitlines()[0].split()[3])
    return {
        "cipher": cipher,
        "rounds": rounds,
        "observation_count": observation_count,
        "trailing_key_count": trailing_key_count,
        "num_vars": num_vars,
        "cnf_clauses": [list(clause) for clause in cnf_clauses],
        "xor_clauses": xor_clauses,
        "native_dimacs": native_dimacs,
        "pure_cnf": pure_cnf,
        "pure_cnf_num_vars": pure_cnf_num_vars,
        "known_observations": serialized_observations,
        "assumption_literals": list(assumptions),
        "counts": {
            "base_cnf": len(cnf_clauses),
            "base_xor": len(xor_clauses),
            "assumption_units": len(assumptions),
            "native_records": native_record_count,
            "pure_cnf_clauses": pure_clause_count,
            "auxiliary_start": auxiliary_start,
            "auxiliary_end": auxiliary_end,
        },
        "varmap": varmap,
        "format_ordering": {
            "native": "CryptoMiniSat DIMACS; x rows mean XOR(signed literals)=True",
            "pure_cnf": "standard DIMACS CNF",
            "variable_ids": "one-based DIMACS IDs",
            "word_bits": "little-endian; each list is least-significant bit first",
            "master_key_word_order": master_order,
            "round_key_order": "k[0], k[1], ... in encryption order",
            "state_word_order": word_names,
            "state_rounds": "round 0 input through round R ciphertext endpoint",
            "observations": "pair 0 member 0, pair 0 member 1, then later pairs",
            "assumptions": "trailing round keys in chronological order, then LSB-first bits",
        },
        "encoder_sources": {
            "speck32_64": "sat.encoding bit encoders and sat.model._encode_round",
            "simon32_64": "simon.sat_model.SimonSatModel recorder with a no-solve sink",
        },
        "limitations": [
            "No SAT solver is constructed or called; this artifact records constraints only.",
            "Conditioned formulas trust explicit trailing_round_keys and do not compare them with a master key.",
            "Master-key truth and verification pairs are not read or exported.",
        ],
    }
