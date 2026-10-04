"""Solver-free checks for canonical cross-cipher formula artifacts."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest

from neural_sat_speck.backend.formula import build_formula, render_native_dimacs
from neural_sat_speck.sat.dimacs import export_speck_to_dimacs
from neural_sat_speck.sat.pure_cnf import convert_native_xor_dimacs_to_pure_cnf
from neural_sat_speck.simon.cipher import encrypt_block as encrypt_simon
from neural_sat_speck.simon.cipher import encrypt_round as simon_round
from neural_sat_speck.simon.key_schedule import expand_key as expand_simon_key
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.speck.cipher import encrypt_block as encrypt_speck
from neural_sat_speck.speck.cipher import encrypt_round as speck_round
from neural_sat_speck.speck.key_schedule import expand_key as expand_speck_key


class _NoSolveSink:
    def add_clause(self, clause: object) -> None:
        del clause

    def add_xor_clause(self, literals: object, rhs: bool) -> None:
        del literals, rhs


def _cipher_parts(cipher: str):
    if cipher == "speck32_64":
        return encrypt_speck, speck_round, expand_speck_key
    return encrypt_simon, simon_round, expand_simon_key


def _make_instance(cipher: str, rounds: int, key: tuple[int, ...]) -> dict[str, Any]:
    encrypt, _, _ = _cipher_parts(cipher)
    pairs = []
    for pair_index in range(2):
        pt0 = (0x1021 + 17 * pair_index, 0x2345 + 29 * pair_index)
        pt1 = (0xFEDC - 31 * pair_index, 0xAB89 - 13 * pair_index)
        pairs.append(
            {
                "plaintext0": list(pt0),
                "plaintext1": list(pt1),
                "ciphertext0": list(encrypt(pt0, key, rounds)),
                "ciphertext1": list(encrypt(pt1, key, rounds)),
            }
        )
    return {"rounds": rounds, "pairs": pairs}


def _literal_value(literal: int, assignment: dict[int, bool]) -> bool:
    value = assignment[abs(literal)]
    return value if literal > 0 else not value


def _bind(assignment: dict[int, bool], variable: int, value: bool) -> tuple[bool, bool]:
    if variable in assignment:
        return assignment[variable] == value, False
    assignment[variable] = value
    return True, True


def _propagate_cnf_and_xor(
    cnf_clauses: list[list[int]],
    xor_clauses: list[dict[str, object]],
    initial: dict[int, bool],
    total_variables: int,
) -> dict[int, bool] | None:
    """Unit-propagate deterministic encoder gates without invoking a solver."""
    assignment = dict(initial)
    for _ in range(total_variables + 1):
        changed = False
        for clause in cnf_clauses:
            if any(
                abs(literal) in assignment and _literal_value(literal, assignment)
                for literal in clause
            ):
                continue
            unassigned = [
                literal for literal in clause if abs(literal) not in assignment
            ]
            variables = {abs(literal) for literal in unassigned}
            if not variables:
                return None
            if len(variables) == 1:
                variable = next(iter(variables))
                signs = {literal > 0 for literal in unassigned}
                if len(signs) == 1:
                    okay, added = _bind(assignment, variable, next(iter(signs)))
                    if not okay:
                        return None
                    changed |= added

        for record in xor_clauses:
            literals = record["literals"]
            rhs = record["rhs"]
            assert isinstance(literals, list) and isinstance(rhs, bool)
            negative_parity = sum(literal < 0 for literal in literals) % 2
            target = int(rhs) ^ negative_parity
            counts = Counter(abs(literal) for literal in literals)
            active_variables = [
                variable for variable, count in counts.items() if count % 2
            ]
            assigned_parity = (
                sum(
                    assignment[variable]
                    for variable in active_variables
                    if variable in assignment
                )
                % 2
            )
            unknown = [
                variable for variable in active_variables if variable not in assignment
            ]
            if not unknown:
                if assigned_parity != target:
                    return None
            elif len(unknown) == 1:
                okay, added = _bind(
                    assignment,
                    unknown[0],
                    bool(target ^ assigned_parity),
                )
                if not okay:
                    return None
                changed |= added

        if not changed:
            break

    return assignment


def _key_assignment(
    formula: dict[str, object], key: tuple[int, ...]
) -> dict[int, bool]:
    varmap = formula["varmap"]
    assert isinstance(varmap, dict)
    key_bits = varmap["master_key"]
    assert isinstance(key_bits, list)
    assignment: dict[int, bool] = {}
    for word, variables in zip(key, key_bits, strict=True):
        assert isinstance(variables, list)
        for bit_index, variable in enumerate(variables):
            assignment[variable] = bool((word >> bit_index) & 1)
    return assignment


def _decode_word(assignment: dict[int, bool], variables: list[int]) -> int:
    return sum(
        int(assignment[variable]) << index for index, variable in enumerate(variables)
    )


@pytest.mark.parametrize(
    ("cipher", "rounds", "key"),
    [
        ("speck32_64", 4, (0x1234, 0x5678, 0x9ABC, 0xDEF0)),
        ("simon32_64", 5, (0x1234, 0x5678, 0x9ABC, 0xDEF0)),
    ],
)
def test_formula_has_independent_reference_witness_and_chronological_assumptions(
    cipher: str, rounds: int, key: tuple[int, ...]
) -> None:
    instance = _make_instance(cipher, rounds, key)
    _, round_fn, expand_key = _cipher_parts(cipher)
    round_keys = expand_key(key, rounds)
    instance["trailing_round_keys"] = [round_keys[-1]]

    formula = build_formula(cipher, instance, 4, 1)
    assert formula["known_observations"] == [
        {
            "plaintext": instance["pairs"][0]["plaintext0"],
            "ciphertext": instance["pairs"][0]["ciphertext0"],
        },
        {
            "plaintext": instance["pairs"][0]["plaintext1"],
            "ciphertext": instance["pairs"][0]["ciphertext1"],
        },
        {
            "plaintext": instance["pairs"][1]["plaintext0"],
            "ciphertext": instance["pairs"][1]["ciphertext0"],
        },
        {
            "plaintext": instance["pairs"][1]["plaintext1"],
            "ciphertext": instance["pairs"][1]["ciphertext1"],
        },
    ]

    assumptions = formula["assumption_literals"]
    assert isinstance(assumptions, list)
    varmap = formula["varmap"]
    assert isinstance(varmap, dict)
    final_key_bits = varmap["round_keys"][-1]
    assert assumptions == [
        variable if (round_keys[-1] >> index) & 1 else -variable
        for index, variable in enumerate(final_key_bits)
    ]
    assert formula["counts"]["assumption_units"] == 16
    native_lines = formula["native_dimacs"].splitlines()
    assert all(f"{literal} 0" in native_lines for literal in assumptions)
    pure_lines = formula["pure_cnf"].splitlines()
    assert all(f"{literal} 0" in pure_lines for literal in assumptions)

    native_assignment = _propagate_cnf_and_xor(
        formula["cnf_clauses"] + [[literal] for literal in assumptions],
        formula["xor_clauses"],
        _key_assignment(formula, key),
        formula["num_vars"],
    )
    assert native_assignment is not None
    assert all(
        variable in native_assignment for variable in range(1, formula["num_vars"] + 1)
    )
    assert all(
        any(_literal_value(literal, native_assignment) for literal in clause)
        for clause in formula["cnf_clauses"] + [[literal] for literal in assumptions]
    )
    for record in formula["xor_clauses"]:
        parity = (
            sum(
                _literal_value(literal, native_assignment)
                for literal in record["literals"]
            )
            % 2
        )
        assert parity == record["rhs"]

    pure_count = formula["pure_cnf_num_vars"]
    pure_assignment = _propagate_cnf_and_xor(
        [[int(literal) for literal in line.split()[:-1]] for line in pure_lines[1:]],
        [],
        _key_assignment(formula, key),
        pure_count,
    )
    assert pure_assignment is not None
    assert all(
        any(_literal_value(literal, pure_assignment) for literal in clause)
        for clause in [
            [int(literal) for literal in line.split()[:-1]] for line in pure_lines[1:]
        ]
    )

    expected_round_keys = round_keys
    assert (
        tuple(
            _decode_word(native_assignment, word_bits)
            for word_bits in varmap["round_keys"]
        )
        == expected_round_keys
    )

    state_maps = varmap["states"]
    assert len(state_maps) == 4
    for observation_index, state_map in enumerate(state_maps):
        observation = formula["known_observations"][observation_index]
        state = tuple(observation["plaintext"])
        expected_states = [state]
        for round_key in round_keys:
            state = round_fn(state, round_key)
            expected_states.append(state)
        assert len(state_map["round_states"]) == rounds + 1
        for round_state, expected_state in zip(
            state_map["round_states"], expected_states, strict=True
        ):
            labels = ("x", "y") if cipher == "speck32_64" else ("left", "right")
            actual = tuple(
                _decode_word(native_assignment, round_state[label]) for label in labels
            )
            assert actual == expected_state


@pytest.mark.parametrize(
    ("cipher", "rounds", "key"),
    [
        ("speck32_64", 4, (0x1234, 0x5678, 0x9ABC, 0xDEF0)),
        ("simon32_64", 5, (0x1234, 0x5678, 0x9ABC, 0xDEF0)),
    ],
)
def test_unguided_build_never_reads_truth_or_verification_fields(
    cipher: str, rounds: int, key: tuple[int, ...]
) -> None:
    class GuardedInstance(dict[str, object]):
        protected = {
            "master_key",
            "final_round_subkey",
            "verification_pairs",
            "trailing_round_keys",
        }

        def __getitem__(self, name: str) -> object:
            if name in self.protected:
                raise AssertionError(f"builder read protected field {name}")
            return super().__getitem__(name)

        def get(self, name: str, default: object = None) -> object:
            if name in self.protected:
                raise AssertionError(f"builder read protected field {name}")
            return super().get(name, default)

    public_input = _make_instance(cipher, rounds, key)
    guarded = GuardedInstance(
        {
            **public_input,
            "master_key": list(key),
            "final_round_subkey": 123,
            "verification_pairs": [[[1, 2], [3, 4]]],
            "trailing_round_keys": [456],
        }
    )
    formula = build_formula(cipher, guarded, 2, 0, include_cnf=False)
    assert formula["assumption_literals"] == []
    assert formula["pure_cnf"] is None
    assert formula["pure_cnf_num_vars"] is None
    assert "verification_pairs" not in formula
    assert "master_key_values" not in formula


def test_observation_prefix_does_not_read_unselected_pairs() -> None:
    source = _make_instance("speck32_64", 4, (0x1234, 0x5678, 0x9ABC, 0xDEF0))

    class PrefixOnly(Sequence[dict[str, object]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int | slice) -> dict[str, object]:
            if index == 0:
                return source["pairs"][0]
            raise AssertionError("builder read an unselected pair")

    instance = {"rounds": 4, "pairs": PrefixOnly()}
    formula = build_formula("speck32_64", instance, 2, 0, include_cnf=False)
    assert formula["observation_count"] == 2


def test_reference_key_conflicts_with_corrupted_known_ciphertext() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("simon32_64", 5, key)
    instance["pairs"][0]["ciphertext0"][0] ^= 1
    formula = build_formula("simon32_64", instance, 2, 0, include_cnf=False)
    witness = _propagate_cnf_and_xor(
        formula["cnf_clauses"],
        formula["xor_clauses"],
        _key_assignment(formula, key),
        formula["num_vars"],
    )
    assert witness is None


def test_speck_native_output_matches_existing_encoder_bytes() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("speck32_64", 4, key)
    formula = build_formula("speck32_64", instance, 4, 0)
    pairs = [
        (tuple(obs["plaintext"]), tuple(obs["ciphertext"]))
        for obs in formula["known_observations"]
    ]
    assert (
        formula["native_dimacs"] == export_speck_to_dimacs(pairs, rounds=4).dimacs_text
    )


def test_speck_conditioned_output_matches_existing_pinned_key_units() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("speck32_64", 4, key)
    final_key = expand_speck_key(key, 4)[-1]
    instance["trailing_round_keys"] = [final_key]
    formula = build_formula("speck32_64", instance, 4, 1)
    pairs = [
        (tuple(obs["plaintext"]), tuple(obs["ciphertext"]))
        for obs in formula["known_observations"]
    ]
    assert (
        formula["native_dimacs"]
        == export_speck_to_dimacs(
            pairs, rounds=4, pinned_round_keys={3: final_key}
        ).dimacs_text
    )


def test_simon_native_output_matches_existing_recorder_bytes() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("simon32_64", 5, key)
    formula = build_formula("simon32_64", instance, 4, 0)
    pairs = [
        (tuple(obs["plaintext"]), tuple(obs["ciphertext"]))
        for obs in formula["known_observations"]
    ]
    model = SimonSatModel.from_known_pairs(pairs, rounds=5, solver=_NoSolveSink())
    assert formula["native_dimacs"] == model.to_dimacs()


def test_simon_conditioned_output_matches_existing_assumption_serializer() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("simon32_64", 5, key)
    trailing = expand_simon_key(key, 5)[-2:]
    instance["trailing_round_keys"] = list(trailing)
    formula = build_formula("simon32_64", instance, 4, 2)
    pairs = [
        (tuple(obs["plaintext"]), tuple(obs["ciphertext"]))
        for obs in formula["known_observations"]
    ]
    model = SimonSatModel.from_known_pairs(pairs, rounds=5, solver=_NoSolveSink())
    assumptions = model.assumptions_for_trailing_round_subkeys(trailing)
    assert formula["native_dimacs"] == model.to_dimacs(assumptions)


def test_simon_builder_never_constructs_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    import neural_sat_speck.simon.sat_model as simon_sat_module

    def fail_if_called():
        raise AssertionError("formula construction tried to create a solver")

    monkeypatch.setattr(simon_sat_module, "_new_solver", fail_if_called)
    instance = _make_instance("simon32_64", 5, (0x1234, 0x5678, 0x9ABC, 0xDEF0))
    build_formula("simon32_64", instance, 2, 0, include_cnf=False)


def test_native_only_build_skips_pure_cnf_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import neural_sat_speck.backend.formula as formula_module

    def fail_if_called(*args: object, **kwargs: object) -> tuple[str, int]:
        del args, kwargs
        raise AssertionError("native-only build invoked XOR-to-CNF conversion")

    monkeypatch.setattr(
        formula_module, "convert_native_xor_dimacs_to_pure_cnf", fail_if_called
    )
    instance = _make_instance("speck32_64", 4, (0x1234, 0x5678, 0x9ABC, 0xDEF0))
    formula = build_formula("speck32_64", instance, 2, 0, include_cnf=False)
    assert formula["pure_cnf"] is None
    assert formula["pure_cnf_num_vars"] is None


def test_renderer_preserves_signed_rhs_and_empty_xor_semantics() -> None:
    signed = render_native_dimacs(
        2,
        [],
        [{"literals": [-1, 2], "rhs": False}],
    )
    assert signed == "p cnf 2 1\nx 1 2 0\n"
    pure, _ = convert_native_xor_dimacs_to_pure_cnf(signed, 2)
    clauses = [
        [int(literal) for literal in line.split()[:-1]]
        for line in pure.splitlines()[1:]
    ]
    for values in itertools.product((False, True), repeat=2):
        assignment = {1: values[0], 2: values[1]}
        expected = (
            _literal_value(-1, assignment) ^ _literal_value(2, assignment)
        ) is False
        actual = all(
            any(_literal_value(literal, assignment) for literal in clause)
            for clause in clauses
        )
        assert actual == expected

    contradiction = render_native_dimacs(0, [], [{"literals": [], "rhs": True}])
    assert contradiction == "p cnf 0 1\n0\n"
    contradiction_cnf, _ = convert_native_xor_dimacs_to_pure_cnf(contradiction, 0)
    assert contradiction_cnf.splitlines()[0] == "p cnf 0 1"
    assert contradiction_cnf.splitlines()[1].split() == ["0"]

    tautology = render_native_dimacs(
        0,
        [],
        [
            {"literals": [], "rhs": False},
        ],
    )
    assert tautology == "p cnf 0 0\n"


def test_long_xor_conversion_allocates_fresh_auxiliaries_repeatably() -> None:
    native = render_native_dimacs(
        8,
        [],
        [{"literals": [1, 2, 3, 4, 5, 6], "rhs": True}],
    )
    first, first_variables = convert_native_xor_dimacs_to_pure_cnf(native, 8)
    second, second_variables = convert_native_xor_dimacs_to_pure_cnf(native, 8)
    assert first == second
    assert first_variables == second_variables == 11
    clause_ids = {
        abs(int(literal))
        for line in first.splitlines()[1:]
        for literal in line.split()[:-1]
    }
    assert {9, 10, 11} <= clause_ids


@pytest.mark.parametrize(
    ("observation_count", "trailing_key_count", "fields"),
    [
        (0, 0, {}),
        (3, 0, {}),
        (6, 0, {}),
        (2, -1, {}),
        (2, 5, {"trailing_round_keys": [1] * 5}),
        (2, 1, {"trailing_round_keys": [True]}),
        (2, 1, {"trailing_round_keys": [0x10000]}),
    ],
)
def test_invalid_counts_and_trailing_values_fail_closed(
    observation_count: int,
    trailing_key_count: int,
    fields: dict[str, object],
) -> None:
    instance = {**_make_instance("speck32_64", 4, (1, 2, 3, 4)), **fields}
    with pytest.raises((TypeError, ValueError)):
        build_formula("speck32_64", instance, observation_count, trailing_key_count)


def test_invalid_rounds_and_observation_words_fail_closed() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("speck32_64", 4, key)
    with pytest.raises(ValueError, match="rounds"):
        build_formula("speck32_64", {**instance, "rounds": 0}, 2, 0)

    malformed = deepcopy(instance)
    malformed["pairs"][0]["plaintext0"][1] = 0x10000
    with pytest.raises(ValueError, match="plaintext0"):
        build_formula("speck32_64", malformed, 2, 0)


def test_repeated_builds_are_deterministic_and_do_not_mutate_inputs() -> None:
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    instance = _make_instance("speck32_64", 4, key)
    instance["trailing_round_keys"] = [expand_speck_key(key, 4)[-1]]
    snapshot = deepcopy(instance)
    first = build_formula("speck32_64", instance, 4, 1)
    second = build_formula("speck32_64", instance, 4, 1)
    assert first == second
    assert instance == snapshot
