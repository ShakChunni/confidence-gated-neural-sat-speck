"""Synthetic lifecycle and small exact-SAT checks for queue traversal workers."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neural_sat_speck.backend.worker import Journal
from neural_sat_speck.followups import worker


def _known_observations(count: int = 128) -> list[list[list[int]]]:
    return [[[index, index + 1], [index + 2, index + 3]] for index in range(count)]


def _public(cipher: str = "simon32_64", rounds: int = 4) -> dict[str, Any]:
    return {
        "cipher": cipher,
        "rounds": rounds,
        "instance_id": "synthetic-public-instance",
        "ciphertext_pairs": [],
        "known_observations": _known_observations(),
    }


def _fresh_observations() -> list[list[list[int]]]:
    return [
        [[0xAAAA, 0xBBBB], [0xCCCC, 0xDDDD]],
        [[0xEEEE, 0x1111], [0x2222, 0x3333]],
    ]


def _job(cipher: str = "simon32_64", rounds: int = 4) -> dict[str, Any]:
    return {
        "cipher": cipher,
        "rounds": rounds,
        "candidate_request_seconds": 10.0,
    }


class _RecordingJournal(Journal):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.transitions: list[str | None] = []
        self.transition_times: list[float] = []

    def transition(self, phase: str | None) -> None:
        self.transitions.append(phase)
        self.transition_times.append(time.perf_counter())
        super().transition(phase)


class _ScriptedModel:
    def __init__(
        self,
        script: list[tuple[str, object | None]],
        events: list[tuple[Any, ...]],
        key: tuple[int, int, int, int] = (0x1111, 0x2222, 0x3333, 0x4444),
    ) -> None:
        self.script = list(script)
        self.events = events
        self.key = key
        self.solve_calls: list[tuple[Any, float]] = []

    def solve(self, assumptions, *, time_limit):
        self.solve_calls.append((assumptions, time_limit))
        self.events.append(("solve", id(self)))
        time.sleep(0.006)
        status, solution = self.script.pop(0)
        return SimpleNamespace(status=status, solution=solution)

    def master_key_from_solution(self, solution):
        assert solution == "model"
        return self.key

    def assumptions_for_trailing_round_subkeys(self, trailing_subkeys):
        self.events.append(("simon_assumptions", tuple(trailing_subkeys)))
        return tuple(trailing_subkeys)


def _run_traverse(
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
    public: dict[str, Any],
    fresh: list,
    candidates: tuple[tuple[int, int], ...],
    factory,
    verify,
    *,
    cipher: str = "simon32_64",
) -> tuple[dict[str, Any], _RecordingJournal]:
    directory.mkdir(parents=True, exist_ok=True)
    journal = _RecordingJournal(directory / "journal.json")
    monkeypatch.setattr(worker, "new_model", factory)
    monkeypatch.setattr(worker, "verify_observations", verify)
    result = worker.traverse(
        public,
        fresh,
        candidates,
        _job(cipher),
        time.perf_counter() + 10.0,
        journal,
        directory,
    )
    return result, journal


def test_traversal_reuses_model_after_unsat_but_starts_a_fresh_model_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = _public()
    fresh = _fresh_observations()
    events: list[tuple[Any, ...]] = []
    models: list[_ScriptedModel] = []
    scripts = [
        [("unsat", None), ("sat", "model")],
        [("unsat", None)],
    ]
    model_inputs = []

    def factory(cipher, known, rounds):
        assert cipher == "simon32_64" and rounds == 4
        assert known is public["known_observations"]
        model_inputs.append(known)
        model = _ScriptedModel(scripts[len(models)], events)
        models.append(model)
        return model

    verifier_calls = []

    def verify(cipher, rounds, key, pairs):
        verifier_calls.append((cipher, rounds, key, pairs))
        time.sleep(0.015)
        return True

    candidates = ((0x1234, 0x5678), (0x9ABC, 0xDEF0))
    result, journal = _run_traverse(
        monkeypatch,
        tmp_path / "ranked-arm",
        public,
        fresh,
        candidates,
        factory,
        verify,
    )

    assert result["outcome"] == "verified_recovery"
    assert result["recovered_candidate_index"] == 2
    assert [attempt["solver_status"] for attempt in result["attempts"]] == [
        "unsat",
        "sat",
    ]
    assert len(models) == 1
    assert [call[0] for call in models[0].solve_calls] == list(candidates)
    assert len(model_inputs[0]) == 128
    assert len(verifier_calls) == 2
    assert verifier_calls[0][3] is public["known_observations"]
    assert verifier_calls[1][3] is fresh
    assert verifier_calls[0][3] is not verifier_calls[1][3]

    # Solve and verification have disjoint journal phases; the sleeps make a
    # boundary error visible instead of relying on tiny timing differences.
    assert journal.transitions == [
        "initial_formula_construction",
        "solving",
        "decoding",
        "solving",
        "decoding",
        "verification",
        None,
    ]
    assert journal.completed["solving"] >= 0.01
    assert journal.completed["verification"] >= 0.025
    phase_span = journal.transition_times[-1] - journal.transition_times[0]
    assert sum(journal.completed.values()) <= phase_span + 0.02

    second, _ = _run_traverse(
        monkeypatch,
        tmp_path / "shuffled-arm",
        public,
        fresh,
        (candidates[0],),
        factory,
        verify,
    )
    assert second["outcome"] == "queue_exhausted"
    assert len(models) == 2
    assert models[0] is not models[1]
    assert len(models[1].solve_calls) == 1


@pytest.mark.parametrize("unresolved_status", ["solver_timeout", "unsat_or_timeout"])
def test_unresolved_status_rebuilds_before_next_candidate(
    unresolved_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _public()
    events: list[tuple[Any, ...]] = []
    models: list[_ScriptedModel] = []
    scripts = [[(unresolved_status, None)], [("unsat", None)]]

    def factory(_cipher, _known, _rounds):
        model = _ScriptedModel(scripts[len(models)], events)
        models.append(model)
        return model

    result, _ = _run_traverse(
        monkeypatch,
        tmp_path / unresolved_status,
        public,
        _fresh_observations(),
        ((0x1234, 0x5678), (0x9ABC, 0xDEF0)),
        factory,
        lambda *_args: pytest.fail("verification is only for SAT"),
    )

    assert result["outcome"] == "unresolved_candidates"
    assert result["unresolved_candidates"] == 1
    assert len(models) == 2
    assert [len(model.solve_calls) for model in models] == [1, 1]
    assert [event[0] for event in events if event[0] == "solve"] == [
        "solve",
        "solve",
    ]


def test_verification_failure_rebuilds_and_is_not_reported_as_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = _public()
    fresh = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    events: list[tuple[Any, ...]] = []
    models: list[_ScriptedModel] = []
    scripts = [[("sat", "model")], [("unsat", None)]]

    def factory(_cipher, _known, _rounds):
        model = _ScriptedModel(scripts[len(models)], events)
        models.append(model)
        return model

    verify_calls = []

    def verify(_cipher, _rounds, _key, pairs):
        verify_calls.append(pairs)
        return pairs is public["known_observations"]

    result, _ = _run_traverse(
        monkeypatch,
        tmp_path / "verification-rebuild",
        public,
        fresh,
        ((0x1234, 0x5678), (0x9ABC, 0xDEF0)),
        factory,
        verify,
    )

    assert result["outcome"] == "verification_failure"
    assert result["success"] is False
    assert result["verification_failures"] == 1
    assert result["attempts"][0]["verified_known"] is True
    assert result["attempts"][0]["verified_fresh"] is False
    assert verify_calls == [public["known_observations"], fresh]
    assert len(models) == 2
    assert len(models[0].solve_calls) == len(models[1].solve_calls) == 1


@pytest.mark.parametrize(
    ("known_count", "fresh_count"),
    [(127, 2), (128, 1)],
)
def test_traverse_rejects_wrong_observation_counts_before_model_construction(
    known_count: int,
    fresh_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _public()
    public["known_observations"] = _known_observations(known_count)
    directory = tmp_path / f"known-{known_count}-fresh-{fresh_count}"
    directory.mkdir()
    journal = _RecordingJournal(directory / "journal.json")
    factory_calls = []

    def forbidden_factory(*args):
        factory_calls.append(args)
        raise AssertionError("invalid observations must fail before model build")

    monkeypatch.setattr(worker, "new_model", forbidden_factory)
    with pytest.raises(
        ValueError, match="Replay needs128 known observations and2 fresh blocks"
    ):
        worker.traverse(
            public,
            _fresh_observations()[:fresh_count],
            ((0x1234, 0x5678),),
            _job(),
            time.perf_counter() + 10.0,
            journal,
            directory,
        )
    assert factory_calls == []
    assert journal.transitions == []


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_new_model_uses_only_the_leading_four_known_observations(
    cipher: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations = _known_observations()
    captured = {}
    sentinel = object()

    if cipher == "speck32_64":
        from neural_sat_speck.sat.model import SpeckSatModel

        model_type = SpeckSatModel
    else:
        from neural_sat_speck.simon.sat_model import SimonSatModel

        model_type = SimonSatModel

    def fake_factory(cls, known_pairs, *, rounds):
        captured["known_pairs"] = known_pairs
        captured["rounds"] = rounds
        return sentinel

    monkeypatch.setattr(
        model_type,
        "from_known_pairs",
        classmethod(fake_factory),
    )

    assert worker.new_model(cipher, observations, rounds=4) is sentinel
    expected = [
        (tuple(pt), tuple(ct))
        for pt, ct in ((item[0], item[1]) for item in observations[:4])
    ]
    assert captured["known_pairs"] == expected
    assert captured["rounds"] == 4


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_traverse_builds_chronological_round_key_assumptions(
    cipher: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _public(cipher)
    events: list[tuple[Any, ...]] = []
    model = _ScriptedModel([("unsat", None)], events)
    candidate = (0x1234, 0x5678)
    monkeypatch.setattr(worker, "new_model", lambda *_args: model)

    if cipher == "speck32_64":
        from neural_sat_speck.sat import multi_round

        seen = []

        def assumptions_spy(received_model, round_key_values):
            seen.append((received_model, dict(round_key_values)))
            return ("speck-assumption-spy",)

        monkeypatch.setattr(
            multi_round,
            "assumptions_for_round_keys",
            assumptions_spy,
        )
        expected_assumptions = ("speck-assumption-spy",)
    else:
        seen = []
        expected_assumptions = candidate

    result, _ = _run_traverse(
        monkeypatch,
        tmp_path / cipher,
        public,
        _fresh_observations(),
        (candidate,),
        lambda *_args: model,
        lambda *_args: pytest.fail("UNSAT candidates are not verified"),
        cipher=cipher,
    )

    assert result["outcome"] == "queue_exhausted"
    assert model.solve_calls[0][0] == expected_assumptions
    if cipher == "speck32_64":
        assert seen == [(model, {2: candidate[0], 3: candidate[1]})]
    else:
        assert events[0] == ("simon_assumptions", candidate)


def _rotl16(word: int, amount: int) -> int:
    word &= 0xFFFF
    return ((word << amount) | (word >> (16 - amount))) & 0xFFFF


def _rotr16(word: int, amount: int) -> int:
    word &= 0xFFFF
    return ((word >> amount) | (word << (16 - amount))) & 0xFFFF


def _reference_speck_encrypt(
    plaintext: tuple[int, int],
    round_keys: tuple[int, ...],
) -> tuple[int, int]:
    x, y = plaintext
    for round_key in round_keys:
        x = ((_rotr16(x, 7) + y) & 0xFFFF) ^ round_key
        y = _rotl16(y, 2) ^ x
    return x & 0xFFFF, y & 0xFFFF


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_four_round_cipher_sat_trailing_assumptions_and_reference_verification(
    cipher: str,
) -> None:
    pytest.importorskip("pycryptosat")

    rounds = 4
    if cipher == "speck32_64":
        from neural_sat_speck.sat.model import SpeckSatModel
        from neural_sat_speck.speck.cipher import (
            SPECK32_64_KAT_KEY,
            SPECK32_64_KAT_PLAINTEXT,
            encrypt_block,
        )
        from neural_sat_speck.speck.key_schedule import expand_key
        from tests.fixtures.speck32_64_vectors import EXPECTED_ROUND_KEYS

        key = SPECK32_64_KAT_KEY
        plaintexts = (
            SPECK32_64_KAT_PLAINTEXT,
            (0x0000, 0x0000),
            (0x1234, 0xABCD),
            (0xDEAD, 0xBEEF),
        )
        round_keys = expand_key(key, rounds)
        assert round_keys == EXPECTED_ROUND_KEYS[:rounds]

        def reference_encrypt(pt):
            return _reference_speck_encrypt(pt, round_keys)

        assert encrypt_block(plaintexts[0], key, rounds) == reference_encrypt(
            plaintexts[0]
        )
        model_type = SpeckSatModel
    else:
        from neural_sat_speck.simon.cipher import (
            SIMON32_64_KAT_KEY,
            SIMON32_64_KAT_PLAINTEXT,
            encrypt_block,
        )
        from neural_sat_speck.simon.key_schedule import expand_key
        from neural_sat_speck.simon.sat_model import SimonSatModel
        from tests.fixtures.simon_reference_oracle import (
            oracle_simon32_64_encrypt,
            oracle_simon32_64_key_schedule,
        )

        key = SIMON32_64_KAT_KEY
        plaintexts = (
            SIMON32_64_KAT_PLAINTEXT,
            (0x0000, 0x0000),
            (0x1234, 0xABCD),
            (0xDEAD, 0xBEEF),
        )
        round_keys = expand_key(key, rounds)
        assert round_keys == oracle_simon32_64_key_schedule(key, rounds)

        def reference_encrypt(pt):
            return oracle_simon32_64_encrypt(pt, key, rounds)

        assert encrypt_block(plaintexts[0], key, rounds) == reference_encrypt(
            plaintexts[0]
        )
        model_type = SimonSatModel

    fresh_plaintexts = ((0x0BAD, 0xF00D), (0x2468, 0x1357))
    known_pairs = tuple(
        (plaintext, reference_encrypt(plaintext)) for plaintext in plaintexts
    )
    fresh_pairs = tuple(
        (plaintext, reference_encrypt(plaintext)) for plaintext in fresh_plaintexts
    )
    known_observations = [[list(pt), list(ct)] for pt, ct in known_pairs]
    fresh_observations = [[list(pt), list(ct)] for pt, ct in fresh_pairs]

    model = worker.new_model(cipher, known_observations, rounds)
    assert isinstance(model, model_type)
    assert len(model.known_pairs) == 4

    master_key_assumptions = model.assumptions_for_master_key(key)
    master_only = model.solve(master_key_assumptions, time_limit=5.0)
    assert master_only.status == "sat"
    assert master_only.solution is not None

    true_trailing = (round_keys[-2], round_keys[-1])
    wrong_trailing = (true_trailing[0] ^ 1, true_trailing[1])
    contradictory = model.solve(
        master_key_assumptions
        + worker.assumptions(model, cipher, wrong_trailing, rounds),
        time_limit=5.0,
    )
    assert contradictory.status == "unsat"
    assert contradictory.timed_out is False

    restored = model.solve(
        master_key_assumptions
        + worker.assumptions(model, cipher, true_trailing, rounds),
        time_limit=5.0,
    )
    assert restored.status == "sat"
    assert restored.solution is not None
    recovered = model.master_key_from_solution(restored.solution)
    assert recovered == key
    if cipher == "simon32_64":
        assert model.round_keys_from_solution(restored.solution) == round_keys

    observations = known_observations + fresh_observations
    assert worker.verify_observations(cipher, rounds, list(recovered), observations)
    assert all(
        reference_encrypt(tuple(pair[0])) == tuple(pair[1]) for pair in observations
    )
