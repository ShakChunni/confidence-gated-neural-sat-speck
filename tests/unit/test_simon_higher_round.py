"""SIMON11 runner contract tests with fixed, opened test-only inputs.

These tests never call the SIMON instance generator, load ranking assets, or
invoke a campaign CLI. Ranking and completion are stubbed for runner-control
tests; one separate native-completer test checks a single planted guess only.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstance,
    SimonAttackPair,
)
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.verification import verify_simon_master_key
from scripts import bruteforce_completion as bf
from scripts import run_simon_higher_round as runner
from tests.fixtures.simon_reference_oracle import (
    oracle_simon32_64_encrypt,
    oracle_simon32_64_key_schedule,
)

ROUNDS = 11
MASTER_KEY = (0x1918, 0x1110, 0x0908, 0x0100)  # (k3, k2, k1, k0)
SIMON_R11_KEYS = (
    0x0100,
    0x0908,
    0x1110,
    0x1918,
    0x71C3,
    0xB649,
    0x56D4,
    0xE070,
    0xF15A,
    0xC535,
    0xDD94,
)
KNOWN_PAIR_COUNT = 64
KNOWN_BLOCK_COUNT = 2 * KNOWN_PAIR_COUNT


def _fixed_instance() -> SimonAttackInstance:
    """Construct one fixed-key fixture without calling any seed/data generator."""
    pairs = []
    for index in range(KNOWN_PAIR_COUNT):
        pt0 = (index, (0x1234 + 257 * index) & 0xFFFF)
        pt1 = (pt0[0], pt0[1] ^ 0x0040)
        pairs.append(
            SimonAttackPair(
                plaintext0=pt0,
                plaintext1=pt1,
                ciphertext0=oracle_simon32_64_encrypt(pt0, MASTER_KEY, ROUNDS),
                ciphertext1=oracle_simon32_64_encrypt(pt1, MASTER_KEY, ROUNDS),
            )
        )

    fresh_plaintexts = ((0x8000, 0x1357), (0x8001, 0x2468))
    verification_pairs = tuple(
        (pt, oracle_simon32_64_encrypt(pt, MASTER_KEY, ROUNDS))
        for pt in fresh_plaintexts
    )
    return SimonAttackInstance(
        instance_id="fixed-simon11-test-key",
        split_role="demo",
        master_key=MASTER_KEY,
        rounds=ROUNDS,
        pairs=tuple(pairs),
        verification_pairs=verification_pairs,
    )


@pytest.fixture
def instance() -> SimonAttackInstance:
    return _fixed_instance()


@pytest.fixture
def native_completer(tmp_path: Path):
    """Build into pytest's temporary directory; never touch runtime assets."""
    if shutil.which("clang") is None:
        pytest.skip("native brute-force completer requires clang")
    return bf.BruteforceLibrary(bf.build_library(tmp_path / "native"))


def _truth_order(target: int) -> np.ndarray:
    domain = np.arange(1 << 16, dtype=np.uint16)
    return np.concatenate(
        (np.asarray([target], dtype=np.uint16), domain[domain != target])
    )[: runner.RESCORE_TOP]


@dataclass(frozen=True)
class _FakeModel:
    rounds: int


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def perf_counter(self) -> float:
        return self.value


class _StubCompleter:
    def __init__(
        self,
        key,
        *,
        clock: _FakeClock | None = None,
        wrong=False,
        events=None,
    ) -> None:
        self.key = tuple(key)
        self.clock = clock
        self.wrong = wrong
        self.events = events
        self.calls: list[dict[str, object]] = []

    def complete(self, cipher, rounds, trailing, pt_blocks, ct_blocks, lo, hi):
        self.calls.append(
            {
                "cipher": cipher,
                "rounds": rounds,
                "trailing": tuple(trailing),
                "pt_blocks": tuple(pt_blocks),
                "ct_blocks": tuple(ct_blocks),
                "lo": lo,
                "hi": hi,
            }
        )
        if self.events is not None:
            self.events.append(
                ("complete", rounds, None if self.clock is None else self.clock.value)
            )
        if self.clock is not None:
            self.clock.value = runner.BUDGET - 0.25
        if self.wrong and len(self.calls) > 1:
            return SimpleNamespace(
                found=False,
                master_key=None,
                tested=hi - lo,
                found_guess=None,
            )
        recovered = list(self.key)
        if self.wrong:
            recovered[0] ^= 1
            found_guess = lo
        else:
            found_guess = oracle_simon32_64_key_schedule(self.key, rounds)[rounds - 4]
        return SimpleNamespace(
            found=True,
            master_key=tuple(recovered),
            tested=found_guess - lo + 1,
            found_guess=found_guess,
        )


def _patch_neural_path(
    monkeypatch, instance, *, completer=None, clock=None, events=None
):
    """Patch every model/ranker boundary; no real asset is opened."""
    words = oracle_simon32_64_key_schedule(instance.master_key, ROUNDS)
    base_cipher_pairs = tuple(
        (pair.ciphertext0, pair.ciphertext1) for pair in instance.pairs
    )

    def load_vv(name: str):
        if events is not None:
            events.append(
                ("load", int(name[2:]), None if clock is None else clock.value)
            )
        return _FakeModel(int(name[2:]))

    def rank_vv(model: _FakeModel, pairs):
        stage = model.rounds
        if events is not None:
            events.append(("rank", stage, None if clock is None else clock.value))

        # A fixed reference check confirms each conditional stage receives the
        # ciphertexts after the already-fixed newer round keys were peeled.
        peeled = list(base_cipher_pairs)
        for key_index in range(10, stage, -1):
            peeled = runner.peel(peeled, words[key_index])
        assert tuple(pairs) == tuple(peeled)

        if (
            clock is not None
            and stage == 10
            and getattr(clock, "expire_after_stage1", False)
        ):
            clock.value = runner.BUDGET + 0.01
        return _truth_order(words[stage])

    monkeypatch.setattr(runner, "_validate_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_vv", load_vv)
    monkeypatch.setattr(runner, "rank_vv", rank_vv)
    monkeypatch.setattr(
        runner,
        "rank_stage1_ondemand",
        lambda *args, **kwargs: pytest.fail("unexpected classical stage-1 dispatch"),
    )
    monkeypatch.setattr(
        runner,
        "rank_ondemand_batch",
        lambda *args, **kwargs: pytest.fail("unexpected classical row dispatch"),
    )
    monkeypatch.setattr(
        runner,
        "rank_classical_domain",
        lambda *args, **kwargs: pytest.fail("unexpected classical-domain dispatch"),
    )
    monkeypatch.setattr(
        runner,
        "FullPredictedDdt",
        lambda *args, **kwargs: pytest.fail("unexpected table load"),
    )
    if completer is not None:
        monkeypatch.setattr(runner, "BF", completer)
    return words


def test_simons11_schedule_word_order_and_chronological_trailing_window():
    expected = oracle_simon32_64_key_schedule(MASTER_KEY, ROUNDS)
    assert expected == SIMON_R11_KEYS
    assert expand_key(MASTER_KEY, ROUNDS) == SIMON_R11_KEYS
    assert SIMON_R11_KEYS[7:11] == (0xE070, 0xF15A, 0xC535, 0xDD94)
    assert SIMON_R11_KEYS[8:11] == (0xF15A, 0xC535, 0xDD94)


def test_peel_order_matches_independent_reduced_round_reference(instance):
    states = [(pair.ciphertext0, pair.ciphertext1) for pair in instance.pairs]
    keys = oracle_simon32_64_key_schedule(MASTER_KEY, ROUNDS)

    for key_index, remaining_rounds in ((10, 10), (9, 9), (8, 8)):
        states = runner.peel(states, keys[key_index])
        expected = [
            (
                oracle_simon32_64_encrypt(
                    pair.plaintext0, MASTER_KEY, remaining_rounds
                ),
                oracle_simon32_64_encrypt(
                    pair.plaintext1, MASTER_KEY, remaining_rounds
                ),
            )
            for pair in instance.pairs
        ]
        assert states == expected


def test_queue_order_ties_are_deterministic_and_duplicate_paths_keep_first():
    paths = [
        (1, 1, 0, (50, 40, 30)),
        (0, 2, 0, (60, 50, 40)),
        (0, 2, 0, (60, 50, 40)),  # duplicate candidate path
        (0, 0, 0, (9, 8, 7)),
        (1, 0, 1, (20, 21, 22)),
        (1, 0, 1, (19, 21, 22)),  # same ranks; deterministic word tie-break
    ]
    expected = [
        (7, 8, 9),
        (40, 50, 60),
        (22, 21, 19),
        (22, 21, 20),
        (30, 40, 50),
    ]
    assert runner.queue_order(paths) == expected
    assert runner.queue_order(list(reversed(paths))) == expected


def test_native_simon11_completion_checks_128_known_blocks_and_two_fresh_pairs(
    native_completer, instance
):
    keys = oracle_simon32_64_key_schedule(MASTER_KEY, ROUNDS)
    assert (
        native_completer.master_from_window("simon", ROUNDS, keys[7:11]) == MASTER_KEY
    )

    known = instance.known_pairs
    assert len(instance.pairs) == 64
    assert len(known) == KNOWN_BLOCK_COUNT == 128
    assert len(instance.verification_pairs) == 2
    known_plaintexts = [pt for pt, _ in known]
    fresh_plaintexts = [pt for pt, _ in instance.verification_pairs]
    assert len(set(known_plaintexts)) == KNOWN_BLOCK_COUNT
    assert len(set(fresh_plaintexts)) == 2
    assert not (set(known_plaintexts) & set(fresh_plaintexts))
    assert _fixed_instance().verification_pairs == instance.verification_pairs

    pts = [pt for pt, _ in known]
    cts = [ct for _, ct in known]
    unknown = keys[7]  # window is [k7, k8, k9, k10]
    result = native_completer.complete(
        "simon",
        ROUNDS,
        keys[8:11],  # chronological [k8, k9, k10]
        pts,
        cts,
        unknown,
        unknown + 1,  # one planted guess; never enumerate the full domain
    )
    assert result.found
    assert result.master_key == MASTER_KEY
    assert result.found_guess == unknown
    assert result.tested == 1

    verified = verify_simon_master_key(
        result.master_key,
        known,
        instance.verification_pairs,
        ROUNDS,
    )
    assert verified.known_pairs_verified
    assert verified.independent_pairs_verified
    assert verified.verified

    # A match on the first 127 known blocks must not pass the C-side check.
    corrupted = list(cts)
    corrupted[-1] = (corrupted[-1][0] ^ 1, corrupted[-1][1])
    rejected = native_completer.complete(
        "simon", ROUNDS, keys[8:11], pts, corrupted, unknown, unknown + 1
    )
    assert not rejected.found
    assert rejected.master_key is None
    assert rejected.tested == 1

    # Fresh verification is a separate Python reference check, not a C input.
    fresh_pt, fresh_ct = instance.verification_pairs[0]
    corrupted_fresh = (
        (fresh_pt, (fresh_ct[0] ^ 1, fresh_ct[1])),
        *instance.verification_pairs[1:],
    )
    fresh_rejected = verify_simon_master_key(MASTER_KEY, known, corrupted_fresh, ROUNDS)
    assert fresh_rejected.known_pairs_verified
    assert not fresh_rejected.independent_pairs_verified
    assert not fresh_rejected.verified


def test_run_cell_success_uses_stubbed_rankers_and_fresh_verification(
    monkeypatch, instance
):
    words = oracle_simon32_64_key_schedule(MASTER_KEY, ROUNDS)
    events = []
    completer = _StubCompleter(MASTER_KEY, events=events)
    _patch_neural_path(monkeypatch, instance, completer=completer, events=events)

    truth_expansions = []
    original_expand = runner.expand_key
    original_verify = runner.verify_simon_master_key

    def tracked_expand(key, rounds):
        truth_expansions.append(tuple(key))
        events.append(("truth", rounds, None))
        return original_expand(key, rounds)

    def tracked_verify(*args, **kwargs):
        events.append(
            (
                "verify",
                args[3] if len(args) > 3 else kwargs.get("rounds"),
                None,
            )
        )
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(runner, "expand_key", tracked_expand)
    monkeypatch.setattr(runner, "verify_simon_master_key", tracked_verify)
    result = runner.run_cell("S11-N", instance)

    assert result["outcome"] == "recovered"
    assert result["attempts"] == 1
    assert result["verification_rejections"] == 0
    assert result["verification"] == {
        "known_pairs_verified": True,
        "independent_pairs_verified": True,
        "verified": True,
    }
    assert tuple(result["recovered_key"]) == MASTER_KEY
    assert result["exact_key_match"] is True
    assert result["true_queue_position"] == 0
    assert result["stage1_true_position"] == 1
    assert result["stage1_true_rank"] == 1
    assert result["total_s"] >= 0
    assert result["prep_s"] >= 0
    assert all(result[f"stage{i}_s"] >= 0 for i in (1, 2, 3))

    assert len(completer.calls) == 1
    call = completer.calls[0]
    assert call["cipher"] == "simon"
    assert call["rounds"] == ROUNDS
    assert call["trailing"] == words[8:11]
    assert len(call["pt_blocks"]) == len(call["ct_blocks"]) == KNOWN_BLOCK_COUNT
    assert 0 <= call["lo"] < call["hi"] <= (1 << 16)
    assert any(event[0] == "rank" and event[1] == 10 for event in events)
    assert any(event[0] == "rank" and event[1] == 9 for event in events)
    assert any(event[0] == "rank" and event[1] == 8 for event in events)
    assert truth_expansions == [MASTER_KEY]
    verify_index = max(i for i, event in enumerate(events) if event[0] == "verify")
    truth_index = min(i for i, event in enumerate(events) if event[0] == "truth")
    assert verify_index < truth_index


def test_run_cell_rejects_completion_that_fails_fresh_verification(
    monkeypatch, instance
):
    completer = _StubCompleter(MASTER_KEY, wrong=True)
    _patch_neural_path(monkeypatch, instance, completer=completer)

    result = runner.run_cell("S11-N", instance)

    assert result["outcome"] != "recovered"
    assert result["recovered_key"] is None
    assert result["verification_rejections"] >= 1
    assert result["verification"]["verified"] is False


def test_run_cell_stops_dispatch_after_ranking_crosses_deadline(monkeypatch, instance):
    clock = _FakeClock()
    clock.expire_after_stage1 = True
    completer = _StubCompleter(MASTER_KEY, clock=clock)
    events = []
    _patch_neural_path(
        monkeypatch, instance, completer=completer, clock=clock, events=events
    )
    monkeypatch.setattr(runner.time, "perf_counter", clock.perf_counter)

    result = runner.run_cell("S11-N", instance)

    assert result["outcome"] == "deadline"
    assert completer.calls == []
    assert [event[1] for event in events if event[0] == "rank"] == [10]
    assert all(event[2] is None or event[2] < runner.BUDGET for event in events)


def test_run_cell_does_not_accept_verification_completed_after_deadline(
    monkeypatch, instance
):
    clock = _FakeClock()
    completer = _StubCompleter(MASTER_KEY, clock=clock)
    _patch_neural_path(monkeypatch, instance, completer=completer, clock=clock)
    monkeypatch.setattr(runner.time, "perf_counter", clock.perf_counter)

    real_verify = runner.verify_simon_master_key
    verification_calls = []

    def late_verify(*args, **kwargs):
        verification_calls.append(clock.value)
        verified = real_verify(*args, **kwargs)
        clock.value = runner.BUDGET + 0.01
        return verified

    monkeypatch.setattr(runner, "verify_simon_master_key", late_verify)
    result = runner.run_cell("S11-N", instance)

    assert verification_calls == [runner.BUDGET - 0.25]
    assert result["outcome"] == "deadline"
    assert result["recovered_key"] is None


@pytest.mark.parametrize("field", ["rounds", "pairs", "verification_pairs", "overlap"])
def test_run_cell_rejects_malformed_instance_before_ranking(
    monkeypatch, instance, field
):
    if field == "rounds":
        malformed = SimpleNamespace(**{**instance.__dict__, "rounds": 10})
    elif field == "pairs":
        malformed = SimpleNamespace(
            **{**instance.__dict__, "pairs": instance.pairs[:-1]}
        )
    elif field == "verification_pairs":
        malformed = SimpleNamespace(
            **{
                **instance.__dict__,
                "verification_pairs": instance.verification_pairs[:1],
            }
        )
    else:
        pairs = list(instance.pairs)
        fresh_pt, fresh_ct = instance.verification_pairs[0]
        first = pairs[0]
        pairs[0] = SimonAttackPair(
            plaintext0=fresh_pt,
            plaintext1=first.plaintext1,
            ciphertext0=fresh_ct,
            ciphertext1=first.ciphertext1,
        )
        malformed = SimpleNamespace(**{**instance.__dict__, "pairs": tuple(pairs)})

    malformed.known_pairs = tuple(
        observation
        for pair in malformed.pairs
        for observation in (
            (pair.plaintext0, pair.ciphertext0),
            (pair.plaintext1, pair.ciphertext1),
        )
    )

    called = []
    monkeypatch.setattr(
        runner, "rank_vv", lambda *args, **kwargs: called.append("rank")
    )
    result = runner.run_cell("S11-N", malformed)
    assert result["outcome"] == "error"
    assert called == []


def test_validate_seed_role_accepts_only_demo_without_explicit_role():
    assert runner.validate_seed_role(999999) == "demo"
    assert runner.validate_seed_role(999999, "demo") == "demo"
    with pytest.raises((TypeError, ValueError)):
        runner.validate_seed_role(1234567)
    assert runner.validate_seed_role(1234567, "development") == "development"
    with pytest.raises((TypeError, ValueError)):
        runner.validate_seed_role(999999, "development")


@pytest.mark.parametrize(
    "seed",
    [
        98001,
        97001,
        99001,
        101001,
        201001,
        300001,
        406737,
        669981,
        6503664708853956672,
        701004,
    ],
)
@pytest.mark.parametrize("role", ["demo", "development", "feasibility", "confirmation"])
def test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds(seed, role):
    with pytest.raises((TypeError, ValueError)):
        runner.validate_seed_role(seed, role)


def test_parse_args_requires_run_parameters_and_limits_tensorflow_threads(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    args = runner.parse_args(
        [
            "--seed",
            "999999",
            "--keys",
            "1",
            "--ledger",
            str(ledger),
            "--tf-intra-op-threads",
            "8",
        ]
    )
    assert args.seed == 999999
    assert args.keys == 1
    assert args.ledger == ledger
    assert args.tf_intra_op_threads == 8
    assert args.role == "demo"

    with pytest.raises(SystemExit):
        runner.parse_args([])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--seed",
                "999999",
                "--keys",
                "1",
                "--ledger",
                str(ledger),
                "--tf-intra-op-threads",
                "4",
            ]
        )


def test_ledger_creation_is_exclusive_and_append_requires_existing_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    header = {"type": "header", "rounds": ROUNDS}
    record = {"type": "cell", "outcome": "deadline"}

    runner.create_ledger(path, header)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == header
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.create_ledger(path, {"type": "replacement"})
    assert path.read_bytes() == before

    runner.append_ledger(path, record)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [header, record]

    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        runner.append_ledger(missing, record)
    assert not missing.exists()


@pytest.mark.parametrize("arm", ["S11-C", "S11-H"])
def test_classical_and_hybrid_stages_batch_parents_once(monkeypatch, instance, arm):
    keys = SIMON_R11_KEYS
    observed = {"stage1": 0, "stage2": 0, "stage3": 0}
    base = [(p.ciphertext0, p.ciphertext1) for p in instance.pairs]
    monkeypatch.setattr(runner, "_validate_table", lambda *a: None)
    monkeypatch.setattr(runner, "BF", _StubCompleter(MASTER_KEY))

    def stage1(pairs, **kwargs):
        observed["stage1"] += 1
        assert pairs == base
        assert kwargs["screen_pairs"] == 16
        assert kwargs["full_pairs"] == 64
        assert kwargs["rescore_top"] == 512
        assert kwargs["ddt_7r_path"] == runner.ASSETS["table8"]
        return SimpleNamespace(order=_truth_order(keys[10])), {"stub": True}

    def neural(model, pairs):
        assert arm == "S11-H" and model.rounds == 10
        assert pairs == base
        return _truth_order(keys[10])

    def stage2(pair_sets, tool, table, *, check_budget):
        observed["stage2"] += 1
        assert len(pair_sets) == 8 and all(len(p) == 64 for p in pair_sets)
        assert pair_sets == [
            runner.peel(base, int(k)) for k in _truth_order(keys[10])[:8]
        ]
        assert tool == runner.ASSETS["row_tool"]
        assert table == runner.ASSETS["table7"]
        assert check_budget() > 0
        return [_truth_order(keys[9]) for _ in pair_sets], {"stub": True}

    def load_table(path, expected_rounds):
        assert path == runner.ASSETS["table7"] and expected_rounds == 7
        return object()

    def stage3(pairs, table, **kwargs):
        observed["stage3"] += 1
        index = observed["stage3"] - 1
        k10 = int(_truth_order(keys[10])[index // 8])
        k9 = int(_truth_order(keys[9])[index % 8])
        assert pairs == runner.peel(runner.peel(base, k10), k9)
        assert kwargs == {
            "expected_rounds": 7,
            "screen_pairs": 16,
            "full_pairs": 64,
            "rescore_top": 512,
        }
        return SimpleNamespace(order=_truth_order(keys[8]))

    monkeypatch.setattr(runner, "rank_stage1_ondemand", stage1)
    monkeypatch.setattr(runner, "load_vv", lambda name: _FakeModel(int(name[2:])))
    monkeypatch.setattr(runner, "rank_vv", neural)
    monkeypatch.setattr(runner, "rank_ondemand_batch", stage2)
    monkeypatch.setattr(runner, "FullPredictedDdt", load_table)
    monkeypatch.setattr(runner, "rank_classical_domain", stage3)
    record = runner.run_cell(arm, instance)
    assert record["outcome"] == "recovered"
    assert record["queue_len"] == 8 * 8 * 16
    assert observed == {"stage1": int(arm == "S11-C"), "stage2": 1, "stage3": 64}


def test_row_score_ties_use_word_order_and_nonfinite_scores_fail():
    pairs = [((0x1234, 0x5678), (0x1234, 0x5638))]
    rows = np.zeros((1, 1 << 16))
    assert runner._score_rows(pairs, rows, screen_pairs=1, rescore_top=4).tolist() == [
        0,
        1,
        2,
        3,
    ]
    rows[0, 0] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        runner._score_rows(pairs, rows, screen_pairs=1)


def test_bad_table_header_is_rejected_without_dense_allocation(tmp_path):
    table = tmp_path / "toy.bin"
    table.write_bytes(b"not a table")
    with pytest.raises(ValueError, match="header"):
        runner._validate_table(table, 8)


@pytest.mark.parametrize("probability", [-1.0, float("nan"), 1.01])
def test_batch_rows_reject_invalid_probabilities(monkeypatch, tmp_path, probability):
    monkeypatch.setattr(runner, "TEMP_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_validate_table", lambda *a: None)

    def fake_tool(command, **kwargs):
        # All toy pairs have the same anchor, so output exactly one queried row.
        assert len(command[3:]) == 1
        np.full(1 << 16, probability, dtype="<f8").tofile(command[2])
        return SimpleNamespace(
            stdout=json.dumps({"input_rounds": 7, "num_query_rows": 1})
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_tool)
    with pytest.raises(ValueError, match="probabilities"):
        runner.rank_ondemand_batch(
            [[((0, 0), (0, 0))] * 64], Path("tool"), Path("table")
        )


def test_header_pins_sources_and_only_assets_used_by_selected_arm(
    monkeypatch, tmp_path
):
    args = runner.parse_args(
        [
            "--seed",
            "999999",
            "--role",
            "demo",
            "--keys",
            "1",
            "--ledger",
            str(tmp_path / "ledger"),
            "--tf-intra-op-threads",
            "1",
            "--arms",
            "S11-N",
        ]
    )
    hashed = []
    monkeypatch.setattr(runner, "sha256", lambda p: hashed.append(Path(p)) or "a" * 64)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout="f" * 40 if "rev-parse" in command else "?? runner"
        ),
    )
    monkeypatch.setattr(runner, "TF", None)
    monkeypatch.setattr(runner.platform, "platform", lambda: "toy-platform")
    header = runner.make_header(args, ["S11-N"], tmp_path / "lib")
    assert set(header["assets"]) == {"vv10", "vv9", "vv8"}
    assert header["tf_intra_op_threads"] == 1
    assert header["tf_inter_op_threads"] == 1
    assert header["runner_dirty"] is True
    assert header["command"][-len(args.command_line) :] == args.command_line
    assert set(header["source_shas"]) == set(runner.SOURCES + runner.SUPPORT_SOURCES)
    assert runner.ASSETS["table8"] not in hashed


@pytest.mark.parametrize(
    "blocked",
    [
        "existing_ledger",
        "protected_seed",
        "missing_threads",
        "role_specific_seed",
        "unfrozen_role",
    ],
)
def test_rejected_cli_never_draws_data_or_initializes_workers(
    monkeypatch, tmp_path, blocked
):
    ledger = tmp_path / "ledger"
    args = ["--seed", "999999", "--keys", "1", "--ledger", str(ledger)]
    if blocked != "missing_threads":
        args += ["--tf-intra-op-threads", "1"]
    if blocked == "existing_ledger":
        ledger.write_text("preserved")
    elif blocked == "protected_seed":
        args[1] = "701004"
        args += ["--role", "confirmation"]
    elif blocked == "role_specific_seed":
        args[1] = "260926"
        args += ["--role", "confirmation"]
    elif blocked == "unfrozen_role":
        args[1] = "1234567"
        args += ["--role", "development"]

    def prohibited(*args, **kwargs):
        pytest.fail("rejected preflight reached a forbidden boundary")

    for name in [
        "generate_simon_attack_instances",
        "configure_tensorflow",
        "build_library",
        "make_header",
        "run_cell",
    ]:
        monkeypatch.setattr(runner, name, prohibited)
    with pytest.raises(SystemExit):
        runner.main(args)
    if blocked == "existing_ledger":
        assert ledger.read_text() == "preserved"
    else:
        assert not ledger.exists()


def test_loader_time_is_charged_and_no_rank_starts_at_exact_deadline(
    monkeypatch, instance
):
    clock = _FakeClock()
    monkeypatch.setattr(runner.time, "perf_counter", clock.perf_counter)
    loaded = []

    def slow_loader(name):
        loaded.append(name)
        clock.value = runner.BUDGET
        return _FakeModel(10)

    monkeypatch.setattr(runner, "load_vv", slow_loader)
    monkeypatch.setattr(
        runner, "rank_vv", lambda *a: pytest.fail("rank after deadline")
    )
    result = runner.run_cell("S11-N", instance)
    assert result["outcome"] == "deadline"
    assert result["total_s"] == runner.BUDGET
    assert result["attempts"] == 0
    assert loaded == ["vv10"]


def test_vv10_pin_registration_is_scoped_to_one_load(monkeypatch):
    before = dict(runner.published_vv._CHECKPOINTS)
    marker = object()

    def loader(path, rounds):
        assert rounds == 10 and path == runner.ASSETS["vv10"]
        assert runner.published_vv._CHECKPOINTS[10] == runner.VV10_PINS
        return marker, {"stub": True}

    monkeypatch.setattr(runner.published_vv, "load_published_vv_checkpoint", loader)
    assert runner.load_vv("vv10") is marker
    assert runner.published_vv._CHECKPOINTS == before


def test_serial_schedule_rotates_by_absolute_key_index_with_stubbed_cells(
    monkeypatch, tmp_path, instance
):
    ledger = tmp_path / "serial.jsonl"
    monkeypatch.setattr(runner, "TEMP_ROOT", tmp_path / "tmp")
    monkeypatch.setattr(runner, "TF", None)
    monkeypatch.setattr(runner, "BF", None)
    selections = []
    monkeypatch.setattr(runner, "configure_tensorflow", selections.append)
    monkeypatch.setattr(runner, "build_library", lambda: tmp_path / "stub-library")
    monkeypatch.setattr(runner, "BruteforceLibrary", lambda p: object())
    monkeypatch.setattr(runner, "make_header", lambda *a: {"type": "header"})
    # No real generator or cell is called: only schedule/ledger plumbing executes.
    configs = []

    def already_constructed(config):
        configs.append(config)
        return [instance] * 3

    monkeypatch.setattr(runner, "generate_simon_attack_instances", already_constructed)
    scheduled = []

    def fake_cell(arm, x):
        assert x is instance
        scheduled.append(arm)
        return {"arm": arm, "outcome": "queue_exhausted", "attempts": 0, "total_s": 0.0}

    monkeypatch.setattr(runner, "run_cell", fake_cell)
    result = runner.main(
        [
            "--seed",
            "999999",
            "--role",
            "demo",
            "--keys",
            "2",
            "--first-key",
            "1",
            "--ledger",
            str(ledger),
            "--tf-intra-op-threads",
            "8",
        ]
    )
    assert result == 0
    assert selections == [8]
    assert len(configs) == 1 and configs[0].instance_count == 3
    assert scheduled == ["S11-H", "S11-N", "S11-C", "S11-N", "S11-C", "S11-H"]
    records = [json.loads(s) for s in ledger.read_text().splitlines()]
    assert len(records) == 7
    assert [r["key_index"] for r in records[1:]] == [1, 1, 1, 2, 2, 2]
    assert [r["arm_position"] for r in records[1:]] == [0, 1, 2, 0, 1, 2]


def test_wrong_input_difference_is_rejected_before_model_load(monkeypatch, instance):
    pairs = list(instance.pairs)
    original = pairs[0]
    pairs[0] = SimonAttackPair(
        original.plaintext0,
        (original.plaintext0[0], original.plaintext0[1] ^ 1),
        original.ciphertext0,
        original.ciphertext1,
    )
    malformed = SimpleNamespace(
        **{
            **instance.__dict__,
            "pairs": tuple(pairs),
            "known_pairs": instance.known_pairs,
        }
    )
    monkeypatch.setattr(
        runner, "load_vv", lambda *a: pytest.fail("load before input validation")
    )
    result = runner.run_cell("S11-N", malformed)
    assert result["outcome"] == "error"
    assert "input difference" in result["error"]


def test_posthoc_truth_time_does_not_turn_ontime_recovery_into_deadline(
    monkeypatch, instance
):
    clock = _FakeClock()
    completer = _StubCompleter(MASTER_KEY, clock=clock)
    _patch_neural_path(monkeypatch, instance, completer=completer, clock=clock)
    monkeypatch.setattr(runner.time, "perf_counter", clock.perf_counter)
    real_expand = runner.expand_key

    def slow_truth(key, rounds):
        clock.value += 10
        return real_expand(key, rounds)

    monkeypatch.setattr(runner, "expand_key", slow_truth)
    result = runner.run_cell("S11-N", instance)
    assert result["outcome"] == "recovered"
    assert result["total_s"] == runner.BUDGET - 0.25
    assert clock.value > runner.BUDGET
