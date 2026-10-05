"""Adversarial synthetic guards for the opened-demo queue-order protocol.

These tests never read the registered dataset and never invoke a ranker or SAT
worker.  The manifest fixtures contain only temporary placeholder files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from neural_sat_speck.backend import guard
from neural_sat_speck.followups import queue_order as protocol
from neural_sat_speck.followups import runner as executor


def _candidate_queue() -> list[tuple[int, int]]:
    # The first coordinate makes all 1,024 candidate tuples distinct.
    return [(word, (37 * word + 11) & 0xFFFF) for word in range(1024)]


def _preparation(job: dict, ranked: list[tuple[int, int]]) -> dict:
    shuffled = list(protocol.permutation(ranked, job))
    return {
        "job_id": job["case_id"] + "__prepare",
        "ranked": ranked,
        "shuffled": shuffled,
        "set_sha256": protocol.ordered_hash(sorted(ranked)),
        "ranked_sha256": protocol.ordered_hash(ranked),
        "shuffled_sha256": protocol.ordered_hash(shuffled),
    }


def _demo_block(index: int, offset: int) -> tuple[int, int]:
    return index + offset, (index * 13 + offset * 3) & 0xFFFF


def _demo_dataset() -> dict:
    """Small in-memory schema fixture; it is unrelated to real key/data bytes."""
    ciphers = {}
    for cipher, rounds in (("speck32_64", 7), ("simon32_64", 10)):
        instances = []
        for index in range(8):
            attack_plaintexts = [
                [_demo_block(pair, 0), _demo_block(pair, 1000)]
                for pair in range(64)
            ]
            attack_ciphertexts = [
                [_demo_block(pair, 20000), _demo_block(pair, 21000)]
                for pair in range(64)
            ]
            instances.append(
                {
                    "instance_index": index,
                    "instance_id": f"synthetic-{cipher}-{index}",
                    "master_key": [index, 1, 2, 3],
                    "round_keys": [index] * 10,
                    "attack_plaintexts": attack_plaintexts,
                    "verification_plaintexts": [
                        _demo_block(0, 40000), _demo_block(1, 41000)
                    ],
                    "rounds_data": {
                        str(rounds): {
                            "attack_ciphertexts": attack_ciphertexts,
                            "verification_ciphertexts": [
                                _demo_block(0, 50000), _demo_block(1, 51000)
                            ],
                        }
                    },
                }
            )
        ciphers[cipher] = {"instances": instances}
    return {
        "seed": 999999,
        "split_role": "demo",
        "metadata": {
            "key_count_per_cipher": 8,
            "max_differential_pairs": 64,
            "verification_blocks_count": 2,
        },
        "ciphers": ciphers,
    }


def _job(cipher: str = "simon32_64", policy: str = "classical_8parent") -> dict:
    return next(
        item
        for item in protocol.fixed_jobs()
        if item["cipher"] == cipher
        and item["policy"] == policy
        and item["key_index"] == 0
        and item["operation"] == "solve"
        and item["order"] == "ranked"
    )


def test_fixed_grid_is_exactly_32_preparations_and_64_replay_arms():
    jobs = protocol.fixed_jobs()

    assert len(jobs) == 96
    assert len({job["job_id"] for job in jobs}) == 96
    preparations = [job for job in jobs if job["operation"] == "rank"]
    replays = [job for job in jobs if job["operation"] == "solve"]
    assert len(preparations) == 32
    assert len(replays) == 64
    assert len({job["case_id"] for job in jobs}) == 32

    for cipher, expected_rounds in (("speck32_64", 7), ("simon32_64", 10)):
        by_key = {}
        for job in jobs:
            if job["cipher"] == cipher:
                by_key.setdefault(job["key_index"], []).append(job)
        assert set(by_key) == set(range(8))
        for key_index, key_jobs in by_key.items():
            expected_policies = {
                policy for pcipher, policy, _, _ in protocol.POLICIES
                if pcipher == cipher
            }
            assert len(key_jobs) == 3 * len(expected_policies)
            assert {job["rounds"] for job in key_jobs} == {expected_rounds}
            assert {job["policy"] for job in key_jobs} == expected_policies
            for policy in {
                job["policy"] for job in key_jobs
            }:
                cell = [job for job in key_jobs if job["policy"] == policy]
                assert len(cell) == 3
                assert sum(job["operation"] == "rank" for job in cell) == 1
                assert {
                    job.get("order") for job in cell if job["operation"] == "solve"
                } == {"ranked", "shuffled"}
                assert len({job["case_id"] for job in cell}) == 1


def test_policy_rotation_and_ranked_shuffled_dispatch_alternate_by_key():
    jobs = protocol.fixed_jobs()
    for key_index in range(8):
        selected = [job for job in jobs if job["key_index"] == key_index]
        policy_order = []
        case_groups = []
        for position in range(0, len(selected), 3):
            group = selected[position : position + 3]
            assert len(group) == 3
            case_groups.append(group)
            policy_order.append(group[0]["policy"])
            assert [job["order"] for job in group[1:]] == (
                ["ranked", "shuffled"]
                if key_index % 2 == 0
                else ["shuffled", "ranked"]
            )
        shift = key_index % len(protocol.POLICIES)
        expected = protocol.POLICIES[shift:] + protocol.POLICIES[:shift]
        assert policy_order == [policy for _, policy, _, _ in expected]
        assert len(case_groups) == 4


def test_shuffle_is_reproducible_and_preserves_the_exact_candidate_set():
    job = _job()
    ranked = _candidate_queue()
    first = protocol.permutation(ranked, job)
    second = protocol.permutation(ranked, job)

    assert first == second
    assert len(first) == 1024
    assert len(set(first)) == 1024
    assert set(first) == set(ranked)
    assert first != tuple(ranked)
    protocol.validate_preparation(_preparation(job, ranked), job)


@pytest.mark.parametrize(
    "bad_word",
    [True, False, -1, 65536, 1.0, "1", None],
)
def test_words_reject_bool_noninteger_and_out_of_range_values(bad_word):
    with pytest.raises(ValueError):
        protocol.word(bad_word)


def test_queue_rejects_bool_words_and_duplicate_tuples():
    ranked = _candidate_queue()
    bad_bool = list(ranked)
    bad_bool[100] = (True, 42)
    with pytest.raises(ValueError, match="genuine 16-bit"):
        protocol.queue(bad_bool)

    duplicate = list(ranked)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.queue(duplicate)


def test_preparation_rejects_same_set_in_the_wrong_shuffle_order_or_bad_hash():
    job = _job()
    ranked = _candidate_queue()
    valid = _preparation(job, ranked)
    protocol.validate_preparation(valid, job)

    wrong_order = copy.deepcopy(valid)
    wrong_order["shuffled"][0], wrong_order["shuffled"][1] = (
        wrong_order["shuffled"][1], wrong_order["shuffled"][0]
    )
    wrong_order["shuffled_sha256"] = protocol.ordered_hash(wrong_order["shuffled"])
    with pytest.raises(ValueError, match="membership/stream"):
        protocol.validate_preparation(wrong_order, job)

    wrong_hash = copy.deepcopy(valid)
    wrong_hash["set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="set hash"):
        protocol.validate_preparation(wrong_hash, job)


def test_preparation_rejects_missing_or_duplicated_members_and_wrong_identity():
    job = _job()
    ranked = _candidate_queue()
    valid = _preparation(job, ranked)

    missing_and_duplicate = copy.deepcopy(valid)
    missing_and_duplicate["ranked"][-1] = missing_and_duplicate["ranked"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.validate_preparation(missing_and_duplicate, job)

    wrong_identity = copy.deepcopy(valid)
    wrong_identity["job_id"] = "another_case__prepare"
    with pytest.raises(ValueError, match="identity"):
        protocol.validate_preparation(wrong_identity, job)


def test_public_projection_has_all_64_pairs_and_strips_truth_and_fresh_data():
    dataset = _demo_dataset()
    job = _job()
    public = protocol.public_case(dataset, job)

    assert public["cipher"] == "simon32_64"
    assert public["rounds"] == 10
    assert public["instance_id"] == "synthetic-simon32_64-0"
    assert len(public["ciphertext_pairs"]) == 64
    assert len(public["known_observations"]) == 128
    assert set(public) == {
        "cipher", "rounds", "instance_id", "ciphertext_pairs", "known_observations"
    }
    forbidden = {
        "master_key", "round_keys", "verification_plaintexts",
        "verification_ciphertexts", "fresh_observations",
    }
    assert forbidden.isdisjoint(public)

    # Verification stores exactly two fresh plaintext blocks, each with its
    # corresponding ciphertext.  They are not four fresh observations.
    fresh = protocol.fresh_observations(dataset, job)
    assert len(fresh) == 2
    assert all(len(record) == 2 for record in fresh)
    assert all(len(record[0]) == len(record[1]) == 2 for record in fresh)
    attack_plaintexts = {
        tuple(block)
        for pair in dataset["ciphers"][job["cipher"]]["instances"][0]["attack_plaintexts"]
        for block in pair
    }
    assert not attack_plaintexts.intersection(tuple(record[0]) for record in fresh)


@pytest.mark.parametrize(
    "mutation",
    [
        ("seed", 777001),
        ("split_role", "validation"),
        ("key_count_per_cipher", 7),
        ("max_differential_pairs", 32),
        ("verification_blocks_count", 4),
    ],
)
def test_public_projection_rejects_wrong_dataset_role_or_dimensions(mutation):
    field, value = mutation
    dataset = _demo_dataset()
    if field in {"key_count_per_cipher", "max_differential_pairs", "verification_blocks_count"}:
        dataset["metadata"][field] = value
    else:
        dataset[field] = value
    with pytest.raises(ValueError):
        protocol.public_case(dataset, _job())


@pytest.mark.parametrize("bad_index", [True, -1, 8, 1.0, "0"])
def test_public_and_fresh_accessors_reject_noncanonical_instance_indices(bad_index):
    dataset = _demo_dataset()
    job = {**_job(), "key_index": bad_index}

    with pytest.raises(ValueError):
        protocol.public_case(dataset, job)
    with pytest.raises(ValueError):
        protocol.fresh_observations(dataset, job)


@pytest.mark.parametrize("accessor", ["public_case", "fresh_observations"])
def test_both_dataset_accessors_reject_reordered_or_incomplete_instances(accessor):
    dataset = _demo_dataset()
    instances = dataset["ciphers"]["simon32_64"]["instances"]
    instances[3]["instance_index"] = 2
    with pytest.raises(ValueError):
        getattr(protocol, accessor)(dataset, _job())

    dataset = _demo_dataset()
    dataset["ciphers"]["simon32_64"]["instances"].pop()
    with pytest.raises(ValueError):
        getattr(protocol, accessor)(dataset, _job())


def test_fresh_accessor_rejects_attack_plaintext_reuse():
    dataset = _demo_dataset()
    dataset["ciphers"]["simon32_64"]["instances"][0]["verification_plaintexts"][0] = (
        dataset["ciphers"]["simon32_64"]["instances"][0]["attack_plaintexts"][0][0]
    )
    with pytest.raises(ValueError, match="fresh blocks"):
        protocol.fresh_observations(dataset, _job())


def _manifest_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """A valid profile backed only by small temporary placeholders."""
    root = tmp_path / "source"
    (root / "src" / "neural_sat_speck" / "followups").mkdir(parents=True)
    (root / "scripts").mkdir()
    source_files = [
        "src/neural_sat_speck/followups/queue_order.py",
        "scripts/run_neural_ranking_worker.py",
        "scripts/run_queue_order_neural_ranking.py",
        "scripts/evaluate_queue_order_diagnostic.py",
    ]
    file_records = {}
    for index, relative in enumerate(source_files):
        path = root / relative
        path.write_text(f"synthetic placeholder {index}\n", encoding="utf-8")
        file_records["source:" + relative] = {
            "path": str(path),
            "sha256": guard.sha256_file(path),
        }

    for index, identity in enumerate(
        (
            "cohort", "python", "pycryptosat", "protocol", "speck_stage1",
            "speck_stage2", "speck_classical", "simon_ddt", "simon_tool",
            "tf_python", "bao_vv9",
        )
    ):
        path = root / "placeholders" / f"{identity}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"placeholder-{index}".encode())
        file_records[identity] = {
            "path": str(path),
            "sha256": guard.sha256_file(path),
        }

    manifest = {
        "status": "APPROVED",
        "execution_authorized": True,
        "execution_kind": "pilot",
        "pilot_gate": {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [j["job_id"] for j in protocol.fixed_jobs()[:12]],
        },
        "profile": copy.deepcopy(protocol.PROFILE),
        "jobs": protocol.fixed_jobs(),
        "files": file_records,
    }
    return root, manifest


def test_valid_synthetic_manifest_checks_only_its_pinned_placeholder_files(tmp_path):
    root, manifest = _manifest_fixture(tmp_path)
    paths = protocol.validate_manifest(manifest, root)
    assert set(manifest["files"]) <= paths.keys()
    assert all(path.is_file() for path in paths.values())
    assert paths["cohort"].name == "cohort.bin"


def test_rejected_preflight_reads_no_dataset_or_worker_boundary(tmp_path, monkeypatch):
    # This source-free manifest names a nonexistent cohort.  Authorization and
    # exact-profile rejection must happen before any pinned identity is opened,
    # any source scan begins, or any data/worker boundary can be called.
    manifest_path = tmp_path / "unapproved.json"
    unopened_dataset = tmp_path / "must_never_be_opened_dataset.json"
    unapproved = {
        "status": "PROPOSED",
        "execution_authorized": False,
        "files": {
            "cohort": {
                "path": str(unopened_dataset),
                "sha256": "0" * 64,
            }
        },
    }
    guard.atomic_json(manifest_path, unapproved)

    identity_reads = []
    source_walks = []
    data_boundaries = []
    worker_boundaries = []
    dataset_reads = []

    def forbid_identity_read(*args, **kwargs):
        identity_reads.append((args, kwargs))
        raise AssertionError("rejected preflight read a pinned identity")

    def forbid_source_walk(*args, **kwargs):
        source_walks.append((args, kwargs))
        raise AssertionError("rejected preflight scanned source identities")

    def forbid_data_boundary(*args, **kwargs):
        data_boundaries.append((args, kwargs))
        raise AssertionError("rejected preflight touched a dataset projection")

    def forbid_worker_boundary(*args, **kwargs):
        worker_boundaries.append((args, kwargs))
        raise AssertionError("rejected preflight crossed a worker boundary")

    original_read_bytes = Path.read_bytes

    def watch_dataset_read(path):
        if path == unopened_dataset:
            dataset_reads.append(path)
            raise AssertionError("rejected preflight loaded the dataset")
        return original_read_bytes(path)

    monkeypatch.setattr(protocol, "checked_path", forbid_identity_read)
    monkeypatch.setattr(protocol.Path, "rglob", forbid_source_walk)
    monkeypatch.setattr(protocol.Path, "read_bytes", watch_dataset_read)
    monkeypatch.setattr(protocol, "public_case", forbid_data_boundary)
    monkeypatch.setattr(protocol, "fresh_observations", forbid_data_boundary)
    monkeypatch.setattr(executor, "run_job", forbid_worker_boundary)
    monkeypatch.setattr(executor.subprocess, "Popen", forbid_worker_boundary)

    with pytest.raises(ValueError, match="Unapproved"):
        executor.run(
            manifest_path,
            tmp_path / "never-created-output.json",
            tmp_path / "source-free-root",
        )

    assert identity_reads == []
    assert source_walks == []
    assert data_boundaries == []
    assert worker_boundaries == []
    assert dataset_reads == []
    assert not unopened_dataset.exists()


def test_corrupted_manifest_checksum_fails_before_source_or_asset_reads(tmp_path, monkeypatch):
    manifest_path = tmp_path / "tampered.json"
    guard.atomic_json(manifest_path, {"status": "PROPOSED"})
    manifest_path.write_text('{"status":"APPROVED"}\n', encoding="utf-8")
    reads = []
    monkeypatch.setattr(
        protocol,
        "checked_path",
        lambda *args, **kwargs: reads.append((args, kwargs)),
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        protocol.load_manifest(manifest_path, tmp_path / "empty-root")
    assert reads == []


def test_checked_path_hashing_does_not_read_the_entire_artifact_at_once(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "large-file-placeholder.bin"
    artifact.write_bytes(b"streamed artifact contents\n")
    record = {"path": str(artifact), "sha256": guard.sha256_file(artifact)}

    def forbidden_read_bytes(_path):
        raise AssertionError("checked_path must hash large artifacts incrementally")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert protocol.checked_path(tmp_path, record) == artifact.absolute()


def _execution_artifacts(tmp_path: Path, job: dict) -> tuple[Path, dict]:
    token = "a" * 32
    directory = tmp_path / "jobs" / job["job_id"] / token
    directory.mkdir(parents=True)
    deadline = 130.0
    payload = {
        "python_path": "/fixture/python",
        "invocation": token,
        "directory": str(directory),
        "job": job,
        "repo_root": "/fixture/repo",
        "deadline_monotonic": deadline,
    }
    guardian = {
        "invocation": token,
        "directory": str(directory),
        "repo_root": "/fixture/repo",
        "worker_arguments": [
            "/fixture/python",
            "-m",
            "neural_sat_speck.followups.worker",
            str(directory / "payload.json"),
        ],
        "receipt": str(directory / "guardian_receipt.json"),
        "deadline_monotonic": deadline,
    }
    process = {"pid": 321, "process_group": 321, "invocation": token}
    receipt = {
        "invocation": token,
        "worker_pid": 321,
        "process_group": 321,
        "process_group_absent": True,
        "status": "worker_exit",
        "deadline_monotonic": deadline,
        "observed_monotonic": 102.0,
        "worker_returncode": 0,
    }
    for filename, value in (
        ("payload.json", payload),
        ("guardian.json", guardian),
        ("process.json", process),
        ("guardian_receipt.json", receipt),
    ):
        guard.atomic_json(directory / filename, value)
    raw = {
        "job_id": job["job_id"],
        "job": job,
        "directory": str(directory),
        "outcome": "ranking_timeout" if job["operation"] == "rank" else "queue_exhausted",
        "physical_worker_wall_seconds": 2.0,
        "deadline_seconds": 30.0,
        "started_monotonic": 100.0,
        "guardian_receipt": receipt,
    }
    return directory, raw


@pytest.mark.parametrize("field", ["job_id", "job", "directory"])
def test_execution_artifacts_reject_raw_result_from_another_cell_or_path(
    tmp_path, field
):
    job = _job()
    directory, raw = _execution_artifacts(tmp_path, job)
    if field == "job_id":
        raw[field] = "another_cell"
    elif field == "job":
        raw[field] = protocol.fixed_jobs()[0]
    else:
        raw[field] = str(tmp_path / "other-invocation")

    with pytest.raises(ValueError, match="identity|job|directory"):
        executor.validate_execution_artifacts(directory, job, raw)


@pytest.mark.parametrize("mutation", ["invocation", "deadline"])
def test_execution_artifacts_reject_forged_receipt_identity_or_deadline(
    tmp_path, mutation
):
    job = _job()
    directory, raw = _execution_artifacts(tmp_path, job)
    receipt = guard.read_checked_json(directory / "guardian_receipt.json")
    if mutation == "invocation":
        receipt["invocation"] = "b" * 32
    else:
        receipt["deadline_monotonic"] += 1.0
    guard.atomic_json(directory / "guardian_receipt.json", receipt)
    raw["guardian_receipt"] = receipt

    with pytest.raises(ValueError, match="identity|deadline"):
        executor.validate_execution_artifacts(directory, job, raw)


def test_execution_artifacts_reject_deadline_receipt_observed_before_deadline(
    tmp_path,
):
    job = next(
        item
        for item in protocol.fixed_jobs()
        if item["operation"] == "rank"
    )
    directory, raw = _execution_artifacts(tmp_path, job)
    receipt = guard.read_checked_json(directory / "guardian_receipt.json")
    receipt["status"] = "deadline_enforced"
    guard.atomic_json(directory / "guardian_receipt.json", receipt)
    raw["guardian_receipt"] = receipt

    with pytest.raises(ValueError, match="deadline"):
        executor.validate_execution_artifacts(directory, job, raw)


def _successful_replay(job: dict, *, common: float = 2.0, wall: float = 5.0) -> dict:
    return {
        "job_id": job["job_id"],
        "outcome": "verified_recovery",
        "success": True,
        "common_preparation_seconds": common,
        "replay_wall_seconds": wall,
        "charged_online_seconds": common + wall,
        "verified_known": True,
        "verified_fresh": True,
        "recovered_key": [0, 1, 2, 3],
        "queue_sha256": "a" * 64,
    }


def _failure_replay(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "outcome": "queue_exhausted",
        "success": False,
        "common_preparation_seconds": 2.0,
        "replay_wall_seconds": 5.0,
        "charged_online_seconds": 7.0,
        "verified_known": False,
        "verified_fresh": False,
        "queue_sha256": "a" * 64,
    }


def test_replay_requires_strict_boolean_flags_consistent_cost_and_queue_identity():
    job = _job()
    preparation = {
        "common_preparation_seconds": 2.0,
        "ranked_sha256": "a" * 64,
        "shuffled_sha256": "b" * 64,
    }
    protocol.validate_replay(_successful_replay(job), job, preparation)

    bad_success_flag = _successful_replay(job)
    bad_success_flag["success"] = "true"
    with pytest.raises(ValueError):
        protocol.validate_replay(bad_success_flag, job, preparation)

    bad_outcome_flag = _successful_replay(job)
    bad_outcome_flag["success"] = False
    with pytest.raises(ValueError, match="agree"):
        protocol.validate_replay(bad_outcome_flag, job, preparation)

    bad_queue = _successful_replay(job)
    bad_queue["queue_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="order"):
        protocol.validate_replay(bad_queue, job, preparation)

    bad_common = _successful_replay(job)
    bad_common["common_preparation_seconds"] = 3.0
    bad_common["charged_online_seconds"] = 8.0
    with pytest.raises(ValueError, match="preparation cost"):
        protocol.validate_replay(bad_common, job, preparation)

    bad_sum = _successful_replay(job)
    bad_sum["charged_online_seconds"] = 8.0
    with pytest.raises(ValueError, match="does not include"):
        protocol.validate_replay(bad_sum, job, preparation)


@pytest.mark.parametrize("field", ["verified_known", "verified_fresh"])
@pytest.mark.parametrize("bad_value", ["false", 0, 1, None])
def test_failure_records_reject_non_boolean_verification_flags(field, bad_value):
    job = _job()
    record = _failure_replay(job)
    record[field] = bad_value
    with pytest.raises(ValueError):
        protocol.validate_replay(record, job, None)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("common_preparation_seconds", True),
        ("common_preparation_seconds", -0.1),
        ("common_preparation_seconds", float("nan")),
        ("replay_wall_seconds", float("inf")),
        ("replay_wall_seconds", -1),
        ("charged_online_seconds", True),
        ("charged_online_seconds", float("nan")),
    ],
)
def test_replay_rejects_boolean_negative_and_nonfinite_durations(field, bad_value):
    job = _job()
    record = _failure_replay(job)
    record[field] = bad_value
    with pytest.raises(ValueError):
        protocol.validate_replay(record, job, None)


def test_unavailable_replay_preserves_common_cost_and_queue_identity():
    job = _job()
    preparation = {
        "common_preparation_seconds": 2.0,
        "ranked_sha256": "a" * 64,
        "shuffled_sha256": "b" * 64,
    }
    record = {
        "job_id": job["job_id"],
        "outcome": "infrastructure_interrupted",
        "success": False,
        "timing_unavailable": True,
        "common_preparation_seconds": 2.0,
        "replay_wall_seconds": None,
        "charged_online_seconds": None,
        "verified_known": False,
        "verified_fresh": False,
        "queue_sha256": "a" * 64,
    }
    protocol.validate_replay(record, job, preparation)

    wrong_common = {**record, "common_preparation_seconds": True}
    with pytest.raises(ValueError):
        protocol.validate_replay(wrong_common, job, preparation)

    wrong_queue = {**record, "queue_sha256": "b" * 64}
    with pytest.raises(ValueError):
        protocol.validate_replay(wrong_queue, job, preparation)

    wrong_outcome = {**record, "outcome": "queue_exhausted"}
    with pytest.raises(ValueError):
        protocol.validate_replay(wrong_outcome, job, preparation)


@pytest.mark.parametrize(
    "common,wall",
    [
        (0.0, 300.0),
        (0.5, 299.5),
        (1.0, 299.0),
        (0.0, 300.001),
    ],
)
def test_success_at_or_after_the_inclusive_300_second_deadline_is_rejected(common, wall):
    job = _job()
    record = _successful_replay(job, common=common, wall=wall)
    with pytest.raises(ValueError, match="late"):
        protocol.validate_replay(record, job, None)


def test_recovery_cannot_pass_without_both_independent_verification_flags_and_key_words():
    job = _job()
    for mutation in (
        {"verified_known": False},
        {"verified_fresh": False},
        {"recovered_key": [0, 1, 2, True]},
        {"recovered_key": [0, 1, 2, 65536]},
    ):
        record = {**_successful_replay(job), **mutation}
        with pytest.raises(ValueError):
            protocol.validate_replay(record, job, None)
