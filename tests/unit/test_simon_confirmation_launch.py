"""Adversarial tests of the final launch gates; no real ranking/SAT workers."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neural_sat_speck.simon import confirmation_guard as guard
from neural_sat_speck.simon import confirmation_profile as profile
from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from scripts import analyze_simon_confirmation as analysis
from scripts.audit_simon_training_key_replay import reconstruct
from tests.unit import test_simon_confirmation_guards as fixtures


@pytest.fixture
def approved_fixture(monkeypatch):
    return fixtures.mock_valid_manifest.__wrapped__(monkeypatch)


@pytest.mark.parametrize('section,field,value', [
    ('resource_policy','solver_threads',2),
    ('random_control_stream','prng_algorithm','another generator'),
    ('statistical_analysis_plan','analysis_freeze_status','changed'),
    ('candidate_sat_specification','formula_variables',1857),
])
def test_nested_profile_drift_is_rejected(approved_fixture, section, field, value):
    manifest = copy.deepcopy(approved_fixture)
    manifest[section][field] = value
    with pytest.raises(guard.ConfirmationPreflightError, match='Scientific profile|formula_variables'):
        guard.validate_confirmation_manifest(manifest)


def test_opened_eight_pair_demo_is_in_overlap_union():
    demo = generate_simon_attack_instances(SimonAttackInstanceConfig(instance_count=8, pair_count=8, rounds=10, seed=999999, split_role='demo'))
    overlaps = guard.check_key_overlaps_with_prior_families([i.master_key for i in demo])
    assert len(overlaps['demo_seed_999999_p8']) == 8


def test_generation_rejects_unapproved_before_any_data_draw(approved_fixture, tmp_path, monkeypatch):
    manifest = copy.deepcopy(approved_fixture)
    manifest.update(status='DRAFT', confirmation_dataset_generation_authorized=True)
    draw = MagicMock()
    monkeypatch.setattr(guard, 'generate_simon_attack_instances', draw)
    with pytest.raises(guard.ConfirmationPreflightError, match="must be 'APPROVED'"):
        guard.generate_confirmation_dataset(manifest, tmp_path/'data.json', tmp_path/'cert.json', authorization_token=True)
    draw.assert_not_called()


def test_generation_refuses_existing_artifact(approved_fixture, tmp_path, monkeypatch):
    manifest = copy.deepcopy(approved_fixture)
    manifest['confirmation_dataset_generation_authorized'] = True
    monkeypatch.setattr(guard, 'verify_runtime_provenance', lambda *_: {})
    data = tmp_path/'data.json'
    data.write_text('preserve this evidence')
    draw = MagicMock()
    monkeypatch.setattr(guard, 'generate_simon_attack_instances', draw)
    with pytest.raises(guard.ConfirmationPreflightError, match='Refusing to overwrite'):
        guard.generate_confirmation_dataset(manifest, data, tmp_path/'cert.json', authorization_token=True)
    assert data.read_text() == 'preserve this evidence'
    draw.assert_not_called()


def test_execution_pins_reject_source_change_before_runtime_probe(monkeypatch):
    monkeypatch.setattr(profile, 'code_identity', lambda: {'code.py': 'new'})
    probe = MagicMock()
    monkeypatch.setattr(profile, 'local_runtime_identity', probe)
    with pytest.raises(ValueError, match='source identity'):
        profile.verify_execution_pins({'execution_pins': {'source_sha256': {'code.py':'old'}}}, tensorflow_python=Path('unused'))
    probe.assert_not_called()


def test_analysis_cannot_accept_missing_confirmation_identity():
    args=SimpleNamespace(allow_drift=False, global_cutoff=300.0, alpha=0.05, manifest=None, dataset=None, certificate=None)
    with pytest.raises(ValueError, match='requires the frozen manifest'):
        analysis.validate_confirmation_analysis_inputs(args, {}, {})


def test_success_flag_cannot_replace_missing_fresh_verification():
    with pytest.raises(ValueError, match='missing or False'):
        analysis.extract_record_success({'outcome':'verified_recovery','counts_as_verified_recovery':True,'complete_online_wall_seconds':1.0})


def test_missing_component_time_is_unknown():
    rec={'instance_id':'fixture','outcome':'queue_exhausted','verified_on_fresh_pairs':False,'complete_online_wall_seconds':2.0}
    summary=analysis.compute_method_summary('random_control_128',[rec])
    assert summary['mean_component_seconds']['solving'] is None


@pytest.mark.parametrize('group_size',[None,64])
def test_training_replay_matches_original_generator(group_size):
    import numpy as np

    from neural_sat_speck.simon.b1_data import (
        generate_simon_b0_pairs,
        generate_simon_b1_groups,
    )
    features, labels, keys = reconstruct(8, 880001, group_size)
    if group_size is None:
        original = generate_simon_b0_pairs(8, 880001, rounds=9)
    else:
        original = generate_simon_b1_groups(8, 880001, rounds=9, group_size=64)
    for actual, expected in zip((features,labels,keys),(original.features,original.labels,original.master_keys),strict=True):
        np.testing.assert_array_equal(actual,expected)


def test_real_frozen_sample_size_cannot_drift():
    manifest=json.loads(Path("docs/SIMON_UNSEEN_CONFIRMATION_MANIFEST_PROPOSED.json").read_text())
    assert manifest["sample_size_specification"]["selected_instances"] == 64
    assert profile.scientific_digest(manifest) == profile.FROZEN_SCIENTIFIC_SHA256
    manifest["sample_size_specification"]["selected_instances"] = 32
    with pytest.raises(guard.ConfirmationPreflightError, match="Scientific profile"):
        guard.validate_scientific_profile(manifest)


def test_same_package_version_changed_native_bytes_changes_identity(tmp_path, monkeypatch):
    package = tmp_path / "native.so"
    package.write_bytes(b"first native build")
    fake = SimpleNamespace(version="fixed-version", files=[Path("native.so")], locate_file=lambda _: package)
    monkeypatch.setattr(profile.importlib.metadata, "distribution", lambda _: fake)
    first = profile.distribution_artifact_identity(("fixture",))
    package.write_bytes(b"different native build")
    assert fake.version == "fixed-version"
    assert first != profile.distribution_artifact_identity(("fixture",))


@pytest.mark.parametrize("state", ["seed", "gate", "blocker"])
def test_contradictory_authorization_is_rejected(approved_fixture, state):
    manifest=copy.deepcopy(approved_fixture)
    if state == "seed":
        manifest["seed_governance"]["seed_status"]="UNASSIGNED_ABSENT_PENDING_PI_AUTHORIZATION"
    elif state == "gate":
        manifest["execution_readiness"]["open_approval_gates"]=["C5_launch_authorization"]
    else:
        manifest["execution_readiness"]["preparation_blockers"]=["unreviewed source"]
    with pytest.raises(guard.ConfirmationPreflightError, match="seed is not registered|remain open"):
        guard.validate_confirmation_manifest(manifest)


def test_confirmation_analysis_rejects_equal_partial_grid(tmp_path, monkeypatch):
    from scripts import evaluate_simon_confirmation as runner
    from tests.unit.test_simon_confirmation_runner import synthetic_runner_environment
    env = synthetic_runner_environment.__wrapped__(tmp_path, monkeypatch)
    manifest = env["manifest"]
    manifest["execution_pins"] = {"source_sha256": profile.code_identity()}
    env["manifest_path"].write_text(json.dumps(manifest))
    guard.build_disjoint_family_certificate(manifest, env["instances"], output_path=env["cert_path"])
    identity = guard.compute_execution_identity(manifest, guard.file_sha256(env["dataset_path"]))
    records = []
    from neural_sat_speck.simon.method_worker import MethodWorkerOutcome
    for method in guard.ALL_CONFIRMATION_METHODS:
        rec = MethodWorkerOutcome(method=method, outcome="queue_exhausted", complete_online_wall_seconds=1.0,
            ranking_wall_seconds=0.0, solving_wall_seconds=1.0, verification_wall_seconds=0.0,
            candidates_tested_count=0, candidates_remaining_count=0, recovered_master_key=None,
            verified_on_fresh_pairs=False, candidate_attempts=(), peak_rss_gib=0.0).to_dict()
        rec.update(instance_id=env["instances"][0].instance_id,instance_index=0)
        records.append(rec)
    ledger=tmp_path/"partial.json"
    runner._save_ledger_atomic(ledger, records, identity, "CONFIRMATION")
    args=SimpleNamespace(allow_drift=False, global_cutoff=300.0, alpha=0.05, manifest=env["manifest_path"], dataset=env["dataset_path"], certificate=env["cert_path"], ledger=ledger)
    with pytest.raises(ValueError, match="full selected N x 9"):
        analysis.validate_confirmation_analysis_inputs(args, {}, {})
