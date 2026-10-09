"""Grid, configuration and cohort-family guards for the D-060/D-061 campaign.

Nothing here draws keys: the 601001/601002 generator is only exercised up to its
refusal checks, which run before any key is generated.
"""

import copy
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

from neural_sat_speck.backend import artifacts
from neural_sat_speck.followups import ranking_depth_control as control

REPO = Path(__file__).resolve().parents[2]
CONFIG_FULL = REPO / control.PROSPECTIVE_FULL_DESIGN
CONFIG_PILOT = REPO / control.PROSPECTIVE_PILOT_DESIGN
GENERATOR = REPO / "scripts" / "generate_dev_depth_dataset_n32.py"
R6_SHA = "e437cd3d7e41afcd5b400257b9d2a009a952313c5410bf6da654e5ee32e9e941"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_dev_depth_dataset_n32", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_grid_has_320_unique_cells_with_the_ten_protocol_arms():
    cells = control.prospective_cells("full")
    assert len(cells) == 320
    assert len({c["cell_id"] for c in cells}) == 320
    assert all(c["trailing_keys"] == 2 for c in cells)
    per_key = {}
    for c in cells:
        per_key.setdefault(c["key_index"], []).append(c)
    assert sorted(per_key) == list(range(32))
    for members in per_key.values():
        speck = {
            (c["stage1"], c["profile"], c["candidate_cap_seconds"])
            for c in members
            if c["cipher"] == "speck32_64"
        }
        simon = {
            (c["stage1"], c["profile"], c["candidate_cap_seconds"])
            for c in members
            if c["cipher"] == "simon32_64"
        }
        assert speck == {
            ("neural", "cms_native_gauss", None),
            ("neural", "cms_native_no_gauss", None),
            ("neural", "cms_native_gauss", 1.5),
            ("neural", "cms_native_no_gauss", 1.5),
            ("classical_sparse_r6", "cms_native_gauss", None),
            ("classical_sparse_r6", "cms_native_no_gauss", None),
        }
        assert simon == {
            ("neural", "cms_native_gauss", None),
            ("neural", "cms_native_no_gauss", None),
            ("neural", "cms_native_gauss", 10.0),
            ("neural", "cms_native_no_gauss", 10.0),
        }


def test_cell_identifiers_follow_the_protocol_format():
    by_id = {c["cell_id"]: c for c in control.prospective_cells("full")}
    assert "speck32_64_00_m2_cms_native_gauss_cap1p5" in by_id
    assert "speck32_64_00_m2_cms_native_no_gauss_capnone" in by_id
    assert "simon32_64_00_m2_cms_native_gauss_cap10" in by_id
    assert "speck32_64_00_m2_classical_cms_native_gauss_capnone" in by_id
    assert "speck32_64_00_m2_classical_cms_native_no_gauss_capnone" in by_id
    assert all(set(i) <= set("abcdefghijklmnopqrstuvwxyz0123456789_") for i in by_id)


def test_rotation_mirrors_fixed_cells_and_alternates_the_leading_cipher():
    cells = control.prospective_cells("full")
    for index in range(32):
        order = [c["cipher"] for c in cells if c["key_index"] == index]
        first = "speck32_64" if index % 2 == 0 else "simon32_64"
        assert order[0] == first
        assert order.count("speck32_64") == 6 and order.count("simon32_64") == 4
    # Each SPECK arm occupies every position at least once across the 32 keys.
    speck_positions = {}
    for index in range(32):
        seq = [
            c["cell_id"]
            for c in cells
            if c["key_index"] == index and c["cipher"] == "speck32_64"
        ]
        for position, cell_id in enumerate(seq):
            speck_positions.setdefault(cell_id.split("_m2_")[1], set()).add(position)
    assert all(positions == set(range(6)) for positions in speck_positions.values())


def test_pilot_is_one_opened_key_per_cipher_across_all_applicable_arms():
    pilot = control.prospective_cells("pilot")
    assert len(pilot) == 10
    assert {c["key_index"] for c in pilot} == {0}
    assert sum(c["cipher"] == "speck32_64" for c in pilot) == 6
    assert sum(c["cipher"] == "simon32_64" for c in pilot) == 4


def test_each_generated_cell_passes_the_field_validator():
    for cell in control.prospective_cells("full"):
        control.validate_prospective_cell(cell)


@pytest.mark.parametrize(
    "change",
    [
        ("candidate_cap_seconds", 10),  # int, not float
        ("candidate_cap_seconds", 0.0),
        ("candidate_cap_seconds", 300.0),
        ("stage1", "classical_sparse_r7"),
        ("cell_id", "wrong_id"),
        ("trailing_keys", 1),
        ("extra", True),
    ],
)
def test_cell_validator_rejects_bad_new_fields(change):
    cell = copy.deepcopy(control.prospective_cells("full")[0])
    field, value = change
    if field == "extra":
        cell["extra"] = value
    else:
        cell[field] = value
    with pytest.raises(ValueError):
        control.validate_prospective_cell(cell)


def test_classical_stage1_is_speck_only():
    speck = next(
        c
        for c in control.prospective_cells("full")
        if c["stage1"] == "classical_sparse_r6"
    )
    simon_version = {
        **speck,
        "cipher": "simon32_64",
        "case_id": "simon32_64_00",
        "rounds": 10,
        "cell_id": "simon32_64_00_m2_classical_cms_native_gauss_capnone",
    }
    with pytest.raises(ValueError):
        control.validate_prospective_cell(simon_version)


def test_d056_fixed_cells_are_untouched_by_the_new_campaign():
    cells = control.fixed_cells("full", num_keys=32)
    assert len(cells) == 256
    assert all("stage1" not in c and "candidate_cap_seconds" not in c for c in cells)
    assert all(
        c["cell_id"].endswith(
            (
                "_m1_cms_native_gauss",
                "_m2_cms_native_gauss",
                "_m1_cms_native_no_gauss",
                "_m2_cms_native_no_gauss",
            )
        )
        for c in cells
    )


def test_configs_pin_the_r6_table_and_ship_unapproved():
    full = json.loads(CONFIG_FULL.read_text())
    pilot = json.loads(CONFIG_PILOT.read_text())
    for profile in (full, pilot):
        assert profile["author_decision"] == "D-060"
        assert profile["status"] == "PENDING_AUTHOR_APPROVAL"
        assert profile["execution_authorized"] is False
        assert profile["candidate_caps_seconds"] == {
            "speck32_64": 1.5,
            "simon32_64": 10.0,
        }
        assert profile["assumed_trailing_key_counts"] == [2]
        assert profile["confirmation_seed"] is None
        assert profile["files"]["speck_classical_r6"] == {
            "path": "models/classical_dist_r6_sparse_twopass_mc20_v1.json",
            "sha256": R6_SHA,
        }
    assert full["seed"] == 601001 and full["campaign_role"] == "full"
    assert full["campaign_ceiling_seconds"] == 100000
    # Pinned after generating the fresh 601001 cohort (D-060 ordering).
    assert full["files"]["cohort"]["sha256"] == (
        "8283e7e47cfc16cfddcf7a9cac6370e433beb33c9a6027c0fe0b4ddaa4d2c4c5"
    )
    assert pilot["seed"] == 501001 and pilot["campaign_role"] == "pilot"
    assert pilot["selected_instances"] == [0]
    assert pilot["campaign_ceiling_seconds"] == 3600
    assert (
        pilot["files"]["cohort"]["path"]
        == "datasets/dev_cross_cipher_depth_n32/dataset.json"
    )
    assert pilot["files"]["cohort"]["sha256"] == (
        "39aad2922a5a2ad0bd5a2fd425af01ab57a2ab4896096435285a9a61cbc05ee1"
    )


def test_pilot_and_full_share_code_config_and_profile_identity():
    def manifest(profile, files):
        return {
            "campaign": control.PROSPECTIVE_CAMPAIGN,
            "author_decision": "D-060",
            "profile": profile,
            "files": files,
            "runtime": {"fixture": 1},
            "source_commit": "a" * 40,
            "python_version": "3",
        }

    full = json.loads(CONFIG_FULL.read_text())
    pilot = json.loads(CONFIG_PILOT.read_text())
    shared = {"speck_classical_r6": {"sha256": R6_SHA}, "cms": {"sha256": "c" * 64}}
    full_manifest = manifest(
        full, {**shared, "cohort": {"sha256": "1" * 64, "path": "x"}}
    )
    pilot_manifest = manifest(
        pilot, {**shared, "cohort": {"sha256": "2" * 64, "path": "y"}}
    )
    assert control.prospective_identity(full_manifest) == control.prospective_identity(
        pilot_manifest
    )
    changed = copy.deepcopy(pilot)
    changed["stage2_child_counts"]["speck32_64"] = 31
    assert control.prospective_identity(
        manifest(changed, shared)
    ) != control.prospective_identity(full_manifest)
    other_files = {**shared, "cms": {"sha256": "d" * 64}}
    assert control.prospective_identity(
        manifest(pilot, other_files)
    ) != control.prospective_identity(full_manifest)


def test_asset_keys_add_the_r6_table_only_for_the_campaign():
    assert control.asset_keys_for({}) == control.ASSET_KEYS
    assert control.asset_keys_for({"campaign": control.PROSPECTIVE_CAMPAIGN}) == (
        control.PROSPECTIVE_ASSET_KEYS
    )
    assert "speck_classical_r6" not in control.ASSET_KEYS


def test_cohort_validator_accepts_601001_label_and_derives_simon_stream():
    # Zero instances fails the 32-key count before any draw, proving the label passes.
    empty = {
        "ciphers": {"speck32_64": {"instances": []}, "simon32_64": {"instances": []}}
    }
    with pytest.raises(ValueError, match="Expected 32 cohort keys"):
        artifacts.validate_cohort(
            {**empty, "seed": 601001, "split_role": "development"}
        )
    with pytest.raises(ValueError, match="Only the recorded"):
        artifacts.validate_cohort({**empty, "seed": 601001, "split_role": "demo"})
    with pytest.raises(ValueError, match="Only the recorded"):
        artifacts.validate_cohort(
            {**empty, "seed": 123456, "split_role": "development"}
        )


def test_seed_family_option_requires_output_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    generator = _generator()
    existing = tmp_path / "dataset.json"
    existing.write_text("{}")
    monkeypatch.setattr(
        generator,
        "generate_cipher_instances",
        lambda *a, **k: pytest.fail("No key may be drawn when the output is refused"),
    )
    with pytest.raises(FileExistsError):
        generator.generate_and_save_dataset(existing, 601001)
    assert existing.read_text() == "{}"
    monkeypatch.setattr(sys, "argv", ["generate", "--seed-family", "601001"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(GENERATOR), run_name="__main__")
    monkeypatch.setattr(
        sys, "argv", ["generate", "--seed-family", "601001", "--output", str(existing)]
    )
    with pytest.raises(SystemExit):
        runpy.run_path(str(GENERATOR), run_name="__main__")
    assert existing.read_text() == "{}"


def test_default_501001_family_is_still_the_default_output_path():
    generator = _generator()
    assert generator.DEV_SEED_501001 == 501001
    assert generator.DATASET_PATH == generator.DATASET_DIR / "dataset.json"
    assert generator.PROSPECTIVE_SEED_601001 == 601001


def _prepare_case(monkeypatch, profile_edit=None):
    """Run prepare far enough to exercise the gates, with hashing forbidden."""
    profile = json.loads(CONFIG_FULL.read_text())
    if profile_edit is not None:
        profile_edit(profile)
    monkeypatch.setattr(control, "policy", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(control, "validate_source_root", lambda *_args: None)

    def no_hashing(*_args, **_kwargs):
        pytest.fail("No file may be hashed before the approval and pin gates pass")

    monkeypatch.setattr(control, "file_identity", no_hashing)
    monkeypatch.setattr(control, "runtime_identity", no_hashing)
    return profile


def test_prepare_refuses_an_unpinned_cohort_before_hashing(tmp_path, monkeypatch):
    _prepare_case(monkeypatch, lambda p: p["files"]["cohort"].update(sha256=None))
    with pytest.raises(ValueError, match="Cohort SHA-256 is not pinned"):
        control.prepare(
            REPO,
            tmp_path / "out",
            "full",
            Path(sys.executable),
            REPO,
            pilot=tmp_path / "audit.json",
            config_path=CONFIG_FULL,
        )
    assert not (tmp_path / "out").exists()


def test_prepare_refuses_a_design_that_carries_its_own_approval(tmp_path, monkeypatch):
    _prepare_case(
        monkeypatch, lambda p: p.update(status="APPROVED", execution_authorized=True)
    )
    with pytest.raises(ValueError, match="ship unapproved"):
        control.prepare(
            REPO,
            tmp_path / "out",
            "full",
            Path(sys.executable),
            REPO,
            pilot=tmp_path / "audit.json",
            config_path=CONFIG_FULL,
            author_approval="D-060",
        )


def test_prepare_requires_the_exact_d060_approval_label(tmp_path, monkeypatch):
    _prepare_case(monkeypatch)
    with pytest.raises(ValueError, match="exactly D-060"):
        control.prepare(
            REPO,
            tmp_path / "out",
            "full",
            Path(sys.executable),
            REPO,
            pilot=tmp_path / "audit.json",
            config_path=CONFIG_FULL,
            author_approval="D-055",
        )


def test_prepare_pilot_and_full_gates_are_mode_specific(tmp_path, monkeypatch):
    _prepare_case(monkeypatch)
    with pytest.raises(ValueError, match="needs the pinned pilot audit"):
        control.prepare(
            REPO,
            tmp_path / "full_out",
            "full",
            Path(sys.executable),
            REPO,
            pilot=None,
            config_path=CONFIG_FULL,
        )
    with pytest.raises(ValueError, match="role differs"):
        control.prepare(
            REPO,
            tmp_path / "pilot_out",
            "pilot",
            Path(sys.executable),
            REPO,
            pilot=None,
            config_path=CONFIG_FULL,
        )


def test_approval_flag_is_refused_for_the_legacy_campaigns(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "validate_source_root", lambda *_args: None)
    with pytest.raises(ValueError, match="applies only to the prospective campaign"):
        control.prepare(
            REPO,
            tmp_path / "legacy",
            "pilot",
            Path(sys.executable),
            REPO,
            pilot=None,
            config_path=REPO / control.D056_DESIGN,
            author_approval="D-060",
        )
