"""Synthetic reference verification and no-truth public boundary."""

import copy

import pytest

from neural_sat_speck.backend.artifacts import case_inputs, cipher_functions, verify_key
from neural_sat_speck.backend.guard import fixed_grid


@pytest.mark.parametrize("cipher,rounds", [("speck32_64", 7), ("simon32_64", 10)])
def test_solver_public_boundary_and_fresh_verification(cipher, rounds):
    encrypt, expand = cipher_functions(cipher)
    key = (1, 2, 3, 4)
    pts = [[[10 + i, 20 + i], [30 + i, 40 + i]] for i in range(64)]
    fresh = [[1234, 5678], [9876, 5432]]
    instance = {
        "master_key": list(key),
        "attack_plaintexts": pts,
        "verification_plaintexts": fresh,
        "rounds_data": {
            str(rounds): {
                "attack_ciphertexts": [
                    [list(encrypt(tuple(pt), key, rounds)) for pt in pair]
                    for pair in pts
                ],
                "verification_ciphertexts": [
                    list(encrypt(tuple(pt), key, rounds)) for pt in fresh
                ],
                "round_keys": list(expand(key, rounds)),
            }
        },
    }
    data = {"ciphers": {cipher: {"instances": [instance]}}}
    cell = next(
        c
        for c in fixed_grid()
        if c["cipher"] == cipher
        and c["trailing_keys"] == 0
        and c["observations"] == 4
        and c["key_index"] == 0
    )
    public, truth = case_inputs(data, cell)
    assert set(public) == {"instance_id", "rounds", "pairs"}
    assert len(public["pairs"]) == 2
    flags = verify_key(cipher, rounds, key, public, truth)
    assert all(flags.values())
    wrong = verify_key(cipher, rounds, [1, 2, 3, 5], public, truth)
    assert not all(wrong.values()) and not wrong["assigned_key_match"]
    fullcell = {**cell, "observations": 128}
    fullpublic, _ = case_inputs(data, fullcell)
    assert fullpublic["pairs"][:2] == public["pairs"]
    conditioned = {**cell, "trailing_keys": 2}
    conditioned_public, _ = case_inputs(data, conditioned)
    assert conditioned_public["trailing_round_keys"] == list(expand(key, rounds)[-2:])
    bad = copy.deepcopy(truth)
    bad["verification_pairs"][0][0] = public["pairs"][0]["plaintext0"]
    with pytest.raises(ValueError, match="overlap"):
        verify_key(cipher, rounds, key, public, bad)
