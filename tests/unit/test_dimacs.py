"""Unit tests for native-XOR DIMACS export and decoding."""

from __future__ import annotations

import subprocess

from neural_sat_speck.sat.dimacs import (
    decode_master_key_from_dimacs_solution,
    export_speck_to_dimacs,
)
from neural_sat_speck.speck.cipher import encrypt_block


def test_dimacs_export_and_solve_recovers_master_key():
    """Verify that cryptominisat5 CLI can solve exported DIMACS and recover exact key."""
    true_key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    plaintexts = [(0x1111, 0x2222), (0x3333, 0x4444), (0x5555, 0x6666), (0x7777, 0x8888)]
    pairs = [(p, encrypt_block(p, true_key, rounds=4)) for p in plaintexts]

    model = export_speck_to_dimacs(pairs, rounds=4)
    proc = subprocess.run(
        ["cryptominisat5", "--verb", "0", "--zero-exit-status"],
        input=model.dimacs_text,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "s SATISFIABLE" in proc.stdout
    recovered_key = decode_master_key_from_dimacs_solution(proc.stdout, model.master_key_bits)
    assert recovered_key == true_key


def test_dimacs_export_with_pinned_round_key():
    """Verify pinning a round key produces a satisfiable instance consistent with ground truth."""
    true_key = (0xCAFE, 0xBABE, 0xDEAD, 0xBEEF)
    plaintexts = [(0x1001, 0x2002), (0x3003, 0x4004)]
    pairs = [(p, encrypt_block(p, true_key, rounds=4)) for p in plaintexts]

    # Pin round key 0 (which is word 3 of master key: 0xBEEF)
    model = export_speck_to_dimacs(pairs, rounds=4, pinned_round_keys={0: 0xBEEF})
    proc = subprocess.run(
        ["cryptominisat5", "--verb", "0", "--zero-exit-status"],
        input=model.dimacs_text,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "s SATISFIABLE" in proc.stdout
    recovered_key = decode_master_key_from_dimacs_solution(proc.stdout, model.master_key_bits)
    assert recovered_key is not None
    assert recovered_key[3] == 0xBEEF
