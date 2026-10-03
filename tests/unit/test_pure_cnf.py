"""Unit tests for pure CNF export and solving of SPECK formulas."""

from __future__ import annotations

import subprocess

from neural_sat_speck.sat.dimacs import decode_master_key_from_dimacs_solution
from neural_sat_speck.sat.pure_cnf import export_speck_to_pure_cnf
from neural_sat_speck.speck.cipher import encrypt_block


def test_speck_pure_cnf_export_solved_by_cadical_and_cms():
    """Verify that pure CNF export (XORs converted) is solved correctly by cadical and CMS --xor 0."""
    true_key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    plaintexts = [(0x1111, 0x2222), (0x3333, 0x4444), (0x5555, 0x6666), (0x7777, 0x8888)]
    pairs = [(p, encrypt_block(p, true_key, rounds=4)) for p in plaintexts]

    cnf_text, total_vars, master_key_bits = export_speck_to_pure_cnf(pairs, rounds=4)
    assert cnf_text != ""
    assert "x " not in cnf_text  # Zero native XOR lines
    assert cnf_text.startswith("p cnf ")

    # 1. Solve with cryptominisat5 in pure CNF mode (--xor 0)
    proc_cms = subprocess.run(
        ["cryptominisat5", "--xor", "0", "--verb", "0", "--zero-exit-status"],
        input=cnf_text,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "s SATISFIABLE" in proc_cms.stdout
    rec_cms = decode_master_key_from_dimacs_solution(proc_cms.stdout, master_key_bits)
    assert rec_cms == true_key

    # 2. Solve with cadical (pure CDCL solver)
    proc_cad = subprocess.run(
        ["cadical", "-q"],
        input=cnf_text,
        text=True,
        capture_output=True,
    )
    assert proc_cad.returncode == 10  # SAT exit code in CaDiCaL
    assert "s SATISFIABLE" in proc_cad.stdout
    rec_cad = decode_master_key_from_dimacs_solution(proc_cad.stdout, master_key_bits)
    assert rec_cad == true_key


def test_speck_pure_cnf_export_with_pinned_keys():
    """Verify pinning round keys in pure CNF produces valid solution."""
    true_key = (0xCAFE, 0xBABE, 0xDEAD, 0xBEEF)
    plaintexts = [(0x1001, 0x2002), (0x3003, 0x4004)]
    pairs = [(p, encrypt_block(p, true_key, rounds=4)) for p in plaintexts]

    cnf_text, total_vars, master_key_bits = export_speck_to_pure_cnf(
        pairs, rounds=4, pinned_round_keys={0: 0xBEEF}
    )
    proc_cad = subprocess.run(
        ["cadical", "-q"],
        input=cnf_text,
        text=True,
        capture_output=True,
    )
    assert proc_cad.returncode == 10
    rec_cad = decode_master_key_from_dimacs_solution(proc_cad.stdout, master_key_bits)
    assert rec_cad is not None
    assert rec_cad[3] == 0xBEEF
