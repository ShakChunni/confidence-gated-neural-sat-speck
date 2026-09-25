"""Exact reduced-round SIMON32/64 Boolean constraint model for SAT solving.

Encodes exact SIMON32/64 round transformations and 100% linear key schedule
constraints using native XOR and CNF clauses for CryptoMiniSat (via pycryptosat
or native CLI with Gaussian elimination). Supports trailing-key assumptions
and formula export to native-XOR DIMACS format.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

from neural_sat_speck.sat.encoding import (
    VariablePool,
    WordBits,
    add_fixed_word,
    add_xor_bits,
    add_xor_constant,
    decode_word,
    rotate_bits_left,
    rotate_bits_right,
)
from neural_sat_speck.sat.model import SatSolveResult
from neural_sat_speck.simon.cipher import Block, _normalise_block
from neural_sat_speck.simon.key_schedule import (
    SIMON32_64_CONSTANT_C,
    SIMON32_64_KEY_WORD_COUNT,
    SIMON32_64_Z0,
    SIMON32_64_Z0_PERIOD,
    _normalise_master_key,
    _validate_round_count,
)
from neural_sat_speck.simon.primitives import mask_word

KnownPair: TypeAlias = tuple[Block, Block]
StateBits: TypeAlias = tuple[WordBits, WordBits]
DimacsClause: TypeAlias = list[int]


def _new_solver():
    """Instantiate a fresh isolated single-threaded pycryptosat solver."""
    try:
        from pycryptosat import Solver
    except ImportError as error:
        raise RuntimeError(
            "CryptoMiniSat is required for the SAT pilot; install pycryptosat"
        ) from error
    return Solver(verbose=0, threads=1)


@dataclass
class DimacsClauseRecorder:
    """Records CNF and native XOR clauses for DIMACS export and statistics."""

    cnf_clauses: list[DimacsClause] = field(default_factory=list)
    xor_clauses: list[tuple[list[int], bool]] = field(default_factory=list)

    def add_clause(self, clause: Sequence[int]) -> None:
        """Record one disjunctive CNF clause."""
        self.cnf_clauses.append(list(clause))

    def add_xor_clause(self, lits: Sequence[int], rhs: bool) -> None:
        """Record one native XOR clause: lits[0] ^ ... ^ lits[k-1] == rhs."""
        self.xor_clauses.append((list(lits), bool(rhs)))

    def to_dimacs(self, num_vars: int, assumptions: Sequence[int] = ()) -> str:
        """Render recorded clauses into CryptoMiniSat native-XOR DIMACS format."""
        extra_unit_clauses = [[lit] for lit in assumptions]
        total_clauses = (
            len(self.cnf_clauses)
            + len(extra_unit_clauses)
            + len(self.xor_clauses)
        )
        lines: list[str] = [f"p cnf {num_vars} {total_clauses}"]

        for clause in self.cnf_clauses:
            lines.append(" ".join(str(lit) for lit in clause) + " 0")
        for unit in extra_unit_clauses:
            lines.append(f"{unit[0]} 0")

        for lits, rhs in self.xor_clauses:
            if not lits:
                continue
            if rhs:
                # Odd parity = True
                lines.append("x " + " ".join(str(lit) for lit in lits) + " 0")
            else:
                # Even parity = False (negate first literal)
                lines.append(
                    f"x {-lits[0]} "
                    + " ".join(str(lit) for lit in lits[1:])
                    + " 0"
                )

        return "\n".join(lines) + "\n"


class RecordingSolverWrapper:
    """Pass-through solver wrapper that mirrors clause additions into a recorder."""

    def __init__(self, solver: object | None, recorder: DimacsClauseRecorder) -> None:
        self.solver = solver
        self.recorder = recorder

    def add_clause(self, clause: Sequence[int]) -> None:
        self.recorder.add_clause(clause)
        if self.solver is not None:
            self.solver.add_clause(clause)

    def add_xor_clause(self, lits: Sequence[int], rhs: bool) -> None:
        self.recorder.add_xor_clause(lits, rhs)
        if self.solver is not None:
            self.solver.add_xor_clause(lits, rhs)

    def solve(self, assumptions: Sequence[int] = (), **kwargs):
        if self.solver is None:
            raise RuntimeError("Underlying solver is not attached")
        if assumptions:
            return self.solver.solve(list(assumptions), **kwargs)
        return self.solver.solve(**kwargs)


@dataclass
class SimonSatModel:
    """Exact reduced-round SIMON32/64 SAT model built from known pairs."""

    solver: object
    pool: VariablePool
    known_pairs: tuple[KnownPair, ...]
    master_key_bits: tuple[WordBits, ...]  # (k3, k2, k1, k0) in little-endian
    round_keys: tuple[WordBits, ...]       # (k0, k1, ..., k_{rounds-1})
    state_bits: tuple[tuple[StateBits, ...], ...]
    rounds: int
    recorder: DimacsClauseRecorder

    @property
    def variable_count(self) -> int:
        """Total allocated SAT variables."""
        return self.pool.variable_count

    @property
    def cnf_clause_count(self) -> int:
        """Total disjunctive CNF clauses."""
        return len(self.recorder.cnf_clauses)

    @property
    def xor_clause_count(self) -> int:
        """Total native XOR clauses."""
        return len(self.recorder.xor_clauses)

    @property
    def total_clause_count(self) -> int:
        """Total clauses (CNF + XOR)."""
        return self.cnf_clause_count + self.xor_clause_count

    @classmethod
    def from_known_pairs(
        cls,
        known_pairs: Sequence[KnownPair],
        rounds: int,
        solver: object | None = None,
    ) -> SimonSatModel:
        """Build exact key-schedule and encryption constraints for SIMON32/64."""
        valid_rounds = _validate_round_count(rounds)
        normalised_pairs = tuple(
            (_normalise_block(pt), _normalise_block(ct))
            for pt, ct in known_pairs
        )
        if not normalised_pairs:
            raise ValueError("At least one known pair is required")

        underlying_solver = solver if solver is not None else _new_solver()
        recorder = DimacsClauseRecorder()
        recording_solver = RecordingSolverWrapper(underlying_solver, recorder)

        pool = VariablePool()
        # Master key in (k3, k2, k1, k0) word order, 16 bits each (little-endian)
        master_key_bits = tuple(pool.new_bits(16) for _ in range(SIMON32_64_KEY_WORD_COUNT))
        k3, k2, k1, k0 = master_key_bits
        round_keys: list[WordBits] = [k0, k1, k2, k3]

        # Key schedule expansion: 100% linear over F_2
        for i in range(valid_rounds - 4):
            # tmp = (k_{i+3} >>> 3) ^ k_{i+1}
            tmp = add_xor_bits(
                recording_solver,
                pool,
                rotate_bits_right(round_keys[i + 3], 3),
                round_keys[i + 1],
            )
            # tmp2 = tmp ^ (tmp >>> 1)
            tmp2 = add_xor_bits(
                recording_solver,
                pool,
                tmp,
                rotate_bits_right(tmp, 1),
            )
            z_bit = (SIMON32_64_Z0 >> (i % SIMON32_64_Z0_PERIOD)) & 1
            c_i = SIMON32_64_CONSTANT_C ^ z_bit
            k_next = add_xor_constant(
                recording_solver,
                pool,
                add_xor_bits(recording_solver, pool, round_keys[i], tmp2),
                c_i,
            )
            round_keys.append(k_next)

        # Cipher encryption rounds for each known observation
        all_state_bits: list[tuple[StateBits, ...]] = []
        for pt, ct in normalised_pairs:
            initial_l = pool.new_bits(16)
            initial_r = pool.new_bits(16)
            add_fixed_word(recording_solver, initial_l, pt[0])
            add_fixed_word(recording_solver, initial_r, pt[1])

            states: list[StateBits] = [(initial_l, initial_r)]
            state_l = initial_l
            state_r = initial_r

            for r in range(valid_rounds):
                k_r = round_keys[r]
                # Feistel branch function f(L) = (L <<< 1 & L <<< 8) ^ (L <<< 2)
                rot1 = rotate_bits_left(state_l, 1)
                rot8 = rotate_bits_left(state_l, 8)
                rot2 = rotate_bits_left(state_l, 2)

                # Bitwise AND: y = rot1 & rot8 (3 CNF clauses per bit)
                and_out = pool.new_bits(16)
                for j in range(16):
                    y_var = and_out[j]
                    u_var = rot1[j]
                    v_var = rot8[j]
                    recording_solver.add_clause([-y_var, u_var])
                    recording_solver.add_clause([-y_var, v_var])
                    recording_solver.add_clause([-u_var, -v_var, y_var])

                # Feistel update: next_L = R ^ and_out ^ rot2 ^ k_r
                # Encoded as 1 native 5-literal XOR clause per bit:
                # next_L ^ R ^ and_out ^ rot2 ^ k_r == 0
                next_l = pool.new_bits(16)
                for j in range(16):
                    recording_solver.add_xor_clause(
                        [next_l[j], state_r[j], and_out[j], rot2[j], k_r[j]],
                        False,
                    )

                state_r = state_l
                state_l = next_l
                states.append((state_l, state_r))

            # Fix ciphertext output
            add_fixed_word(recording_solver, state_l, ct[0])
            add_fixed_word(recording_solver, state_r, ct[1])
            all_state_bits.append(tuple(states))

        return cls(
            solver=underlying_solver,
            pool=pool,
            known_pairs=normalised_pairs,
            master_key_bits=master_key_bits,
            round_keys=tuple(round_keys[:valid_rounds]),
            state_bits=tuple(all_state_bits),
            rounds=valid_rounds,
            recorder=recorder,
        )

    def assumptions_for_master_key(
        self,
        master_key: Sequence[int],
    ) -> tuple[int, ...]:
        """Return literals fixing master-key bits to specified 4 words (k3, k2, k1, k0)."""
        words = _normalise_master_key(master_key)
        assumptions: list[int] = []
        for word, bits in zip(words, self.master_key_bits, strict=True):
            assumptions.extend(
                var if (word >> bit) & 1 else -var
                for bit, var in enumerate(bits)
            )
        return tuple(assumptions)

    def assumptions_for_round_key(
        self,
        round_index: int,
        value: int,
    ) -> tuple[int, ...]:
        """Return literals fixing the generated round key at round_index."""
        if not 0 <= round_index < len(self.round_keys):
            raise IndexError(f"round_index {round_index} outside model round keys")
        word = mask_word(value)
        bits = self.round_keys[round_index]
        return tuple(
            var if (word >> bit) & 1 else -var
            for bit, var in enumerate(bits)
        )

    def assumptions_for_trailing_round_subkeys(
        self,
        trailing_subkeys: Sequence[int],
    ) -> tuple[int, ...]:
        """Return literals fixing the last A consecutive round keys.

        Args:
            trailing_subkeys: Sequence of A words in chronological round order:
                              e.g. for A=2, (k_{rounds-2}, k_{rounds-1}).
        """
        count = len(trailing_subkeys)
        if not 1 <= count <= self.rounds:
            raise ValueError(f"trailing_subkeys count must be between 1 and {self.rounds}")
        start_round = self.rounds - count
        assumptions: list[int] = []
        for offset, val in enumerate(trailing_subkeys):
            r_idx = start_round + offset
            assumptions.extend(self.assumptions_for_round_key(r_idx, val))
        return tuple(assumptions)

    def solve(
        self,
        assumptions: Sequence[int] = (),
        *,
        time_limit: float | None = None,
    ) -> SatSolveResult:
        """Solve exact model via pycryptosat under temporary assumptions."""
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        started = perf_counter()
        assumption_list = list(assumptions)
        if time_limit is None:
            if assumption_list:
                satisfiable, solution = self.solver.solve(assumption_list)
            else:
                satisfiable, solution = self.solver.solve()
        else:
            satisfiable, solution = self.solver.solve(
                assumption_list,
                time_limit=time_limit,
            )
        elapsed = perf_counter() - started
        if satisfiable is True:
            status = "sat"
            timed_out: bool | None = False
        elif satisfiable is None:
            status = "solver_timeout"
            timed_out = True
        elif time_limit is not None and elapsed >= time_limit:
            status = "unsat_or_timeout"
            timed_out = None
        else:
            status = "unsat"
            timed_out = False

        return SatSolveResult(
            satisfiable=bool(satisfiable),
            solution=tuple(solution) if solution is not None else None,
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            status=status,
        )

    def solve_native_cli(
        self,
        assumptions: Sequence[int] = (),
        *,
        time_limit: float = 10.0,
        enable_gauss: bool = True,
        max_matrix_cols: int = 2500,
        binary_path: str = "cryptominisat5",
    ) -> SatSolveResult:
        """Solve model via native cryptominisat5 CLI with explicit Gauss settings."""
        dimacs_text = self.to_dimacs(assumptions=assumptions)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cnf", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(dimacs_text)
            tmp_path = Path(tmp.name)

        cmd = [
            binary_path,
            "--verb", "0",
            "--threads", "1",
            "--maxtime", str(time_limit),
        ]
        if enable_gauss:
            cmd.extend(["--maxmatrixcols", str(max_matrix_cols), "--autodisablegauss", "0"])

        cmd.append(str(tmp_path))

        started = perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=time_limit + 5.0,
            )
            elapsed = perf_counter() - started
            stdout = proc.stdout
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            elapsed = perf_counter() - started
            stdout = ""
            returncode = 0
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Parse CryptoMiniSat CLI output
        # Return code 10 = SAT, 20 = UNSAT, 0 = timeout/interrupted
        if returncode == 10 or "s SATISFIABLE" in stdout:
            # Extract solution literals from 'v ...' lines
            solution_map: dict[int, bool] = {}
            for line in stdout.splitlines():
                if line.startswith("v "):
                    for tok in line[2:].split():
                        lit = int(tok)
                        if lit != 0:
                            var = abs(lit)
                            solution_map[var] = lit > 0
            sol_array = [None] * (self.variable_count + 1)
            for var, val in solution_map.items():
                if var < len(sol_array):
                    sol_array[var] = val
            return SatSolveResult(
                satisfiable=True,
                solution=tuple(sol_array),
                elapsed_seconds=elapsed,
                timed_out=False,
                status="sat",
            )
        elif returncode == 20 or "s UNSATISFIABLE" in stdout:
            return SatSolveResult(
                satisfiable=False,
                solution=None,
                elapsed_seconds=elapsed,
                timed_out=False,
                status="unsat",
            )
        else:
            return SatSolveResult(
                satisfiable=False,
                solution=None,
                elapsed_seconds=elapsed,
                timed_out=True,
                status="solver_timeout",
            )

    def master_key_from_solution(
        self,
        solution: tuple[bool | None, ...],
    ) -> tuple[int, int, int, int]:
        """Decode (k3, k2, k1, k0) master key from SAT solution."""
        return (
            decode_word(solution, self.master_key_bits[0]),
            decode_word(solution, self.master_key_bits[1]),
            decode_word(solution, self.master_key_bits[2]),
            decode_word(solution, self.master_key_bits[3]),
        )

    def round_keys_from_solution(
        self,
        solution: tuple[bool | None, ...],
    ) -> tuple[int, ...]:
        """Decode all round keys from SAT solution."""
        return tuple(
            decode_word(solution, r_bits)
            for r_bits in self.round_keys
        )

    def to_dimacs(self, assumptions: Sequence[int] = ()) -> str:
        """Render model to native-XOR DIMACS format, including temporary assumptions."""
        return self.recorder.to_dimacs(self.variable_count, assumptions=assumptions)

    def export_dimacs(
        self,
        path: Path | str,
        assumptions: Sequence[int] = (),
    ) -> None:
        """Save model to a DIMACS file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_dimacs(assumptions=assumptions))
