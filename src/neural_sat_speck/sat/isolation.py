"""Enforce strict solver isolation across evaluated cryptanalytic methods.

Under the scientific integrity rules of COMP6002 (AGENTS.md and PLAN.md Phase 9),
reusing a SAT solver or SpeckSatModel instance across compared methods is strictly
forbidden. This module provides a validated factory boundary ensuring every method
receives a fresh, independent CryptoMiniSat solver instance and mutable SAT state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from neural_sat_speck.sat.model import KnownPair, SpeckSatModel


@dataclass
class IsolatedSolverFactory:
    """Factory guaranteeing per-method solver isolation for experimental benchmarks.

    Attributes
    ----------
    known_pairs : tuple[KnownPair, ...]
        The known plaintext/ciphertext constraint pairs for this instance.
    rounds : int
        Cipher round count.
    instance_id : str
        Unique instance identifier (e.g. 'test-0000').
    _issued_models : dict[str, int]
        Mapping from method_name to id(SpeckSatModel).
    _issued_solvers : set[int]
        Set of underlying C++ solver object IDs (id(pycryptosat.Solver)).
    """

    known_pairs: tuple[KnownPair, ...]
    rounds: int
    instance_id: str = ""
    _issued_models: dict[str, int] = field(default_factory=dict, init=False)
    _issued_solvers: set[int] = field(default_factory=set, init=False)

    @classmethod
    def from_pairs(
        cls,
        known_pairs: Sequence[KnownPair],
        rounds: int,
        instance_id: str = "",
    ) -> IsolatedSolverFactory:
        """Construct factory with validated pairs and rounds."""
        return cls(
            known_pairs=tuple(known_pairs),
            rounds=rounds,
            instance_id=instance_id,
        )

    def build_isolated_model(self, method_name: str) -> SpeckSatModel:
        """Construct a guaranteed fresh, unpolluted SpeckSatModel for a named method.

        Parameters
        ----------
        method_name : str
            The distinct method being evaluated (e.g. 'sat_only', 'b1_sat').

        Returns
        -------
        SpeckSatModel
            A brand new model with fresh C++ CryptoMiniSat solver and variable pool.

        Raises
        ------
        ValueError
            If method_name has already claimed a solver in this evaluation context.
        """
        if not method_name or not isinstance(method_name, str):
            raise ValueError("method_name must be a non-empty string")

        if method_name in self._issued_models:
            raise ValueError(
                f"Method '{method_name}' has already been issued a solver model for instance "
                f"'{self.instance_id}'. Shared or duplicate solver allocation is forbidden."
            )

        # Construct fresh independent model
        model = SpeckSatModel.from_known_pairs(self.known_pairs, rounds=self.rounds)

        model_id = id(model)
        solver_id = id(model.solver)

        if solver_id in self._issued_solvers:
            raise RuntimeError(
                f"Critical isolation violation: underlying solver ID {solver_id} was previously "
                "issued to another method. Memory reuse or object aliasing detected."
            )

        self._issued_models[method_name] = model_id
        self._issued_solvers.add(solver_id)
        return model

    def verify_isolation(self, active_models: Mapping[str, SpeckSatModel]) -> None:
        """Verify that all active models have distinct model and solver object identities.

        Raises
        ------
        ValueError
            If any object identity is shared across distinct method names.
        """
        seen_model_ids: dict[int, str] = {}
        seen_solver_ids: dict[int, str] = {}

        for method, model in active_models.items():
            m_id = id(model)
            s_id = id(model.solver)

            if m_id in seen_model_ids:
                raise ValueError(
                    f"Cross-method model reuse detected! Methods '{seen_model_ids[m_id]}' and "
                    f"'{method}' share the same SpeckSatModel instance ({m_id:#x})."
                )
            if s_id in seen_solver_ids:
                raise ValueError(
                    f"Cross-method solver reuse detected! Methods '{seen_solver_ids[s_id]}' and "
                    f"'{method}' share the same underlying CryptoMiniSat solver ({s_id:#x})."
                )

            seen_model_ids[m_id] = method
            seen_solver_ids[s_id] = method


def create_isolated_sat_model(
    known_pairs: Sequence[KnownPair],
    rounds: int,
) -> SpeckSatModel:
    """Convenience helper returning a fresh independent SpeckSatModel."""
    return SpeckSatModel.from_known_pairs(known_pairs, rounds=rounds)
