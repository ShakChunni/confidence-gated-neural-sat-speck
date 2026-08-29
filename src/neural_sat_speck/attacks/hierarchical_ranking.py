"""Solver-budget-aware hierarchical final-round subkey ranking.

The H1 method is deliberately a composition of the frozen Phase 6A B1 and
B0 rankers.  B1 ranks the complete 16-bit candidate domain and supplies a
high-recall top-16 window.  The existing independent-pair B0 ranker then
scores only that window by summed raw logits.  H1 changes the ordering inside
the window and leaves the B1 tail byte/order-identical; it never creates a
fused candidate score and never uses the hidden true subkey.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.neural.numpy_inference import NumpyInferenceModule
from neural_sat_speck.speck.primitives import WORD_MASK

from .group_ranking import (
    PAIR_GROUP_SCORE_AGGREGATION,
    PairGroupSubkeyRanking,
)
from .ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    SCORE_AGGREGATION,
    SubkeyRanking,
    rank_final_round_subkeys,
)

HIERARCHICAL_METHOD_ID = "h1_b1_top16_b0_rerank"
HIERARCHICAL_WINDOW_SIZE = 16
HIERARCHICAL_PAIR_COUNT = 64
HIERARCHICAL_TIE_RULE = "descending_b0_sum_logits_then_ascending_candidate"


def _validate_seconds(value: float, name: str) -> float:
    """Validate and normalise a non-negative finite timing measurement."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalised = float(value)
    if not math.isfinite(normalised) or normalised < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalised


def _validate_optional_text(value: str | None, name: str) -> str | None:
    """Validate optional provenance text without imposing a hash format."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    """Resolve the requested device or the current model parameter device."""

    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _validate_full_b1_input(
    instance: AttackInstance,
    b1_ranking: PairGroupSubkeyRanking,
) -> tuple[int, ...]:
    """Validate the complete-domain B1 ranking and return its candidate order."""

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if len(instance.pairs) != HIERARCHICAL_PAIR_COUNT:
        raise ValueError("H1 requires an attack instance with exactly 64 pairs")
    if not isinstance(b1_ranking, PairGroupSubkeyRanking):
        raise TypeError("b1_ranking must be a PairGroupSubkeyRanking")
    if b1_ranking.pair_count != HIERARCHICAL_PAIR_COUNT:
        raise ValueError("B1 ranking must contain exactly 64 pairs")
    if b1_ranking.group_size != HIERARCHICAL_PAIR_COUNT:
        raise ValueError("B1 ranking group size must be exactly 64")
    if b1_ranking.score_aggregation != PAIR_GROUP_SCORE_AGGREGATION:
        raise ValueError("B1 ranking must use group_logit aggregation")
    if len(b1_ranking.candidate_values) != len(ALL_FINAL_ROUND_SUBKEYS):
        raise ValueError("B1 ranking must score the complete 16-bit domain")
    if b1_ranking.candidate_values != ALL_FINAL_ROUND_SUBKEYS:
        raise ValueError(
            "B1 ranking candidate values must be the ascending complete domain"
        )

    candidate_order = tuple(b1_ranking.candidate_order)
    if len(candidate_order) != len(ALL_FINAL_ROUND_SUBKEYS):
        raise ValueError("B1 candidate order must contain all 65,536 candidates")
    if set(candidate_order) != set(ALL_FINAL_ROUND_SUBKEYS):
        raise ValueError("B1 candidate order must be a complete candidate permutation")
    if len(set(candidate_order)) != len(candidate_order):
        raise ValueError("B1 candidate order must not contain duplicates")

    for name, value in (
        ("partial_decryption_seconds", b1_ranking.partial_decryption_seconds),
        ("inference_seconds", b1_ranking.inference_seconds),
        ("sorting_seconds", b1_ranking.sorting_seconds),
        ("total_seconds", b1_ranking.total_seconds),
    ):
        _validate_seconds(value, f"B1 {name}")
    return candidate_order


def _parse_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Parse an integer sequence used by :meth:`from_dict`."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    for item in result:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers")
    return result


def _parse_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    """Parse and validate a finite floating-point sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    parsed: list[float] = []
    for item in result:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain numbers")
        normalised = float(item)
        if not math.isfinite(normalised):
            raise ValueError(f"{name} items must be finite")
        parsed.append(normalised)
    return tuple(parsed)


@dataclass(frozen=True)
class HierarchicalSubkeyRanking:
    """Complete H1 order and the provenance of its two-stage construction.

    ``candidate_order`` is the complete H1 permutation.  The first 16 values
    are ``reranked_top16_order`` and the remaining values are exactly
    ``b1_tail``.  ``b0_window_candidate_values`` and ``b0_window_scores`` are
    parallel ascending-value tuples containing only the 16 candidates that B0
    evaluated.  No fused score is stored because H1 is an order composition,
    not a new scalar scoring rule.
    """

    method_id: str
    candidate_values: tuple[int, ...]
    candidate_order: tuple[int, ...]
    b1_top16_input_order: tuple[int, ...]
    b1_tail: tuple[int, ...]
    reranked_top16_order: tuple[int, ...]
    b0_window_candidate_values: tuple[int, ...]
    b0_window_scores: tuple[float, ...]
    pair_count: int
    group_size: int
    candidate_budget: int
    window_size: int
    b1_score_aggregation: str
    b0_score_aggregation: str
    b1_partial_decryption_seconds: float
    b1_inference_seconds: float
    b1_sorting_seconds: float
    b1_ranking_seconds: float
    b0_window_partial_decryption_seconds: float
    b0_window_inference_seconds: float
    b0_window_sorting_seconds: float
    b0_window_seconds: float
    composition_seconds: float
    invocation_seconds: float
    total_seconds: float
    device: str
    b1_inference_backend: str
    b0_inference_backend: str
    b1_device: str | None = None
    b1_checkpoint_sha256: str | None = None
    b0_checkpoint_sha256: str | None = None
    b1_model_id: str | None = None
    b0_model_id: str | None = None

    def __post_init__(self) -> None:
        """Enforce H1's complete-domain and head/tail invariants."""

        if self.method_id != HIERARCHICAL_METHOD_ID:
            raise ValueError(f"method_id must be {HIERARCHICAL_METHOD_ID!r}")
        if self.candidate_values != ALL_FINAL_ROUND_SUBKEYS:
            raise ValueError("H1 candidate_values must be the complete 16-bit domain")
        if len(self.candidate_order) != len(ALL_FINAL_ROUND_SUBKEYS):
            raise ValueError("H1 candidate order must contain all 65,536 candidates")
        if set(self.candidate_order) != set(ALL_FINAL_ROUND_SUBKEYS):
            raise ValueError("H1 candidate order must be a complete permutation")
        if len(set(self.candidate_order)) != len(self.candidate_order):
            raise ValueError("H1 candidate order must not contain duplicates")
        if self.candidate_order[:HIERARCHICAL_WINDOW_SIZE] != self.reranked_top16_order:
            raise ValueError("H1 head must equal reranked_top16_order")
        if self.candidate_order[HIERARCHICAL_WINDOW_SIZE:] != self.b1_tail:
            raise ValueError("H1 tail must equal the unchanged B1 tail")
        if len(self.b1_top16_input_order) != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B1 input window must contain exactly 16 candidates")
        if len(set(self.b1_top16_input_order)) != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B1 input window must not contain duplicates")
        if set(self.b1_top16_input_order) != set(self.reranked_top16_order):
            raise ValueError("H1 must preserve the B1 top-16 candidate set")
        if len(self.b1_tail) != len(ALL_FINAL_ROUND_SUBKEYS) - HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B1 tail has the wrong length")
        if len(set(self.b1_tail)) != len(self.b1_tail):
            raise ValueError("B1 tail must not contain duplicates")
        if set(self.b1_tail) | set(self.b1_top16_input_order) != set(
            ALL_FINAL_ROUND_SUBKEYS
        ):
            raise ValueError("B1 head and tail must cover the complete domain")
        if set(self.b1_tail) & set(self.b1_top16_input_order):
            raise ValueError("B1 head and tail must be disjoint")
        if len(self.b0_window_candidate_values) != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B0 window must contain exactly 16 candidates")
        if self.b0_window_candidate_values != tuple(
            sorted(self.b0_window_candidate_values)
        ):
            raise ValueError("B0 window candidate values must be ascending")
        if len(set(self.b0_window_candidate_values)) != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B0 window candidate values must not repeat")
        if set(self.b0_window_candidate_values) != set(self.b1_top16_input_order):
            raise ValueError("B0 must score exactly the B1 top-16 candidate set")
        if len(self.b0_window_scores) != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("B0 must record one score per window candidate")
        expected_reranked_top16 = tuple(
            candidate
            for candidate, _ in sorted(
                zip(
                    self.b0_window_candidate_values,
                    self.b0_window_scores,
                    strict=True,
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )
        if self.reranked_top16_order != expected_reranked_top16:
            raise ValueError(
                "H1 reranked order must follow the recorded B0 scores and tie rule"
            )
        if self.pair_count != HIERARCHICAL_PAIR_COUNT:
            raise ValueError("H1 requires exactly 64 pairs")
        if self.group_size != HIERARCHICAL_PAIR_COUNT:
            raise ValueError("H1 group size must be exactly 64")
        if self.candidate_budget != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("H1 candidate budget must be exactly 16")
        if self.window_size != HIERARCHICAL_WINDOW_SIZE:
            raise ValueError("H1 window size must be exactly 16")
        if self.b1_score_aggregation != PAIR_GROUP_SCORE_AGGREGATION:
            raise ValueError("H1 B1 score convention must be group_logit")
        if self.b0_score_aggregation != SCORE_AGGREGATION:
            raise ValueError("H1 B0 score convention must be sum_logits")
        for index, score in enumerate(self.b0_window_scores):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"B0 window score {index} must be a number")
            if not math.isfinite(float(score)):
                raise ValueError(f"B0 window score {index} must be finite")
        for name, value in (
            ("b1_partial_decryption_seconds", self.b1_partial_decryption_seconds),
            ("b1_inference_seconds", self.b1_inference_seconds),
            ("b1_sorting_seconds", self.b1_sorting_seconds),
            ("b1_ranking_seconds", self.b1_ranking_seconds),
            (
                "b0_window_partial_decryption_seconds",
                self.b0_window_partial_decryption_seconds,
            ),
            ("b0_window_inference_seconds", self.b0_window_inference_seconds),
            ("b0_window_sorting_seconds", self.b0_window_sorting_seconds),
            ("b0_window_seconds", self.b0_window_seconds),
            ("composition_seconds", self.composition_seconds),
            ("invocation_seconds", self.invocation_seconds),
            ("total_seconds", self.total_seconds),
        ):
            _validate_seconds(value, name)
        for name, value in (
            ("device", self.device),
            ("b1_inference_backend", self.b1_inference_backend),
            ("b0_inference_backend", self.b0_inference_backend),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _validate_optional_text(self.b1_device, "b1_device")
        _validate_optional_text(self.b1_checkpoint_sha256, "b1_checkpoint_sha256")
        _validate_optional_text(self.b0_checkpoint_sha256, "b0_checkpoint_sha256")
        _validate_optional_text(self.b1_model_id, "b1_model_id")
        _validate_optional_text(self.b0_model_id, "b0_model_id")
        expected_total = (
            self.b1_ranking_seconds
            + self.b0_window_seconds
            + self.composition_seconds
        )
        if not math.isclose(self.total_seconds, expected_total, rel_tol=1e-12):
            raise ValueError("H1 total time must equal its recorded stage totals")

    @property
    def b1_top16(self) -> tuple[int, ...]:
        """Alias for the deterministic B1 input window order."""

        return self.b1_top16_input_order

    @property
    def reranked_top16(self) -> tuple[int, ...]:
        """Alias for H1's reranked head."""

        return self.reranked_top16_order

    @property
    def h1_order(self) -> tuple[int, ...]:
        """Alias for the complete H1 candidate order."""

        return self.candidate_order

    @property
    def b0_scores(self) -> dict[int, float]:
        """Return B0's recorded summed-logit score for each window candidate."""

        return dict(zip(self.b0_window_candidate_values, self.b0_window_scores, strict=True))

    def b0_score_for(self, candidate: int) -> float:
        """Return a B0 window score, rejecting candidates outside the window."""

        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("candidate must be an integer")
        position = bisect_left(self.b0_window_candidate_values, candidate)
        if (
            position == len(self.b0_window_candidate_values)
            or self.b0_window_candidate_values[position] != candidate
        ):
            raise KeyError(f"candidate {candidate:#06x} was not scored by B0")
        return self.b0_window_scores[position]

    def rank(self, candidate: int) -> int:
        """Return the one-based H1 rank of a complete-domain candidate."""

        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("candidate must be an integer")
        try:
            return self.candidate_order.index(candidate) + 1
        except ValueError as error:
            raise KeyError(f"candidate {candidate:#06x} was not ranked") from error

    def top_k(self, k: int) -> tuple[int, ...]:
        """Return the first ``k`` candidates from the H1 order."""

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if not 0 <= k <= len(self.candidate_order):
            raise ValueError("k must be between zero and the candidate count")
        return self.candidate_order[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize H1 order, B0 window scores, timing, and provenance."""

        return {
            "method_id": self.method_id,
            "candidate_values": list(self.candidate_values),
            "candidate_order": list(self.candidate_order),
            "b1_top16_input_order": list(self.b1_top16_input_order),
            "b1_tail": list(self.b1_tail),
            "reranked_top16_order": list(self.reranked_top16_order),
            "b0_window_candidate_values": list(self.b0_window_candidate_values),
            "b0_window_scores": list(self.b0_window_scores),
            "pair_count": self.pair_count,
            "group_size": self.group_size,
            "candidate_budget": self.candidate_budget,
            "window_size": self.window_size,
            "score_aggregation": {
                "b1": self.b1_score_aggregation,
                "b0": self.b0_score_aggregation,
            },
            "timing_seconds": {
                "b1_ranking": {
                    "partial_decryption": self.b1_partial_decryption_seconds,
                    "inference": self.b1_inference_seconds,
                    "sorting": self.b1_sorting_seconds,
                    "total": self.b1_ranking_seconds,
                },
                "b0_window": {
                    "partial_decryption": self.b0_window_partial_decryption_seconds,
                    "inference": self.b0_window_inference_seconds,
                    "sorting": self.b0_window_sorting_seconds,
                    "total": self.b0_window_seconds,
                },
                "composition": self.composition_seconds,
                "invocation": self.invocation_seconds,
                "total": self.total_seconds,
            },
            "candidate_domain": {
                "first": 0,
                "last": WORD_MASK,
                "count": len(ALL_FINAL_ROUND_SUBKEYS),
                "complete_16_bit_domain": True,
            },
            "tie_rule": HIERARCHICAL_TIE_RULE,
            "provenance": {
                "b1_checkpoint_sha256": self.b1_checkpoint_sha256,
                "b0_checkpoint_sha256": self.b0_checkpoint_sha256,
                "b1_model_id": self.b1_model_id,
                "b0_model_id": self.b0_model_id,
                "b1_device": self.b1_device,
                "device": self.device,
                "b1_inference_backend": self.b1_inference_backend,
                "b0_inference_backend": self.b0_inference_backend,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HierarchicalSubkeyRanking:
        """Reload an H1 result serialized by :meth:`to_dict`."""

        if not isinstance(payload, Mapping):
            raise TypeError("hierarchical ranking payload must be a mapping")
        timing = payload.get("timing_seconds")
        if not isinstance(timing, Mapping):
            raise ValueError("hierarchical ranking payload is missing timings")
        b1_timing = timing.get("b1_ranking")
        b0_timing = timing.get("b0_window")
        if not isinstance(b1_timing, Mapping) or not isinstance(b0_timing, Mapping):
            raise ValueError("hierarchical ranking payload has invalid stage timings")
        aggregation = payload.get("score_aggregation")
        if not isinstance(aggregation, Mapping):
            raise ValueError("hierarchical ranking payload is missing score conventions")
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("hierarchical ranking payload has invalid provenance")
        if payload.get("tie_rule") != HIERARCHICAL_TIE_RULE:
            raise ValueError("hierarchical ranking payload has the wrong tie rule")
        candidate_domain = payload.get("candidate_domain")
        if not isinstance(candidate_domain, Mapping) or candidate_domain != {
            "first": 0,
            "last": WORD_MASK,
            "count": len(ALL_FINAL_ROUND_SUBKEYS),
            "complete_16_bit_domain": True,
        }:
            raise ValueError(
                "hierarchical ranking payload must describe the complete domain"
            )
        candidate_order = _parse_int_tuple(
            payload.get("candidate_order"),
            "candidate_order",
        )
        b1_tail_value = payload.get("b1_tail")
        if b1_tail_value is None:
            b1_tail_value = candidate_order[HIERARCHICAL_WINDOW_SIZE:]
        return cls(
            method_id=payload.get("method_id"),
            candidate_values=_parse_int_tuple(
                payload.get("candidate_values"),
                "candidate_values",
            ),
            candidate_order=candidate_order,
            b1_top16_input_order=_parse_int_tuple(
                payload.get("b1_top16_input_order"),
                "b1_top16_input_order",
            ),
            b1_tail=_parse_int_tuple(b1_tail_value, "b1_tail"),
            reranked_top16_order=_parse_int_tuple(
                payload.get("reranked_top16_order"),
                "reranked_top16_order",
            ),
            b0_window_candidate_values=_parse_int_tuple(
                payload.get("b0_window_candidate_values"),
                "b0_window_candidate_values",
            ),
            b0_window_scores=_parse_float_tuple(
                payload.get("b0_window_scores"),
                "b0_window_scores",
            ),
            pair_count=payload.get("pair_count"),
            group_size=payload.get("group_size"),
            candidate_budget=payload.get("candidate_budget"),
            window_size=payload.get("window_size"),
            b1_score_aggregation=aggregation.get("b1"),
            b0_score_aggregation=aggregation.get("b0"),
            b1_partial_decryption_seconds=b1_timing.get("partial_decryption"),
            b1_inference_seconds=b1_timing.get("inference"),
            b1_sorting_seconds=b1_timing.get("sorting"),
            b1_ranking_seconds=b1_timing.get("total"),
            b0_window_partial_decryption_seconds=b0_timing.get(
                "partial_decryption"
            ),
            b0_window_inference_seconds=b0_timing.get("inference"),
            b0_window_sorting_seconds=b0_timing.get("sorting"),
            b0_window_seconds=b0_timing.get("total"),
            composition_seconds=timing.get("composition"),
            invocation_seconds=timing.get("invocation"),
            total_seconds=timing.get("total"),
            device=provenance.get("device"),
            b1_inference_backend=provenance.get("b1_inference_backend"),
            b0_inference_backend=provenance.get("b0_inference_backend"),
            b1_device=provenance.get("b1_device"),
            b1_checkpoint_sha256=provenance.get("b1_checkpoint_sha256"),
            b0_checkpoint_sha256=provenance.get("b0_checkpoint_sha256"),
            b1_model_id=provenance.get("b1_model_id"),
            b0_model_id=provenance.get("b0_model_id"),
        )


def rank_hierarchical_subkeys(
    instance: AttackInstance,
    b1_ranking: PairGroupSubkeyRanking,
    b0_model: nn.Module,
    *,
    window_size: int = HIERARCHICAL_WINDOW_SIZE,
    b0_candidate_batch_size: int = HIERARCHICAL_WINDOW_SIZE,
    device: torch.device | str | None = None,
    b1_device: torch.device | str | None = None,
    b1_inference_backend: str = "precomputed",
    b1_checkpoint_sha256: str | None = None,
    b0_checkpoint_sha256: str | None = None,
    b1_model_id: str | None = None,
    b0_model_id: str | None = "speck32_64_b0_gohr_baseline",
) -> HierarchicalSubkeyRanking:
    """Build H1 from a complete B1 order and a frozen B0 model.

    The B1 ranking is an input and is never recomputed here.  Its first 16
    candidates are passed to :func:`rank_final_round_subkeys`, which performs
    the existing B0 partial-decryption, independent-pair inference, raw-logit
    summation, and deterministic score sorting.  No true subkey or candidate
    label is accepted by this function.
    """

    b1_order = _validate_full_b1_input(instance, b1_ranking)
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size != HIERARCHICAL_WINDOW_SIZE:
        raise ValueError("H1 window_size is fixed at 16")
    if isinstance(b0_candidate_batch_size, bool) or not isinstance(
        b0_candidate_batch_size, int
    ):
        raise TypeError("b0_candidate_batch_size must be an integer")
    if b0_candidate_batch_size <= 0:
        raise ValueError("b0_candidate_batch_size must be positive")
    if not isinstance(b0_model, nn.Module):
        raise TypeError("b0_model must be a torch.nn.Module")
    if not isinstance(b1_inference_backend, str) or not b1_inference_backend:
        raise ValueError("b1_inference_backend must be a non-empty string")
    b1_device_value = None if b1_device is None else str(torch.device(b1_device))
    b0_device = _resolve_device(b0_model, device)
    b0_backend = "numpy" if isinstance(b0_model, NumpyInferenceModule) else "pytorch"
    _validate_optional_text(b1_checkpoint_sha256, "b1_checkpoint_sha256")
    _validate_optional_text(b0_checkpoint_sha256, "b0_checkpoint_sha256")
    _validate_optional_text(b1_model_id, "b1_model_id")
    _validate_optional_text(b0_model_id, "b0_model_id")

    started = time.perf_counter()
    b1_top16 = b1_order[:HIERARCHICAL_WINDOW_SIZE]
    b0_ranking: SubkeyRanking = rank_final_round_subkeys(
        instance,
        b0_model,
        candidate_values=b1_top16,
        candidate_batch_size=b0_candidate_batch_size,
        device=b0_device,
    )
    if len(b0_ranking.candidate_values) != HIERARCHICAL_WINDOW_SIZE:
        raise RuntimeError("B0 did not score exactly the H1 window")
    if set(b0_ranking.candidate_values) != set(b1_top16):
        raise RuntimeError("B0 scored candidates outside the B1 H1 window")
    if b0_ranking.pair_count != HIERARCHICAL_PAIR_COUNT:
        raise RuntimeError("B0 did not score all 64 attack pairs")

    composition_started = time.perf_counter()
    reranked_top16 = tuple(b0_ranking.candidate_order)
    b1_tail = b1_order[HIERARCHICAL_WINDOW_SIZE:]
    candidate_order = reranked_top16 + b1_tail
    composition_seconds = time.perf_counter() - composition_started
    invocation_seconds = time.perf_counter() - started
    b1_ranking_seconds = float(b1_ranking.total_seconds)
    b0_window_seconds = float(b0_ranking.total_seconds)
    total_seconds = b1_ranking_seconds + b0_window_seconds + composition_seconds

    return HierarchicalSubkeyRanking(
        method_id=HIERARCHICAL_METHOD_ID,
        candidate_values=ALL_FINAL_ROUND_SUBKEYS,
        candidate_order=candidate_order,
        b1_top16_input_order=b1_top16,
        b1_tail=b1_tail,
        reranked_top16_order=reranked_top16,
        b0_window_candidate_values=b0_ranking.candidate_values,
        b0_window_scores=b0_ranking.candidate_scores,
        pair_count=HIERARCHICAL_PAIR_COUNT,
        group_size=HIERARCHICAL_PAIR_COUNT,
        candidate_budget=HIERARCHICAL_WINDOW_SIZE,
        window_size=HIERARCHICAL_WINDOW_SIZE,
        b1_score_aggregation=b1_ranking.score_aggregation,
        b0_score_aggregation=b0_ranking.score_aggregation,
        b1_partial_decryption_seconds=float(
            b1_ranking.partial_decryption_seconds
        ),
        b1_inference_seconds=float(b1_ranking.inference_seconds),
        b1_sorting_seconds=float(b1_ranking.sorting_seconds),
        b1_ranking_seconds=b1_ranking_seconds,
        b0_window_partial_decryption_seconds=float(
            b0_ranking.partial_decryption_seconds
        ),
        b0_window_inference_seconds=float(b0_ranking.inference_seconds),
        b0_window_sorting_seconds=float(b0_ranking.sorting_seconds),
        b0_window_seconds=b0_window_seconds,
        composition_seconds=composition_seconds,
        invocation_seconds=invocation_seconds,
        total_seconds=total_seconds,
        device=str(b0_device),
        b1_inference_backend=b1_inference_backend,
        b0_inference_backend=b0_backend,
        b1_device=b1_device_value,
        b1_checkpoint_sha256=b1_checkpoint_sha256,
        b0_checkpoint_sha256=b0_checkpoint_sha256,
        b1_model_id=b1_model_id,
        b0_model_id=b0_model_id,
    )


__all__ = [
    "HIERARCHICAL_METHOD_ID",
    "HIERARCHICAL_PAIR_COUNT",
    "HIERARCHICAL_TIE_RULE",
    "HIERARCHICAL_WINDOW_SIZE",
    "HierarchicalSubkeyRanking",
    "rank_hierarchical_subkeys",
]
