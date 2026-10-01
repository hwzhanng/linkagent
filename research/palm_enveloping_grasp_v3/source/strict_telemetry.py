"""Pure PyTorch helpers for auditable per-condition strict telemetry."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch


GATE_ORDER = (
    "palm_up",
    "palm_support",
    "all_five_physical",
    "coverage",
    "opposition",
    "hemisphere",
    "palm_load_share",
    "force_safe",
    "valid_support",
    "not_dropped",
    "distal_participation",
    "palm_first",
)
GATE_BITS = {name: 1 << index for index, name in enumerate(GATE_ORDER)}


def encode_failed_gates(gates: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Encode all false gates as a per-environment integer bit mask."""

    missing = [name for name in GATE_ORDER if name not in gates]
    if missing:
        raise KeyError(f"missing strict gates: {missing}")
    reference = gates[GATE_ORDER[0]]
    if reference.dtype != torch.bool or reference.ndim != 1:
        raise ValueError("strict gate tensors must be boolean vectors")
    result = torch.zeros_like(reference, dtype=torch.long)
    for name in GATE_ORDER:
        gate = gates[name]
        if gate.shape != reference.shape or gate.dtype != torch.bool:
            raise ValueError(f"gate {name} must match the reference boolean vector")
        result |= (~gate).to(torch.long) * GATE_BITS[name]
    return result


def decode_failed_gates(code: int) -> list[str]:
    return [name for name in GATE_ORDER if int(code) & GATE_BITS[name]]


def update_streak(
    condition: torch.Tensor, current: torch.Tensor, maximum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance/reset a boolean streak and its episode maximum."""

    if condition.dtype != torch.bool or condition.shape != current.shape or current.shape != maximum.shape:
        raise ValueError("condition/current/maximum must be matching vectors")
    updated = torch.where(condition, current + 1, torch.zeros_like(current))
    return updated, torch.maximum(maximum, updated)


def failure_histogram(codes: list[int]) -> dict[str, int]:
    """Count individual and exact-combination failure labels."""

    individual: Counter[str] = Counter()
    combinations: Counter[str] = Counter()
    for code in codes:
        labels = decode_failed_gates(code)
        if not labels:
            combinations["pass"] += 1
            continue
        for label in labels:
            individual[label] += 1
        combinations["+".join(labels)] += 1
    return {
        **{f"gate:{key}": value for key, value in sorted(individual.items())},
        **{f"combination:{key}": value for key, value in sorted(combinations.items())},
    }
