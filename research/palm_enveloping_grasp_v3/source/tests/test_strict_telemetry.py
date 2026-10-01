from __future__ import annotations

import pytest
import torch

from simulation.ppo_tomato_grasp_v3.strict_telemetry import (
    GATE_BITS,
    GATE_ORDER,
    decode_failed_gates,
    encode_failed_gates,
    failure_histogram,
    update_streak,
)


def _gates() -> dict[str, torch.Tensor]:
    return {name: torch.tensor([True, True]) for name in GATE_ORDER}


def test_failure_mask_preserves_multiple_causes() -> None:
    gates = _gates()
    gates["all_five_physical"][1] = False
    gates["opposition"][1] = False
    code = encode_failed_gates(gates)
    assert int(code[0]) == 0
    assert int(code[1]) == GATE_BITS["all_five_physical"] | GATE_BITS["opposition"]
    assert decode_failed_gates(int(code[1])) == ["all_five_physical", "opposition"]


def test_streak_resets_and_retains_maximum() -> None:
    current = torch.zeros(2, dtype=torch.long)
    maximum = torch.zeros_like(current)
    for condition in (
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    ):
        current, maximum = update_streak(condition, current, maximum)
    assert current.tolist() == [0, 1]
    assert maximum.tolist() == [2, 1]


def test_histogram_reports_individual_and_joint_failures() -> None:
    code = GATE_BITS["coverage"] | GATE_BITS["opposition"]
    report = failure_histogram([0, code, code])
    assert report["combination:pass"] == 1
    assert report["gate:coverage"] == 2
    assert report["combination:coverage+opposition"] == 2


def test_missing_gate_is_rejected() -> None:
    gates = _gates()
    gates.pop("palm_first")
    with pytest.raises(KeyError, match="palm_first"):
        encode_failed_gates(gates)
