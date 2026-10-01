from __future__ import annotations

from typing import TypedDict

import pytest
import torch

from research.small_fruit_pinch_v1.source import contract
from research.small_fruit_pinch_v1.source.verifier import TimStrictVerifier


class Inputs(TypedDict, total=False):
    force_n: torch.Tensor
    palm_contact: torch.Tensor
    center_drift_m: torch.Tensor
    relative_speed_m_s: torch.Tensor
    safety_projection: torch.Tensor
    self_collision: torch.Tensor
    support_removed: torch.Tensor
    disturbance_applied: torch.Tensor
    topology_valid: torch.Tensor
    physical_contact_valid: torch.Tensor
    forbidden_contact: torch.Tensor


def _inputs(n: int = 1) -> Inputs:
    return Inputs(
        force_n=torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.0]]).repeat(n, 1),
        palm_contact=torch.zeros(n, dtype=torch.bool),
        center_drift_m=torch.full((n,), 0.001),
        relative_speed_m_s=torch.full((n,), 0.001),
        safety_projection=torch.zeros(n, dtype=torch.bool),
        self_collision=torch.zeros(n, dtype=torch.bool),
        support_removed=torch.ones(n, dtype=torch.bool),
        disturbance_applied=torch.zeros(n, dtype=torch.bool),
        topology_valid=torch.ones(n, dtype=torch.bool),
        physical_contact_valid=torch.ones(n, dtype=torch.bool),
        forbidden_contact=torch.zeros(n, dtype=torch.bool),
    )


def _advance(verifier: TimStrictVerifier, inputs: Inputs, frames: int) -> None:
    for _ in range(frames):
        verifier.update(**inputs)


def test_soft_band_loss_after_acquisition_cannot_pass_final() -> None:
    # 前提：83 帧 0.5 N 获取接触，并在当前窗内确认真实扰动。
    verifier = TimStrictVerifier(1, device="cpu")
    inputs = _inputs()
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 41)
    assert verifier.success.item()
    # 操作：后续 46 帧 2 N 已脱离软带，但尚未硬过载。
    inputs["force_n"][:, :3] = 2.0
    _advance(verifier, inputs, 46)
    # 预期：历史接触获取不能充当最终保持证明。
    assert not verifier.final_success.item()
    assert verifier.support_hold_age.item() == verifier.post_hold_age.item() == 0
    assert verifier.retention_lost.item()


def test_disturbance_before_scheduled_hold_step_cannot_qualify() -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    _advance(verifier, inputs, 10)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 150)
    assert verifier.contact_acquired.item()
    assert not verifier.support_retained.item()


@pytest.mark.parametrize("key", ["support_removed", "disturbance_applied", "topology_valid", "physical_contact_valid"])
@pytest.mark.parametrize("missing", [False, True])
def test_required_evidence_fails_closed(key: str, missing: bool) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    if missing:
        inputs.pop(key)
    else:
        inputs[key].fill_(False)
    _advance(verifier, inputs, 42)
    if key != "disturbance_applied":
        inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 138)
    assert not verifier.final_success.item()
    if key in ("topology_valid", "physical_contact_valid"):
        assert not verifier.success.item()


@pytest.mark.parametrize("digit", [0, 1, 2])
def test_missing_required_digit_blocks_acquisition(digit: int) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    inputs["force_n"][:, digit] = 0.0
    _advance(verifier, inputs, 180)
    assert not verifier.success.item()
    assert not verifier.final_success.item()


@pytest.mark.parametrize("key,value", [
    ("palm_contact", True), ("forbidden_contact", True),
    ("safety_projection", True), ("self_collision", True),
    ("topology_valid", False), ("physical_contact_valid", False),
    ("support_removed", False), ("center_drift_m", 0.02),
    ("relative_speed_m_s", 0.04), ("force_n", 0.0),
    ("force_n", 2.0), ("force_n", 5.0),
])
def test_invalid_frame_revokes_final_and_window(key: str, value: float) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 79)
    assert verifier.final_success.item()
    inputs[key].fill_(value)
    state = verifier.update(**inputs)
    assert not state["final_success"].item()
    assert state["support_hold_age"].item() == state["post_hold_age"].item() == 0
    assert not state["success_event"].item()


@pytest.mark.parametrize("key", ["force_n", "palm_contact", "center_drift_m", "relative_speed_m_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_cannot_pass(key: str, value: float) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    _advance(verifier, inputs, 83)
    inputs[key] = torch.full(inputs[key].shape, value)
    _advance(verifier, inputs, 46)
    assert not verifier.final_success.item()
    assert verifier.support_hold_age.item() == 0


@pytest.mark.parametrize("key", list(_inputs()))
def test_shape_errors_are_rejected_before_state_changes(key: str) -> None:
    verifier, inputs = TimStrictVerifier(2, device="cpu"), _inputs(2)
    inputs[key] = torch.zeros((2, 1))
    with pytest.raises(ValueError):
        verifier.update(**inputs)
    assert not verifier.enter_age.any()
    assert not verifier.hold_age.any()


def test_positive_real_sequence_and_output_aliases() -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    inputs.pop("forbidden_contact")
    inputs["support_removed"].fill_(False)
    _advance(verifier, inputs, 81)
    assert verifier.success.item()
    assert verifier.hold_age.item() == 80
    assert verifier.support_hold_age.item() == 0
    _advance(verifier, inputs, 5)
    assert not verifier.retention_lost.item()
    inputs["support_removed"].fill_(True)
    _advance(verifier, inputs, 39)
    assert not verifier.disturbance_due.item()
    _advance(verifier, inputs, 1)
    assert verifier.disturbance_due.item()
    _advance(verifier, inputs, 40)
    assert verifier.disturbance_due.item()
    assert verifier.post_hold_age.item() == 0
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 1)
    assert not verifier.disturbance_due.item()
    assert verifier.post_hold_age.item() == 0
    _advance(verifier, inputs, 39)
    assert verifier.support_hold_age.item() == 120
    assert not verifier.final_success.item()
    state = verifier.update(**inputs)
    assert state["post_hold_age"].item() == 40
    assert state["final_success"].item() and state["success_event"].item()
    assert state["contact_acquired"] is verifier.contact_acquired is verifier.success
    assert state["support_retained"] is verifier.support_retained is verifier.final_success
    assert state["support_hold_age"] is verifier.support_hold_age
    assert set(state) == {"tim_contact", "force_in_band", "hard_overload", "candidate", "hold_age", "contact_ratio", "band_ratio", "palm_ratio", "disturbance_due", "success", "final_success", "success_event", "post_hold_age", "retention_lost", "support_hold_age", "contact_acquired", "support_retained"}
    assert not verifier.update(**inputs)["success_event"].item()


@pytest.mark.parametrize("supported_prefix", [0, 90])
def test_initial_true_disturbance_is_not_current_window_evidence(supported_prefix: int) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    inputs["disturbance_applied"].fill_(True)
    inputs["support_removed"].fill_(False)
    _advance(verifier, inputs, supported_prefix)
    inputs["support_removed"].fill_(True)
    _advance(verifier, inputs, 180)
    assert not verifier.final_success.item()
    assert verifier.disturbance_due.item()
    assert verifier.post_hold_age.item() == 0


def test_recovery_requires_new_window_and_fresh_disturbance() -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 79)
    inputs["force_n"][:, 0] = 0.0
    _advance(verifier, inputs, 1)
    inputs["force_n"][:, 0] = 0.5
    _advance(verifier, inputs, 120)
    assert verifier.success.item() and verifier.retention_lost.item()
    assert not verifier.final_success.item()
    assert verifier.disturbance_due.item()
    inputs["disturbance_applied"].fill_(False)
    _advance(verifier, inputs, 1)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 40)
    assert not verifier.final_success.item()
    _advance(verifier, inputs, 1)
    assert verifier.final_success.item()
    assert not verifier.retention_lost.item()


@pytest.mark.parametrize("key", ["force_n", "self_collision", "safety_projection", "forbidden_contact"])
def test_safety_violation_requires_episode_reset(key: str) -> None:
    verifier, inputs = TimStrictVerifier(1, device="cpu"), _inputs()
    inputs[key].fill_(5)
    _advance(verifier, inputs, 1)
    inputs = _inputs()
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 140)
    assert not verifier.final_success.item()


def test_vector_reset_isolates_all_window_state() -> None:
    verifier, inputs = TimStrictVerifier(2, device="cpu"), _inputs(2)
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 79)
    verifier.reset(torch.tensor([0]))
    assert verifier.success.tolist() == [False, True]
    assert verifier.final_success.tolist() == [False, True]
    assert verifier.support_hold_age.tolist() == [0, 120]
    _advance(verifier, inputs, 180)
    assert verifier.final_success.tolist() == [False, True]
    assert verifier.disturbance_due.tolist() == [True, False]
    assert verifier.post_hold_age[0].item() == 0
    verifier.reset()
    assert not verifier.success.any() and not verifier.final_success.any()
    assert not verifier.support_hold_age.any() and not verifier.post_hold_age.any()
    assert not verifier.disturbance_due.any() and not verifier.retention_lost.any()


def test_one_environment_fault_does_not_invalidate_its_neighbor() -> None:
    verifier, inputs = TimStrictVerifier(2, device="cpu"), _inputs(2)
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 79)
    inputs["force_n"][0, 0] = float("nan")
    verifier.update(**inputs)
    assert verifier.final_success.tolist() == [False, True]
    assert verifier.support_hold_age.tolist() == [0, 121]
    verifier.reset(torch.tensor([0]))
    inputs = _inputs(2)
    _advance(verifier, inputs, 42)
    inputs["disturbance_applied"].fill_(True)
    _advance(verifier, inputs, 79)
    assert verifier.final_success.tolist() == [True, True]


def test_contract_version_changes_without_tensor_dimension_changes() -> None:
    assert contract.TASK_VERSION == "small_fruit_pinch_v1.2.0"
    assert contract.VERIFICATION_CONTRACT_VERSION == "support_retention_v1"
    assert (contract.ACTION_DIM, contract.ACTOR_OBSERVATION_DIM, contract.CRITIC_STATE_DIM) == (14, 114, 175)
