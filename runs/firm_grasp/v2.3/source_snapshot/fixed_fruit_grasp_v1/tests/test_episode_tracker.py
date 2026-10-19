"""逐回合记录累加器的 CPU 测试（含与主线程账本 run_utils 的兼容性）。"""

from __future__ import annotations

import torch
import pytest

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1.env import _EpisodeTracker


def _state(
    num_envs: int = 1,
    *,
    forces: torch.Tensor | None = None,
    required: bool = True,
    hold_age: torch.Tensor | None = None,
    released: bool = False,
    height_gain_m: float = 0.0,
    faults: dict[str, bool] | None = None,
) -> dict[str, torch.Tensor]:
    count = num_envs
    force = (
        forces
        if forces is not None
        else torch.tensor([[0.5, 0.4, 0.3, 0.0, 0.0]] * count, dtype=torch.float32)
    )
    fault_flags = {
        name: torch.zeros(count, dtype=torch.bool) for name in _TRACKED_FLAG_NAMES
    }
    for name, value in (faults or {}).items():
        fault_flags[name] = torch.full((count,), bool(value), dtype=torch.bool)
    return {
        "forces_n": force,
        "raw_forces_n": force,
        "palm_force_n": torch.zeros(count),
        "forbidden_total_n": torch.zeros(count),
        "required_contact": torch.full((count,), bool(required), dtype=torch.bool),
        "contact_count": (force >= C.CONTACT_FORCE_N).sum(dim=1),
        "hold_age": (
            hold_age
            if hold_age is not None
            else torch.full((count,), 10, dtype=torch.long)
        ),
        "hold_ok": torch.full((count,), bool(required), dtype=torch.bool),
        "support_released": torch.full((count,), bool(released), dtype=torch.bool),
        "height_gain_m": torch.full((count,), float(height_gain_m)),
        "drift_m": torch.zeros(count),
        "rel_speed_m_s": torch.zeros(count),
        "joint_excursion_rad": torch.zeros(count),
        "max_applied_effort_nm": torch.zeros(count),
        "self_collision_clearance_m": torch.full((count,), 0.01),
        "collider_distance_m": torch.full((count,), 0.034),
        "penetration_depth_m": torch.zeros(count),
        "open_hand_baseline": torch.zeros(count, dtype=torch.bool),
        # v1.4.0 逐物理子步语义字段（peak 在回合内单调不减）。
        "physical_hold_streak": torch.full((count,), 120, dtype=torch.long),
        "physical_hold_peak_streak": torch.full((count,), 120, dtype=torch.long),
        "physical_hard_fault_flags": {
            name: torch.zeros(count, dtype=torch.bool) for name in _HARD_FAULT_NAMES
        },
        "faults": fault_flags,
    }


_HARD_FAULT_NAMES = (
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
)


_TRACKED_FLAG_NAMES = (
    "overload",
    "nonfinite",
    "penetration",
    "joint_limit",
    "self_collision",
    "instability",
    "joint_effort",
    "stem_fracture",
)


def _tracker(num_envs: int = 1, *, hold_steps: int = 60, stage: str = "hold") -> _EpisodeTracker:
    spec = C.stage_spec(stage)
    return _EpisodeTracker(
        num_envs,
        device="cpu",
        policy_dt_s=C.POLICY_DT_S,
        stage=spec.stage,
        success_criterion=spec.success_criterion,
        required_hold_steps=hold_steps,
        stage_min_height_gain_m=spec.min_height_gain_m,
        stage_success_requires_height_gain=spec.success_requires_height_gain,
        stage_requires_lift_program=spec.requires_lift_program,
        stage_is_goal=spec.counts_as_goal,
        goal_min_height_gain_m=C.MIN_HEIGHT_GAIN_M,
        goal_status=spec.goal_status,
        action_effect_ok=True,
        open_hand_baseline=False,
        immediate_failure_steps=5,
        contact_force_threshold_n=C.CONTACT_FORCE_N,
        min_contact_fingers=C.MIN_CONTACT_FINGERS,
        required_hold_substeps=hold_steps * C.DECIMATION,
        sim_dt_s=C.SIM_DT_S,
    )


def test_initial_reset_produces_no_row() -> None:
    tracker = _tracker(2)
    rows = tracker.finalize(
        torch.tensor([0, 1]),
        success=torch.zeros(2, dtype=torch.bool),
        reason_code=torch.zeros(2, dtype=torch.long),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        stem_fracture=torch.zeros(2, dtype=torch.bool),
        support_release_step=40,
    )
    assert rows == []
    assert tracker.episode_index.tolist() == [0, 0]


def test_row_matches_schema_and_is_bool_typed() -> None:
    tracker = _tracker(1)
    for _ in range(4):
        tracker.observe(_state(released=True))
    rows = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([True]),
        reason_code=torch.tensor([1]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        stem_fracture=torch.tensor([False]),
        support_release_step=40,
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(C.EPISODE_RECORD_FIELDS)
    assert isinstance(row["success"], bool) and row["success"] is True
    assert isinstance(row["terminated"], bool)
    assert isinstance(row["immediate_failure"], bool)
    assert row["reason"] == "success"
    assert row["reason_code"] == 1
    assert row["goal_success"] is False
    assert row["goal_status"] == C.GOAL_STATUS_LIFT_NOT_IMPLEMENTED
    assert row["stage_requires_lift_program"] is False
    assert row["stage_is_goal"] is False
    assert row["stage_success_requires_height_gain"] is False
    assert row["goal_min_height_gain_m"] == C.MIN_HEIGHT_GAIN_M
    assert row["steps"] == 4
    assert row["contact_fraction"] == 1.0
    assert row["finger_contact_fraction"][:3] == [1.0, 1.0, 1.0]
    assert row["finger_contact_fraction"][3:] == [0.0, 0.0]
    assert row["support_released"] is True
    assert row["stem_fracture_before_release"] is False
    # ``max_unsupported_hold_s`` 来自满足判据的连续保持计数（hold_age=10），
    # 不是回合步数。
    assert row["max_unsupported_hold_s"] == pytest.approx(10 * C.POLICY_DT_S)
    assert row["episode_id"] == 0


def test_episode_index_increments_and_rows_are_separate() -> None:
    tracker = _tracker(1)
    tracker.observe(_state(released=True))
    first = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([False]),
        reason_code=torch.tensor([3]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        stem_fracture=torch.tensor([False]),
        support_release_step=40,
    )
    tracker.observe(_state(released=True))
    tracker.observe(_state(released=True))
    second = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([False]),
        reason_code=torch.tensor([2]),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        stem_fracture=torch.tensor([False]),
        support_release_step=40,
    )
    assert first[0]["episode_id"] == 0 and second[0]["episode_id"] == 1
    assert first[0]["reason"] == "drop" and second[0]["reason"] == "timeout"
    assert first[0]["steps"] == 1 and second[0]["steps"] == 2
    assert second[0]["truncated"] is True and second[0]["terminated"] is False


def test_faults_and_stem_fracture_are_recorded() -> None:
    tracker = _tracker(1)
    tracker.observe(_state(released=True, faults={"joint_effort": True}))
    tracker.observe(
        _state(released=True, faults={"stem_fracture": True}, height_gain_m=0.05)
    )
    row = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([False]),
        reason_code=torch.tensor([11]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        stem_fracture=torch.tensor([True]),
        support_release_step=40,
    )[0]
    assert row["joint_effort"] is True
    assert row["stem_fracture"] is True
    assert row["stem_fracture_before_release"] is True
    assert row["first_fault_step"] == 0
    assert row["immediate_failure"] is True
    assert row["failure_phase"] == "immediate"
    assert row["height_gain_sufficient"] is True  # 0.05 m 超过声明目标 0.03 m
    assert row["max_height_gain_m"] == pytest.approx(0.05)


def test_late_failure_is_not_marked_immediate() -> None:
    tracker = _tracker(1)
    for _ in range(10):
        tracker.observe(_state(released=True))
    tracker.observe(_state(released=True, faults={"overload": True}))
    row = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([False]),
        reason_code=torch.tensor([4]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        stem_fracture=torch.tensor([False]),
        support_release_step=40,
    )[0]
    assert row["first_fault_step"] == 10
    assert row["immediate_failure"] is False
    assert row["failure_phase"] == "late"


def test_clear_is_per_environment() -> None:
    tracker = _tracker(2)
    tracker.observe(_state(2, released=True))
    tracker.clear(torch.tensor([0]))
    assert tracker.steps.tolist() == [0, 1]
    rows = tracker.finalize(
        torch.tensor([1]),
        success=torch.tensor([False, False]),
        reason_code=torch.tensor([3, 3]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        stem_fracture=torch.tensor([False, False]),
        support_release_step=40,
    )
    assert len(rows) == 1 and rows[0]["env_id"] == 1


def test_rows_satisfy_main_thread_ledger_contract() -> None:
    """主线程 run_utils 账本必须能直接消费这些行（含 append-only 语义）。"""

    from research.fixed_fruit_grasp_v1 import run_utils as RU

    tracker = _tracker(1)
    tracker.observe(_state(released=True))
    row = tracker.finalize(
        torch.tensor([0]),
        success=torch.tensor([False]),
        reason_code=torch.tensor([3]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        stem_fracture=torch.tensor([False]),
        support_release_step=40,
    )[0]
    normalized = RU.normalize_episode_record(row)
    assert "missing_reason" not in normalized
    accumulator = RU.EpisodeAccumulator()
    accumulator.add(row)
    summary = accumulator.summary()
    assert summary["episodes_completed"] == 1
    assert summary["success_count"] == 0
    assert summary["failure_reason_counts"] == {"drop": 1}

    class _Env:
        completed_episodes = [row]

    rows, cursor = RU.completed_episode_delta(_Env(), 0)
    assert len(rows) == 1 and cursor == 1
