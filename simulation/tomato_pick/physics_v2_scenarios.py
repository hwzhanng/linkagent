"""Wrench-level scripted scenarios for Physics-v2 pre-simulator gates.

The oracle exercises the exact stem damage, support-credit, break-credit, and
curriculum reward primitives.  It proves that a coherent signal path exists;
it does not prove that the robot can realize the required pose or wrench in
Isaac/PhysX.
"""

from __future__ import annotations

import torch

from simulation.tomato_pick.physics_v1 import (
    load_physics_v1_profile,
    secure_kinematic_break_credit_gate,
    secure_support_credit_gate,
    stem_load_damage_progress_score,
    update_stem_damage,
)
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    initial_fruit_damage_state,
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
    stem_load_components_v2,
    update_fruit_damage_diagnostics,
)
from simulation.tomato_pick.reward import (
    break_grasp_quality,
    grasp_contact_score,
    opposed_contact_score,
    tomato_pick_reward,
)


def physical_task_predicates(
    physical_finger_force_n: torch.Tensor,
    *,
    stem_detached: torch.Tensor,
    center_distance_m: torch.Tensor,
    relative_speed_m_s: torch.Tensor,
    hold_steps: torch.Tensor,
    maximum_fingertip_force_n: float = 8.0,
    physical_palm_force_n: torch.Tensor | None = None,
    minimum_palm_support_force_n: float = 0.25,
    palm_support_requires_detach: bool = True,
) -> dict[str, torch.Tensor]:
    """Evaluate a conservative support/retention oracle from physical state.

    Unlike measured tactile values, ``physical_finger_force_n`` is the actual
    contact force.  The force ceiling mirrors the runtime excess-force guard;
    without it, an arbitrarily destructive squeeze could pass this offline
    oracle even though the integrated environment correctly terminates it.
    """

    if physical_finger_force_n.ndim != 2 or physical_finger_force_n.shape[1] != 5:
        raise ValueError("physical_finger_force_n must have shape (N, 5)")
    if maximum_fingertip_force_n <= 0.0:
        raise ValueError("maximum_fingertip_force_n must be positive")
    secure_cfg = load_physics_v1_profile()["secure_success"]
    minimum_force_n = float(secure_cfg["minimum_support_force_n"])
    minimum_fingertips = int(secure_cfg["minimum_supported_fingertips"])
    maximum_distance_m = float(secure_cfg["maximum_center_distance_m"])
    maximum_speed_m_s = float(secure_cfg["maximum_relative_speed_m_s"])
    success_hold_steps = int(secure_cfg["hold_policy_steps"])
    contact = physical_finger_force_n >= minimum_force_n
    fingertip_support = (
        (contact.sum(dim=1) >= minimum_fingertips)
        & contact[:, 0]
        & contact[:, 1:].any(dim=1)
    )
    if physical_palm_force_n is None:
        palm_force = torch.zeros(
            physical_finger_force_n.shape[0],
            dtype=physical_finger_force_n.dtype,
            device=physical_finger_force_n.device,
        )
    else:
        if physical_palm_force_n.shape not in {
            (physical_finger_force_n.shape[0],),
            (physical_finger_force_n.shape[0], 1),
        }:
            raise ValueError("physical_palm_force_n must have shape (N,) or (N, 1)")
        palm_force = physical_palm_force_n.reshape(-1)
    if minimum_palm_support_force_n <= 0.0:
        raise ValueError("minimum_palm_support_force_n must be positive")
    palm_contact = palm_force >= minimum_palm_support_force_n
    palm_support = stem_detached.bool() & palm_contact
    if not palm_support_requires_detach:
        palm_support = palm_contact
    secure_support = fingertip_support | palm_support
    kinematic_retained = (
        center_distance_m <= maximum_distance_m
    ) & (relative_speed_m_s <= maximum_speed_m_s)
    within_force_limit = (
        (physical_finger_force_n.amax(dim=1) <= maximum_fingertip_force_n)
        & (palm_force <= maximum_fingertip_force_n)
    )
    retained = (
        stem_detached.bool()
        & secure_support
        & kinematic_retained
        & within_force_limit
    )
    success = retained & (hold_steps >= success_hold_steps)
    return {
        "secure_support": secure_support,
        "fingertip_support": fingertip_support,
        "palm_support": palm_support,
        "palm_contact": palm_contact,
        "kinematic_retained": kinematic_retained,
        "within_force_limit": within_force_limit,
        "retained": retained,
        "success": success,
    }


def _reward_transition(
    best: dict[str, torch.Tensor],
    *,
    reach: torch.Tensor,
    finger_force_n: torch.Tensor,
    stem_progress: torch.Tensor,
    stable_detach: torch.Tensor,
    retention: torch.Tensor,
    break_credit: torch.Tensor,
    success: torch.Tensor,
    dropped: torch.Tensor | None = None,
    excess_force: torch.Tensor | None = None,
    timed_out: torch.Tensor | None = None,
    stem_detached: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zeros = torch.zeros_like(reach)
    grasp = grasp_contact_score(finger_force_n, full_contact_force_n=4.0)
    opposed = opposed_contact_score(finger_force_n, full_contact_force_n=4.0)
    values = tomato_pick_reward(
        reach_score=reach,
        previous_best_reach=best["reach"],
        grasp_score=grasp,
        previous_best_grasp=best["grasp"],
        opposed_score=opposed,
        previous_best_opposed=best["opposed"],
        stem_load_score=stem_progress,
        previous_best_stem_load=best["stem"],
        stable_detach_score=stable_detach,
        previous_best_stable_detach=best["stable_detach"],
        retention_score=retention,
        previous_best_retention=best["retention"],
        break_grasped_score=break_credit,
        success=success,
        dropped=zeros.bool() if dropped is None else dropped,
        excess_force=zeros.bool() if excess_force is None else excess_force,
        out_of_bounds=zeros.bool(),
        timed_out=zeros.bool() if timed_out is None else timed_out,
        stem_detached=zeros.bool() if stem_detached is None else stem_detached,
        alignment_alpha=1.0,
    )
    reward, next_reach, next_grasp, next_opposed, next_stem, next_detach, next_retention, _ = values
    return reward, {
        "reach": next_reach,
        "grasp": next_grasp,
        "opposed": next_opposed,
        "stem": next_stem,
        "stable_detach": next_detach,
        "retention": next_retention,
    }


def run_wrench_level_pick_scenario(
    *,
    strength_quantile: float,
    seed: int = 42,
    sample_count: int = 8192,
) -> dict[str, float | int | bool]:
    """Run a gentle twist-detach-retain oracle for one stem quantile."""

    if not 0.0 <= strength_quantile <= 1.0 or sample_count < 2:
        raise ValueError("invalid quantile or sample count")
    profile = load_physics_v2_profile()
    fruit = sample_correlated_fruit_contact_profiles(
        sample_count,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    stem = sample_stem_profiles_v2(fruit, mode="learnability")
    ordered = torch.argsort(stem["stem_torsional_moment_nm"])
    rank = round(strength_quantile * (sample_count - 1))
    index = int(ordered[rank])
    limit = {
        key: value[index : index + 1]
        for key, value in stem.items()
    }

    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    force_w = torch.zeros((1, 3), dtype=torch.float64)
    torque_w = axis * limit["stem_torsional_moment_nm"].unsqueeze(1) * 1.05
    load = stem_load_components_v2(
        force_w,
        torque_w,
        axis,
        limit["stem_tensile_force_n"],
        limit["stem_shear_force_n"],
        limit["stem_bending_moment_nm"],
        limit["stem_torsional_moment_nm"],
    )["combined"]

    finger_force = torch.tensor([[4.0, 4.0, 4.0, 0.0, 0.0]], dtype=torch.float64)
    predicates = physical_task_predicates(
        finger_force,
        stem_detached=torch.zeros(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03], dtype=torch.float64),
        relative_speed_m_s=torch.tensor([0.02], dtype=torch.float64),
        hold_steps=torch.zeros(1, dtype=torch.long),
    )
    opposed = opposed_contact_score(finger_force, full_contact_force_n=4.0)
    damage = torch.zeros(1, dtype=torch.float64)
    break_now = torch.zeros(1, dtype=torch.bool)
    substeps = 0
    one_substep_broken = False
    common = profile["stem_modes_v2"]["common"]
    while not bool(break_now.item()) and substeps < 128:
        damage, break_now = update_stem_damage(
            damage,
            load,
            dt_s=1.0 / 120.0,
            critical_duration_s=limit["stem_critical_duration_s"],
            onset_ratio=float(common["damage_onset_ratio"]),
            recovery_time_s=float(common["damage_recovery_time_s"]),
            attached=~break_now,
        )
        substeps += 1
        if substeps == 1:
            one_substep_broken = bool(break_now.item())

    kinematic = torch.ones(1, dtype=torch.bool)
    quality = break_grasp_quality(finger_force, contact_force_n=0.25)
    break_credit = secure_kinematic_break_credit_gate(
        break_now.float() * quality,
        predicates["secure_support"],
        kinematic,
    )
    stem_progress = secure_support_credit_gate(
        stem_load_damage_progress_score(load, damage, opposed),
        predicates["secure_support"],
    )

    diagnostic = initial_fruit_damage_state(torch.zeros((1, 5), dtype=torch.float64))
    for ramp_step in range(1, 21):
        ramp_force = torch.full((1, 5), 0.2 * ramp_step, dtype=torch.float64)
        ramp_force[:, 3:] = 0.0
        diagnostic = update_fruit_damage_diagnostics(
            diagnostic,
            ramp_force,
            ramp_force / float(fruit["contact_stiffness_n_m"][index]),
            dt_s=1.0 / 120.0,
        )

    best = {
        name: torch.zeros(1, dtype=torch.float64)
        for name in ("reach", "grasp", "opposed", "stem", "stable_detach", "retention")
    }
    total_reward = torch.zeros(1, dtype=torch.float64)
    zero_force = torch.zeros_like(finger_force)
    reward, best = _reward_transition(
        best,
        reach=torch.ones(1, dtype=torch.float64),
        finger_force_n=zero_force,
        stem_progress=torch.zeros(1, dtype=torch.float64),
        stable_detach=torch.zeros(1, dtype=torch.float64),
        retention=torch.zeros(1, dtype=torch.float64),
        break_credit=torch.zeros(1, dtype=torch.float64),
        success=torch.zeros(1, dtype=torch.bool),
    )
    total_reward += reward
    reward, best = _reward_transition(
        best,
        reach=torch.ones(1, dtype=torch.float64),
        finger_force_n=finger_force,
        stem_progress=stem_progress,
        stable_detach=break_now.float(),
        retention=torch.zeros(1, dtype=torch.float64),
        break_credit=break_credit,
        success=torch.zeros(1, dtype=torch.bool),
        stem_detached=break_now,
    )
    total_reward += reward
    final_predicates: dict[str, torch.Tensor] | None = None
    for hold_step in range(1, 21):
        final_predicates = physical_task_predicates(
            finger_force,
            stem_detached=break_now,
            center_distance_m=torch.tensor([0.03], dtype=torch.float64),
            relative_speed_m_s=torch.tensor([0.02], dtype=torch.float64),
            hold_steps=torch.tensor([hold_step]),
        )
        reward, best = _reward_transition(
            best,
            reach=torch.ones(1, dtype=torch.float64),
            finger_force_n=finger_force,
            stem_progress=stem_progress,
            stable_detach=torch.ones(1, dtype=torch.float64),
            retention=torch.tensor([hold_step / 20.0], dtype=torch.float64),
            break_credit=torch.zeros(1, dtype=torch.float64),
            success=final_predicates["success"],
            stem_detached=break_now,
        )
        total_reward += reward
    assert final_predicates is not None
    return {
        "stem_torsional_limit_nm": float(limit["stem_torsional_moment_nm"]),
        "stem_combined_load_ratio": float(load),
        "break_substeps_120hz": substeps,
        "one_substep_broken": one_substep_broken,
        "detached": bool(break_now.item()),
        "secure_support": bool(predicates["secure_support"].item()),
        "break_credit": float(break_credit),
        "retained": bool(final_predicates["retained"].item()),
        "success": bool(final_predicates["success"].item()),
        "fruit_damage_score": float(diagnostic["damage_score"].max()),
        "aligned_return": float(total_reward),
    }


__all__ = ["physical_task_predicates", "run_wrench_level_pick_scenario"]
