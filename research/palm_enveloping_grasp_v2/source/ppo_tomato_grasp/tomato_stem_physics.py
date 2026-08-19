"""Pure-torch stem physics, load decomposition, damage, and success verifier.

This module is the CPU-testable physics core of the palm-up tomato grasp task.
The stem is a virtual spring-damper between the tomato attachment point and a
fixed plant anchor; it is not a PhysX joint.  The model is migrated from the
team Clean-v5 tomato-pick environment (per HAND_ONLY_TOMATO_PPO_PLAN.md §3):

- ``stem_attachment_wrench`` adapts the team's attachment dynamics with one
  deliberate change: no gravity-compensation term.  The tomato rests on the
  palm, so the palm carries the weight and the stem only provides the
  attachment constraint and the pick resistance.  A hanging-fruit model would
  suspend the tomato above the palm and suppress the palm-support signal.
- ``stem_load_components`` and ``update_stem_damage`` follow the team's
  four-mode (tension/shear/bending/torsion) damage-integration break model.
- ``evaluate_success_conditions`` implements the independent eight-condition
  success Verifier from the plan §6.

No Isaac or hardware imports; runnable on CPU under any torch build.
"""

from __future__ import annotations

import torch


def quat_conjugate_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Conjugate of a (N, 4) wxyz quaternion."""
    result = quat_wxyz.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (N, 4) wxyz quaternions."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_to_rot_matrix_6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix as a (N, 6) vector.

    The block is the actor's ``fruit_orientation_6d`` observation.  For an
    ellipsoid the first two columns span the local x-y plane, which is enough
    to reconstruct the full rotation.
    """

    if quat_wxyz.ndim != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError("quat_wxyz must have shape (N, 4)")
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return torch.stack((r00, r01, r10, r11, r20, r21), dim=-1)


def quat_box_minus_wxyz(quat_to: torch.Tensor, quat_from: torch.Tensor) -> torch.Tensor:
    """Rotation vector from ``quat_from`` to ``quat_to`` (N, 3) axis-angle."""

    relative = quat_mul_wxyz(quat_to, quat_conjugate_wxyz(quat_from))
    angle = 2.0 * torch.atan2(
        torch.linalg.vector_norm(relative[..., 1:], dim=-1),
        relative[..., 0].abs().clamp_min(1.0e-12),
    )
    axis = torch.nn.functional.normalize(relative[..., 1:], dim=-1)
    return axis * angle.unsqueeze(-1)


def quat_rotate_vector_wxyz(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate (N, 3) vectors by (N, 4) wxyz quaternions."""

    qw = quat_wxyz[..., 0].unsqueeze(-1)
    qv = quat_wxyz[..., 1:]
    cross = torch.linalg.cross(qv, vectors, dim=-1)
    return vectors + 2.0 * (qw * cross + torch.linalg.cross(qv, cross, dim=-1))


def stem_attachment_wrench(
    *,
    mass_kg: torch.Tensor,
    inertia_kg_m2: torch.Tensor,
    axes_m: torch.Tensor,
    tomato_pos_w: torch.Tensor,
    tomato_quat_wxyz: torch.Tensor,
    tomato_lin_vel_w: torch.Tensor,
    tomato_ang_vel_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    anchor_quat_wxyz: torch.Tensor,
    omega_linear_rad_s: float,
    zeta_linear: float,
    omega_angular_rad_s: float,
    zeta_angular: float,
    maximum_force_n: float,
    maximum_torque_nm: float,
    gravity_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the stem wrench applied to the tomato and its attachment frame.

    Returns ``(force_w, torque_w, attachment_pos_w, pull_axis_w)`` where the
    force/torque are what the stem applies to the tomato (N, 3) each.
    ``pull_axis_w`` points from the anchor toward the tomato and is the
    reference used by ``stem_load_components``.

    The stem is a critically-damped spring in translation and rotation.  By
    default it carries no resting gravity load (the palm-supported branch),
    while callers can pass the world gravity vector to reproduce the team's
    hanging-fruit baseline where the attached stem carries the fruit weight.
    """

    offset_local = torch.zeros_like(axes_m)
    offset_local[..., 2] = axes_m[..., 2]
    attachment_offset_w = quat_rotate_vector_wxyz(tomato_quat_wxyz, offset_local)
    attachment_pos_w = tomato_pos_w + attachment_offset_w
    position_error = anchor_pos_w - attachment_pos_w
    attachment_vel_w = tomato_lin_vel_w + torch.linalg.cross(
        tomato_ang_vel_w, attachment_offset_w, dim=-1
    )
    if gravity_w is None:
        gravity_w = torch.zeros_like(tomato_pos_w)
    force = mass_kg.unsqueeze(-1) * (
        omega_linear_rad_s * omega_linear_rad_s * position_error
        - 2.0 * zeta_linear * omega_linear_rad_s * attachment_vel_w
        - gravity_w
    )
    force = force.clamp(-maximum_force_n, maximum_force_n)

    rotation_error = quat_box_minus_wxyz(anchor_quat_wxyz, tomato_quat_wxyz)
    torque = inertia_kg_m2.unsqueeze(-1) * (
        omega_angular_rad_s * omega_angular_rad_s * rotation_error
        - 2.0 * zeta_angular * omega_angular_rad_s * tomato_ang_vel_w
    )
    torque = torque.clamp(-maximum_torque_nm, maximum_torque_nm)

    pull_axis = torch.nn.functional.normalize(
        attachment_pos_w - anchor_pos_w, dim=-1
    )
    return force, torque, attachment_pos_w, pull_axis


def stem_load_components(
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    tensile_limit_n: torch.Tensor,
    shear_limit_n: torch.Tensor,
    bending_limit_nm: torch.Tensor,
    torsional_limit_nm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return normalized tension, shear, bending, torsion, and interaction load.

    Migrated from the team Clean-v5 ``stem_load_components_v2``.  The pull
    axis points from the plant anchor toward the fruit, so a force holding the
    fruit against the anchor reads as positive tension.
    """

    if force_on_fruit_w.shape != torque_on_fruit_w.shape:
        raise ValueError("force and torque tensors must have the same shape")
    if force_on_fruit_w.ndim != 2 or force_on_fruit_w.shape[1] != 3:
        raise ValueError("force and torque tensors must have shape (N, 3)")
    if pull_axis_w.shape != force_on_fruit_w.shape:
        raise ValueError("pull_axis_w must have shape (N, 3)")
    axis = torch.nn.functional.normalize(pull_axis_w, dim=-1)
    signed_axial = (force_on_fruit_w * axis).sum(dim=-1)
    tension = (-signed_axial).clamp_min(0.0)
    shear_vector = force_on_fruit_w - signed_axial.unsqueeze(-1) * axis
    shear = torch.linalg.vector_norm(shear_vector, dim=-1)
    torsion = torch.abs((torque_on_fruit_w * axis).sum(dim=-1))
    bending_vector = torque_on_fruit_w - (
        torque_on_fruit_w * axis
    ).sum(dim=-1, keepdim=True) * axis
    bending = torch.linalg.vector_norm(bending_vector, dim=-1)
    normalized = {
        "tension": tension / tensile_limit_n.clamp_min(1.0e-8),
        "shear": shear / shear_limit_n.clamp_min(1.0e-8),
        "bending": bending / bending_limit_nm.clamp_min(1.0e-8),
        "torsion": torsion / torsional_limit_nm.clamp_min(1.0e-8),
    }
    normalized["combined"] = torch.sqrt(
        sum(torch.square(value) for value in normalized.values())
    )
    return normalized


def update_stem_damage(
    previous_damage: torch.Tensor,
    load_ratio: torch.Tensor,
    *,
    dt_s: float,
    critical_duration_s: torch.Tensor,
    onset_ratio: float,
    recovery_time_s: float,
    attached: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate load-duration damage and return ``(damage, broken)``.

    Migrated from the team Clean-v5 ``update_stem_damage``.  At unit load
    ratio, damage reaches one after ``critical_duration_s``, so a single
    solver spike cannot detach the fruit.  Subcritical load below
    ``onset_ratio`` heals gradually instead of resetting discontinuously.
    """

    if dt_s <= 0.0 or recovery_time_s <= 0.0:
        raise ValueError("time constants must be positive")
    if not 0.0 <= onset_ratio < 1.0:
        raise ValueError("onset_ratio must be in [0, 1)")
    drive = ((load_ratio - onset_ratio) / (1.0 - onset_ratio)).clamp_min(0.0)
    accumulated = previous_damage + dt_s * drive / critical_duration_s.clamp_min(dt_s)
    recovering = (previous_damage - dt_s / recovery_time_s).clamp_min(0.0)
    damage = torch.where(load_ratio >= onset_ratio, accumulated, recovering)
    damage = torch.where(attached, damage, torch.zeros_like(damage))
    broken = attached & (damage >= 1.0)
    return damage.clamp(0.0, 2.0), broken


def evaluate_success_conditions(
    *,
    secure_before_pick: torch.Tensor,
    break_opposition_ok: torch.Tensor,
    stem_broken: torch.Tensor,
    hold_streak: torch.Tensor,
    center_distance: torch.Tensor,
    relative_speed: torch.Tensor,
    dropped: torch.Tensor,
    force_safe: torch.Tensor,
    damage_ok: torch.Tensor,
    maximum_center_distance_m: float,
    maximum_relative_speed_m_s: float,
    success_hold_steps: int,
) -> dict[str, torch.Tensor]:
    """Independent eight-condition success Verifier (plan §6).

    Every condition is a (N,) boolean tensor.  The environment latches its
    per-episode state before calling; this function is stateless.  The first
    two conditions record *break-time* facts (stable contact before the pick
    started, and thumb opposition plus two other digits at the break substep);
    the rest evaluate the current retention state.
    """

    distance_ok = center_distance <= maximum_center_distance_m
    speed_ok = relative_speed <= maximum_relative_speed_m_s
    conditions = {
        "stable_contact_before_pick": secure_before_pick,
        "opposition_support_at_break": break_opposition_ok,
        "stem_broke_by_load": stem_broken,
        "post_break_hold_20_steps": stem_broken & (hold_streak >= success_hold_steps),
        "retained_distance": stem_broken & distance_ok,
        "no_drop_no_slip": stem_broken & ~dropped & speed_ok,
        "force_within_safe_ceiling": force_safe,
        "damage_below_threshold": damage_ok,
    }
    conditions["success"] = torch.stack(
        tuple(conditions.values()), dim=-1
    ).all(dim=-1)
    return conditions


def evaluate_stage1_hold_conditions(
    *,
    enclosure_contact: torch.Tensor,
    palm_support: torch.Tensor,
    hold_streak: torch.Tensor,
    center_distance: torch.Tensor,
    relative_speed: torch.Tensor,
    dropped: torch.Tensor,
    force_safe: torch.Tensor,
    damage_ok: torch.Tensor,
    maximum_center_distance_m: float,
    maximum_relative_speed_m_s: float,
    success_hold_steps: int,
) -> dict[str, torch.Tensor]:
    """Verify the stage-1 palm-supported five-finger hold.

    Stage 1 deliberately has no stem-break requirement.  The fruit is already
    placed over the upward-facing palm, so this gate isolates the skill PPO is
    meant to learn first: all five digit channels contact the tomato while the
    palm carries support, the object stays near the palm center, and pressure
    remains safe for a complete hold streak.  Keeping this verifier separate
    from the stage-2 pick verifier prevents a policy from receiving zero
    success signal merely because stem physics has not yet been enabled.
    """

    if enclosure_contact.ndim != 2 or enclosure_contact.shape[-1] != 5:
        raise ValueError("enclosure_contact must have shape (N, 5)")
    all_digits = enclosure_contact.bool().all(dim=-1)
    distance_ok = center_distance <= maximum_center_distance_m
    speed_ok = relative_speed <= maximum_relative_speed_m_s
    conditions = {
        "five_digit_enclosure": all_digits,
        "palm_support": palm_support.bool(),
        "hold_streak": hold_streak >= success_hold_steps,
        "retained_distance": distance_ok,
        "no_drop_no_slip": (~dropped) & speed_ok,
        "force_within_safe_ceiling": force_safe.bool(),
        "damage_below_threshold": damage_ok.bool(),
    }
    conditions["success"] = torch.stack(tuple(conditions.values()), dim=-1).all(dim=-1)
    return conditions


__all__ = [
    "evaluate_success_conditions",
    "evaluate_stage1_hold_conditions",
    "quat_box_minus_wxyz",
    "quat_conjugate_wxyz",
    "quat_mul_wxyz",
    "quat_rotate_vector_wxyz",
    "quat_to_rot_matrix_6d",
    "stem_attachment_wrench",
    "stem_load_components",
    "update_stem_damage",
]
