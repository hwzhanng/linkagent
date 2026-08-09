"""Pure, testable physics primitives for the independent Physics-v1 task.

The numerical ranges are literature and vendor priors, not claims about the
user's exact hand, arm, cultivar, or fingertip material.  Formal PPO remains
blocked until the named calibration gates replace those priors with measured
values.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from simulation.tomato_pick.config import load_object


PHYSICS_V1_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "config/tomato_pick_physics_v1.yaml"
)


def secure_support_credit_gate(
    score: torch.Tensor, secure_support: torch.Tensor
) -> torch.Tensor:
    """Allow downstream task credit only under measured secure support.

    This is deliberately a credit gate, not a contact reward: reach, grasp,
    and opposition keep their continuous gradients, while stem-load and break
    credit cannot make an unsupported fracture locally optimal.
    """

    if score.shape != secure_support.shape:
        raise ValueError("score and secure_support must have identical shapes")
    if secure_support.dtype is not torch.bool:
        raise ValueError("secure_support must be boolean")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("score must be finite")
    return score * secure_support.to(dtype=score.dtype)


def secure_kinematic_break_credit_gate(
    score: torch.Tensor,
    secure_support: torch.Tensor,
    kinematic_retained: torch.Tensor,
) -> torch.Tensor:
    """Credit a break only when support and retained motion agree.

    Physics-v1 success requires both load-bearing thumb opposition and a fruit
    that is already inside the retained distance/speed envelope. Requiring
    their intersection at the exact 120 Hz fracture substep prevents a secure
    but high-speed pull from receiving the one-shot break reward.
    """

    if secure_support.shape != kinematic_retained.shape:
        raise ValueError(
            "secure_support and kinematic_retained must have identical shapes"
        )
    if kinematic_retained.dtype is not torch.bool:
        raise ValueError("kinematic_retained must be boolean")
    return secure_support_credit_gate(
        score, secure_support & kinematic_retained
    )


def load_physics_v1_profile(path: Path = PHYSICS_V1_PROFILE_PATH) -> dict[str, Any]:
    profile = load_object(path)
    if profile.get("schema_version") != 1:
        raise ValueError("Physics-v1 profile schema_version must be 1")
    if profile.get("profile_id") != "tomato_pick_physics_v1":
        raise ValueError("unexpected Physics-v1 profile_id")
    hand = profile.get("hand")
    if not isinstance(hand, dict) or hand.get("active_joint_count") != 16:
        raise ValueError("Physics-v1 requires sixteen active L20 joints")
    if hand.get("passive_follower_count") != 5:
        raise ValueError("Physics-v1 requires five passive L20 follower joints")
    velocity = hand.get("joint_max_velocity_deg_s_in_policy_order")
    if not isinstance(velocity, list) or len(velocity) != 16:
        raise ValueError("Physics-v1 requires sixteen joint velocity limits")
    gate = profile.get("training_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("ready"), bool):
        raise ValueError("Physics-v1 training_gate.ready must be boolean")
    blockers = gate.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("Physics-v1 training_gate.blockers must be a list")
    if gate["ready"] and blockers:
        raise ValueError("a ready Physics-v1 profile cannot retain training blockers")
    if gate["ready"] and gate.get("qualification") != "literature_prior_provisional":
        raise ValueError("Physics-v1 ready state must preserve provisional qualification")
    return profile


def sample_correlated_fruit_profiles(
    count: int,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Sample correlated geometry, mass, contact, and abscission priors.

    A small set of latent variables creates physically related values.  In
    particular, mass follows geometry, abscission load follows diameter, and
    surface/compliance values vary smoothly rather than as unrelated uniforms.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    profile = load_physics_v1_profile()
    fruit = profile["fruit_prior"]
    stem = profile["stem_prior"]
    contact = profile["contact"]

    latent = torch.rand((count, 6), generator=generator, dtype=dtype)
    size = latent[:, 0]
    ripeness = latent[:, 1]
    shape = 2.0 * latent[:, 2:4] - 1.0
    surface = 2.0 * latent[:, 4] - 1.0
    biological = 2.0 * latent[:, 5] - 1.0

    radius_lo, radius_hi = fruit["equatorial_radius_m"]
    radius = radius_lo + size * (radius_hi - radius_lo)
    x = radius * (1.0 + 0.035 * shape[:, 0])
    y = radius * (1.0 - 0.035 * shape[:, 0])
    aspect_lo, aspect_hi = fruit["polar_aspect_ratio"]
    aspect = aspect_lo + (0.65 * size + 0.35 * latent[:, 3]) * (aspect_hi - aspect_lo)
    z = radius * aspect
    axes = torch.stack((x, y, z), dim=1)

    density_lo, density_hi = fruit["density_kg_m3"]
    density_center = 0.5 * (density_lo + density_hi)
    density = (
        density_center
        + 0.22 * (density_hi - density_lo) * (0.5 - ripeness)
        + 0.18 * (density_hi - density_lo) * biological
    ).clamp(density_lo, density_hi)
    mass = 4.0 / 3.0 * math.pi * axes.prod(dim=1) * density

    static_friction = float(contact["tomato_static_friction"]) * (
        1.0 + 0.12 * surface
    )

    diameter_lo, diameter_hi = stem["abscission_diameter_m"]
    stem_diameter = diameter_lo + (0.70 * size + 0.30 * latent[:, 5]) * (
        diameter_hi - diameter_lo
    )
    strength_lo, strength_hi = stem["tensile_strength_pa"]
    tensile_strength = (
        0.5 * (strength_lo + strength_hi)
        * (1.0 - 0.08 * ripeness + 0.10 * biological)
    ).clamp(strength_lo, strength_hi)
    intact_tensile_force = (
        tensile_strength * math.pi * torch.square(stem_diameter) / 4.0
    )
    detach_lo, detach_hi = stem["abscission_detachment_force_n"]
    detach_difficulty = (
        0.45 * size
        + 0.25 * (1.0 - ripeness)
        + 0.30 * (0.5 * (biological + 1.0))
    ).clamp(0.0, 1.0)
    tensile_force = detach_lo + detach_difficulty * (detach_hi - detach_lo)
    safe_capacity = conservative_tangential_capacity(
        static_friction,
        maximum_fingertip_force_n=float(
            stem["safe_wrench_maximum_fingertip_force_n"]
        ),
        supported_fingertips=int(stem["safe_wrench_supported_fingertips"]),
        utilization_fraction=float(stem["safe_wrench_utilization_fraction"]),
    )
    if bool((tensile_force > safe_capacity).any()):
        raise RuntimeError(
            "abscission detachment force exceeds conservative hand wrench capacity"
        )
    shear_lo, shear_hi = stem["shear_to_tension_ratio"]
    shear_ratio = shear_lo + latent[:, 2] * (shear_hi - shear_lo)
    shear_force = tensile_force * shear_ratio
    lever_lo, lever_hi = stem["bending_lever_arm_m"]
    lever = lever_lo + latent[:, 3] * (lever_hi - lever_lo)
    bending_moment = tensile_force * lever

    duration_lo, duration_hi = stem["critical_duration_s"]
    critical_duration = duration_lo + ripeness * (duration_hi - duration_lo)
    stiffness = float(contact["compliant_contact_stiffness_n_m"]) * (
        1.0 + 0.18 * biological - 0.12 * ripeness
    )
    damping = float(contact["compliant_contact_damping_n_s_m"]) * (
        1.0 + 0.15 * biological
    )
    return {
        "axes_m": axes,
        "density_kg_m3": density,
        "mass_kg": mass,
        "ripeness": ripeness,
        "surface_condition": surface,
        "contact_stiffness_n_m": stiffness,
        "contact_damping_n_s_m": damping,
        "static_friction": static_friction,
        "stem_diameter_m": stem_diameter,
        "stem_tensile_force_n": tensile_force,
        "stem_intact_tensile_force_n": intact_tensile_force,
        "stem_safe_tangential_capacity_n": safe_capacity,
        "stem_shear_force_n": shear_force,
        "stem_bending_moment_nm": bending_moment,
        "stem_critical_duration_s": critical_duration,
    }


def conservative_tangential_capacity(
    static_friction: torch.Tensor,
    *,
    maximum_fingertip_force_n: float,
    supported_fingertips: int,
    utilization_fraction: float,
) -> torch.Tensor:
    """Return a conservative friction-limited hand tangential capacity.

    The utilization factor reserves margin for contact direction, unequal load
    sharing, dynamics, and geometry.  It is a feasibility gate, not a claim
    that pad friction and stem strength are physically correlated.
    """

    if maximum_fingertip_force_n <= 0.0:
        raise ValueError("maximum fingertip force must be positive")
    if isinstance(supported_fingertips, bool) or supported_fingertips < 1:
        raise ValueError("supported fingertips must be a positive integer")
    if not 0.0 < utilization_fraction <= 1.0:
        raise ValueError("utilization fraction must lie in (0, 1]")
    if bool((static_friction <= 0.0).any()):
        raise ValueError("static friction must be positive")
    return (
        static_friction
        * float(maximum_fingertip_force_n)
        * int(supported_fingertips)
        * float(utilization_fraction)
    )


def combined_stem_load_ratio(
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    tensile_limit_n: torch.Tensor,
    shear_limit_n: torch.Tensor,
    bending_limit_nm: torch.Tensor,
) -> torch.Tensor:
    """Return a dimensionless axial/shear/bending interaction ratio."""

    if force_on_fruit_w.shape != torque_on_fruit_w.shape:
        raise ValueError("force and torque tensors must have the same shape")
    if force_on_fruit_w.ndim != 2 or force_on_fruit_w.shape[1] != 3:
        raise ValueError("force and torque tensors must have shape (N, 3)")
    axis = torch.nn.functional.normalize(pull_axis_w, dim=1)
    signed_axial = (force_on_fruit_w * axis).sum(dim=1)
    tension = (-signed_axial).clamp_min(0.0)
    shear_vector = force_on_fruit_w - signed_axial.unsqueeze(1) * axis
    shear = torch.linalg.vector_norm(shear_vector, dim=1)
    torsional = (torque_on_fruit_w * axis).sum(dim=1)
    bending_vector = torque_on_fruit_w - torsional.unsqueeze(1) * axis
    bending = torch.linalg.vector_norm(bending_vector, dim=1)
    return torch.sqrt(
        torch.square(tension / tensile_limit_n.clamp_min(1.0e-8))
        + torch.square(shear / shear_limit_n.clamp_min(1.0e-8))
        + torch.square(bending / bending_limit_nm.clamp_min(1.0e-8))
    )


def stem_load_damage_progress_score(
    combined_load_ratio: torch.Tensor,
    accumulated_damage: torch.Tensor,
    opposed_grasp_score: torch.Tensor,
    *,
    instantaneous_load_fraction: float = 0.75,
) -> torch.Tensor:
    """Return the bounded Physics-v1 stem progress used by PPO.

    The physical stem breaks from combined tension, shear, and bending after
    load-duration damage reaches one.  A pure instantaneous-load reward is
    therefore incomplete: it can be exhausted by a brief impulse that cannot
    break the stem.  This score preserves a strong dense load gradient for
    early exploration while reserving part of the same unit reward budget for
    sustained damage.  Opposition gates both parts so pulling without a
    load-bearing grasp cannot earn stem progress.
    """

    if not 0.0 < instantaneous_load_fraction < 1.0:
        raise ValueError("instantaneous_load_fraction must lie strictly in (0, 1)")
    if (
        combined_load_ratio.shape != accumulated_damage.shape
        or combined_load_ratio.shape != opposed_grasp_score.shape
    ):
        raise ValueError("load, damage, and opposition tensors must have matching shapes")
    load = combined_load_ratio.clamp(0.0, 1.0)
    damage = accumulated_damage.clamp(0.0, 1.0)
    opposed = opposed_grasp_score.clamp(0.0, 1.0)
    sustained_fraction = 1.0 - instantaneous_load_fraction
    return opposed * (
        instantaneous_load_fraction * load + sustained_fraction * damage
    )


def quantize_contact_profiles(
    profiles: dict[str, torch.Tensor],
    *,
    friction_bins: int,
    compliance_bins: int,
) -> dict[str, torch.Tensor]:
    """Quantize correlated contact samples into a bounded material bank.

    Authoring a unique USD material for every one of thousands of RL
    environments is wasteful.  A small two-dimensional bank preserves the
    sampled friction/compliance ordering while keeping stage size bounded.
    """

    if friction_bins < 2 or compliance_bins < 2:
        raise ValueError("contact material bin counts must each be at least two")
    profile = load_physics_v1_profile()
    contact = profile["contact"]
    static = profiles["static_friction"]
    stiffness = profiles["contact_stiffness_n_m"]
    damping = profiles["contact_damping_n_s_m"]
    if (
        static.ndim != 1
        or stiffness.shape != static.shape
        or damping.shape != static.shape
    ):
        raise ValueError("contact profile tensors must be one-dimensional and aligned")

    nominal_static = float(contact["tomato_static_friction"])
    nominal_dynamic = float(contact["tomato_dynamic_friction"])
    nominal_stiffness = float(contact["compliant_contact_stiffness_n_m"])
    nominal_damping = float(contact["compliant_contact_damping_n_s_m"])
    static_low, static_high = 0.88 * nominal_static, 1.12 * nominal_static
    stiffness_low, stiffness_high = 0.70 * nominal_stiffness, 1.18 * nominal_stiffness
    damping_low, damping_high = 0.85 * nominal_damping, 1.15 * nominal_damping

    friction_index = torch.floor(
        (static - static_low) / (static_high - static_low) * friction_bins
    ).long().clamp(0, friction_bins - 1)
    compliance_index = torch.floor(
        (stiffness - stiffness_low)
        / (stiffness_high - stiffness_low)
        * compliance_bins
    ).long().clamp(0, compliance_bins - 1)
    friction_fraction = (friction_index.to(static.dtype) + 0.5) / friction_bins
    compliance_fraction = (
        compliance_index.to(stiffness.dtype) + 0.5
    ) / compliance_bins
    material_static = static_low + friction_fraction * (static_high - static_low)
    material_dynamic = material_static * (nominal_dynamic / nominal_static)
    material_stiffness = stiffness_low + compliance_fraction * (
        stiffness_high - stiffness_low
    )
    material_damping = damping_low + compliance_fraction * (
        damping_high - damping_low
    )
    return {
        "friction_bin": friction_index,
        "compliance_bin": compliance_index,
        "material_bin": friction_index * compliance_bins + compliance_index,
        "material_static_friction": material_static,
        "material_dynamic_friction": material_dynamic,
        "material_stiffness_n_m": material_stiffness,
        "material_damping_n_s_m": material_damping,
    }


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

    At unit load ratio, damage reaches one after ``critical_duration_s``.
    A single 120 Hz spike therefore cannot detach the fruit.  Subcritical load
    below ``onset_ratio`` heals gradually instead of resetting discontinuously.
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


__all__ = [
    "PHYSICS_V1_PROFILE_PATH",
    "combined_stem_load_ratio",
    "conservative_tangential_capacity",
    "load_physics_v1_profile",
    "quantize_contact_profiles",
    "sample_correlated_fruit_profiles",
    "secure_kinematic_break_credit_gate",
    "secure_support_credit_gate",
    "stem_load_damage_progress_score",
    "update_stem_damage",
]
