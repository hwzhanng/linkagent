"""Tensor physics priors and diagnostic models for offline Physics-v2.

All values remain literature or numerical priors.  The module keeps the first
offline iteration deliberately simple: a rigid ellipsoid, linear compliant
contact, and a bounded damage-exposure indicator rather than a deformable fruit.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile


def fixed_fruit_contact_cases(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> dict[str, dict[str, torch.Tensor]]:
    """Return deterministic soft/nominal/firm order-check cases."""

    cases = load_physics_v2_profile()["fruit_contact_v2"]["fixed_cases"]
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in cases.items():
        axes = torch.tensor(values["axes_m"], device=device, dtype=dtype)
        density = torch.tensor(values["density_kg_m3"], device=device, dtype=dtype)
        result[name] = {
            "axes_m": axes,
            "density_kg_m3": density,
            "mass_kg": 4.0 / 3.0 * math.pi * axes.prod() * density,
            **{
                key: torch.tensor(values[key], device=device, dtype=dtype)
                for key in (
                    "static_friction",
                    "dynamic_friction",
                    "effective_stiffness_n_m",
                    "effective_damping_n_s_m",
                )
            },
        }
    return result


def sample_correlated_fruit_contact_profiles(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Sample correlated geometry, mass, ripeness, wetness, and contact priors."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    cfg = load_physics_v2_profile()["fruit_contact_v2"]
    latent = torch.rand((count, 7), generator=generator, device=device, dtype=dtype)
    size = latent[:, 0]
    ripeness = latent[:, 1]
    shape = 2.0 * latent[:, 2:4] - 1.0
    wetness = latent[:, 4]
    surface_texture = 2.0 * latent[:, 5] - 1.0
    biological = 2.0 * latent[:, 6] - 1.0

    radius_lo, radius_hi = cfg["equatorial_radius_m"]
    radius = radius_lo + size * (radius_hi - radius_lo)
    x = radius * (1.0 + 0.035 * shape[:, 0])
    y = radius * (1.0 - 0.035 * shape[:, 0])
    aspect_lo, aspect_hi = cfg["polar_aspect_ratio"]
    aspect_latent = (0.55 * size + 0.30 * latent[:, 3] + 0.15 * (1.0 - ripeness))
    aspect = aspect_lo + aspect_latent * (aspect_hi - aspect_lo)
    z = radius * aspect
    axes = torch.stack((x, y, z), dim=1)

    density_lo, density_hi = cfg["density_kg_m3"]
    density_center = 0.5 * (density_lo + density_hi)
    density = (
        density_center
        + 0.22 * (density_hi - density_lo) * (0.5 - ripeness)
        + 0.16 * (density_hi - density_lo) * biological
    ).clamp(density_lo, density_hi)
    mass = 4.0 / 3.0 * math.pi * axes.prod(dim=1) * density

    stiffness_lo, stiffness_hi = cfg["effective_stiffness_n_m"]
    firmness = (0.62 * (1.0 - ripeness) + 0.23 * (0.5 * (biological + 1.0)) + 0.15 * size)
    stiffness = (stiffness_lo + firmness * (stiffness_hi - stiffness_lo)).clamp(
        stiffness_lo, stiffness_hi
    )
    damping_lo, damping_hi = cfg["effective_damping_n_s_m"]
    damping_fraction = (0.72 * firmness + 0.28 * size).clamp(0.0, 1.0)
    damping = damping_lo + damping_fraction * (damping_hi - damping_lo)

    friction_lo, friction_hi = cfg["static_friction"]
    friction_fraction = (
        0.56 + 0.18 * surface_texture - 0.40 * wetness + 0.08 * (1.0 - ripeness)
    ).clamp(0.0, 1.0)
    static_friction = friction_lo + friction_fraction * (friction_hi - friction_lo)
    ratio_lo, ratio_hi = cfg["dynamic_to_static_ratio"]
    dynamic_ratio = ratio_lo + latent[:, 2] * (ratio_hi - ratio_lo)
    dynamic_friction = static_friction * dynamic_ratio
    return {
        "axes_m": axes,
        "density_kg_m3": density,
        "mass_kg": mass,
        "size_latent": size,
        "ripeness": ripeness,
        "wetness": wetness,
        "surface_texture": surface_texture,
        "biological_latent": biological,
        "contact_stiffness_n_m": stiffness,
        "contact_damping_n_s_m": damping,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
    }


def quantize_contact_material_bank(
    profiles: dict[str, torch.Tensor], *, friction_bins: int, compliance_bins: int
) -> dict[str, torch.Tensor]:
    """Quantize v2 contact samples while retaining sampled ordering."""

    if friction_bins < 2 or compliance_bins < 2:
        raise ValueError("material-bank dimensions must be >= 2")
    cfg = load_physics_v2_profile()["fruit_contact_v2"]
    static = profiles["static_friction"]
    dynamic = profiles["dynamic_friction"]
    stiffness = profiles["contact_stiffness_n_m"]
    damping = profiles["contact_damping_n_s_m"]
    if static.ndim != 1 or any(item.shape != static.shape for item in (dynamic, stiffness, damping)):
        raise ValueError("contact tensors must be aligned one-dimensional arrays")

    friction_lo, friction_hi = cfg["static_friction"]
    stiffness_lo, stiffness_hi = cfg["effective_stiffness_n_m"]
    friction_index = torch.floor(
        (static - friction_lo) / (friction_hi - friction_lo) * friction_bins
    ).long().clamp(0, friction_bins - 1)
    compliance_index = torch.floor(
        (stiffness - stiffness_lo) / (stiffness_hi - stiffness_lo) * compliance_bins
    ).long().clamp(0, compliance_bins - 1)
    material_bin = friction_index * compliance_bins + compliance_index

    output: dict[str, torch.Tensor] = {
        "friction_bin": friction_index,
        "compliance_bin": compliance_index,
        "material_bin": material_bin,
    }
    for key, values in (
        ("material_static_friction", static),
        ("material_dynamic_friction", dynamic),
        ("material_stiffness_n_m", stiffness),
        ("material_damping_n_s_m", damping),
    ):
        sums = torch.zeros(friction_bins * compliance_bins, device=values.device, dtype=values.dtype)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, material_bin, values)
        counts.scatter_add_(0, material_bin, torch.ones_like(values))
        means = sums / counts.clamp_min(1.0)
        output[key] = means[material_bin]
    return output


def sample_stem_profiles_v2(
    fruit_profiles: dict[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Create one explicitly labeled stem prior without cross-mode clamping."""

    profile = load_physics_v2_profile()["stem_modes_v2"]
    if mode not in {"learnability", "literature_axial", "bend_twist_prior"}:
        raise ValueError(f"unsupported Physics-v2 stem mode: {mode}")
    cfg = profile[mode]
    common = profile["common"]
    required = ("size_latent", "ripeness", "biological_latent", "static_friction")
    if any(key not in fruit_profiles for key in required):
        raise ValueError("fruit profile is missing correlated stem latents")
    size = fruit_profiles["size_latent"]
    ripeness = fruit_profiles["ripeness"]
    biological = fruit_profiles["biological_latent"]
    difficulty = (
        0.45 * size
        + 0.25 * (1.0 - ripeness)
        + 0.30 * (0.5 * (biological + 1.0))
    ).clamp(0.0, 1.0)
    diameter_lo, diameter_hi = common["abscission_diameter_m"]
    diameter = diameter_lo + (0.70 * size + 0.30 * (0.5 * (biological + 1.0))) * (
        diameter_hi - diameter_lo
    )
    tangential_capacity = (
        fruit_profiles["static_friction"]
        * float(common["safe_maximum_fingertip_force_n"])
        * int(common["safe_supported_fingertips"])
        * float(common["safe_wrench_utilization_fraction"])
    )
    moment_capacity = tangential_capacity * float(common["conservative_moment_arm_m"])
    if mode in {"learnability", "literature_axial"}:
        tensile_lo, tensile_hi = cfg["tensile_force_n"]
        tensile = tensile_lo + difficulty * (tensile_hi - tensile_lo)
        if mode == "learnability":
            tensile = torch.minimum(
                tensile,
                float(cfg["safe_capacity_fraction"]) * tangential_capacity,
            )
        shear_lo, shear_hi = cfg["shear_to_tension_ratio"]
        shear_ratio = shear_lo + size * (shear_hi - shear_lo)
        shear = tensile * shear_ratio
        lever_lo, lever_hi = cfg["bending_lever_arm_m"]
        lever = lever_lo + ripeness * (lever_hi - lever_lo)
        bending = tensile * lever
        torsion_lo, torsion_hi = cfg["torsion_to_bending_ratio"]
        torsion_ratio = torsion_lo + (0.5 * (biological + 1.0)) * (
            torsion_hi - torsion_lo
        )
        torsional = bending * torsion_ratio
    else:
        tensile_lo, tensile_hi = cfg["tensile_force_n"]
        shear_lo, shear_hi = cfg["shear_force_n"]
        bending_lo, bending_hi = cfg["bending_moment_nm"]
        torsional_lo, torsional_hi = cfg["torsional_moment_nm"]
        tensile = tensile_lo + difficulty * (tensile_hi - tensile_lo)
        shear = shear_lo + difficulty * (shear_hi - shear_lo)
        # Riper fruit is assigned the easier end of the uncalibrated mixed-mode
        # envelope, while size/biology preserve within-mode variation.
        mixed_difficulty = (0.60 * difficulty + 0.40 * (1.0 - ripeness)).clamp(0.0, 1.0)
        bending = bending_lo + mixed_difficulty * (bending_hi - bending_lo)
        torsional = torsional_lo + mixed_difficulty * (torsional_hi - torsional_lo)

    duration_lo, duration_hi = common["critical_duration_s"]
    critical_duration = duration_lo + ripeness * (duration_hi - duration_lo)
    mode_code = {"learnability": 0, "literature_axial": 1, "bend_twist_prior": 2}[mode]
    return {
        "stem_mode_code": torch.full_like(size, mode_code, dtype=torch.long),
        "stem_diameter_m": diameter,
        "stem_tensile_force_n": tensile,
        "stem_shear_force_n": shear,
        "stem_bending_moment_nm": bending,
        "stem_torsional_moment_nm": torsional,
        "stem_critical_duration_s": critical_duration,
        "stem_safe_tangential_capacity_n": tangential_capacity,
        "stem_safe_moment_capacity_nm": moment_capacity,
        "stem_translation_feasible": tensile <= tangential_capacity,
        "stem_bending_feasible": bending <= moment_capacity,
        "stem_torsion_feasible": torsional <= moment_capacity,
    }


def stem_load_components_v2(
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    tensile_limit_n: torch.Tensor,
    shear_limit_n: torch.Tensor,
    bending_limit_nm: torch.Tensor,
    torsional_limit_nm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return normalized tension, shear, bending, torsion, and interaction load."""

    if force_on_fruit_w.shape != torque_on_fruit_w.shape:
        raise ValueError("force and torque tensors must have the same shape")
    if force_on_fruit_w.ndim != 2 or force_on_fruit_w.shape[1] != 3:
        raise ValueError("force and torque tensors must have shape (N, 3)")
    if pull_axis_w.shape != force_on_fruit_w.shape:
        raise ValueError("pull_axis_w must have shape (N, 3)")
    axis = torch.nn.functional.normalize(pull_axis_w, dim=1)
    signed_axial = (force_on_fruit_w * axis).sum(dim=1)
    tension = (-signed_axial).clamp_min(0.0)
    shear_vector = force_on_fruit_w - signed_axial.unsqueeze(1) * axis
    shear = torch.linalg.vector_norm(shear_vector, dim=1)
    torsion = torch.abs((torque_on_fruit_w * axis).sum(dim=1))
    bending_vector = torque_on_fruit_w - (
        torque_on_fruit_w * axis
    ).sum(dim=1, keepdim=True) * axis
    bending = torch.linalg.vector_norm(bending_vector, dim=1)
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


def linear_compliant_contact_force(
    indentation_m: torch.Tensor,
    closing_speed_m_s: torch.Tensor,
    stiffness_n_m: torch.Tensor,
    damping_n_s_m: torch.Tensor,
    *,
    maximum_force_n: float | None = None,
) -> torch.Tensor:
    """Return the unilateral linear spring-damper order-check force."""

    if any(item.shape != indentation_m.shape for item in (closing_speed_m_s, stiffness_n_m, damping_n_s_m)):
        raise ValueError("contact tensors must have identical shapes")
    if bool((stiffness_n_m <= 0.0).any()) or bool((damping_n_s_m < 0.0).any()):
        raise ValueError("contact stiffness must be positive and damping non-negative")
    force = (
        stiffness_n_m * indentation_m.clamp_min(0.0)
        + damping_n_s_m * closing_speed_m_s.clamp_min(0.0)
    )
    force = torch.where(indentation_m > 0.0, force, torch.zeros_like(force))
    if maximum_force_n is not None:
        if maximum_force_n <= 0.0:
            raise ValueError("maximum_force_n must be positive")
        force = force.clamp_max(maximum_force_n)
    return force


def initial_fruit_damage_state(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Allocate per-contact diagnostic exposure state."""

    zeros = torch.zeros_like(reference)
    return {
        "peak_force_n": zeros.clone(),
        "peak_indentation_m": zeros.clone(),
        "force_exposure_n_s": zeros.clone(),
        "compression_exposure_m_s": zeros.clone(),
        "impact_exposure_n": zeros.clone(),
        "previous_force_n": zeros.clone(),
        "damage_score": zeros.clone(),
    }


def sample_uncertainty_profiles_v2(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    hand_joint_count: int = 16,
    tactile_channel_count: int = 5,
) -> dict[str, torch.Tensor]:
    """Sample bounded episode-fixed actuator and tactile robustness priors."""

    if count < 1 or hand_joint_count < 1 or tactile_channel_count < 1:
        raise ValueError("profile dimensions must be positive")
    cfg = load_physics_v2_profile()["uncertainty_v2"]["episode_fixed"]

    def uniform(key: str, shape: tuple[int, ...]) -> torch.Tensor:
        low, high = cfg[key]
        return low + torch.rand(
            shape, generator=generator, device=device, dtype=dtype
        ) * (high - low)

    delay_low, delay_high = cfg["tactile_delay_policy_steps"]
    return {
        "hand_response_scale": uniform("hand_response_scale", (count, hand_joint_count)),
        "hand_target_bias_rad": uniform("hand_target_bias_rad", (count, hand_joint_count)),
        "hand_deadband_rad": uniform("hand_deadband_rad", (count, hand_joint_count)),
        "hand_backlash_rad": uniform("hand_backlash_rad", (count, hand_joint_count)),
        "arm_response_scale": uniform("arm_response_scale", (count, 1)),
        "tactile_gain": uniform("tactile_gain", (count, tactile_channel_count)),
        "tactile_bias_n": uniform("tactile_bias_n", (count, tactile_channel_count)),
        "tactile_drift_rate_n_s": uniform(
            "tactile_drift_rate_n_s", (count, tactile_channel_count)
        ),
        "tactile_saturation_n": uniform(
            "tactile_saturation_n", (count, tactile_channel_count)
        ),
        "tactile_release_time_constant_s": uniform(
            "tactile_release_time_constant_s", (count, tactile_channel_count)
        ),
        "tactile_delay_policy_steps": torch.randint(
            int(delay_low),
            int(delay_high) + 1,
            (count,),
            generator=generator,
            device=device,
        ),
        "tactile_dropout_probability": uniform(
            "tactile_dropout_probability", (count, tactile_channel_count)
        ),
    }


def apply_joint_target_uncertainty_v2(
    desired_rad: torch.Tensor,
    current_rad: torch.Tensor,
    lower_limit_rad: torch.Tensor,
    upper_limit_rad: torch.Tensor,
    maximum_delta_rad: torch.Tensor,
    response_scale: torch.Tensor,
    target_bias_rad: torch.Tensor,
    deadband_rad: torch.Tensor,
    backlash_rad: torch.Tensor,
) -> torch.Tensor:
    """Apply bounded target response, deadband, backlash, and bias."""

    tensors = (
        current_rad,
        lower_limit_rad,
        upper_limit_rad,
        maximum_delta_rad,
        response_scale,
        target_bias_rad,
        deadband_rad,
        backlash_rad,
    )
    if any(value.shape != desired_rad.shape for value in tensors):
        raise ValueError("joint uncertainty tensors must have identical shapes")
    if bool((maximum_delta_rad <= 0.0).any()) or bool((response_scale <= 0.0).any()):
        raise ValueError("joint target limits and response scales must be positive")
    error = desired_rad - current_rad
    active = error.abs() > deadband_rad
    transmitted = torch.sign(error) * (error.abs() - backlash_rad).clamp_min(0.0)
    uncertain_delta = response_scale * transmitted + target_bias_rad
    uncertain_delta = uncertain_delta.clamp(-maximum_delta_rad, maximum_delta_rad)
    target = torch.where(active, current_rad + uncertain_delta, current_rad)
    return torch.maximum(lower_limit_rad, torch.minimum(upper_limit_rad, target))


def initial_tactile_uncertainty_state(
    reference: torch.Tensor, *, maximum_delay_steps: int
) -> dict[str, torch.Tensor]:
    """Allocate tactile history, release-lag, drift, and held-sample state."""

    if reference.ndim != 2 or maximum_delay_steps < 0:
        raise ValueError("reference must be (N,C) and delay non-negative")
    return {
        "history_n": torch.zeros(
            (maximum_delay_steps + 1, *reference.shape),
            device=reference.device,
            dtype=reference.dtype,
        ),
        "release_state_n": torch.zeros_like(reference),
        "drift_n": torch.zeros_like(reference),
        "measured_n": torch.zeros_like(reference),
    }


def update_tactile_uncertainty_v2(
    state: dict[str, torch.Tensor],
    physical_force_n: torch.Tensor,
    profiles: dict[str, torch.Tensor],
    *,
    dt_s: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Apply delay, release lag, gain, bias, drift, dropout, and saturation."""

    if dt_s <= 0.0 or physical_force_n.ndim != 2:
        raise ValueError("dt_s must be positive and physical_force_n must be (N,C)")
    history = state["history_n"].clone()
    history[1:] = history[:-1].clone()
    history[0] = physical_force_n.clamp_min(0.0)
    delay = profiles["tactile_delay_policy_steps"]
    if int(delay.max()) >= history.shape[0] or int(delay.min()) < 0:
        raise ValueError("tactile delay exceeds allocated history")
    env_ids = torch.arange(physical_force_n.shape[0], device=physical_force_n.device)
    delayed = history[delay, env_ids]
    tau = profiles["tactile_release_time_constant_s"].clamp_min(dt_s)
    alpha = 1.0 - torch.exp(-dt_s / tau)
    release = state["release_state_n"] + alpha * (
        delayed - state["release_state_n"]
    )
    drift = state["drift_n"] + profiles["tactile_drift_rate_n_s"] * dt_s
    candidate = (
        release * profiles["tactile_gain"] + profiles["tactile_bias_n"] + drift
    ).clamp_min(0.0)
    random = torch.rand(
        physical_force_n.shape,
        generator=generator,
        device=physical_force_n.device,
        dtype=physical_force_n.dtype,
    )
    dropout = random < profiles["tactile_dropout_probability"]
    measured = torch.where(dropout, state["measured_n"], candidate)
    measured = torch.minimum(measured, profiles["tactile_saturation_n"])
    return {
        "history_n": history,
        "release_state_n": release,
        "drift_n": drift,
        "measured_n": measured,
        "dropout_mask": dropout,
    }


def update_fruit_damage_diagnostics(
    state: dict[str, torch.Tensor],
    force_n: torch.Tensor,
    indentation_m: torch.Tensor,
    *,
    dt_s: float,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Integrate a bounded monotonic exposure score without claiming damage probability."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if force_n.shape != indentation_m.shape or not bool(torch.isfinite(force_n).all()):
        raise ValueError("force and indentation must be finite aligned tensors")
    cfg = config or load_physics_v2_profile()["fruit_damage_proxy_v2"]
    force = force_n.clamp_min(0.0)
    indentation = indentation_m.clamp_min(0.0)
    previous = state["previous_force_n"]
    force_rate = (force - previous).clamp_min(0.0) / dt_s
    impact_increment = (
        force_rate - float(cfg["impact_force_rate_threshold_n_s"])
    ).clamp_min(0.0) * dt_s
    force_exposure = state["force_exposure_n_s"] + (
        force - float(cfg["safe_force_n"])
    ).clamp_min(0.0) * dt_s
    compression_exposure = state["compression_exposure_m_s"] + (
        indentation - float(cfg["safe_indentation_m"])
    ).clamp_min(0.0) * dt_s
    impact_exposure = state["impact_exposure_n"] + impact_increment
    peak_force = torch.maximum(state["peak_force_n"], force)
    peak_indentation = torch.maximum(state["peak_indentation_m"], indentation)

    peak_force_ratio = (
        (peak_force - float(cfg["safe_force_n"]))
        / (float(cfg["reference_peak_force_n"]) - float(cfg["safe_force_n"]))
    ).clamp(0.0, 1.0)
    peak_indentation_ratio = (
        (peak_indentation - float(cfg["safe_indentation_m"]))
        / (float(cfg["reference_indentation_m"]) - float(cfg["safe_indentation_m"]))
    ).clamp(0.0, 1.0)
    force_time_ratio = (
        force_exposure / float(cfg["safe_force_exposure_budget_n_s"])
    ).clamp(0.0, 1.0)
    compression_time_ratio = (
        compression_exposure / float(cfg["safe_compression_exposure_budget_m_s"])
    ).clamp(0.0, 1.0)
    impact_ratio = (
        impact_exposure / float(cfg["impact_exposure_budget_n"])
    ).clamp(0.0, 1.0)
    weights = cfg["score_weights"]
    score = (
        float(weights["peak_force"]) * peak_force_ratio
        + float(weights["peak_indentation"]) * peak_indentation_ratio
        + float(weights["force_time_exposure"]) * force_time_ratio
        + float(weights["compression_time_exposure"]) * compression_time_ratio
        + float(weights["impact"]) * impact_ratio
    ).clamp(0.0, 1.0)
    return {
        "peak_force_n": peak_force,
        "peak_indentation_m": peak_indentation,
        "force_exposure_n_s": force_exposure,
        "compression_exposure_m_s": compression_exposure,
        "impact_exposure_n": impact_exposure,
        "previous_force_n": force,
        "damage_score": torch.maximum(state["damage_score"], score),
    }


__all__ = [
    "fixed_fruit_contact_cases",
    "initial_fruit_damage_state",
    "initial_tactile_uncertainty_state",
    "linear_compliant_contact_force",
    "apply_joint_target_uncertainty_v2",
    "quantize_contact_material_bank",
    "sample_correlated_fruit_contact_profiles",
    "sample_stem_profiles_v2",
    "sample_uncertainty_profiles_v2",
    "stem_load_components_v2",
    "update_fruit_damage_diagnostics",
    "update_tactile_uncertainty_v2",
]
