"""Public-prior mechanics for the isolated Physics-v2 tomato task.

The module is intentionally independent of :mod:`physics_v1`.  It consumes
the frozen evidence package under ``research/physics_v2_public_priors`` and
keeps every non-measured modelling choice explicit in the V2 profile.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from simulation.tomato_pick.config import load_object


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_V2_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "simulation/config/tomato_pick_physics_v2_public_prior_v1.json"
)


def v2_outcome_alignment_alpha_for_iteration(
    learning_iteration: float,
    *,
    discovery_until_iteration: float = 10.0,
    fully_aligned_iteration: float = 40.0,
) -> float:
    """Blend V2 discovery into outcome alignment inside the audited gate.

    The inherited Clean-v5 schedule keeps drop penalties and stable-detach
    credit disabled until update 250, which is beyond V2's mandatory
    50-update first-success gate.  V2 evidence shows that this rewards an
    increasingly fast break-and-drop shortcut even after transient retention
    has been discovered.  Preserve ten discovery updates, then smoothstep the
    unchanged objective endpoints to full alignment when public mechanics are
    also restored at update 40.
    """

    if not 0.0 <= discovery_until_iteration < fully_aligned_iteration:
        raise ValueError(
            "discovery cutoff must precede the full-alignment iteration"
        )
    fraction = min(
        1.0,
        max(
            0.0,
            (float(learning_iteration) - discovery_until_iteration)
            / (fully_aligned_iteration - discovery_until_iteration),
        ),
    )
    return float(fraction * fraction * (3.0 - 2.0 * fraction))


def discovery_capacity_scale(
    learning_iteration: float,
    *,
    initial_scale: float,
    hold_initial_scale_iterations: float,
    full_capacity_iteration: float,
) -> float:
    """Return the auditable early-training abscission-capacity scale.

    The scale is a training curriculum, never a crop prior or an evaluation
    relaxation.  It starts near the released Physics-v1 discovery regime and
    reaches the complete public-prior capacities before the first-success
    gate's ten-update tail window.  Smoothstep avoids an abrupt mechanics
    discontinuity while preserving exact endpoints.
    """

    if not 0.0 < initial_scale <= 1.0:
        raise ValueError("initial_scale must lie in (0, 1]")
    if full_capacity_iteration <= 0.0:
        raise ValueError("full_capacity_iteration must be positive")
    if not 0.0 <= hold_initial_scale_iterations < full_capacity_iteration:
        raise ValueError(
            "hold_initial_scale_iterations must lie in "
            "[0, full_capacity_iteration)"
        )
    fraction = min(
        1.0,
        max(
            0.0,
            (
                float(learning_iteration)
                - float(hold_initial_scale_iterations)
            )
            / (
                float(full_capacity_iteration)
                - float(hold_initial_scale_iterations)
            ),
        ),
    )
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return float(initial_scale + (1.0 - initial_scale) * fraction)


def retention_ready_conditioned_capacity_scale(
    curriculum_scale: float,
    retention_ready: torch.Tensor,
) -> torch.Tensor:
    """Apply training assistance only to a retention-ready current state.

    The caller combines strict thumb-plus-other-finger support with the
    current 120 Hz distance and relative-speed retention predicates.  Every
    other environment retains full public-prior capacity.  This prevents the
    curriculum from rewarding a supported but dynamically unsafe fracture.
    Evaluation uses a curriculum scale of one and is unchanged for every
    environment.
    """

    if not 0.0 < float(curriculum_scale) <= 1.0:
        raise ValueError("curriculum_scale must lie in (0, 1]")
    if retention_ready.dtype is not torch.bool or retention_ready.ndim != 1:
        raise ValueError("retention_ready must be a one-dimensional bool tensor")
    assisted = torch.full_like(
        retention_ready, float(curriculum_scale), dtype=torch.float32
    )
    return torch.where(retention_ready, assisted, torch.ones_like(assisted))


def kinematic_retention_margin_score(
    distance_m: torch.Tensor,
    relative_speed_m_s: torch.Tensor,
    *,
    maximum_distance_m: float,
    maximum_relative_speed_m_s: float,
) -> torch.Tensor:
    """Return centered-and-slow margin inside the strict retention envelope.

    This is a bounded modifier for the existing public-load progress credit,
    not a replacement retention predicate or a new reward term.  The score is
    exactly zero when either unchanged retention limit is reached and cannot
    provide credit outside either limit.  A geometric mean preserves a useful
    gradient only when both safety margins are present.
    """

    if distance_m.shape != relative_speed_m_s.shape or distance_m.ndim != 1:
        raise ValueError(
            "distance and relative speed must have matching one-dimensional shapes"
        )
    if not bool(torch.isfinite(distance_m).all()) or bool(
        (distance_m < 0.0).any()
    ):
        raise ValueError("distance must be finite and non-negative")
    if not bool(torch.isfinite(relative_speed_m_s).all()) or bool(
        (relative_speed_m_s < 0.0).any()
    ):
        raise ValueError("relative speed must be finite and non-negative")
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise ValueError("maximum_distance_m must be finite and positive")
    if (
        not math.isfinite(maximum_relative_speed_m_s)
        or maximum_relative_speed_m_s <= 0.0
    ):
        raise ValueError(
            "maximum_relative_speed_m_s must be finite and positive"
        )

    distance_margin = (
        1.0 - distance_m / float(maximum_distance_m)
    ).clamp(0.0, 1.0)
    speed_margin = (
        1.0 - relative_speed_m_s / float(maximum_relative_speed_m_s)
    ).clamp(0.0, 1.0)
    return torch.sqrt(distance_margin * speed_margin)


def issued_command_history_observation(
    action_history: torch.Tensor,
    *,
    history_steps: int,
) -> torch.Tensor:
    """Flatten newest-to-oldest issued commands for a feed-forward actor.

    ``action_history`` is controller-owned state, not a simulator truth.  A
    delayed actuator can still have newer commands queued after the command
    reported as previously applied.  Exposing those issued commands restores
    the finite-order Markov state without revealing the sampled delay.
    """

    if action_history.ndim != 3:
        raise ValueError("action_history must have shape (slots, envs, actions)")
    if not isinstance(history_steps, int) or not 1 <= history_steps <= int(
        action_history.shape[0]
    ):
        raise ValueError("history_steps must select available action-history slots")
    selected = action_history[:history_steps].permute(1, 0, 2)
    return selected.reshape(selected.shape[0], -1)


def relative_position_history_observation(
    current_relative_position: torch.Tensor,
    previous_relative_position: torch.Tensor,
    reset_pending: torch.Tensor,
) -> torch.Tensor:
    """Return a reset-safe current/previous deployable position pair."""

    if (
        current_relative_position.ndim != 2
        or current_relative_position.shape[1] != 3
        or previous_relative_position.shape != current_relative_position.shape
    ):
        raise ValueError("relative positions must have matching shape (envs, 3)")
    if (
        reset_pending.dtype is not torch.bool
        or reset_pending.shape != current_relative_position.shape[:1]
    ):
        raise ValueError("reset_pending must be a one-dimensional bool tensor")
    initialized_previous = torch.where(
        reset_pending.unsqueeze(1),
        current_relative_position,
        previous_relative_position,
    )
    return torch.cat((current_relative_position, initialized_previous), dim=1)


def normalized_controller_tracking_error(
    position_error_w: torch.Tensor,
    rotation_error_w: torch.Tensor,
    grasp_basis_w: torch.Tensor,
    action_scales: torch.Tensor,
) -> torch.Tensor:
    """Encode command tracking error in local one-action-step units.

    Command target and encoder forward kinematics are controller-owned at
    deployment. Their error is a proprioceptive load/release signal that does
    not reveal the simulated stem wrench, damage, detachment, or gate state.
    """

    if (
        position_error_w.ndim != 2
        or position_error_w.shape[1] != 3
        or rotation_error_w.shape != position_error_w.shape
        or grasp_basis_w.shape
        != (position_error_w.shape[0], 3, 3)
        or action_scales.shape != (6,)
    ):
        raise ValueError(
            "tracking errors, basis, and scales must have shapes "
            "(envs,3), (envs,3,3), and (6,)"
        )
    if not bool(
        torch.isfinite(position_error_w).all()
        and torch.isfinite(rotation_error_w).all()
        and torch.isfinite(grasp_basis_w).all()
        and torch.isfinite(action_scales).all()
        and (action_scales > 0.0).all()
    ):
        raise ValueError("tracking-error inputs must be finite with positive scales")
    position_error_g = torch.bmm(
        grasp_basis_w.transpose(1, 2), position_error_w.unsqueeze(2)
    ).squeeze(2)
    rotation_error_g = torch.bmm(
        grasp_basis_w.transpose(1, 2), rotation_error_w.unsqueeze(2)
    ).squeeze(2)
    return (
        torch.cat((position_error_g, rotation_error_g), dim=1)
        / action_scales.unsqueeze(0)
    ).clamp(-1.0, 1.0)


def load_bearing_opposition_target_n(
    fruit_mass_kg: torch.Tensor,
    effective_pair_static_friction: torch.Tensor,
    *,
    full_contact_force_n: float,
    safety_factor: float,
    gravity_m_s2: float = 9.81,
) -> torch.Tensor:
    """Return the per-side pinch force needed to support each fruit.

    A thumb/finger pinch supplies two friction interfaces, so the static load
    balance is ``2 * mu * normal_force >= mass * gravity``.  The target is a
    training-only normalization scale: it never changes contact mechanics,
    termination, retention, or success.  The inherited full-contact force is
    retained as a floor so easy profiles do not receive denser credit than the
    released Physics-v1 shaping signal.
    """

    if fruit_mass_kg.shape != effective_pair_static_friction.shape:
        raise ValueError("mass and pair friction must have the same shape")
    if fruit_mass_kg.ndim != 1:
        raise ValueError("mass and pair friction must have shape (num_envs,)")
    if not bool(torch.isfinite(fruit_mass_kg).all()) or bool(
        (fruit_mass_kg <= 0.0).any()
    ):
        raise ValueError("fruit mass must be finite and positive")
    if not bool(torch.isfinite(effective_pair_static_friction).all()) or bool(
        (effective_pair_static_friction <= 0.0).any()
    ):
        raise ValueError("pair friction must be finite and positive")
    if not math.isfinite(full_contact_force_n) or full_contact_force_n <= 0.0:
        raise ValueError("full_contact_force_n must be finite and positive")
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    static_load_target = (
        float(safety_factor)
        * fruit_mass_kg
        * float(gravity_m_s2)
        / (2.0 * effective_pair_static_friction)
    )
    return torch.maximum(
        static_load_target,
        torch.full_like(static_load_target, float(full_contact_force_n)),
    )


def load_bearing_opposition_score(
    finger_force_n: torch.Tensor,
    opposition_target_n: torch.Tensor,
    *,
    full_contact_force_n: float,
) -> torch.Tensor:
    """Return staged contact-discovery then load-bearing opposition progress.

    Half of the bounded score preserves the inherited 0.60 N contact-discovery
    landscape.  The other half continues smoothly toward the profile-specific
    static-load target.  This reallocates the existing unit score instead of
    adding reward weight or weakening any outcome predicate.
    """

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 5:
        raise ValueError("finger_force_n must have shape (num_envs, 5)")
    if opposition_target_n.shape != (finger_force_n.shape[0],):
        raise ValueError("opposition_target_n must have shape (num_envs,)")
    if not bool(torch.isfinite(finger_force_n).all()):
        raise ValueError("finger force must be finite")
    if not bool(torch.isfinite(opposition_target_n).all()) or bool(
        (opposition_target_n <= 0.0).any()
    ):
        raise ValueError("opposition target must be finite and positive")
    if not math.isfinite(full_contact_force_n) or full_contact_force_n <= 0.0:
        raise ValueError("full_contact_force_n must be finite and positive")

    contact_normalized = (
        finger_force_n / float(full_contact_force_n)
    ).clamp(0.0, 1.0)
    load_normalized = (
        finger_force_n / opposition_target_n.unsqueeze(1)
    ).clamp(0.0, 1.0)
    contact_score = contact_normalized[:, 0] * contact_normalized[
        :, 1:
    ].amax(dim=1)
    load_score = load_normalized[:, 0] * load_normalized[:, 1:].amax(dim=1)
    return 0.5 * (contact_score + load_score)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_physics_v2_profile(
    path: Path = PHYSICS_V2_PROFILE_PATH,
) -> dict[str, Any]:
    """Load V2 and its hash-pinned public evidence package."""

    profile = load_object(path)
    if profile.get("schema_version") != 1:
        raise ValueError("Physics-v2 profile schema_version must be 1")
    if profile.get("profile_id") != "tomato_pick_physics_v2_public_prior_v1":
        raise ValueError("unexpected Physics-v2 profile_id")
    if profile.get("task_id") != "LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1":
        raise ValueError("unexpected Physics-v2 task_id")

    public_spec = profile.get("public_prior")
    if not isinstance(public_spec, dict):
        raise ValueError("Physics-v2 public_prior must be an object")
    public_path = REPOSITORY_ROOT / str(public_spec["path"])
    if _sha256(public_path) != public_spec.get("sha256"):
        raise ValueError("Physics-v2 public-prior evidence hash changed")
    public = load_object(public_path)
    if public.get("profile_id") != public_spec.get("profile_id"):
        raise ValueError("Physics-v2 public-prior profile_id changed")

    engineering_spec = profile.get("engineering_prior")
    if not isinstance(engineering_spec, dict):
        raise ValueError("Physics-v2 engineering_prior must be an object")
    engineering_path = REPOSITORY_ROOT / str(engineering_spec["path"])
    if _sha256(engineering_path) != engineering_spec.get("sha256"):
        raise ValueError("Physics-v2 engineering-prior supplement hash changed")
    engineering = load_object(engineering_path)
    if engineering.get("profile_id") != engineering_spec.get("profile_id"):
        raise ValueError("Physics-v2 engineering-prior profile_id changed")

    parent = profile.get("frozen_parent")
    if not isinstance(parent, dict):
        raise ValueError("Physics-v2 frozen_parent must be an object")
    parent_path = REPOSITORY_ROOT / str(parent["physics_v1_profile"])
    if _sha256(parent_path) != parent.get("physics_v1_profile_sha256"):
        raise ValueError("frozen Physics-v1 parent profile hash changed")

    morphology = profile.get("morphology", {})
    weights = morphology.get("core_profile_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 3
        or any(float(value) <= 0.0 for value in weights)
        or not math.isclose(sum(float(value) for value in weights), 1.0)
    ):
        raise ValueError("Physics-v2 morphology weights must be three positive probabilities")
    correlation = float(morphology.get("shared_size_latent_correlation", -1.0))
    if not 0.0 <= correlation < 1.0:
        raise ValueError("Physics-v2 size-latent correlation must lie in [0, 1)")
    if profile.get("training", {}).get("from_scratch_required") is not True:
        raise ValueError("Physics-v2 must require from-scratch training")
    training = profile.get("training", {})
    if training.get("physics_v1_checkpoint_resume_allowed") is not False:
        raise ValueError("Physics-v2 must reject every Physics-v1 checkpoint")
    if training.get("same_profile_full_state_resume_allowed") is not True:
        raise ValueError("Physics-v2 must allow auditable same-profile continuation")
    training_curriculum = profile.get("training_curriculum", {})
    tensile_shear_scale = float(
        training_curriculum.get("initial_tensile_shear_capacity_scale", 0.0)
    )
    bending_torsion_scale = float(
        training_curriculum.get(
            "initial_bending_torsion_capacity_scale", 0.0
        )
    )
    hold_iteration = float(
        training_curriculum.get("hold_initial_scale_iterations", -1.0)
    )
    full_iteration = float(
        training_curriculum.get("full_public_prior_iteration", -1.0)
    )
    if tensile_shear_scale != 1.0:
        raise ValueError(
            "Physics-v2 discovery must preserve full tensile/shear capacity"
        )
    if not 0.0 < bending_torsion_scale <= 1.0:
        raise ValueError(
            "initial bending/torsion capacity scale must lie in (0, 1]"
        )
    if not 0.0 <= hold_iteration < full_iteration:
        raise ValueError("Physics-v2 capacity curriculum interval is invalid")
    if (
        training_curriculum.get(
            "bending_torsion_assistance_requires_secure_support"
        )
        is not True
    ):
        raise ValueError(
            "Physics-v2 bending/torsion assistance must require secure support"
        )
    if (
        training_curriculum.get(
            "bending_torsion_assistance_requires_kinematic_retention"
        )
        is not True
    ):
        raise ValueError(
            "Physics-v2 bending/torsion assistance must require current "
            "kinematic retention"
        )
    if training_curriculum.get("stem_load_progress_uses_public_capacity") is not True:
        raise ValueError(
            "Physics-v2 stem-load progress must use public capacity"
        )
    if (
        training_curriculum.get(
            "stem_load_progress_requires_kinematic_retention"
        )
        is not True
    ):
        raise ValueError(
            "Physics-v2 stem-load progress must require current kinematic "
            "retention during assisted training"
        )
    if (
        training_curriculum.get(
            "stem_load_progress_uses_retention_margin"
        )
        is not True
    ):
        raise ValueError(
            "Physics-v2 stem-load progress must use the strict-envelope "
            "retention margin during assisted training"
        )
    training_observation = profile.get("training_observation", {})
    expected_observation_contract = {
        "base_actor_dimension": 66,
        "physics_v2_actor_dimension": 103,
        "base_critic_dimension": 83,
        "physics_v2_critic_dimension": 120,
        "issued_command_history_steps": 2,
        "maximum_sampled_action_delay_steps": 2,
        "controller_tracking_error_exposed_to_actor": True,
        "controller_tracking_error_history_steps": 1,
        "privileged_velocity_exposed_to_actor": False,
        "privileged_action_delay_exposed_to_actor": False,
        "privileged_stem_wrench_exposed_to_actor": False,
        "privileged_detachment_exposed_to_actor": False,
    }
    if any(
        training_observation.get(key) != value
        for key, value in expected_observation_contract.items()
    ):
        raise ValueError(
            "Physics-v2 temporal observation contract changed or exposed "
            "privileged simulator state"
        )
    opposition_shaping = profile.get("training_shaping", {}).get(
        "load_bearing_opposition", {}
    )
    if opposition_shaping.get("enabled") is not True:
        raise ValueError("Physics-v2 load-bearing opposition shaping must be enabled")
    opposition_safety_factor = float(opposition_shaping.get("safety_factor", 0.0))
    if not math.isfinite(opposition_safety_factor) or opposition_safety_factor < 1.0:
        raise ValueError("load-bearing opposition safety factor must be at least one")
    shear = engineering["abscission_zone_shear"][
        "capacity_ratio_to_diameter_conditioned_tensile"
    ]["parameters"]
    if profile.get("abscission", {}).get("shear_ratio_triangular") != shear:
        raise ValueError("V2 shear sampler diverged from engineering supplement")
    delay = engineering["a7_l20_effective_actuator"][
        "action_delay_steps_uniform_inclusive"
    ]
    if profile.get("actuator_surrogate", {}).get(
        "action_delay_steps_uniform_inclusive"
    ) != delay:
        raise ValueError("V2 action delay diverged from engineering supplement")

    merged = dict(profile)
    merged["public_prior_data"] = public
    merged["engineering_prior_data"] = engineering
    return merged


def _sample_truncated_standard_normal(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
    maximum_attempts: int = 64,
) -> torch.Tensor:
    if not low < high:
        raise ValueError("truncated-normal bounds must be increasing")
    result = torch.empty(shape, device=device, dtype=dtype)
    flat = result.reshape(-1)
    pending = torch.arange(flat.numel(), device=device)
    for _ in range(maximum_attempts):
        if pending.numel() == 0:
            break
        candidate = torch.randn(
            pending.numel(), generator=generator, device=device, dtype=dtype
        )
        valid = (candidate >= low) & (candidate <= high)
        if bool(valid.any()):
            flat[pending[valid]] = candidate[valid]
        pending = pending[~valid]
    if pending.numel() != 0:
        raise RuntimeError("could not sample the requested truncated normal")
    return result


def _sample_variable_truncated_normal(
    mean: torch.Tensor,
    sd: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    generator: torch.Generator | None,
    maximum_attempts: int = 128,
) -> torch.Tensor:
    if not (mean.shape == sd.shape == low.shape == high.shape):
        raise ValueError("variable truncated-normal tensors must be aligned")
    if bool((sd <= 0.0).any()) or bool((low >= high).any()):
        raise ValueError("invalid variable truncated-normal distribution")
    result = torch.empty_like(mean)
    pending = torch.arange(mean.numel(), device=mean.device)
    for _ in range(maximum_attempts):
        if pending.numel() == 0:
            break
        candidate = mean[pending] + sd[pending] * torch.randn(
            pending.numel(),
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        valid = (candidate >= low[pending]) & (candidate <= high[pending])
        if bool(valid.any()):
            result[pending[valid]] = candidate[valid]
        pending = pending[~valid]
    if pending.numel() != 0:
        raise RuntimeError("could not sample the requested variable truncated normal")
    return result


def _sample_triangular(
    count: int,
    minimum: float,
    mode: float,
    maximum: float,
    *,
    generator: torch.Generator | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not minimum <= mode <= maximum or minimum == maximum:
        raise ValueError("triangular parameters must satisfy min <= mode <= max")
    unit = torch.rand(count, generator=generator, device=device, dtype=dtype)
    fraction = (mode - minimum) / (maximum - minimum)
    left = minimum + torch.sqrt(
        unit * (maximum - minimum) * (mode - minimum)
    )
    right = maximum - torch.sqrt(
        (1.0 - unit) * (maximum - minimum) * (maximum - mode)
    )
    return torch.where(unit < fraction, left, right)


def _piecewise_linear(
    values: torch.Tensor,
    nodes_x: torch.Tensor,
    nodes_y: torch.Tensor,
) -> torch.Tensor:
    if nodes_x.ndim != 1 or nodes_y.shape != nodes_x.shape or nodes_x.numel() < 2:
        raise ValueError("piecewise-linear nodes must be aligned one-dimensional tensors")
    if bool((nodes_x[1:] <= nodes_x[:-1]).any()):
        raise ValueError("piecewise-linear x nodes must be strictly increasing")
    upper = torch.searchsorted(nodes_x, values).clamp(1, nodes_x.numel() - 1)
    lower = upper - 1
    fraction = (values - nodes_x[lower]) / (nodes_x[upper] - nodes_x[lower])
    return nodes_y[lower] + fraction * (nodes_y[upper] - nodes_y[lower])


def sample_public_morphology_profiles(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Sample correlated public fruit dimensions and mass with rejection."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    profile = load_physics_v2_profile()
    morphology = profile["morphology"]
    public_profiles = profile["public_prior_data"]["fruit"][
        "standard_harvest_ready_profiles"
    ]
    by_id = {item["id"]: item for item in public_profiles}
    selected_profiles = [by_id[item] for item in morphology["core_profile_ids"]]
    means = torch.tensor(
        [item["mean_axes_m"] + [item["mean_mass_kg"]] for item in selected_profiles],
        device=device,
        dtype=dtype,
    )
    standard_deviations = torch.tensor(
        [item["sd_axes_m"] + [item["sd_mass_kg"]] for item in selected_profiles],
        device=device,
        dtype=dtype,
    )
    weights = torch.tensor(
        morphology["core_profile_weights"], device=device, dtype=dtype
    )
    low_z, high_z = morphology["truncated_standard_normal_bounds"]
    correlation = float(morphology["shared_size_latent_correlation"])
    independent_scale = math.sqrt(1.0 - correlation * correlation)
    aspect_low, aspect_high = morphology["polar_aspect_ratio_gate"]
    mass_low, mass_high = morphology["mass_gate_kg"]
    density_low, density_high = morphology["density_gate_kg_m3"]

    diameters = torch.empty((count, 3), device=device, dtype=dtype)
    mass = torch.empty(count, device=device, dtype=dtype)
    density = torch.empty(count, device=device, dtype=dtype)
    aspect = torch.empty(count, device=device, dtype=dtype)
    profile_index = torch.empty(count, device=device, dtype=torch.long)
    sample_attempt = torch.empty(count, device=device, dtype=torch.long)
    pending = torch.arange(count, device=device)
    rejection_count = 0
    for attempt in range(1, int(morphology["maximum_sampling_attempts"]) + 1):
        if pending.numel() == 0:
            break
        pending_count = int(pending.numel())
        candidate_profile = torch.multinomial(
            weights, pending_count, replacement=True, generator=generator
        )
        shared = _sample_truncated_standard_normal(
            (pending_count, 1),
            low=float(low_z),
            high=float(high_z),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        independent = _sample_truncated_standard_normal(
            (pending_count, 4),
            low=float(low_z),
            high=float(high_z),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latent = correlation * shared + independent_scale * independent
        candidate = means[candidate_profile] + standard_deviations[candidate_profile] * latent
        candidate_diameter = candidate[:, :3]
        candidate_mass = candidate[:, 3]
        radii = 0.5 * candidate_diameter
        candidate_volume = 4.0 / 3.0 * math.pi * radii.prod(dim=1)
        candidate_density = candidate_mass / candidate_volume.clamp_min(1.0e-12)
        candidate_aspect = candidate_diameter[:, 2] / torch.sqrt(
            (candidate_diameter[:, 0] * candidate_diameter[:, 1]).clamp_min(1.0e-12)
        )
        valid = (
            (candidate_diameter > 0.0).all(dim=1)
            & (candidate_aspect >= float(aspect_low))
            & (candidate_aspect <= float(aspect_high))
            & (candidate_mass >= float(mass_low))
            & (candidate_mass <= float(mass_high))
            & (candidate_density >= float(density_low))
            & (candidate_density <= float(density_high))
        )
        if bool(valid.any()):
            selected = pending[valid]
            diameters[selected] = candidate_diameter[valid]
            mass[selected] = candidate_mass[valid]
            density[selected] = candidate_density[valid]
            aspect[selected] = candidate_aspect[valid]
            profile_index[selected] = candidate_profile[valid]
            sample_attempt[selected] = attempt
        rejection_count += int((~valid).sum().item())
        pending = pending[~valid]
    if pending.numel() != 0:
        raise RuntimeError(
            "Physics-v2 morphology rejection sampler exhausted with "
            f"{int(pending.numel())} samples pending"
        )
    return {
        "diameters_m": diameters,
        "radii_m": 0.5 * diameters,
        "mass_kg": mass,
        "density_kg_m3": density,
        "polar_aspect_ratio": aspect,
        "source_profile_index": profile_index,
        "sampling_attempt": sample_attempt,
        "rejection_count": torch.tensor(rejection_count, device=device),
    }


def sample_public_abscission_profiles(
    count: int,
    *,
    diameter_m: float | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Sample diameter-conditioned tension, bending, shear and torsion."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    profile = load_physics_v2_profile()
    config = profile["abscission"]
    public = profile["public_prior_data"]["pedicel_and_abscission"]
    diameter_low, diameter_high = config["diameter_sampler_m"]
    if diameter_m is None:
        diameter = diameter_low + torch.rand(
            count, generator=generator, device=device, dtype=dtype
        ) * (diameter_high - diameter_low)
    elif isinstance(diameter_m, torch.Tensor):
        diameter = diameter_m.to(device=device, dtype=dtype).broadcast_to((count,)).clone()
    else:
        diameter = torch.full((count,), float(diameter_m), device=device, dtype=dtype)
    if bool((diameter < diameter_low).any()) or bool((diameter > diameter_high).any()):
        raise ValueError("abscission diameter lies outside the V2 sampler range")

    axial_nodes = public["pure_axial_pull_off"]["diameter_nodes"]
    nodes_x = torch.tensor(
        [node["diameter_m"] for node in axial_nodes], device=device, dtype=dtype
    )
    nodes_mean = torch.tensor(
        [node["force_mean_n"] for node in axial_nodes], device=device, dtype=dtype
    )
    nodes_sd = torch.tensor(
        [node["force_sd_n"] for node in axial_nodes], device=device, dtype=dtype
    )
    axial_mean = _piecewise_linear(diameter, nodes_x, nodes_mean)
    axial_sd = _piecewise_linear(diameter, nodes_x, nodes_sd)
    axial_low = torch.maximum(torch.full_like(axial_mean, 5.0), axial_mean - 2.5 * axial_sd)
    axial_high = axial_mean + 2.5 * axial_sd
    axial_limit = _sample_variable_truncated_normal(
        axial_mean,
        axial_sd,
        axial_low,
        axial_high,
        generator=generator,
    )

    bending_groups = public["three_point_bending"]["groups"]
    group_index = torch.where(
        diameter < 0.006,
        torch.zeros_like(diameter, dtype=torch.long),
        torch.where(
            diameter < 0.007,
            torch.ones_like(diameter, dtype=torch.long),
            torch.full_like(diameter, 2, dtype=torch.long),
        ),
    )
    bending_mean_table = torch.tensor(
        [group["derived_center_load_moment_mean_nm"] for group in bending_groups],
        device=device,
        dtype=dtype,
    )
    bending_sd_table = torch.tensor(
        [group["derived_moment_sd_nm"] for group in bending_groups],
        device=device,
        dtype=dtype,
    )
    bending_mean = bending_mean_table[group_index]
    bending_sd = bending_sd_table[group_index]
    bending_low = torch.maximum(
        torch.full_like(bending_mean, 0.005), bending_mean - 2.5 * bending_sd
    )
    bending_high = bending_mean + 2.5 * bending_sd
    bending_limit = _sample_variable_truncated_normal(
        bending_mean,
        bending_sd,
        bending_low,
        bending_high,
        generator=generator,
    )

    engineering = profile["engineering_prior_data"]
    shear_minimum, shear_mode, shear_maximum = engineering[
        "abscission_zone_shear"
    ]["capacity_ratio_to_diameter_conditioned_tensile"]["parameters"]
    shear_ratio = _sample_triangular(
        count,
        shear_minimum,
        shear_mode,
        shear_maximum,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    torsion_minimum, torsion_mode, torsion_maximum = config[
        "torsion_limit_triangular_nm"
    ]
    torsion_limit = _sample_triangular(
        count,
        torsion_minimum,
        torsion_mode,
        torsion_maximum,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    fatigue = engineering["cyclic_fatigue"]
    onset = _sample_triangular(
        count,
        *fatigue["subcritical_onset_ratio"]["parameters"],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    exponent = _sample_triangular(
        count,
        *fatigue["damage_exponent"]["parameters"],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    critical_duration = _sample_triangular(
        count,
        *fatigue["time_constant_s"]["parameters"],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    pedicel_low, pedicel_high = public["geometry"][
        "greenhouse_pedicel_length_m"
    ]["varietal_mean_range"]
    pedicel_length = pedicel_low + torch.rand(
        count, generator=generator, device=device, dtype=dtype
    ) * (pedicel_high - pedicel_low)
    az_low, az_high = public["geometry"]["distance_from_abscission_layer_m"][
        "varietal_mean_range"
    ]
    distance_from_az = az_low + torch.rand(
        count, generator=generator, device=device, dtype=dtype
    ) * (az_high - az_low)
    return {
        "stem_diameter_m": diameter,
        "stem_tensile_force_mean_n": axial_mean,
        "stem_tensile_force_sd_n": axial_sd,
        "stem_tensile_force_n": axial_limit,
        "stem_intact_tensile_force_n": axial_limit,
        "stem_shear_ratio": shear_ratio,
        "stem_shear_force_n": axial_limit * shear_ratio,
        "stem_bending_moment_mean_nm": bending_mean,
        "stem_bending_moment_sd_nm": bending_sd,
        "stem_bending_moment_nm": bending_limit,
        "stem_torsion_moment_nm": torsion_limit,
        "pedicel_length_m": pedicel_length,
        "distance_from_abscission_layer_m": distance_from_az,
        "stem_subcritical_onset_ratio": onset,
        "stem_subcritical_damage_exponent": exponent,
        "stem_critical_duration_s": critical_duration,
    }


def sample_public_task_orientation_categories(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample S09 world categories without exposing them to the local policy."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    probabilities = torch.tensor(
        load_physics_v2_profile()["task_distribution"][
            "world_orientation_probabilities"
        ],
        device=device,
    )
    return torch.multinomial(
        probabilities, count, replacement=True, generator=generator
    )


def sample_engineering_task_orientations(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Smooth the measured S09 categories with explicit bounded gap priors."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    profile = load_physics_v2_profile()
    config = profile["engineering_prior_data"]["natural_pedicel_orientation"]
    categories = sample_public_task_orientation_categories(
        count, generator=generator, device=device
    )
    centers = torch.deg2rad(
        torch.tensor(
            config["category_azimuth_center_deg"], device=device, dtype=dtype
        )
    )
    jitter_config = config["within_category_azimuth_jitter_deg"]
    jitter_sd = math.radians(float(jitter_config["sd"]))
    jitter_bounds = [math.radians(float(value)) for value in jitter_config["bounds"]]
    jitter = jitter_sd * _sample_truncated_standard_normal(
        (count,),
        low=jitter_bounds[0] / jitter_sd,
        high=jitter_bounds[1] / jitter_sd,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    tilt_degrees = config["polar_tilt_from_vertical_deg"]["parameters"]
    tilt = _sample_triangular(
        count,
        *(math.radians(float(value)) for value in tilt_degrees),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    azimuth = centers[categories] + jitter
    azimuth = torch.atan2(torch.sin(azimuth), torch.cos(azimuth))
    return {
        "world_orientation_category": categories,
        "orientation_azimuth_rad": azimuth,
        "orientation_azimuth_jitter_rad": jitter,
        "orientation_polar_tilt_rad": tilt,
    }


def quantize_engineering_contact_profiles(
    profiles: dict[str, torch.Tensor],
    *,
    friction_bins: int,
    compliance_bins: int,
) -> dict[str, torch.Tensor]:
    """Build a bounded material bank for the explicit V2 contact surrogate."""

    if friction_bins < 2 or compliance_bins < 2:
        raise ValueError("contact material bin counts must each be at least two")
    profile = load_physics_v2_profile()
    engineering = profile["engineering_prior_data"]["g20_pad_tomato_contact"]
    parent = load_object(
        REPOSITORY_ROOT / profile["frozen_parent"]["physics_v1_profile"]
    )
    parent_contact = parent["contact"]
    if parent_contact["friction_combine_mode"] != "average":
        raise ValueError("V2 pair-friction mapping requires average combination")
    baked = engineering["baked_finger_material_static_dynamic"]
    if not math.isclose(float(baked[0]), float(parent_contact["finger_static_friction"])):
        raise ValueError("baked finger static friction changed")
    if not math.isclose(float(baked[1]), float(parent_contact["finger_dynamic_friction"])):
        raise ValueError("baked finger dynamic friction changed")

    pair_static = profiles["effective_pair_static_friction"]
    stiffness = profiles["contact_stiffness_n_m"]
    friction_low, _friction_mode, friction_high = engineering[
        "effective_pair_static_friction"
    ]["parameters"]
    stiffness_low, stiffness_mode, stiffness_high = engineering[
        "effective_linear_contact_stiffness_n_m"
    ]["parameters"]
    damping_config = engineering["effective_contact_damping_n_s_m"]
    damping_low, damping_high = damping_config["bounds"]
    damping_mode = damping_config["nominal"]
    friction_index = torch.floor(
        (pair_static - friction_low)
        / (friction_high - friction_low)
        * friction_bins
    ).long().clamp(0, friction_bins - 1)
    compliance_index = torch.floor(
        (stiffness - stiffness_low)
        / (stiffness_high - stiffness_low)
        * compliance_bins
    ).long().clamp(0, compliance_bins - 1)
    friction_fraction = (friction_index.to(pair_static.dtype) + 0.5) / friction_bins
    compliance_fraction = (
        compliance_index.to(stiffness.dtype) + 0.5
    ) / compliance_bins
    material_pair_static = friction_low + friction_fraction * (
        friction_high - friction_low
    )
    material_pair_dynamic = material_pair_static * float(
        engineering["effective_dynamic_to_static_ratio"]
    )
    material_stiffness = stiffness_low + compliance_fraction * (
        stiffness_high - stiffness_low
    )
    damping_nodes_x = torch.tensor(
        (stiffness_low, stiffness_mode, stiffness_high),
        device=stiffness.device,
        dtype=stiffness.dtype,
    )
    damping_nodes_y = torch.tensor(
        (damping_low, damping_mode, damping_high),
        device=stiffness.device,
        dtype=stiffness.dtype,
    )
    material_damping = _piecewise_linear(
        material_stiffness, damping_nodes_x, damping_nodes_y
    )
    material_static = 2.0 * material_pair_static - float(baked[0])
    material_dynamic = 2.0 * material_pair_dynamic - float(baked[1])
    if bool((material_static < 0.0).any()) or bool((material_dynamic < 0.0).any()):
        raise RuntimeError("V2 pair friction cannot be represented by material mapping")
    return {
        "friction_bin": friction_index,
        "compliance_bin": compliance_index,
        "material_bin": friction_index * compliance_bins + compliance_index,
        "effective_pair_static_friction": material_pair_static,
        "effective_pair_dynamic_friction": material_pair_dynamic,
        "material_static_friction": material_static,
        "material_dynamic_friction": material_dynamic,
        "material_stiffness_n_m": material_stiffness,
        "material_damping_n_s_m": material_damping,
    }


def sample_physics_v2_profiles(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return one episode-fixed V2 crop profile per environment."""

    morphology = sample_public_morphology_profiles(
        count, generator=generator, device=device, dtype=dtype
    )
    abscission = sample_public_abscission_profiles(
        count, generator=generator, device=device, dtype=dtype
    )
    engineering = load_physics_v2_profile()["engineering_prior_data"]
    contact = engineering["g20_pad_tomato_contact"]
    pair_static = _sample_triangular(
        count,
        *contact["effective_pair_static_friction"]["parameters"],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    contact_stiffness = _sample_triangular(
        count,
        *contact["effective_linear_contact_stiffness_n_m"]["parameters"],
        generator=generator,
        device=device,
        dtype=dtype,
    )
    stiffness_nodes = contact["effective_linear_contact_stiffness_n_m"][
        "parameters"
    ]
    damping = contact["effective_contact_damping_n_s_m"]
    contact_damping = _piecewise_linear(
        contact_stiffness,
        torch.tensor(stiffness_nodes, device=device, dtype=dtype),
        torch.tensor(
            (damping["bounds"][0], damping["nominal"], damping["bounds"][1]),
            device=device,
            dtype=dtype,
        ),
    )
    orientations = sample_engineering_task_orientations(
        count, generator=generator, device=device, dtype=dtype
    )
    profiles = dict(morphology)
    profiles.update(abscission)
    profiles.update(
        {
            # Isaac's existing tomato authoring expects ellipsoid radii under
            # the historical ``axes_m`` key.
            "axes_m": morphology["radii_m"],
            "effective_pair_static_friction": pair_static,
            "effective_pair_dynamic_friction": pair_static
            * float(contact["effective_dynamic_to_static_ratio"]),
            "static_friction": pair_static,
            "contact_stiffness_n_m": contact_stiffness,
            "contact_damping_n_s_m": contact_damping,
        }
    )
    profiles.update(orientations)
    return profiles


def decompose_abscission_load(
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if force_on_fruit_w.shape != torque_on_fruit_w.shape:
        raise ValueError("force and torque tensors must have the same shape")
    if force_on_fruit_w.ndim != 2 or force_on_fruit_w.shape[1] != 3:
        raise ValueError("force and torque tensors must have shape (N, 3)")
    if pull_axis_w.shape != force_on_fruit_w.shape:
        raise ValueError("pull axis must have shape (N, 3)")
    axis = torch.nn.functional.normalize(pull_axis_w, dim=1)
    signed_axial = (force_on_fruit_w * axis).sum(dim=1)
    tension = (-signed_axial).clamp_min(0.0)
    shear_vector = force_on_fruit_w - signed_axial.unsqueeze(1) * axis
    signed_torsion = (torque_on_fruit_w * axis).sum(dim=1)
    bending_vector = torque_on_fruit_w - signed_torsion.unsqueeze(1) * axis
    return {
        "tension_n": tension,
        "shear_n": torch.linalg.vector_norm(shear_vector, dim=1),
        "bending_nm": torch.linalg.vector_norm(bending_vector, dim=1),
        "torsion_nm": signed_torsion.abs(),
        "signed_torsion_nm": signed_torsion,
    }


def combined_abscission_load_ratio(
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    tensile_limit_n: torch.Tensor,
    shear_limit_n: torch.Tensor,
    bending_limit_nm: torch.Tensor,
    torsion_limit_nm: torch.Tensor,
    integrity: torch.Tensor,
    *,
    capacity_floor_fraction: float = 0.001,
) -> torch.Tensor:
    """Return the V2 tension/shear/bending/torsion interaction ratio."""

    components = decompose_abscission_load(
        force_on_fruit_w, torque_on_fruit_w, pull_axis_w
    )
    capacity_scale = integrity.clamp_min(capacity_floor_fraction)
    return torch.sqrt(
        torch.square(components["tension_n"] / (tensile_limit_n * capacity_scale))
        + torch.square(components["shear_n"] / (shear_limit_n * capacity_scale))
        + torch.square(components["bending_nm"] / (bending_limit_nm * capacity_scale))
        + torch.square(components["torsion_nm"] / (torsion_limit_nm * capacity_scale))
    )


def update_twist_integrity(
    previous_integrity: torch.Tensor,
    torsion_nm: torch.Tensor,
    intact_torsion_limit_nm: torch.Tensor,
    absolute_twist_increment_rad: torch.Tensor,
    *,
    k_theta_per_rad: float,
    exponent_q: float,
    attached: torch.Tensor,
) -> torch.Tensor:
    """Apply the public-target-constrained multiplicative twist law."""

    if k_theta_per_rad <= 0.0 or exponent_q <= 0.0:
        raise ValueError("twist-integrity parameters must be positive")
    if bool((absolute_twist_increment_rad < 0.0).any()):
        raise ValueError("absolute twist increment cannot be negative")
    # ``Mt*`` in the integrity law is the capacity available at this state,
    # while the sampled tensor is the intact public-prior capacity.  This
    # keeps yielded plastic twist at unit utilization as C falls and preserves
    # the calibrated exponential C(2*pi)=0.2604 target.
    current_capacity = (
        intact_torsion_limit_nm * previous_integrity.clamp_min(1.0e-8)
    )
    utilization = (torsion_nm / current_capacity.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    updated = previous_integrity * torch.exp(
        -k_theta_per_rad
        * absolute_twist_increment_rad
        * torch.pow(utilization, exponent_q)
    )
    return torch.where(attached, updated, previous_integrity).clamp(0.0, 1.0)


def update_multimode_abscission(
    previous_integrity: torch.Tensor,
    previous_load_damage: torch.Tensor,
    force_on_fruit_w: torch.Tensor,
    torque_on_fruit_w: torch.Tensor,
    pull_axis_w: torch.Tensor,
    tensile_limit_n: torch.Tensor,
    shear_limit_n: torch.Tensor,
    bending_limit_nm: torch.Tensor,
    torsion_limit_nm: torch.Tensor,
    absolute_twist_increment_rad: torch.Tensor,
    *,
    dt_s: float,
    attached: torch.Tensor,
    k_theta_per_rad: float,
    twist_exponent_q: float,
    subcritical_onset_ratio: float | torch.Tensor,
    subcritical_exponent: float | torch.Tensor,
    subcritical_time_constant_s: float | torch.Tensor,
    instantaneous_failure_ratio: float,
    torsion_detach_integrity_threshold: float,
    capacity_floor_fraction: float,
) -> dict[str, torch.Tensor]:
    """Advance V2 integrity and non-torsional duration damage.

    Torsional yielding is represented by ``integrity``.  The duration term is
    intentionally driven by tension/shear/bending only so the same torsion is
    not counted twice.  That explicit method choice is required for Q5: a
    measured 360-degree twist must leave a residual pull capacity to test.
    """

    onset = torch.as_tensor(
        subcritical_onset_ratio,
        device=previous_integrity.device,
        dtype=previous_integrity.dtype,
    )
    exponent = torch.as_tensor(
        subcritical_exponent,
        device=previous_integrity.device,
        dtype=previous_integrity.dtype,
    )
    time_constant = torch.as_tensor(
        subcritical_time_constant_s,
        device=previous_integrity.device,
        dtype=previous_integrity.dtype,
    )
    if dt_s <= 0.0 or bool((time_constant <= 0.0).any()):
        raise ValueError("time constants must be positive")
    if bool(((onset < 0.0) | (onset >= 1.0)).any()):
        raise ValueError("subcritical onset must lie in [0, 1)")
    if bool((exponent <= 0.0).any()) or instantaneous_failure_ratio < 1.0:
        raise ValueError("invalid abscission damage parameters")
    if not 0.0 < torsion_detach_integrity_threshold < 1.0:
        raise ValueError("torsion detach threshold must lie in (0, 1)")

    components = decompose_abscission_load(
        force_on_fruit_w, torque_on_fruit_w, pull_axis_w
    )
    integrity = update_twist_integrity(
        previous_integrity,
        components["torsion_nm"],
        torsion_limit_nm,
        absolute_twist_increment_rad,
        k_theta_per_rad=k_theta_per_rad,
        exponent_q=twist_exponent_q,
        attached=attached,
    )
    capacity_scale = integrity.clamp_min(capacity_floor_fraction)
    tension_ratio = components["tension_n"] / (tensile_limit_n * capacity_scale)
    shear_ratio = components["shear_n"] / (shear_limit_n * capacity_scale)
    bending_ratio = components["bending_nm"] / (bending_limit_nm * capacity_scale)
    # The attachment is elastoplastic in torsion: once the elastic demand
    # reaches the current capacity, additional rotation is plastic twist, not
    # an unbounded torsional spring.  This prevents the integrity model from
    # immediately double-counting its own capacity reduction as overstress.
    current_torsion_capacity = torsion_limit_nm * capacity_scale
    yielded_torsion_nm = torch.minimum(
        components["torsion_nm"], current_torsion_capacity
    )
    torsion_ratio = yielded_torsion_nm / current_torsion_capacity
    combined_ratio = torch.sqrt(
        tension_ratio.square()
        + shear_ratio.square()
        + bending_ratio.square()
        + torsion_ratio.square()
    )
    duration_ratio = torch.sqrt(
        tension_ratio.square() + shear_ratio.square() + bending_ratio.square()
    )
    drive = (
        (duration_ratio - onset)
        / (1.0 - onset)
    ).clamp_min(0.0)
    load_damage = previous_load_damage + (
        dt_s / time_constant
    ) * torch.pow(drive, exponent)
    load_damage = torch.where(attached, load_damage, previous_load_damage)
    broken = attached & (
        (combined_ratio >= instantaneous_failure_ratio)
        | (load_damage >= 1.0)
        | (integrity <= torsion_detach_integrity_threshold)
    )
    return {
        "integrity": integrity,
        "load_damage": load_damage.clamp(0.0, 2.0),
        "combined_load_ratio": combined_ratio,
        "duration_load_ratio": duration_ratio,
        "tension_ratio": tension_ratio,
        "shear_ratio": shear_ratio,
        "bending_ratio": bending_ratio,
        "torsion_ratio": torsion_ratio,
        "yielded_torsion_nm": yielded_torsion_nm,
        "broken": broken,
    }


__all__ = [
    "PHYSICS_V2_PROFILE_PATH",
    "combined_abscission_load_ratio",
    "discovery_capacity_scale",
    "issued_command_history_observation",
    "normalized_controller_tracking_error",
    "relative_position_history_observation",
    "retention_ready_conditioned_capacity_scale",
    "decompose_abscission_load",
    "load_physics_v2_profile",
    "quantize_engineering_contact_profiles",
    "sample_physics_v2_profiles",
    "sample_engineering_task_orientations",
    "sample_public_abscission_profiles",
    "sample_public_morphology_profiles",
    "sample_public_task_orientation_categories",
    "update_multimode_abscission",
    "update_twist_integrity",
    "v2_outcome_alignment_alpha_for_iteration",
]
