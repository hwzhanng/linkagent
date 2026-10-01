"""CPU-testable left G20 grasp manifold and tactile residual adapter.

The five coarse coordinates retain the semantic structure used by LinkAgent's
successful arm-hand policy.  The numerical ranges here are rebuilt from this
repository's 2026-07-26 *left-hand* motion calibration instead of copying the
right-hand radians.  This module is simulation-only until the explicit gates
in ``docs/PPO_HAND_5D_MANIFOLD_MIGRATION.md`` have passed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


MAPPING_ID = "left_g20_semantic_5d_plus_digit_residual_v1_sim_only"
SOURCE_LINKAGENT_MAPPER_SHA256 = (
    "b3d2458ada8fe4b6129e6d3ce32d74072b4cbbb0c85490340660377869a0b489"
)
SOURCE_LEFT_CALIBRATION = (
    "simulation/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/config/left_g20_motion_calibration_20260726.yaml"
)
VALIDATION_STATUS = "cpu_validated_simulation_only_left_thumb_axes_pending"

SYNERGY_ORDER = (
    "global_finger_flexion",
    "proximal_distal_curl_shape",
    "radial_ulnar_flexion_balance",
    "finger_spread",
    "thumb_opposition",
)
DIGIT_RESIDUAL_ORDER = ("thumb", "index", "middle", "ring", "little")

# SDK policy order, matching the source 5D implementation.  Keeping this
# order avoids accidental coupling when an articulation exposes joints in a
# different order.
LEFT_ACTIVE_JOINT_ORDER = (
    "left_hand_thumb_joint2",
    "left_hand_index_joint1",
    "left_hand_middle_joint1",
    "left_hand_ring_joint1",
    "left_hand_little_joint1",
    "left_hand_thumb_joint0",
    "left_hand_index_joint0",
    "left_hand_middle_joint0",
    "left_hand_ring_joint0",
    "left_hand_little_joint0",
    "left_hand_thumb_joint1",
    "left_hand_thumb_joint3",
    "left_hand_index_joint2",
    "left_hand_middle_joint2",
    "left_hand_ring_joint2",
    "left_hand_little_joint2",
)

# Conservative radian intervals converted from the connected left G20's
# validated raw ranges.  All four thumb axes are widened only for the offline
# PhysX task, staying inside the vendor URDF limits with roughly 5% margin:
# the connected-device calibration deliberately kept them narrow, but a
# hand-only grasp simulator needs a physically reachable thumb/finger
# intersection.  These priors must not be sent to hardware until a dedicated
# thumb-axis calibration is recorded.
LEFT_SAFE_LOWER_RAD = (
    0.020000000,
    0.027450980,
    0.027450980,
    0.027450980,
    0.027450980,
    -1.000000000,
    -0.147843137,
    -0.058117647,
    0.029568627,
    0.158039216,
    0.000000000,
    0.025294118,
    0.021176471,
    0.021176471,
    0.021176471,
    0.021176471,
)
LEFT_SAFE_UPPER_RAD = (
    0.800000000,
    1.317647059,
    1.372549020,
    1.372549020,
    1.317647059,
    0.450000000,
    -0.117254902,
    -0.025490196,
    0.062196078,
    0.174352941,
    1.380000000,
    1.250000000,
    1.058823529,
    1.058823529,
    1.037647059,
    1.058823529,
)
LEFT_SAFE_VELOCITY_RAD_S = (
    1.0685,
    3.1523,
    3.1158,
    3.0806,
    3.0013,
    0.5,
    0.2,
    0.2,
    0.2,
    0.2,
    0.4,
    1.6148,
    2.4061,
    2.3382,
    2.5159,
    2.4225,
)

PREGRASP_FLEXION_FRACTION = 0.15
PROXIMAL_CURL_SHAPE_GAIN = 0.20
DISTAL_CURL_SHAPE_GAIN = 0.30
RADIAL_ULNAR_BALANCE_GAIN = 0.22
RADIAL_ULNAR_BALANCE_PATTERN = (1.0, 0.45, -0.45, -1.0)
FINGER_SPREAD_PATTERN = (-1.0, -0.35, 0.35, 1.0)
# Simulation-only opposition reach.  A 0.50 cap left the fully commanded
# thumb tip about 11 cm from the palm-centred tomato, making five-digit
# enclosure impossible by construction.  The joint still cannot leave the
# calibrated simulation safe interval; hardware use remains explicitly gated.
THUMB_ROOT_MAX_FRACTION = 1.00
# Preserve the established effective-closure scale used by the CPU proxy and
# observation contract; opposition reach and normalized flexion are separate
# quantities after widening the simulation-only thumb workspace.
THUMB_CLOSURE_NORMALIZATION_FRACTION = 0.50
DEFAULT_RESIDUAL_SPAN_FRACTION = 0.08
COARSE_INTERIOR_MARGIN_FRACTION = 0.04


def _require_last_dim(values: torch.Tensor, expected: int, name: str) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim < 1 or values.shape[-1] != expected:
        raise ValueError(
            f"{name} must end with dimension {expected}, got {tuple(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _constants_like(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower = values.new_tensor(LEFT_SAFE_LOWER_RAD)
    upper = values.new_tensor(LEFT_SAFE_UPPER_RAD)
    return lower, upper


def _centered_unit(action: torch.Tensor) -> torch.Tensor:
    action = action.clamp(-1.0, 1.0)
    return torch.where(
        action >= 0.0,
        PREGRASP_FLEXION_FRACTION
        + action * (1.0 - PREGRASP_FLEXION_FRACTION),
        PREGRASP_FLEXION_FRACTION * (1.0 + action),
    )


def _coarse_interior(unit: torch.Tensor) -> torch.Tensor:
    """Keep the coarse policy away from hard limits; residuals retain reach."""

    margin = COARSE_INTERIOR_MARGIN_FRACTION
    return margin + (1.0 - 2.0 * margin) * unit.clamp(0.0, 1.0)


def synergy_to_joint_targets(synergy: torch.Tensor) -> torch.Tensor:
    """Map semantic 5D actions to 16 conservative left-hand targets.

    This preserves LinkAgent's coarse eigengrasp structure but deliberately
    uses this machine's measured left-hand ranges.  The fifth coordinate
    drives thumb opposition; the widened yaw/roll prior is explicitly
    simulation-only until those axes are calibrated on hardware.
    """

    _require_last_dim(synergy, 5, "synergy")
    action = synergy.clamp(-1.0, 1.0)
    lower, upper = _constants_like(action)
    span = upper - lower

    global_flexion = _centered_unit(action[..., 0])
    curl_shape = action[..., 1]
    balance = action[..., 2]
    spread = action[..., 3]
    # Workspace sweeps against the converted left USD establish that the
    # high side of this normalized coordinate is the usable across-palm
    # branch (thumb tip approaches the palm-centred fruit).  Keep the public
    # semantic contract literal: positive action means more opposition.
    thumb_opposition = _centered_unit(action[..., 4])

    balance_pattern = action.new_tensor(RADIAL_ULNAR_BALANCE_PATTERN)
    balance_offset = (
        RADIAL_ULNAR_BALANCE_GAIN * balance.unsqueeze(-1) * balance_pattern
    )
    proximal = _coarse_interior(
        global_flexion.unsqueeze(-1)
        - PROXIMAL_CURL_SHAPE_GAIN * curl_shape.unsqueeze(-1)
        + balance_offset
    )
    distal = _coarse_interior(
        global_flexion.unsqueeze(-1)
        + DISTAL_CURL_SHAPE_GAIN * curl_shape.unsqueeze(-1)
        + balance_offset
    )
    thumb_opposition = _coarse_interior(thumb_opposition)

    result = lower.expand(*action.shape[:-1], 16).clone()
    # Thumb joint2 is the complementary root axis in the converted USD: it
    # must advance as the lateral axes swing inward.
    result[..., 0] = lower[0] + (
        THUMB_ROOT_MAX_FRACTION * (1.0 - thumb_opposition) * span[0]
    )
    result[..., 1:5] = lower[1:5] + proximal * span[1:5]

    # The hardware calibration did not authorize a wide yaw/roll sweep.  In
    # this offline PhysX proxy, however, thumb opposition must actually move
    # across the finger tips; otherwise no object pose can be enclosed by the
    # imported geometry.  Keep the wider values isolated to this manifold and
    # gate them from any real-device controller.
    result[..., 5] = lower[5] + thumb_opposition * span[5]
    spread_pattern = action.new_tensor(FINGER_SPREAD_PATTERN)
    spread_unit = (
        (1.0 - 2.0 * COARSE_INTERIOR_MARGIN_FRACTION)
        * spread.unsqueeze(-1)
        * spread_pattern
    ).clamp(-1.0, 1.0)
    centers = 0.5 * (lower[6:10] + upper[6:10])
    result[..., 6:10] = torch.where(
        spread_unit >= 0.0,
        centers + spread_unit * (upper[6:10] - centers),
        centers + (-spread_unit) * (lower[6:10] - centers),
    )
    # Thumb joint1 provides the forward opposition sweep and therefore uses
    # the complementary side of the mirrored axis.
    result[..., 10] = lower[10] + (1.0 - thumb_opposition) * span[10]
    # Opposition is not only a lateral root sweep on this mechanism.  The
    # distal link must curl as the root crosses the palm; otherwise reaching
    # the fruit requires raising global_flexion so far that all four opposing
    # fingers over-close first.  The max preserves ordinary global closure
    # while letting the dedicated thumb coordinate complete opposition.
    thumb_distal = torch.maximum(
        _coarse_interior(global_flexion),
        0.92 * thumb_opposition,
    ).clamp(0.0, 1.0)
    result[..., 11] = lower[11] + thumb_distal * span[11]
    result[..., 12:16] = lower[12:16] + distal * span[12:16]
    return torch.maximum(lower, torch.minimum(upper, result))


def hybrid_action_to_joint_targets(
    action: torch.Tensor,
    *,
    residual_span_fraction: float = DEFAULT_RESIDUAL_SPAN_FRACTION,
) -> torch.Tensor:
    """Map 5D coarse synergies plus five per-digit tactile residuals.

    Action layout is ``[five synergies, thumb/index/middle/ring/little]``.
    Residuals are intentionally small and cannot escape the measured safe
    ranges.  They restore independent tactile micro-adjustment that a strict
    five-dimensional manifold cannot express.
    """

    _require_last_dim(action, 10, "action")
    if not 0.0 <= residual_span_fraction <= 0.25:
        raise ValueError("residual_span_fraction must lie in [0, 0.25]")
    targets = synergy_to_joint_targets(action[..., :5])
    residual = action[..., 5:].clamp(-1.0, 1.0)
    lower, upper = _constants_like(action)
    span = upper - lower
    digit_joint_indices = ((0, 11), (1, 12), (2, 13), (3, 14), (4, 15))
    for digit_index, joint_indices in enumerate(digit_joint_indices):
        for joint_index in joint_indices:
            targets[..., joint_index] += (
                residual[..., digit_index]
                * residual_span_fraction
                * span[joint_index]
            )
    return torch.maximum(lower, torch.minimum(upper, targets))


def effective_digit_closure(action: torch.Tensor) -> torch.Tensor:
    """Return normalized thumb-to-little closure for proxy/contact tests."""

    if action.shape[-1] == 5:
        targets = synergy_to_joint_targets(action)
    elif action.shape[-1] == 10:
        targets = hybrid_action_to_joint_targets(action)
    else:
        raise ValueError("action must contain 5 synergy or 10 hybrid values")
    return joint_targets_to_digit_closure(targets)


def joint_targets_to_digit_closure(targets: torch.Tensor) -> torch.Tensor:
    """Project canonical 16-joint state back to five effective closures."""

    _require_last_dim(targets, 16, "targets")
    lower, upper = _constants_like(targets)
    normalized = ((targets - lower) / (upper - lower)).clamp(0.0, 1.0)
    return torch.stack(
        (
            0.5 * (
                normalized[..., 0] / THUMB_CLOSURE_NORMALIZATION_FRACTION
                + normalized[..., 11]
            ),
            0.5 * (normalized[..., 1] + normalized[..., 12]),
            0.5 * (normalized[..., 2] + normalized[..., 13]),
            0.5 * (normalized[..., 3] + normalized[..., 14]),
            0.5 * (normalized[..., 4] + normalized[..., 15]),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def rate_limit_joint_targets(
    previous: torch.Tensor,
    desired: torch.Tensor,
    *,
    policy_dt_s: float,
    velocity_scale: float = 0.50,
    absolute_velocity_cap_rad_s: float | None = None,
) -> torch.Tensor:
    """Bound one policy step using measured no-load joint velocities."""

    _require_last_dim(previous, 16, "previous")
    _require_last_dim(desired, 16, "desired")
    if previous.shape != desired.shape:
        raise ValueError("previous and desired must have identical shapes")
    if policy_dt_s <= 0.0:
        raise ValueError("policy_dt_s must be positive")
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError("velocity_scale must lie in (0, 1]")
    if absolute_velocity_cap_rad_s is not None and absolute_velocity_cap_rad_s <= 0.0:
        raise ValueError("absolute_velocity_cap_rad_s must be positive when provided")
    maximum_velocity = desired.new_tensor(LEFT_SAFE_VELOCITY_RAD_S) * velocity_scale
    if absolute_velocity_cap_rad_s is not None:
        maximum_velocity = maximum_velocity.clamp_max(absolute_velocity_cap_rad_s)
    maximum_delta = maximum_velocity * policy_dt_s
    bounded = previous + (desired - previous).clamp(-maximum_delta, maximum_delta)
    lower, upper = _constants_like(desired)
    return torch.maximum(lower, torch.minimum(upper, bounded))


def reorder_joint_targets(
    targets: torch.Tensor,
    target_joint_order: Sequence[str],
) -> torch.Tensor:
    """Reorder canonical targets by explicit names, failing on mismatch."""

    _require_last_dim(targets, 16, "targets")
    order = tuple(target_joint_order)
    if len(order) != 16 or len(set(order)) != 16:
        raise ValueError("target_joint_order must contain 16 unique names")
    missing = set(LEFT_ACTIVE_JOINT_ORDER) - set(order)
    extra = set(order) - set(LEFT_ACTIVE_JOINT_ORDER)
    if missing or extra:
        raise ValueError(f"joint-order mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    indices = [LEFT_ACTIVE_JOINT_ORDER.index(name) for name in order]
    return targets[..., indices]


__all__ = [
    "COARSE_INTERIOR_MARGIN_FRACTION",
    "DEFAULT_RESIDUAL_SPAN_FRACTION",
    "DIGIT_RESIDUAL_ORDER",
    "LEFT_ACTIVE_JOINT_ORDER",
    "LEFT_SAFE_LOWER_RAD",
    "LEFT_SAFE_UPPER_RAD",
    "LEFT_SAFE_VELOCITY_RAD_S",
    "MAPPING_ID",
    "SOURCE_LEFT_CALIBRATION",
    "SOURCE_LINKAGENT_MAPPER_SHA256",
    "SYNERGY_ORDER",
    "VALIDATION_STATUS",
    "effective_digit_closure",
    "hybrid_action_to_joint_targets",
    "joint_targets_to_digit_closure",
    "rate_limit_joint_targets",
    "reorder_joint_targets",
    "synergy_to_joint_targets",
]
