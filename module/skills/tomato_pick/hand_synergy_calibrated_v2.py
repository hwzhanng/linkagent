"""User-calibrated, continuous five-dimensional G20 grasp manifold v2.

The eleven joint-space anchors were captured in the hand-only URDF viewer.
Each normalized coordinate interpolates independently from the captured center
to its negative or positive endpoint.  Summed multi-axis commands are clipped
only at the physical joint limits.  The frozen Clean-v5 mapper remains intact.
"""

from __future__ import annotations

from typing import Sequence

from .hand_synergy import HAND_LOWER_RAD, HAND_UPPER_RAD


MAPPING_ID = "tomato_g20_user_calibrated_5d_v2_1_collision_safe"
SOURCE_DRAFT_SHA256 = "75d8122975dac780110505596419a1a98ed13bb3a6e06c3c5206f844e2fb30a7"
THUMB_ACTIVE_JOINT_INDICES = (0, 5, 10, 11)
FINGER_ACTIVE_JOINT_INDICES = tuple(
    index for index in range(16) if index not in THUMB_ACTIVE_JOINT_INDICES
)

CENTER_RAD = (
    0.135882669, 0.262019600, 0.271784300, 0.271784300,
    0.262019600, 0.093000000, -0.087254910, -0.016388526,
    0.013151839, 0.073459640, -1.300000000, 0.354264588,
    0.222666900, 0.222666900, 0.218667000, 0.222666900,
)

NEGATIVE_ENDPOINTS_RAD = (
    (0.074282718, 0.032549000, 0.032549000, 0.032549000, 0.032549000, 0.156000000, -0.012745110, 0.003337754, -0.006859971, -0.007779560, -1.300000000, 0.184864466, 0.026667000, 0.026667000, 0.026667000, 0.026667000),
    (0.161082648, 0.642022914, 0.625501666, 0.556027734, 0.493937886, 0.115000000, 0.043137240, 0.018132464, -0.021868828, -0.068708960, -1.300000000, 0.423564637, 0.026667000, 0.026667000, 0.026667000, 0.026667000),
    (0.345882500, 0.416223843, 0.540620981, 0.717463715, 0.793167549, 0.062000000, 0.043137240, 0.018132464, -0.021868828, -0.068708960, -1.300000000, 0.931765000, 0.400112143, 0.488651831, 0.621150200, 0.722074645),
    (0.219882601, 0.990512265, 1.038995958, 1.051627582, 1.017436815, -0.012000000, 0.141145080, 0.047353488, -0.051318276, -0.172100320, -1.300000000, 0.585264752, 0.759967959, 0.766292223, 0.761335800, 0.782965281),
    (0.016471000, 0.032549000, 0.032549000, 0.032549000, 0.032549000, -0.021000000, 0.024509790, 0.013200894, -0.016865876, -0.048399160, -1.300000000, 0.025882000, 0.492885429, 0.486561165, 0.467038200, 0.469888107),
)

POSITIVE_ENDPOINTS_RAD = (
    (0.345882500, 1.562353000, 1.627451000, 1.627451000, 1.562353000, 0.129000000, 0.132549000, 0.041804000, -0.045883000, -0.166196000, -1.297000000, 0.931765000, 1.333333000, 1.333333000, 1.306667000, 1.333333000),
    (0.222682599, 0.292615680, 0.110699198, 0.032549000, 0.032549000, 0.046000000, 0.043137240, 0.018132464, -0.021868828, -0.068708960, -1.300000000, 0.592964758, 0.902133220, 0.744026634, 0.475947000, 0.327200180),
    (0.298282538, 0.772974136, 0.611498426, 0.295707830, 0.099860376, 0.061000000, 0.072941160, 0.026022976, -0.029873552, -0.101204640, -1.300000000, 0.800864906, 1.286293024, 1.128186438, 0.852267000, 0.711359984),
    (0.133082671, 0.262019600, 0.271784300, 0.271784300, 0.262019600, 0.052000000, -0.240000000, -0.056827400, 0.054176050, 0.240000000, -1.300000000, 0.346564582, 0.222666900, 0.222666900, 0.218667000, 0.222666900),
    (0.345882500, 0.148814104, 0.153761552, 0.153761552, 0.148814104, 0.075000000, 0.035686260, 0.016159836, -0.019867647, -0.060585040, -1.300000000, 0.931765000, 0.367706826, 0.367706826, 0.360747000, 0.367706826),
)


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def g20_synergy_to_joint_targets_v2(values: Sequence[float]) -> tuple[float, ...]:
    """Map five normalized coordinates to the calibrated sixteen-joint pose."""

    if len(values) != 5:
        raise ValueError("G20 grasp synergy must contain exactly five values")
    result = list(CENTER_RAD)
    for coordinate_index, (coordinate, negative, positive) in enumerate(
        zip(
            (_clip_unit(value) for value in values),
            NEGATIVE_ENDPOINTS_RAD,
            POSITIVE_ENDPOINTS_RAD,
            strict=True,
        )
    ):
        endpoint = positive if coordinate >= 0.0 else negative
        magnitude = abs(coordinate)
        controlled_indices = (
            THUMB_ACTIVE_JOINT_INDICES
            if coordinate_index == 4
            else FINGER_ACTIVE_JOINT_INDICES
        )
        for joint_index in controlled_indices:
            center = CENTER_RAD[joint_index]
            target = endpoint[joint_index]
            result[joint_index] += magnitude * (target - center)
    return tuple(
        max(HAND_LOWER_RAD[index], min(HAND_UPPER_RAD[index], target))
        for index, target in enumerate(result)
    )


__all__ = [
    "CENTER_RAD",
    "FINGER_ACTIVE_JOINT_INDICES",
    "MAPPING_ID",
    "NEGATIVE_ENDPOINTS_RAD",
    "POSITIVE_ENDPOINTS_RAD",
    "SOURCE_DRAFT_SHA256",
    "THUMB_ACTIVE_JOINT_INDICES",
    "g20_synergy_to_joint_targets_v2",
]
