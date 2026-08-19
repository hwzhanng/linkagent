"""Contracts for the second palm-supported enclosure experiment.

This module has no Isaac Lab dependency.  Keeping the task contract here makes
it possible to validate observation/action names and phase semantics in CPU
unit tests before starting Isaac Sim.
"""

from __future__ import annotations

from types import MappingProxyType

V2_CONTRACT_VERSION = "palm_enveloping_grasp_v2.0"
V2_ACTION_DIM = 14
V2_ACTOR_OBSERVATION_DIM = 128
V2_CRITIC_STATE_DIM = 177

# The first ten entries preserve the validated coarse hand manifold.  The
# final four are independent distal residuals for index/middle/ring/little;
# they are bounded residuals, not direct joint positions.  V2 gives these
# residuals their own rate limit in the environment adapter.
ACTION_ORDER = (
    "global_finger_flexion",
    "proximal_distal_curl_shape",
    "radial_ulnar_flexion_balance",
    "finger_spread",
    "thumb_opposition",
    "thumb_residual",
    "index_residual",
    "middle_residual",
    "ring_residual",
    "little_residual",
    "index_distal_residual",
    "middle_distal_residual",
    "ring_distal_residual",
    "little_distal_residual",
)

OBSERVATION_BLOCKS = MappingProxyType(
    {
        "fruit_center_in_palm": slice(0, 3),
        "fruit_semi_axes": slice(3, 6),
        "fruit_orientation_6d": slice(6, 12),
        "fruit_twist_in_palm": slice(12, 18),
        "hand_joint_position": slice(18, 34),
        "hand_joint_velocity": slice(34, 50),
        "tactile_per_digit": slice(50, 85),  # five digits x seven measured features
        "palm_tactile": slice(85, 90),
        "contact_geometry": slice(90, 100),
        "phase_state": slice(100, 108),
        "previous_action": slice(108, 122),
        "safety_margin": slice(122, 128),
    }
)

if len(ACTION_ORDER) != V2_ACTION_DIM:
    raise AssertionError("v2 action layout arithmetic drifted")
if OBSERVATION_BLOCKS["safety_margin"].stop != V2_ACTOR_OBSERVATION_DIM:
    raise AssertionError("v2 observation layout arithmetic drifted")

PHASES = ("approach", "palm_support", "coarse_enclosure", "distal_enclosure", "retention", "pick")
PROFILE_NAMES = (
    "v1_training_small",
    "v1_nominal_reliable",
    "v2_band1_match",
    "v2_band2_shape",
    "v2_band3_sensor_actuator",
    "v2_band4_stress",
)
