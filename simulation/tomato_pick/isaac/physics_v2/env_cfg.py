"""Step-1 Physics-v2 configuration with no physical override."""

from __future__ import annotations

import copy
import math

from isaaclab.utils.configclass import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from simulation.tomato_pick.isaac.physics_v1.env_cfg import (
    TomatoPickPhysicsV1EventCfg,
    TomatoPickPhysicsV1EnvCfg,
)
from simulation.tomato_pick.isaac.env_cfg import RIGHT_FINGERTIP_PATHS
from simulation.tomato_pick.physics_v2 import PHYSICS_V2_ROBOT_USD_PATH

from .assets import author_correlated_fruit_profiles_v2


_PHYSICS_V1_CFG = TomatoPickPhysicsV1EnvCfg()
_TIP_SUFFIX = {
    "thumb": "/right_hand_thumb_link5",
    "index": "/right_hand_index_link4",
    "middle": "/right_hand_middle_link4",
    "ring": "/right_hand_ring_link4",
    "little": "/right_hand_little_link4",
}


def _physics_v2_fingertip_contacts():
    sensors = copy.deepcopy(_PHYSICS_V1_CFG.fingertip_contacts)
    for finger_name, sensor in sensors.items():
        sensor.prim_path = RIGHT_FINGERTIP_PATHS[finger_name] + _TIP_SUFFIX[
            finger_name
        ]
    return sensors


@configclass
class TomatoPickPhysicsV2EventCfg(TomatoPickPhysicsV1EventCfg):
    fruit_profile = EventTerm(
        func=author_correlated_fruit_profiles_v2,
        mode="prestartup",
        params={"asset_name": "tomato"},
    )


@configclass
class TomatoPickPhysicsV2EnvCfg(TomatoPickPhysicsV1EnvCfg):
    """Initial near-nominal Physics-v2 contact curriculum."""

    robot = copy.deepcopy(_PHYSICS_V1_CFG.robot)
    robot.spawn.usd_path = str(PHYSICS_V2_ROBOT_USD_PATH)
    fingertip_contacts = _physics_v2_fingertip_contacts()
    events = TomatoPickPhysicsV2EventCfg()
    action_delay_steps = (0, 2)
    # Re-evaluate the inherited hand-viewer center on the formal USD using the
    # Sim 6 xyzw convention and the true distal fingertip bodies.  The v2
    # candidate preserves its x/y location while correcting the depth so a
    # nominal four-finger enclosure produces low-force physical contact.  It
    # remains simulation-derived until hardware calibration is available.
    closed_grasp_fruit_center_m = (0.13834628, -0.06018020, 0.00303458)
    # Start the first numerical probe near the nominal A7 pose.  Broad arm and
    # stem-orientation randomization is a later curriculum stage, not a
    # prerequisite for demonstrating a reachable contact-learning signal.
    reset_arm_joint_ellipsoid_radius_rad = (math.radians(1.0),) * 7
    reset_arm_correlated_modes_rad = ()
    reset_arm_correlated_mode_coefficient_ranges = ()
    reset_fruit_stem_tilt_rad_max = math.radians(5.0)
    reset_fruit_approach_yaw_rad = (
        math.radians(-5.0),
        math.radians(5.0),
    )
    reset_fruit_approach_pitch_rad = (
        math.radians(-5.0),
        math.radians(5.0),
    )
    maximum_closure_translation_norm_m = 0.20
    minimum_closure_translation_headroom_fraction = 0.05
    open_hand_before_workspace_reset = False


__all__ = ["TomatoPickPhysicsV2EnvCfg", "TomatoPickPhysicsV2EventCfg"]
