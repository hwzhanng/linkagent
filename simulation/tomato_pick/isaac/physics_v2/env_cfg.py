"""Isaac configuration for the public-prior Physics-v2 task."""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass
from module.skills.tomato_pick import ACTION_DIM, CRITIC_STATE_DIM, OBSERVATION_DIM

from simulation.tomato_pick.isaac.physics_v1.assets import (
    verify_baked_hand_physics_material,
    verify_baked_hand_self_collision_filters,
    verify_baked_passive_hand_mimics,
)
from simulation.tomato_pick.isaac.physics_v1.env_cfg import (
    TomatoPickPhysicsV1EnvCfg,
)
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile

from .assets import author_public_prior_fruit_profiles


PROFILE = load_physics_v2_profile()
ABSCISSION = PROFILE["abscission"]
ACTUATOR = PROFILE["engineering_prior_data"]["a7_l20_effective_actuator"]
PAPER_STRATEGY = PROFILE["paper_strategy_audit"]
TRAINING_CURRICULUM = PROFILE["training_curriculum"]
TRAINING_SHAPING = PROFILE["training_shaping"]

# Physics-v2 trains from scratch and has a harder dynamic fracture transition
# than the frozen 66D Clean-v5/V1 interface. Keep the released interface
# untouched and extend only this isolated task with one previous 3D perceived
# relative-position sample, current 6D controller tracking error, and the two 11D
# commands still relevant to its 0--2-step actuator-delay queue. All are
# deployable controller inputs; exact simulator fruit orientation/velocity,
# stem wrench, detachment, and sampled delay remain critic-only/hidden.
PHYSICS_V2_OBSERVATION_DIM = OBSERVATION_DIM + 3 + 6 + 6 + 2 * ACTION_DIM
PHYSICS_V2_CRITIC_STATE_DIM = CRITIC_STATE_DIM + 3 + 6 + 6 + 2 * ACTION_DIM


@configclass
class TomatoPickPhysicsV2EventCfg:
    hand_mimics = EventTerm(
        func=verify_baked_passive_hand_mimics,
        mode="prestartup",
        params={"asset_name": "robot", "hand_side": "right"},
    )
    hand_material = EventTerm(
        func=verify_baked_hand_physics_material,
        mode="prestartup",
        params={"asset_name": "robot"},
    )
    hand_self_collision_filters = EventTerm(
        func=verify_baked_hand_self_collision_filters,
        mode="prestartup",
        params={"asset_name": "robot"},
    )
    fruit_profile = EventTerm(
        func=author_public_prior_fruit_profiles,
        mode="prestartup",
        params={"asset_name": "tomato"},
    )


@configclass
class TomatoPickPhysicsV2EnvCfg(TomatoPickPhysicsV1EnvCfg):
    """Frozen V1 robot/interface plus isolated public crop mechanics."""

    events = TomatoPickPhysicsV2EventCfg()
    observation_space = PHYSICS_V2_OBSERVATION_DIM
    state_space = PHYSICS_V2_CRITIC_STATE_DIM
    physics_profile_id = PROFILE["profile_id"]
    public_prior_profile_id = PROFILE["public_prior"]["profile_id"]
    engineering_prior_profile_id = PROFILE["engineering_prior"]["profile_id"]
    action_delay_steps = tuple(ACTUATOR["action_delay_steps_uniform_inclusive"])
    maximum_stem_force_n = 180.0
    maximum_stem_torque_nm = 0.40
    stem_twist_integrity_k_per_rad = ABSCISSION["twist_integrity_k_per_rad"]
    stem_twist_integrity_q = ABSCISSION["twist_integrity_q"]
    stem_subcritical_damage_onset_ratio = ABSCISSION[
        "subcritical_damage_onset_ratio"
    ]
    stem_subcritical_damage_exponent = ABSCISSION[
        "subcritical_damage_exponent"
    ]
    stem_subcritical_time_constant_s = ABSCISSION[
        "subcritical_time_constant_s"
    ]
    stem_instantaneous_failure_ratio = ABSCISSION[
        "instantaneous_failure_ratio"
    ]
    stem_torsion_detach_integrity_threshold = ABSCISSION[
        "torsion_detach_integrity_threshold"
    ]
    stem_capacity_floor_fraction = ABSCISSION["capacity_floor_fraction"]
    paper_strategy_effective_twist_minimum_rad = PAPER_STRATEGY[
        "effective_twist_minimum_rad"
    ]
    paper_strategy_bending_utilization_minimum = PAPER_STRATEGY[
        "bending_utilization_minimum"
    ]
    paper_strategy_final_axial_pull_n = tuple(
        PAPER_STRATEGY["final_axial_pull_n"]
    )
    discovery_capacity_curriculum_enabled = (
        os.environ.get("TOMATO_V2_DISCOVERY_CAPACITY_CURRICULUM", "0") == "1"
    )
    discovery_tensile_shear_initial_scale = TRAINING_CURRICULUM[
        "initial_tensile_shear_capacity_scale"
    ]
    discovery_bending_torsion_initial_scale = TRAINING_CURRICULUM[
        "initial_bending_torsion_capacity_scale"
    ]
    discovery_capacity_hold_initial_scale_iterations = TRAINING_CURRICULUM[
        "hold_initial_scale_iterations"
    ]
    discovery_capacity_full_iteration = TRAINING_CURRICULUM[
        "full_public_prior_iteration"
    ]
    discovery_bending_torsion_requires_secure_support = TRAINING_CURRICULUM[
        "bending_torsion_assistance_requires_secure_support"
    ]
    discovery_bending_torsion_requires_kinematic_retention = (
        TRAINING_CURRICULUM[
            "bending_torsion_assistance_requires_kinematic_retention"
        ]
    )
    load_bearing_opposition_safety_factor = TRAINING_SHAPING[
        "load_bearing_opposition"
    ]["safety_factor"]


__all__ = ["TomatoPickPhysicsV2EnvCfg"]
