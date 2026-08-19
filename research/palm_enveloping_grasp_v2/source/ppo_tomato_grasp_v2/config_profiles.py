"""Explicit, reproducible PhysX randomization profiles for v2.

The old evaluator called a broad stress domain "mild" and reversed the
Band-1 isotropic geometry switch.  Profiles here are named and resolved into
the saved report so an improvement or regression can be attributed to one
factor at a time.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "v1_training_small": {
        "axis_range_m": (0.032, 0.032),
        "isotropic_axes": True,
        "accepted_mass_kg": (0.050, 0.180),
        "density_range_kg_m3": (850.0, 850.0),
        "position_jitter": (0.0, 0.0, 0.0),
        "friction_range": (0.70, 0.70), "restitution_range": (0.0, 0.0),
        "stiffness_range_n_m": (18000.0, 18000.0), "tactile_gain_range": (1.0, 1.0),
        "tactile_bias_range_n": (0.0, 0.0), "sensor_delay_steps_max": 0,
        "sensor_dropout_probability": 0.0, "sensor_stale_probability": 0.0,
        "actuator_response_scale_range": (1.0, 1.0), "actuator_bias_range_rad": (0.0, 0.0),
        "actuator_deadband_range_rad": (0.0, 0.0), "actuator_backlash_range_rad": (0.0, 0.0),
    },
    # Exact deterministic reference used to check the evaluator itself.
    "v1_nominal_reliable": {
        "axis_range_m": (0.0375, 0.0375),
        "isotropic_axes": True,
        # 37.5 mm isotropic at 850 kg/m^3 is about 0.188 kg; the v1 0.180 kg
        # upper bound made the nominal profile mathematically unsamplable.
        "accepted_mass_kg": (0.050, 0.250),
        "density_range_kg_m3": (850.0, 850.0),
        "position_jitter": (0.0, 0.0, 0.0),
        "friction_range": (0.70, 0.70),
        "restitution_range": (0.0, 0.0),
        "stiffness_range_n_m": (18000.0, 18000.0),
        "tactile_gain_range": (1.0, 1.0),
        "tactile_bias_range_n": (0.0, 0.0),
        "sensor_delay_steps_max": 0,
        "sensor_dropout_probability": 0.0,
        "sensor_stale_probability": 0.0,
        "actuator_response_scale_range": (1.0, 1.0),
        "actuator_bias_range_rad": (0.0, 0.0),
        "actuator_deadband_range_rad": (0.0, 0.0),
        "actuator_backlash_range_rad": (0.0, 0.0),
    },
    # Corrected Band-1: the 35--40 mm object is isotropic, not ellipsoidal.
    "v2_band1_match": {
        "axis_range_m": (0.035, 0.040),
        "isotropic_axes": True,
        "accepted_mass_kg": (0.050, 0.250),
        "density_range_kg_m3": (750.0, 1000.0),
        "position_jitter": (0.001, 0.001, 0.0005),
        "friction_range": (0.45, 0.85),
        "restitution_range": (0.0, 0.04),
        "stiffness_range_n_m": (10000.0, 26000.0),
        "tactile_gain_range": (0.90, 1.10),
        "tactile_bias_range_n": (-0.02, 0.02),
        "sensor_delay_steps_max": 1,
        "sensor_dropout_probability": 0.005,
        "sensor_stale_probability": 0.005,
        "actuator_response_scale_range": (0.96, 1.04),
        "actuator_bias_range_rad": (-0.002, 0.002),
        "actuator_deadband_range_rad": (0.0, 0.0015),
        "actuator_backlash_range_rad": (0.0, 0.002),
    },
    "v2_band2_shape": {
        "axis_range_m": (0.035, 0.045),
        "isotropic_axes": False,
        "accepted_mass_kg": (0.050, 0.220),
        "density_range_kg_m3": (700.0, 1050.0),
        "position_jitter": (0.002, 0.002, 0.001),
        "friction_range": (0.40, 0.88),
        "restitution_range": (0.0, 0.05),
        "stiffness_range_n_m": (9000.0, 28000.0),
        "tactile_gain_range": (0.88, 1.12),
        "tactile_bias_range_n": (-0.025, 0.025),
        "sensor_delay_steps_max": 2,
        "sensor_dropout_probability": 0.01,
        "sensor_stale_probability": 0.01,
        "actuator_response_scale_range": (0.92, 1.06),
        "actuator_bias_range_rad": (-0.004, 0.004),
        "actuator_deadband_range_rad": (0.0, 0.003),
        "actuator_backlash_range_rad": (0.0, 0.004),
    },
    "v2_band3_sensor_actuator": {
        "axis_range_m": (0.035, 0.045),
        "isotropic_axes": False,
        "accepted_mass_kg": (0.050, 0.240),
        "density_range_kg_m3": (680.0, 1080.0),
        "position_jitter": (0.003, 0.003, 0.0015),
        "friction_range": (0.32, 0.90),
        "restitution_range": (0.0, 0.06),
        "stiffness_range_n_m": (7000.0, 30000.0),
        "tactile_gain_range": (0.82, 1.18),
        "tactile_bias_range_n": (-0.03, 0.03),
        "sensor_delay_steps_max": 3,
        "sensor_dropout_probability": 0.02,
        "sensor_stale_probability": 0.02,
        "actuator_response_scale_range": (0.84, 1.08),
        "actuator_bias_range_rad": (-0.008, 0.008),
        "actuator_deadband_range_rad": (0.0, 0.006),
        "actuator_backlash_range_rad": (0.0, 0.008),
    },
    "v2_band4_stress": {
        "axis_range_m": (0.038, 0.050),
        "isotropic_axes": False,
        "accepted_mass_kg": (0.020, 1.20),
        "density_range_kg_m3": (650.0, 1150.0),
        "position_jitter": (0.004, 0.004, 0.002),
        "friction_range": (0.25, 0.90),
        "restitution_range": (0.0, 0.06),
        "stiffness_range_n_m": (2500.0, 28000.0),
        "tactile_gain_range": (0.82, 1.18),
        "tactile_bias_range_n": (-0.03, 0.03),
        "sensor_delay_steps_max": 3,
        "sensor_dropout_probability": 0.02,
        "sensor_stale_probability": 0.02,
        "actuator_response_scale_range": (0.72, 1.08),
        "actuator_bias_range_rad": (-0.012, 0.012),
        "actuator_deadband_range_rad": (0.0, 0.010),
        "actuator_backlash_range_rad": (0.0, 0.015),
    },
}

# Matched P2 ablations.  Each profile adds exactly one family relative to the
# previous one so a degradation can be attributed rather than hidden inside a
# broad ``random`` label.
_p2_core = PROFILE_DEFINITIONS["v1_nominal_reliable"]
PROFILE_DEFINITIONS.update({
    "p2_e0_core": deepcopy(_p2_core),
    "p2_e1_size": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.040)},
    "p2_e2_position": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.040), "position_jitter": (0.001, 0.001, 0.0005)},
    "p2_e3_shape": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.045), "isotropic_axes": False, "accepted_mass_kg": (0.05, 0.50), "position_jitter": (0.001, 0.001, 0.0005)},
    "p2_e4_material": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.045), "isotropic_axes": False, "accepted_mass_kg": (0.05, 0.50), "position_jitter": (0.001, 0.001, 0.0005), "density_range_kg_m3": (700.0, 1050.0), "friction_range": (0.45, 0.85), "restitution_range": (0.0, 0.04)},
    "p2_e5_sensor": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.045), "isotropic_axes": False, "accepted_mass_kg": (0.05, 0.50), "position_jitter": (0.001, 0.001, 0.0005), "density_range_kg_m3": (700.0, 1050.0), "friction_range": (0.45, 0.85), "restitution_range": (0.0, 0.04), "tactile_gain_range": (0.90, 1.10), "tactile_bias_range_n": (-0.02, 0.02), "sensor_delay_steps_max": 1, "sensor_dropout_probability": 0.005, "sensor_stale_probability": 0.005},
    "p2_e6_actuator": {**deepcopy(_p2_core), "axis_range_m": (0.035, 0.045), "isotropic_axes": False, "accepted_mass_kg": (0.05, 0.50), "position_jitter": (0.001, 0.001, 0.0005), "density_range_kg_m3": (700.0, 1050.0), "friction_range": (0.45, 0.85), "restitution_range": (0.0, 0.04), "tactile_gain_range": (0.90, 1.10), "tactile_bias_range_n": (-0.02, 0.02), "sensor_delay_steps_max": 1, "sensor_dropout_probability": 0.005, "sensor_stale_probability": 0.005, "actuator_response_scale_range": (0.96, 1.04), "actuator_bias_range_rad": (-0.002, 0.002), "actuator_deadband_range_rad": (0.0, 0.0015), "actuator_backlash_range_rad": (0.0, 0.002)},
    "p2_e7_stress": deepcopy(PROFILE_DEFINITIONS["v2_band4_stress"]),
})


def profile_definition(name: str) -> dict[str, Any]:
    try:
        return deepcopy(PROFILE_DEFINITIONS[name])
    except KeyError as exc:
        raise ValueError(f"unknown v2 profile {name!r}; choose from {sorted(PROFILE_DEFINITIONS)}") from exc


def apply_profile(cfg, name: str, *, num_envs: int | None = None, seed: int | None = None):
    """Apply one named profile to a fresh v1/v2-compatible config.

    The contact stiffness range is retained as a *damage/tactile prior* until
    the Isaac Lab material API is proven to update compliant stiffness at
    reset.  ``profile_resolution`` records this explicitly in reports.
    """

    values = profile_definition(name)
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    if seed is not None:
        cfg.seed = int(seed)
    geom = cfg.events["tomato_geometry"].params
    for key in ("axis_range_m", "isotropic_axes", "accepted_mass_kg", "density_range_kg_m3"):
        geom[key] = values[key]
        # Keep geometry priors first-class on v2 config as well as in the
        # prestartup event.  This makes checkpoint snapshots and explicit
        # matrix overrides reproducible instead of relying on a dynamic
        # attribute that configclass may not expose.
        if hasattr(cfg, key):
            setattr(cfg, key, values[key])
    for key, value in values.items():
        if key in geom:
            continue
        setattr(cfg, key, value)
    cfg.v2_profile_name = name
    cfg.v2_profile_values = deepcopy(values)
    cfg.v2_profile_resolution = {
        "geometry": "sampled per environment from prestartup event",
        "friction_restitution": "PhysX material properties applied at reset",
        "tactile_actuator": "sampled and applied by existing v1 reset path",
        "compliant_contact_stiffness": "damage_only_prior_pending_physx_material_reset_probe",
    }
    return cfg
