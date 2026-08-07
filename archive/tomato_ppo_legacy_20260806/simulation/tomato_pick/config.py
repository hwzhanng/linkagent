"""Load and validate C2 configuration without optional runtime dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from module.skills.tomato_pick import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    INTERFACE_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT_ID,
    PHASE_ORDER,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
C2_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "botSDK/urdf/02_project_derived/simulation_baselines/tomato_pick_v1.yaml"
)
ASSET_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/assets_v1.yaml"
PPO_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/ppo_v1.yaml"
SIM2REAL_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/sim2real_v1.yaml"
RUNTIME_MANIFEST_PATH = (
    REPOSITORY_ROOT / "simulation/environments/runtime_manifest.yaml"
)
ISAAC_RUNTIME_PROFILE_PATH = (
    REPOSITORY_ROOT / "simulation/environments/isaaclab_runtime_v1.yaml"
)


class SimulationConfigError(ValueError):
    """A machine-readable simulation configuration is inconsistent."""


def load_object(path: Path) -> dict[str, Any]:
    """Read a JSON-compatible YAML object using the standard library."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimulationConfigError(f"configuration file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SimulationConfigError(
            f"configuration must use JSON-compatible YAML: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SimulationConfigError(f"configuration root must be an object: {path}")
    return value


def load_c2_profile(path: Path = C2_PROFILE_PATH) -> dict[str, Any]:
    """Load the C2-0 contract and reject interface drift."""

    profile = load_object(path)
    policy = profile.get("policy")
    environment = profile.get("environment")
    if not isinstance(policy, dict) or not isinstance(environment, dict):
        raise SimulationConfigError("C2 profile requires policy and environment objects")
    expected = {
        "interface_version": INTERFACE_VERSION,
        "observation_layout_id": OBSERVATION_LAYOUT_ID,
        "observation_dim": OBSERVATION_DIM,
        "action_layout_id": ACTION_LAYOUT_ID,
        "action_dim": ACTION_DIM,
        "policy_rate_hz": 20,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise SimulationConfigError(
                f"C2 policy.{key} must be {value!r}, got {policy.get(key)!r}"
            )
    if tuple(policy.get("phases", ())) != PHASE_ORDER:
        raise SimulationConfigError(f"C2 policy.phases must be {PHASE_ORDER!r}")
    physics_rate = environment.get("physics_rate_hz")
    decimation = environment.get("policy_decimation")
    if (
        isinstance(physics_rate, bool)
        or not isinstance(physics_rate, int)
        or isinstance(decimation, bool)
        or not isinstance(decimation, int)
        or decimation <= 0
        or physics_rate // decimation != 20
    ):
        raise SimulationConfigError("physics rate and decimation must produce 20 Hz")
    return profile


def load_asset_profile(path: Path = ASSET_PROFILE_PATH) -> dict[str, Any]:
    """Load the separately reviewable robot/task asset selection."""

    return load_object(path)


def load_ppo_profile(path: Path = PPO_PROFILE_PATH) -> dict[str, Any]:
    """Load the versioned PPO hyperparameter and compatibility profile."""

    return load_object(path)


def load_sim2real_profile(path: Path = SIM2REAL_PROFILE_PATH) -> dict[str, Any]:
    """Load and validate the single deployable measurement/randomization profile."""

    profile = load_object(path)
    interface = profile.get("policy_interface")
    perception = profile.get("perception_bridge")
    tomato = profile.get("correlated_tomato_sampling")
    control = profile.get("robot_and_control_randomization")
    tactile = profile.get("tactile_measurement_model")
    contact = profile.get("contact_and_stem_randomization")
    if not all(
        isinstance(item, dict)
        for item in (interface, perception, tomato, control, tactile, contact)
    ):
        raise SimulationConfigError(
            "sim2real profile is missing a required runtime section"
        )
    expected = {
        "observation_layout_id": OBSERVATION_LAYOUT_ID,
        "action_layout_id": ACTION_LAYOUT_ID,
        "policy_rate_hz": 20,
    }
    for key, value in expected.items():
        if interface.get(key) != value:
            raise SimulationConfigError(
                f"sim2real policy_interface.{key} must be {value!r}"
            )
    if interface.get("actor_privileged_state_allowed") is not False:
        raise SimulationConfigError("sim2real actor must not receive privileged state")
    if perception.get("raw_point_cloud_in_actor") is not False:
        raise SimulationConfigError("raw point clouds are outside the actor contract")
    if perception.get("exact_sim_pose_in_actor") is not False:
        raise SimulationConfigError("exact simulation pose must not enter the actor")
    if tomato.get("do_not_sample_mass_independently") is not True:
        raise SimulationConfigError("tomato mass must stay correlated with ellipsoid volume")
    if tactile.get("actor_channels") != [
        "normal_pressure_norm",
        "normal_pressure_rate_norm",
    ]:
        raise SimulationConfigError(
            "actor tactile channels must be pressure and pressure-rate only"
        )
    if tactile.get("contact_threshold_n") != 0.60:
        raise SimulationConfigError("tactile contact threshold must remain 0.60 N")
    randomization_policy = profile.get("training_randomization_policy")
    if not isinstance(randomization_policy, dict) or randomization_policy.get("mode") != (
        "full_target_distribution_from_first_rollout"
    ):
        raise SimulationConfigError("training must use the full target distribution from rollout one")
    if randomization_policy.get("curriculum_or_later_patch") is not False:
        raise SimulationConfigError("sim2real curriculum/later patching is disabled")
    return profile


def load_isaac_runtime_profile(
    path: Path = ISAAC_RUNTIME_PROFILE_PATH,
) -> dict[str, Any]:
    """Load the machine-local, smoke-tested Isaac runtime record."""

    return load_object(path)


__all__ = [
    "ASSET_PROFILE_PATH",
    "C2_PROFILE_PATH",
    "ISAAC_RUNTIME_PROFILE_PATH",
    "PPO_PROFILE_PATH",
    "SIM2REAL_PROFILE_PATH",
    "REPOSITORY_ROOT",
    "RUNTIME_MANIFEST_PATH",
    "SimulationConfigError",
    "load_asset_profile",
    "load_c2_profile",
    "load_isaac_runtime_profile",
    "load_object",
    "load_ppo_profile",
    "load_sim2real_profile",
]
