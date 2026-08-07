"""Load the single active tomato-picking training profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from module.skills.tomato_pick import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    CRITIC_STATE_DIM,
    INTERFACE_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT_ID,
    POLICY_RATE_HZ,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
C2_PROFILE_PATH = REPOSITORY_ROOT / "botSDK/urdf/02_project_derived/simulation_baselines/tomato_pick_v1.yaml"
ASSET_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/assets_v1.yaml"
PPO_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/ppo_v1.yaml"
SIM2REAL_PROFILE_PATH = REPOSITORY_ROOT / "simulation/config/sim2real_v1.yaml"
RUNTIME_MANIFEST_PATH = REPOSITORY_ROOT / "simulation/environments/runtime_manifest.yaml"
ISAAC_RUNTIME_PROFILE_PATH = REPOSITORY_ROOT / "simulation/environments/isaaclab_runtime_v1.yaml"


class SimulationConfigError(ValueError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SimulationConfigError(f"invalid configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimulationConfigError(f"configuration root must be an object: {path}")
    return value


def _validate_interface(interface: object, *, label: str) -> dict[str, Any]:
    if not isinstance(interface, dict):
        raise SimulationConfigError(f"{label} must be an object")
    expected = {
        "interface_version": INTERFACE_VERSION,
        "observation_layout_id": OBSERVATION_LAYOUT_ID,
        "observation_dim": OBSERVATION_DIM,
        "action_layout_id": ACTION_LAYOUT_ID,
        "action_dim": ACTION_DIM,
        "critic_state_dim": CRITIC_STATE_DIM,
        "policy_rate_hz": POLICY_RATE_HZ,
    }
    for key, expected_value in expected.items():
        if interface.get(key) != expected_value:
            raise SimulationConfigError(
                f"{label}.{key} must be {expected_value!r}, got {interface.get(key)!r}"
            )
    return interface


def load_c2_profile(path: Path = C2_PROFILE_PATH) -> dict[str, Any]:
    profile = load_object(path)
    _validate_interface(profile.get("policy"), label="policy")
    return profile


def load_asset_profile(path: Path = ASSET_PROFILE_PATH) -> dict[str, Any]:
    return load_object(path)


def load_ppo_profile(path: Path = PPO_PROFILE_PATH) -> dict[str, Any]:
    profile = load_object(path)
    _validate_interface(profile.get("compatibility"), label="compatibility")
    return profile


def load_sim2real_profile(path: Path = SIM2REAL_PROFILE_PATH) -> dict[str, Any]:
    profile = load_object(path)
    _validate_interface(profile.get("policy_interface"), label="policy_interface")
    if profile["policy_interface"].get("actor_privileged_state_allowed") is not False:
        raise SimulationConfigError("Actor privileged state must remain disabled")
    return profile


def load_isaac_runtime_profile(path: Path = ISAAC_RUNTIME_PROFILE_PATH) -> dict[str, Any]:
    return load_object(path)


__all__ = [
    "ASSET_PROFILE_PATH",
    "C2_PROFILE_PATH",
    "ISAAC_RUNTIME_PROFILE_PATH",
    "PPO_PROFILE_PATH",
    "REPOSITORY_ROOT",
    "RUNTIME_MANIFEST_PATH",
    "SIM2REAL_PROFILE_PATH",
    "SimulationConfigError",
    "load_asset_profile",
    "load_c2_profile",
    "load_isaac_runtime_profile",
    "load_object",
    "load_ppo_profile",
    "load_sim2real_profile",
]
