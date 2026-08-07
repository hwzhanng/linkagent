"""Read-only environment preflight for isolated C2 runtime roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
from typing import Any

from module.skills.tomato_pick import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    CRITIC_STATE_DIM,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT_ID,
)

from .tomato_pick.config import (
    ASSET_PROFILE_PATH,
    C2_PROFILE_PATH,
    PPO_PROFILE_PATH,
    REPOSITORY_ROOT,
    RUNTIME_MANIFEST_PATH,
    load_asset_profile,
    load_c2_profile,
    load_isaac_runtime_profile,
    load_object,
    load_ppo_profile,
)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One machine-checkable runtime prerequisite."""

    check_id: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Read-only result; preflight never installs or activates anything."""

    schema_version: int
    role: str
    ready: bool
    checks: tuple[PreflightCheck, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight(
    role: str,
    *,
    isaaclab_root: Path | None = None,
) -> PreflightReport:
    """Inspect one role without mutating Conda, Pip, files, GPU, or devices."""

    manifest = load_object(RUNTIME_MANIFEST_PATH)
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or role not in roles:
        choices = ", ".join(sorted(roles)) if isinstance(roles, dict) else "<none>"
        raise ValueError(f"unknown runtime role {role!r}; choices: {choices}")
    checks: list[PreflightCheck] = []
    checks.append(
        PreflightCheck(
            "python_supported",
            sys.version_info >= (3, 11),
            f"current={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    for check_id, path in (
        ("c2_contract_present", C2_PROFILE_PATH),
        ("asset_profile_present", ASSET_PROFILE_PATH),
        ("ppo_profile_present", PPO_PROFILE_PATH),
    ):
        checks.append(PreflightCheck(check_id, path.is_file(), str(path)))
    load_c2_profile()
    asset_profile = load_asset_profile()
    ppo_profile = load_ppo_profile()
    isaac_runtime_profile = load_isaac_runtime_profile()
    compatibility = ppo_profile.get("compatibility", {})
    checks.append(
        PreflightCheck(
            "frozen_dimensions_match",
            compatibility.get("observation_layout_id")
            == OBSERVATION_LAYOUT_ID
            and compatibility.get("action_layout_id") == ACTION_LAYOUT_ID
            and compatibility.get("policy_rate_hz") == 20,
            (
                f"expected actor={OBSERVATION_DIM}D, critic={CRITIC_STATE_DIM}D, "
                f"action={ACTION_DIM}D at 20 Hz"
            ),
        )
    )

    if role == "agent_core":
        checks.append(
            PreflightCheck(
                "simulation_not_loaded",
                "isaaclab" not in sys.modules and "torch" not in sys.modules,
                "agent core must not preload Isaac Lab or Torch",
            )
        )
    elif role == "contract_smoke":
        checks.append(
            PreflightCheck(
                "contract_backend_allowed",
                asset_profile.get("interface", {}).get("actor_observation_shape")
                == [OBSERVATION_DIM]
                and asset_profile.get("interface", {}).get("critic_observation_shape")
                == [CRITIC_STATE_DIM]
                and asset_profile.get("interface", {}).get("action_shape") == [ACTION_DIM],
                "minimal interface dimensions must match the frozen contract",
            )
        )
    elif role == "isaac_ppo":
        role_config = roles[role]
        runtime_primary = isaac_runtime_profile.get("primary", {})
        runtime_verification = isaac_runtime_profile.get("verification", {})
        checks.append(
            PreflightCheck(
                "managed_runtime_smoke_passed",
                isaac_runtime_profile.get("status") == "installed_and_smoke_tested"
                and runtime_verification.get("app_launcher_headless_passed") is True
                and runtime_verification.get("cuda_physx_steps_passed") is True
                and runtime_verification.get("formal_robot_usd_passed") is True,
                str(isaac_runtime_profile.get("status")),
            )
        )
        env_name = role_config["root_environment_variable"]
        configured = isaaclab_root
        if configured is None:
            raw = os.environ.get(env_name)
            configured = Path(raw).expanduser() if raw else None
        checks.append(
            PreflightCheck(
                "isaaclab_root_configured",
                configured is not None,
                f"pass --isaaclab-root or set {env_name}",
            )
        )
        if configured is not None:
            root = configured.resolve()
            managed_environment = root / "env_isaaclab"
            launcher = root / role_config["launcher_relative_path"]
            train_script = root / role_config["upstream_train_relative_path"]
            managed_cli = managed_environment / "bin" / "isaaclab"
            checks.extend(
                (
                    PreflightCheck(
                        "isaaclab_runtime_environment_present",
                        managed_environment.is_dir(),
                        str(managed_environment),
                    ),
                    PreflightCheck("isaaclab_launcher_present", launcher.is_file(), str(launcher)),
                    PreflightCheck(
                        "managed_training_cli_present",
                        managed_cli.is_file(),
                        str(managed_cli),
                    ),
                    PreflightCheck(
                        "training_dispatcher_present", train_script.is_file(), str(train_script)
                    ),
                )
            )
        checks.extend(
            (
                PreflightCheck(
                    "task_assets_ready",
                    bool(asset_profile.get("gates", {}).get("physics")),
                    "the new 11D environment must pass its physical reset/control gate",
                ),
                PreflightCheck(
                    "ppo_gate_open",
                    bool(asset_profile.get("ready_for_formal_training"))
                    and bool(asset_profile.get("gates", {}).get("ppo_numeric_probe"))
                    and ppo_profile.get("status")
                    in {
                        "ready_waiting_for_explicit_training_authorization",
                        "formal_training_active",
                    },
                    "unit, physics and short numerical gates must pass before PPO training",
                ),
                PreflightCheck(
                    "isaac_adapter_implemented",
                    role_config.get("current_adapter_status") == "implemented_and_tested",
                    str(role_config.get("current_adapter_status")),
                ),
                PreflightCheck(
                    "wandb_runtime_ready",
                    runtime_primary.get("wandb") == "0.28.1"
                    and runtime_verification.get("wandb_import_passed") is True
                    and runtime_verification.get("wandb_login_verified") is True,
                    "managed runtime must import W&B and have a verified user login",
                ),
            )
        )

    blockers = tuple(check.detail for check in checks if not check.passed)
    return PreflightReport(
        schema_version=1,
        role=role,
        ready=not blockers,
        checks=tuple(checks),
        blockers=blockers,
    )


__all__ = ["PreflightCheck", "PreflightReport", "preflight"]
