"""Plan and explicitly launch the isolated Isaac/RSL-RL training process."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import netrc
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .env_manager import PreflightReport, preflight
from .tomato_pick.config import (
    REPOSITORY_ROOT,
    load_isaac_runtime_profile,
    load_object,
    load_ppo_profile,
)


@dataclass(frozen=True, slots=True)
class TrainingLaunchPlan:
    """Resolved, reviewable training command; an empty command is not runnable."""

    schema_version: int
    ready: bool
    task_id: str
    trainer_backend: str
    command: tuple[str, ...]
    working_directory: str
    runtime_environment: str
    output_root: str
    deployable_root: str
    monitoring_provider: str
    monitoring_entity: str
    monitoring_project: str
    monitoring_authenticated: bool
    blockers: tuple[str, ...]
    preflight: PreflightReport

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_plan(
    *,
    isaaclab_root: Path | None = None,
    num_environments: int | None = None,
    seed: int | None = None,
    maximum_iterations: int | None = None,
    resume_run: str | None = None,
    resume_checkpoint: str | None = None,
    headless: bool = True,
) -> TrainingLaunchPlan:
    """Resolve a training command after every C2 activation gate passes."""

    profile = load_ppo_profile()
    trainer = profile.get("trainer")
    output = profile.get("output")
    monitoring = profile.get("monitoring")
    if not all(isinstance(value, Mapping) for value in (trainer, output, monitoring)):
        raise ValueError("PPO profile requires trainer, monitoring, and output objects")
    assert isinstance(trainer, Mapping)
    assert isinstance(output, Mapping)
    assert isinstance(monitoring, Mapping)
    if monitoring.get("provider") != "wandb" or monitoring.get("mode") != "online":
        raise ValueError("PPO V1 requires online W&B monitoring")
    resolved_envs = trainer["num_environments"] if num_environments is None else num_environments
    resolved_seed = trainer["seed"] if seed is None else seed
    resolved_iterations = (
        trainer["maximum_iterations"] if maximum_iterations is None else maximum_iterations
    )
    for field_name, value in (
        ("num_environments", resolved_envs),
        ("seed", resolved_seed),
        ("maximum_iterations", resolved_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be an integer >= 0")
    if resolved_envs == 0:
        raise ValueError("num_environments must be > 0")
    if resolved_iterations == 0:
        raise ValueError("maximum_iterations must be > 0")
    if (resume_run is None) != (resume_checkpoint is None):
        raise ValueError("resume_run and resume_checkpoint must be provided together")
    if resume_run is not None:
        for field_name, value in (
            ("resume_run", resume_run),
            ("resume_checkpoint", resume_checkpoint),
        ):
            assert value is not None
            if Path(value).name != value or value in {".", ".."}:
                raise ValueError(f"{field_name} must be one file or directory name")

    configured_root = isaaclab_root
    if configured_root is None:
        root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
        configured_root = Path(root_value).expanduser() if root_value else None
    report = preflight("isaac_ppo", isaaclab_root=configured_root)
    runtime_profile = load_isaac_runtime_profile()
    primary = runtime_profile["primary"]
    runtime_environment = (
        configured_root.resolve() / "env_isaaclab"
        if configured_root is not None
        else Path(str(primary["virtual_environment"])).resolve()
    )
    managed_python = runtime_environment / "bin" / "python"
    credential_name = str(monitoring["credential_environment_variable"])
    monitoring_authenticated = bool(os.environ.get(credential_name)) or _wandb_netrc_login()
    blockers = list(report.blockers)
    if resume_run is not None and resume_checkpoint is not None:
        resume_path = REPOSITORY_ROOT / str(output["root"]) / resume_run / resume_checkpoint
        if not resume_path.is_file():
            blockers.append(f"resume checkpoint not found: {resume_path}")
    if not monitoring_authenticated:
        blockers.append(
            f"W&B login required: set {credential_name} or run wandb login in the managed runtime"
        )
    command: tuple[str, ...] = ()
    ready = report.ready and not blockers
    if ready:
        if configured_root is None:  # Preflight and this branch must agree.
            raise RuntimeError("Isaac Lab root disappeared after preflight")
        runtime_manifest = load_object(
            REPOSITORY_ROOT / "simulation/environments/runtime_manifest.yaml"
        )
        role = runtime_manifest["roles"]["isaac_ppo"]
        root = configured_root.resolve()
        train_script = root / role["upstream_train_relative_path"]
        command_items = [
            str(managed_python),
            str(train_script),
            "--rl_library",
            "rsl_rl",
            "--external_callback",
            "simulation.tomato_pick.isaac.register_task",
            "--task",
            str(profile["task_id"]),
            "--num_envs",
            str(resolved_envs),
            "--seed",
            str(resolved_seed),
            "--max_iterations",
            str(resolved_iterations),
            "--logger",
            str(monitoring["provider"]),
            "--log_project_name",
            str(monitoring["project"]),
            "--run_name",
            str(monitoring["run_name"]),
        ]
        if headless:
            command_items.extend(role["headless_arguments"])
        else:
            command_items.extend(role["render_arguments"])
        if resume_run is not None and resume_checkpoint is not None:
            command_items.extend(
                (
                    "--resume",
                    "--load_run",
                    f"^{re.escape(resume_run)}$",
                    "--checkpoint",
                    f"^{re.escape(resume_checkpoint)}$",
                )
            )
        command = tuple(command_items)
    return TrainingLaunchPlan(
        schema_version=1,
        ready=ready,
        task_id=str(profile["task_id"]),
        trainer_backend=str(trainer["backend"]),
        command=command,
        working_directory=str(REPOSITORY_ROOT),
        runtime_environment=str(runtime_environment),
        output_root=str(output["root"]),
        deployable_root=str(output["deployable_root"]),
        monitoring_provider=str(monitoring["provider"]),
        monitoring_entity=str(monitoring["entity"]),
        monitoring_project=str(monitoring["project"]),
        monitoring_authenticated=monitoring_authenticated,
        blockers=tuple(blockers),
        preflight=report,
    )


def execute_training(plan: TrainingLaunchPlan, *, explicitly_confirmed: bool) -> int:
    """Execute an already-reviewed plan without a shell or environment install."""

    if not isinstance(plan, TrainingLaunchPlan):
        raise TypeError("plan must be a TrainingLaunchPlan")
    if not explicitly_confirmed:
        raise PermissionError("training execution requires explicit confirmation")
    if not plan.ready or not plan.command:
        raise RuntimeError("training plan is blocked; resolve preflight blockers first")
    environment = dict(os.environ)
    for secret_name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(secret_name, None)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        plan.working_directory
        if not existing_python_path
        else os.pathsep.join((plan.working_directory, existing_python_path))
    )
    environment["VIRTUAL_ENV"] = plan.runtime_environment
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (
            str(Path(plan.runtime_environment) / "bin"),
            environment.get("PATH", ""),
        )
    )
    environment["WANDB_MODE"] = "online"
    environment["WANDB_ENTITY"] = plan.monitoring_entity
    completed = subprocess.run(
        list(plan.command),
        cwd=plan.working_directory,
        env=environment,
        check=False,
    )
    return completed.returncode


def _wandb_netrc_login() -> bool:
    """Return only credential presence; never expose or copy the key."""

    try:
        authentication = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(authentication and authentication[2])


__all__ = ["TrainingLaunchPlan", "build_training_plan", "execute_training"]
