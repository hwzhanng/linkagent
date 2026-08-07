"""Composition root and CLI for offline or explicitly selected provider runs."""

from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import inspect
import logging
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from module.agent import MODEL_REGISTRY, TopAgent, build_model
from module.redaction import redact_text
from module.runtime import AgentRuntime, RuntimeContext
from module.tools.env import ENV_REGISTRY, BaseEnv, build_env
from module.tools.manipulation import MANIPULATION_REGISTRY
from module.tools.perception import PERCEPTION_REGISTRY
from script.utils import load_mapping_file
from task import TASK_REGISTRY, BaseTask, build_task


logger = logging.getLogger(__name__)


def build_runtime_from_file(path: str | Path) -> AgentRuntime:
    """Load configuration and assemble one single-use runtime."""

    config_path = Path(path).expanduser().resolve()
    config = load_mapping_file(config_path)
    return build_runtime(config, config_dir=config_path.parent)


def build_runtime(
    config: Mapping[str, Any],
    *,
    config_dir: str | Path,
) -> AgentRuntime:
    """Assemble Env, Task dependencies, model, TopAgent, and Runtime.

    This is the only place that knows concrete factories. If assembly fails,
    already-created resource owners are closed before the error is re-raised.
    """

    if not isinstance(config, Mapping):
        raise TypeError("configuration must be a mapping")
    _reject_unknown_keys(
        config,
        allowed={
            "models_file",
            "perception_file",
            "agent",
            "env",
            "task",
            "runtime",
        },
        label="root",
    )
    root = Path(config_dir).expanduser().resolve()
    model_profiles = _load_profiles(config, "models_file", root)
    perception_profiles = _load_profiles(config, "perception_file", root)
    env_config = _required_mapping(config, "env")
    task_config = _required_mapping(config, "task")
    agent_config = _required_mapping(config, "agent")
    runtime_config = config.get("runtime", {})
    if not isinstance(runtime_config, Mapping):
        raise TypeError("runtime config must be a mapping")

    model_config, agent_limits, max_steps = _validate_assembly_config(
        env_config=env_config,
        task_config=task_config,
        agent_config=agent_config,
        runtime_config=runtime_config,
        model_profiles=model_profiles,
        perception_profiles=perception_profiles,
    )

    env: BaseEnv | None = None
    task: BaseTask | None = None
    try:
        # Model construction validates credential environment before any Env or
        # Task owner is acquired. Provider constructors perform no network I/O.
        model = build_model(model_config)
        env = build_env(env_config)
        task = build_task(
            task_config,
            perception_profiles=perception_profiles,
        )
        context = RuntimeContext(env)
        tools = task.build_tools(context)
        agent = TopAgent(
            system_prompt=task.build_prompt(),
            model=model,
            tools=tools,
            **agent_limits,
        )
        return AgentRuntime(
            context=context,
            task=task,
            agent=agent,
            max_steps=max_steps,
        )
    except BaseException as primary_error:
        errors: list[BaseException] = [primary_error]
        if task is not None:
            try:
                task.close()
            except BaseException as exc:
                errors.append(exc)
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) > 1:
            raise BaseExceptionGroup("composition and cleanup failed", errors) from None
        raise


def configure_logging(
    log_dir: str | Path,
    *,
    verbose: bool = False,
    color: bool = True,
) -> Path:
    """Configure console/file logging and return the created log path."""

    directory = Path(log_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log_path = directory / f"run_{timestamp}_pid{os.getpid()}.log"

    root_logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(_ConsoleFormatter(use_color=color))

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except BaseException:
        console.close()
        raise
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        _RedactingFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    previous_level = root_logger.level
    previous_handlers = tuple(root_logger.handlers)
    try:
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(console)
        root_logger.addHandler(file_handler)
    except BaseException:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)
        console.close()
        file_handler.close()
        raise
    return log_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured system; return 0 complete, 2 incomplete, 1 error."""

    parser = argparse.ArgumentParser(description="Run the configured RealAgent system")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="root JSON-compatible YAML configuration",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    try:
        log_path = configure_logging(
            args.log_dir,
            verbose=args.verbose,
            color=not args.no_color and sys.stderr.isatty(),
        )
    except Exception as exc:
        summary = redact_text(
            f"logging setup failed type={type(exc).__name__}: {exc}"
        )
        sys.stderr.write(f"ERROR __main__: {summary}\n")
        return 1
    logger.info("log file=%s", log_path)
    try:
        runtime = build_runtime_from_file(args.config)
        report = runtime.run(seed=args.seed)
    except Exception:
        logger.exception("run failed")
        return 1
    if report.completed:
        logger.info("task completed in %d control step(s)", report.control_steps)
        return 0
    logger.warning("task incomplete after %d control step(s)", report.control_steps)
    return 2


def _load_profiles(
    config: Mapping[str, Any], key: str, config_dir: Path
) -> dict[str, Mapping[str, Any]]:
    file_name = config.get(key)
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"{key} must be a non-empty path string")
    raw_profiles = load_mapping_file(config_dir / file_name)
    profiles: dict[str, Mapping[str, Any]] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, Mapping):
            raise TypeError(f"profile {name!r} in {file_name} must be a mapping")
        profiles[name] = raw_profile
    return profiles


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} config must be a mapping")
    return value


def _validate_assembly_config(
    *,
    env_config: Mapping[str, Any],
    task_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    model_profiles: Mapping[str, Mapping[str, Any]],
    perception_profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, int], int]:
    """Validate the complete configuration before acquiring any resources."""

    _reject_unknown_keys(
        agent_config,
        allowed={
            "model",
            "max_iterations",
            "max_tool_calls",
            "max_tool_calls_per_turn",
        },
        label="agent",
    )
    model_profile_name = agent_config.get("model")
    if not isinstance(model_profile_name, str) or not model_profile_name.strip():
        raise ValueError("agent.model must be a non-empty profile name")
    if model_profile_name != model_profile_name.strip():
        raise ValueError("agent.model must not contain surrounding whitespace")
    try:
        model_config = model_profiles[model_profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(model_profiles)) or "<none>"
        raise ValueError(
            f"unknown model profile {model_profile_name!r}; profiles: {choices}"
        ) from exc

    _validate_registered_config(
        model_config,
        label=f"model profile {model_profile_name!r}",
        registry=MODEL_REGISTRY,
    )
    _validate_registered_config(env_config, label="env", registry=ENV_REGISTRY)
    _validate_task_config(task_config, perception_profiles=perception_profiles)

    _reject_unknown_keys(runtime_config, allowed={"max_steps"}, label="runtime")
    max_steps = _bounded_integer(
        runtime_config.get("max_steps", 3),
        label="runtime.max_steps",
        upper_bound=1_000,
    )
    agent_limits = {
        "max_iterations": _bounded_integer(
            agent_config.get("max_iterations", 8),
            label="agent.max_iterations",
            upper_bound=128,
        ),
        "max_tool_calls": _bounded_integer(
            agent_config.get("max_tool_calls", 8),
            label="agent.max_tool_calls",
            upper_bound=256,
        ),
        "max_tool_calls_per_turn": _bounded_integer(
            agent_config.get("max_tool_calls_per_turn", 4),
            label="agent.max_tool_calls_per_turn",
            upper_bound=64,
        ),
    }
    if agent_limits["max_tool_calls_per_turn"] > agent_limits["max_tool_calls"]:
        raise ValueError(
            "agent.max_tool_calls_per_turn must not exceed agent.max_tool_calls"
        )
    return model_config, agent_limits, max_steps


def _validate_task_config(
    config: Mapping[str, Any],
    *,
    perception_profiles: Mapping[str, Mapping[str, Any]],
) -> None:
    task_type_name = config.get("type")
    if not isinstance(task_type_name, str) or not task_type_name.strip():
        raise ValueError("task.type must be a non-empty string")
    try:
        task_type = TASK_REGISTRY[task_type_name]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown task type {task_type_name!r}; registered types: {choices}"
        ) from exc

    task_options = {
        key: value
        for key, value in config.items()
        if key not in {"type", "perception_tools", "manipulation_tools"}
    }
    _bind_options(task_type, task_options, label=f"task type {task_type_name!r}")
    task_type.validate_config_options(task_options)

    perception_specs = _component_specs(config, "perception_tools")
    seen_perception_names: set[str] = set()
    for index, spec in enumerate(perception_specs):
        name = _component_name(spec, collection="perception_tools", index=index)
        if name in seen_perception_names:
            raise ValueError(f"duplicate perception tool name: {name!r}")
        seen_perception_names.add(name)
        if "type" in spec:
            raise ValueError(
                f"perception tool {name!r} cannot override profile type"
            )
        profile_name = spec.get("profile")
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError(
                f"perception tool {name!r} requires a non-empty profile"
            )
        try:
            profile = perception_profiles[profile_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown perception profile: {profile_name!r}"
            ) from exc
        overrides = {
            key: value
            for key, value in spec.items()
            if key not in {"name", "profile"}
        }
        _validate_registered_config(
            {**dict(profile), **overrides},
            label=f"perception tool {name!r}",
            registry=PERCEPTION_REGISTRY,
        )

    manipulation_specs = _component_specs(config, "manipulation_tools")
    seen_manipulation_names: set[str] = set()
    for index, spec in enumerate(manipulation_specs):
        name = _component_name(spec, collection="manipulation_tools", index=index)
        if name in seen_manipulation_names:
            raise ValueError(f"duplicate manipulation tool name: {name!r}")
        seen_manipulation_names.add(name)
        component_config = {
            key: value for key, value in spec.items() if key != "name"
        }
        _validate_registered_config(
            component_config,
            label=f"manipulation tool {name!r}",
            registry=MANIPULATION_REGISTRY,
        )


def _component_specs(
    task_config: Mapping[str, Any], key: str
) -> list[Mapping[str, Any]]:
    value = task_config.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"task.{key} must be a list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"task.{key}[{index}] must be a mapping")
        result.append(item)
    return result


def _component_name(
    spec: Mapping[str, Any], *, collection: str, index: int
) -> str:
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"task.{collection}[{index}] requires a non-empty name"
        )
    return name


def _validate_registered_config(
    config: Mapping[str, Any],
    *,
    label: str,
    registry: Mapping[str, Any],
) -> None:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    options = dict(config)
    component_type = options.pop("type", None)
    if not isinstance(component_type, str) or not component_type.strip():
        raise ValueError(f"{label}.type must be a non-empty string")
    try:
        builder = registry[component_type]
    except KeyError as exc:
        choices = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"unknown {label} type {component_type!r}; registered types: {choices}"
        ) from exc
    _bind_options(builder, options, label=f"{label} type {component_type!r}")


def _bind_options(builder: Any, options: Mapping[str, Any], *, label: str) -> None:
    if any(not isinstance(key, str) for key in options):
        raise TypeError(f"{label} option keys must be strings")
    try:
        inspect.signature(builder).bind(**dict(options))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"invalid options for {label}: {exc}") from exc


def _bounded_integer(value: Any, *, label: str, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not 1 <= value <= upper_bound:
        raise ValueError(f"{label} must be between 1 and {upper_bound}")
    return value


def _reject_unknown_keys(
    config: Mapping[str, Any], *, allowed: set[str], label: str
) -> None:
    non_string_keys = [key for key in config if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"{label} config keys must be strings")
    unknown = set(config) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported {label} config keys: {names}")


def _redact_log_text(value: str) -> str:
    return redact_text(value)


class _RedactingFormatter(logging.Formatter):
    """Redact the complete rendered record, including exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact_log_text(super().format(record))


class _ConsoleFormatter(_RedactingFormatter):
    _colors = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[35m",
    }

    def __init__(self, *, use_color: bool) -> None:
        super().__init__("%(levelname)s %(name)s: %(message)s")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Keep the console concise while the file handler retains the complete,
        # redacted traceback for diagnostics.
        console_record = copy(record)
        console_record.exc_info = None
        console_record.exc_text = None
        console_record.stack_info = None
        rendered = super().format(console_record)
        if not self._use_color:
            return rendered
        color = self._colors.get(record.levelno, "")
        return f"{color}{rendered}\x1b[0m" if color else rendered


if __name__ == "__main__":
    raise SystemExit(main())
