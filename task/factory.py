"""Explicit Task registry; no import-time directory scanning."""

from __future__ import annotations

from typing import Any, Mapping

from task._base_task import BaseTask


TASK_REGISTRY: dict[str, type[BaseTask]] = {}


def register_task(name: str, task_type: type[BaseTask]) -> None:
    """Register a concrete Task under a stable configuration name."""

    if not isinstance(name, str) or not name:
        raise ValueError("task registry name must be a non-empty string")
    if not isinstance(task_type, type) or not issubclass(task_type, BaseTask):
        raise TypeError("task_type must be a BaseTask subclass")
    normalized_name = name.strip()
    if normalized_name in TASK_REGISTRY:
        raise ValueError(f"task type is already registered: {normalized_name!r}")
    TASK_REGISTRY[normalized_name] = task_type


def build_task(
    config: Mapping[str, Any],
    *,
    perception_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> BaseTask:
    """Build a registered Task from composition-root configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("task config must be a mapping")
    task_name = config.get("type")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("task.type must be a non-empty string")
    try:
        task_type = TASK_REGISTRY[task_name]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown task type {task_name!r}; registered types: {choices}"
        ) from exc
    return task_type.from_config(
        config,
        perception_profiles=perception_profiles,
    )


# Concrete imports are explicit registrations, not directory scanning.
from task.locate_target import LocateTargetTask  # noqa: E402
from task.agri_pick import AgriPickTask  # noqa: E402


register_task("locate_target", LocateTargetTask)
register_task("agri_pick", AgriPickTask)
