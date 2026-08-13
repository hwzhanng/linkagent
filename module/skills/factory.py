"""Reviewed, explicit registry for atomic execution skills."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from module.skills.core import AtomicSkill
from module.skills.harvest_cycle import (
    ActiveViewRepositionSkill,
    LocalizeDroppedPayloadSkill,
    MoveToPickReadySkill,
    PayloadPlaceSkill,
    PayloadRetreatSkill,
    PregraspClearanceCheckSkill,
    ResetForNextTargetSkill,
    SafeRetractSkill,
    SelectHarvestTargetSkill,
    TargetReobserveSkill,
    TransportPayloadSkill,
)
from module.skills.system_self_check import SystemSelfCheckSkill


AtomicSkillBuilder = Callable[..., AtomicSkill]
ATOMIC_SKILL_REGISTRY: dict[str, AtomicSkillBuilder] = {
    "active_view_reposition": ActiveViewRepositionSkill,
    "localize_dropped_payload": LocalizeDroppedPayloadSkill,
    "move_to_pick_ready": MoveToPickReadySkill,
    "payload_place": PayloadPlaceSkill,
    "payload_retreat": PayloadRetreatSkill,
    "pregrasp_clearance_check": PregraspClearanceCheckSkill,
    "reset_for_next_target": ResetForNextTargetSkill,
    "safe_retract": SafeRetractSkill,
    "select_harvest_target": SelectHarvestTargetSkill,
    "system_self_check": SystemSelfCheckSkill,
    "target_reobserve": TargetReobserveSkill,
    "transport_payload": TransportPayloadSkill,
}


def build_atomic_skill(config: Mapping[str, Any], **injected: object) -> AtomicSkill:
    if not isinstance(config, Mapping):
        raise TypeError("atomic skill config must be a mapping")
    options = dict(config)
    kind = options.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("atomic skill type must be a non-empty string")
    try:
        builder = ATOMIC_SKILL_REGISTRY[kind.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(ATOMIC_SKILL_REGISTRY)) or "<none>"
        raise ValueError(f"unknown atomic skill type {kind!r}; registered: {choices}") from exc
    try:
        inspect.signature(builder).bind(**injected, **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for atomic skill type {kind!r}: {exc}") from exc
    skill = builder(**injected, **options)
    if not isinstance(skill, AtomicSkill):
        raise TypeError("atomic skill builder must return an AtomicSkill")
    return skill


def validate_atomic_skill_config(
    config: object,
    *,
    label: str,
    injected_names: tuple[str, ...],
) -> None:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    options = dict(config)
    kind = options.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{label}.type must be a non-empty string")
    try:
        builder = ATOMIC_SKILL_REGISTRY[kind.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(ATOMIC_SKILL_REGISTRY)) or "<none>"
        raise ValueError(f"unknown {label} type {kind!r}; registered: {choices}") from exc
    injected = {name: object() for name in injected_names}
    try:
        inspect.signature(builder).bind(**injected, **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for {label} type {kind!r}: {exc}") from exc


__all__ = ["ATOMIC_SKILL_REGISTRY", "build_atomic_skill", "validate_atomic_skill_config"]
