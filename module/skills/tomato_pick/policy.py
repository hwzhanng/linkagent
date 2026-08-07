"""Policy protocol and deterministic mock used by the offline Agent slice."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .contracts import PolicyManifest, TomatoPickAction, TomatoPickObservation


@runtime_checkable
class TomatoPickPolicy(Protocol):
    @property
    def manifest(self) -> PolicyManifest: ...
    def reset(self) -> None: ...
    def act(self, observation: TomatoPickObservation) -> TomatoPickAction: ...


class MockTomatoPickPolicy:
    """Simple deterministic stub; it is not a training expert or deployment policy."""

    def __init__(self, *, manifest: PolicyManifest | None = None) -> None:
        self._manifest = manifest or PolicyManifest.mock_default()

    @property
    def manifest(self) -> PolicyManifest:
        return self._manifest

    def reset(self) -> None:
        return None

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        if not isinstance(observation, TomatoPickObservation):
            raise TypeError("observation must be a TomatoPickObservation")
        pressure = observation.block("finger_pressure")
        # Keep the mock useful for the offline task without treating it as an
        # RL demonstration: approach, close, and pull are all constant outputs.
        pull = 0.35 if max(pressure) > 0.03 else -0.20
        hand = (0.45, 0.55, 0.10, 0.0, 0.0)
        return TomatoPickAction((0.0, 0.0, pull, 0.0, 0.0, 0.15, *hand))


PolicyBuilder = Callable[..., TomatoPickPolicy]
SKILL_REGISTRY: dict[str, PolicyBuilder] = {"mock": MockTomatoPickPolicy}


def build_policy(config: Mapping[str, Any]) -> TomatoPickPolicy:
    if not isinstance(config, Mapping):
        raise TypeError("skill config must be a mapping")
    values = dict(config)
    kind = values.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("skill.type must be a non-empty string")
    try:
        builder = SKILL_REGISTRY[kind.strip()]
    except KeyError as exc:
        raise ValueError(f"unknown skill type {kind!r}") from exc
    policy = builder(**values)
    if not isinstance(policy, TomatoPickPolicy):
        raise TypeError("skill builder must return a TomatoPickPolicy")
    return policy


__all__ = [
    "MockTomatoPickPolicy",
    "PolicyBuilder",
    "SKILL_REGISTRY",
    "TomatoPickPolicy",
    "build_policy",
]
