"""Policy boundary plus the deterministic C1 mock policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any, Protocol, runtime_checkable

from .contracts import PolicyManifest, TomatoPickAction, TomatoPickObservation


@runtime_checkable
class TomatoPickPolicy(Protocol):
    """Minimal deployable actor boundary; no simulator-private inputs."""

    @property
    def manifest(self) -> PolicyManifest:
        """Return the immutable policy/interface manifest."""

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        """Return one normalized action for a frozen observation."""

    def reset(self) -> None:
        """Reset episode-local policy state."""


class MockTomatoPickPolicy:
    """Small sensor-driven smoke policy; never used as a trained checkpoint."""

    def __init__(self, *, manifest: PolicyManifest | None = None) -> None:
        self._manifest = manifest or PolicyManifest.mock_default()

    @property
    def manifest(self) -> PolicyManifest:
        return self._manifest

    def reset(self) -> None:
        return None

    def act(self, observation: TomatoPickObservation) -> TomatoPickAction:
        if not isinstance(observation, TomatoPickObservation):
            raise TypeError("observation must be TomatoPickObservation")
        pressure = observation.block(58, 63)
        action = [0.0] * 22
        if max(pressure) < 0.03:
            action[2] = -1.0
        elif max(pressure) < 0.15:
            for index in (0, 1, 2, 3, 4, 11, 12, 13, 14, 15):
                action[6 + index] = 1.0
        else:
            action[2] = 1.0
            action[5] = 0.2
            for index in (0, 1, 2, 3, 4, 11, 12, 13, 14, 15):
                action[6 + index] = 0.1
        return TomatoPickAction(tuple(action))


PolicyBuilder = Callable[..., TomatoPickPolicy]

SKILL_REGISTRY: dict[str, PolicyBuilder] = {
    "mock": MockTomatoPickPolicy,
}


def build_policy(config: Mapping[str, Any]) -> TomatoPickPolicy:
    """Build a reviewed policy implementation from an explicit registry."""

    if not isinstance(config, Mapping):
        raise TypeError("skill config must be a mapping")
    options = dict(config)
    policy_type = options.pop("type", None)
    if not isinstance(policy_type, str) or not policy_type.strip():
        raise ValueError("skill.type must be a non-empty string")
    try:
        builder = SKILL_REGISTRY[policy_type.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(SKILL_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown skill type {policy_type!r}; registered types: {choices}"
        ) from exc
    try:
        inspect.signature(builder).bind(**options)
    except TypeError as exc:
        raise TypeError(f"invalid options for skill type {policy_type!r}: {exc}") from exc
    policy = builder(**options)
    if not isinstance(policy, TomatoPickPolicy):
        raise TypeError("skill builder must return a TomatoPickPolicy")
    return policy


__all__ = [
    "MockTomatoPickPolicy",
    "SKILL_REGISTRY",
    "TomatoPickPolicy",
    "build_policy",
]
