"""Environment capability boundary used by the runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod

from proto import Action, Observation, Result


class BaseEnv(ABC):
    """Minimal SDK-neutral environment contract."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Reset the world and return its first stable observation."""

    @abstractmethod
    def get_obs(self) -> Observation:
        """Capture and return one current observation."""

    @abstractmethod
    def take_action(self, action: Action) -> Result:
        """Execute one generic action without interpreting Task semantics."""

    @abstractmethod
    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""
