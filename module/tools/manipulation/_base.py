"""Grasp-planning capability boundary without Agent-framework coupling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from proto import GraspCandidate, Observation


class BaseManipulation(ABC):
    """Plan score-ranked grasp candidates from an Observation.

    This class deliberately does not create an Agent tool.  A concrete Task must
    wrap the method it needs and place that wrapper on its own capability allowlist.
    """

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("manipulation name must be a non-empty string")
        self._name = name.strip()
        self._closed = False

    @property
    def name(self) -> str:
        """Stable backend name."""

        return self._name

    @property
    def closed(self) -> bool:
        """Whether close has been attempted for this backend."""

        return self._closed

    def rank_grasp_candidates(
        self,
        observation: Observation,
        *,
        options: Mapping[str, Any] | None = None,
        **option_overrides: Any,
    ) -> tuple[GraspCandidate, ...]:
        """Return candidates ordered by descending ``score``."""

        if self._closed:
            raise RuntimeError(f"manipulation backend {self._name!r} is closed")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if options is None:
            prepared_options: dict[str, Any] = {}
        elif isinstance(options, Mapping):
            prepared_options = dict(options)
        else:
            raise TypeError("options must be a mapping or None")
        overlap = prepared_options.keys() & option_overrides.keys()
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise TypeError(
                f"manipulation options provided more than once: {duplicate}"
            )
        prepared_options.update(option_overrides)

        candidates = tuple(self._rank_grasp_candidates(observation, prepared_options))
        if not all(isinstance(item, GraspCandidate) for item in candidates):
            raise TypeError(
                f"{type(self).__name__}._rank_grasp_candidates() must return "
                "only GraspCandidate values"
            )
        return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))

    @abstractmethod
    def _rank_grasp_candidates(
        self,
        observation: Observation,
        options: dict[str, Any],
    ) -> tuple[GraspCandidate, ...]:
        """Implement backend-specific candidate generation."""

    def close(self) -> None:
        """Attempt to release owned resources exactly once.

        The closed flag is set before the hook so partial cleanup failures are
        reported once and never cause a second release attempt.
        """

        if self._closed:
            return
        self._closed = True
        self._close()

    def _close(self) -> None:
        """Optional adapter cleanup hook called once by :meth:`close`."""

    @staticmethod
    def _reject_options(options: Mapping[str, Any]) -> None:
        """Reject options a backend did not explicitly consume."""

        if options:
            names = ", ".join(sorted(str(name) for name in options))
            raise TypeError(f"unsupported manipulation option(s): {names}")
