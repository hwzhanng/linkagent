"""Common perception lifecycle without model or array dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
import logging
from time import perf_counter
from typing import Any, Mapping

from proto import Observation, VisionResult


logger = logging.getLogger(__name__)


class BasePerceptionTool(ABC):
    """Validate, time, and lifecycle-guard a perception implementation."""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("perception name must be a non-empty string")
        self._name = name.strip()
        self._closed = False

    @property
    def name(self) -> str:
        """Stable implementation name reported in logs and results."""

        return self._name

    @property
    def closed(self) -> bool:
        """Whether close has been attempted for this perception instance."""

        return self._closed

    def run(
        self,
        observation: Observation,
        *,
        options: Mapping[str, Any] | None = None,
        **option_overrides: Any,
    ) -> VisionResult:
        """Run perception against an Observation using copied call options."""

        if self._closed:
            raise RuntimeError(f"perception tool {self._name!r} is closed")
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
            raise TypeError(f"perception options provided more than once: {duplicate}")
        prepared_options.update(option_overrides)

        started_at = perf_counter()
        result = self._run(observation, prepared_options)
        elapsed_s = perf_counter() - started_at
        if not isinstance(result, VisionResult):
            raise TypeError(
                f"{type(self).__name__}._run() must return VisionResult"
            )
        if (
            result.observation_sequence is not None
            and result.observation_sequence != observation.sequence
        ):
            raise ValueError(
                "VisionResult.observation_sequence does not match the input "
                "Observation.sequence"
            )
        result = replace(
            result,
            elapsed_s=elapsed_s,
            observation_sequence=observation.sequence,
        )
        logger.info(
            "perception completed name=%s instances=%d elapsed_s=%.6f",
            self._name,
            len(result.instances),
            elapsed_s,
        )
        return result

    @abstractmethod
    def _run(
        self,
        observation: Observation,
        options: dict[str, Any],
    ) -> VisionResult:
        """Implement one model invocation using already-copied options."""

    def close(self) -> None:
        """Attempt to release owned resources exactly once.

        The closed flag is set before the adapter hook so a partially failing
        hook is never executed twice. Adapters must make one hook invocation
        release as much as possible before raising an aggregated error.
        """

        if self._closed:
            return
        self._closed = True
        self._close()

    def _close(self) -> None:
        """Optional adapter cleanup hook called once by :meth:`close`."""

    @staticmethod
    def _reject_options(options: Mapping[str, Any]) -> None:
        """Reject options an implementation did not explicitly consume."""

        if options:
            names = ", ".join(sorted(str(name) for name in options))
            raise TypeError(f"unsupported perception option(s): {names}")
