"""Deterministic in-memory environment used by tests and local development."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from proto import Action, Observation, Result

from ._base import BaseEnv


def _string_set(value: Iterable[str], *, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an iterable of action names")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of action names") from exc
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return frozenset(item.strip() for item in items)


def _coerce_observation(value: object, *, index: int) -> Observation:
    if isinstance(value, Observation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"observations[{index}] must be an Observation or a mapping"
        )
    try:
        return Observation(**dict(value))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid observations[{index}]: {exc}") from exc


class MockEnv(BaseEnv):
    """Replay configured observations and record every attempted action.

    Once the configured observations are exhausted, the final observation is
    repeated with an incremented ``sequence``.  Actions named by ``fail_actions``
    (or the first ``fail_next_actions`` attempts) return ``Result(False, ...)``.
    ``raise_actions`` is available for simulating an infrastructure exception.
    """

    def __init__(
        self,
        *,
        observations: Iterable[Observation | Mapping[str, Any]] | None = None,
        fail_actions: Iterable[str] = (),
        fail_next_actions: int = 0,
        raise_actions: Iterable[str] = (),
        failure_message: str = "simulated action failure",
    ) -> None:
        if observations is None:
            prepared_observations = (Observation(sequence=0),)
        else:
            if isinstance(observations, (str, bytes, bytearray, Mapping)):
                raise TypeError("observations must be an iterable of observations")
            try:
                prepared_observations = tuple(
                    _coerce_observation(item, index=index)
                    for index, item in enumerate(observations)
                )
            except TypeError:
                raise
            if not prepared_observations:
                raise ValueError("observations must contain at least one observation")

        previous_sequence = -1
        for index, observation in enumerate(prepared_observations):
            if observation.sequence <= previous_sequence:
                raise ValueError(
                    "observations must have strictly increasing sequence values; "
                    f"observations[{index}].sequence={observation.sequence}"
                )
            previous_sequence = observation.sequence

        if (
            isinstance(fail_next_actions, bool)
            or not isinstance(fail_next_actions, int)
            or fail_next_actions < 0
        ):
            raise ValueError("fail_next_actions must be an integer >= 0")
        if not isinstance(failure_message, str) or not failure_message:
            raise ValueError("failure_message must be a non-empty string")

        self._observations = prepared_observations
        self._fail_actions = _string_set(
            fail_actions,
            field_name="fail_actions",
        )
        self._raise_actions = _string_set(
            raise_actions,
            field_name="raise_actions",
        )
        self._configured_fail_next_actions = fail_next_actions
        self._remaining_fail_next_actions = fail_next_actions
        self._failure_message = failure_message
        self._actions: list[Action] = []
        self._cursor = 0
        self._last_observation: Observation | None = None
        self._closed = False

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return an immutable snapshot of all attempted actions."""

        return tuple(self._actions)

    @property
    def closed(self) -> bool:
        """Whether this environment has been closed."""

        return self._closed

    def reset(self, seed: int | None = None) -> Observation:
        """Restart observation replay and return the first configured sample."""

        self._require_open()
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        self._cursor = 1
        self._remaining_fail_next_actions = self._configured_fail_next_actions
        self._last_observation = self._observations[0]
        return self._last_observation

    def get_obs(self) -> Observation:
        """Return the next replay sample or a sequenced copy of the final one."""

        self._require_open()
        if self._last_observation is None:
            raise RuntimeError("MockEnv must be reset before get_obs()")
        if self._cursor < len(self._observations):
            observation = self._observations[self._cursor]
            self._cursor += 1
        else:
            observation = replace(
                self._last_observation,
                sequence=self._last_observation.sequence + 1,
            )
        self._last_observation = observation
        return observation

    def take_action(self, action: Action) -> Result:
        """Record an action and return its configured simulated outcome."""

        self._require_open()
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        self._actions.append(action)
        attempt = len(self._actions)

        if action.name in self._raise_actions:
            raise RuntimeError(
                f"simulated infrastructure failure for action {action.name!r}"
            )

        should_fail = action.name in self._fail_actions
        if self._remaining_fail_next_actions > 0:
            self._remaining_fail_next_actions -= 1
            should_fail = True
        return Result(
            success=not should_fail,
            message=self._failure_message if should_fail else "action accepted",
            data={"action_name": action.name, "attempt": attempt},
        )

    def close(self) -> None:
        """Close this in-memory environment; repeated calls are no-ops."""

        if self._closed:
            return
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MockEnv is closed")
