"""SDK-neutral runtime lifecycle and the outer bounded control loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from module.agent._contracts import AgentInvoker
from module.tools.env._base import BaseEnv
from proto._base_proto import Action, Observation, Result

logger = logging.getLogger(__name__)


class RuntimeState(str, Enum):
    """Observable lifecycle states for a single-use runtime."""

    READY = "ready"
    RUNNING = "running"
    CLOSED = "closed"


class CompletionReason(str, Enum):
    """Reasons a normal runtime execution can stop."""

    TASK_COMPLETE = "task_complete"
    STEP_LIMIT = "step_limit"


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    """Small, binary-free summary returned by a normal runtime execution."""

    completed: bool
    reason: CompletionReason
    control_steps: int
    last_observation_sequence: int
    last_agent_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a bool")
        if not isinstance(self.reason, CompletionReason):
            raise TypeError("reason must be a CompletionReason")
        if isinstance(self.control_steps, bool) or not isinstance(
            self.control_steps, int
        ):
            raise TypeError("control_steps must be an integer")
        if self.control_steps < 0:
            raise ValueError("control_steps must be >= 0")
        if isinstance(self.last_observation_sequence, bool) or not isinstance(
            self.last_observation_sequence, int
        ):
            raise TypeError("last_observation_sequence must be an integer")
        if self.last_observation_sequence < 0:
            raise ValueError("last_observation_sequence must be >= 0")
        if self.last_agent_result is not None:
            if not isinstance(self.last_agent_result, Mapping):
                raise TypeError("last_agent_result must be a mapping or None")
            object.__setattr__(
                self,
                "last_agent_result",
                MappingProxyType(dict(self.last_agent_result)),
            )
        if self.completed != (self.reason is CompletionReason.TASK_COMPLETE):
            raise ValueError("completed must agree with reason")


class RuntimeContext:
    """Holds the environment and the latest observation for task tools.

    The context is deliberately small. It is the only shared mutable state between
    the outer runtime loop and closures exposed as Agent tools.
    """

    def __init__(self, env: BaseEnv) -> None:
        if not isinstance(env, BaseEnv):
            raise TypeError("env must implement BaseEnv")
        self._env = env
        self._observation: Observation | None = None
        self._observation_version = 0

    @property
    def env(self) -> BaseEnv:
        """Return the environment capability boundary."""

        return self._env

    @property
    def observation(self) -> Observation | None:
        """Return the latest observation, if the environment has been reset."""

        return self._observation

    @property
    def observation_version(self) -> int:
        """Return a monotonic revision used to avoid duplicate observations."""

        return self._observation_version

    def update_observation(self, observation: Observation) -> None:
        """Replace the current observation at the environment boundary."""

        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        self._observation = observation
        self._observation_version += 1

    def require_observation(self) -> Observation:
        """Return the latest observation or fail before environment reset."""

        if self._observation is None:
            raise RuntimeError("no observation is available; reset the runtime first")
        return self._observation


@runtime_checkable
class RuntimeTask(Protocol):
    """Structural Task boundary consumed by :class:`AgentRuntime`.

    Keeping this protocol beside the runtime avoids importing the Task package
    (and therefore any concrete Task implementations) into the orchestration
    layer while still rejecting an invalid owner before execution starts.
    """

    def run(self, runtime: AgentRuntime) -> RuntimeReport:
        """Execute the Task's outer-loop policy."""

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        """Build one bounded Agent input from the current observation."""

    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: Mapping[str, Any],
    ) -> bool:
        """Decide completion from verifiable state and Agent output."""

    def close(self) -> None:
        """Attempt to release Task-owned resources exactly once."""


class AgentRuntime:
    """Own the Env/Task lifecycle and run a bounded outer control loop.

    Instances are intentionally single-use because :meth:`run` always closes the
    Task and Env. A new run must be assembled with fresh owners.
    """

    def __init__(
        self,
        *,
        context: RuntimeContext,
        task: RuntimeTask,
        agent: AgentInvoker,
        max_steps: int = 3,
    ) -> None:
        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be a RuntimeContext")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be an integer >= 1")
        if not isinstance(agent, AgentInvoker):
            raise TypeError("agent must implement AgentInvoker")
        if not isinstance(task, RuntimeTask):
            raise TypeError("task must implement RuntimeTask")

        self._context = context
        self._task = task
        self._agent = agent
        self._max_steps = max_steps
        self._state = RuntimeState.READY

    @property
    def context(self) -> RuntimeContext:
        """Return the shared runtime context."""

        return self._context

    @property
    def state(self) -> RuntimeState:
        """Return the current lifecycle state."""

        return self._state

    @property
    def max_steps(self) -> int:
        """Return the configured outer control-step limit."""

        return self._max_steps

    def run(self, *, seed: int | None = None) -> RuntimeReport:
        """Reset, execute the Task, then close Task and Env on every path.

        Normal step exhaustion is represented by an incomplete
        :class:`RuntimeReport`. Boundary and programming errors are re-raised after
        both resource owners have had a chance to close.
        """

        if self._state is not RuntimeState.READY:
            raise RuntimeError("AgentRuntime instances can only be run once")

        self._state = RuntimeState.RUNNING
        report: RuntimeReport | None = None
        errors: list[BaseException] = []
        try:
            logger.info("runtime starting max_steps=%d", self._max_steps)
            first_observation = self._context.env.reset(seed=seed)
            self._context.update_observation(first_observation)
            report = self._task.run(self)
            if not isinstance(report, RuntimeReport):
                raise TypeError("task.run(runtime) must return RuntimeReport")
        except BaseException as exc:
            logger.error("runtime execution failed type=%s", type(exc).__name__)
            errors.append(exc)
        finally:
            errors.extend(self._close_owned_resources())

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("runtime execution and cleanup failed", errors)

        if report is None:  # pragma: no cover - guarded by the error path above.
            raise AssertionError("runtime finished without a report")
        logger.info(
            "runtime stopped completed=%s reason=%s steps=%d",
            report.completed,
            report.reason.value,
            report.control_steps,
        )
        return report

    def close(self) -> None:
        """Close a READY runtime without executing it; repeated calls are no-ops.

        This gives composition callers an explicit release path if they build a
        runtime but later decide not to call :meth:`run`. A synchronous run owns
        its own cleanup, so externally closing while it is RUNNING is rejected.
        """

        if self._state is RuntimeState.CLOSED:
            return
        if self._state is RuntimeState.RUNNING:
            raise RuntimeError("cannot close a runtime while it is running")
        errors = self._close_owned_resources()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("runtime cleanup failed", errors)

    def run_default(self) -> RuntimeReport:
        """Execute the Task/Agent interaction using the default outer loop."""

        self._require_running()
        last_result: Mapping[str, Any] | None = None

        for control_step in range(1, self._max_steps + 1):
            observation = self._context.require_observation()
            version_before_invoke = self._context.observation_version
            agent_input = self._task.build_input(observation)
            last_result = self._agent.invoke(agent_input)
            if not isinstance(last_result, Mapping):
                raise TypeError("agent.invoke() must return a mapping")

            if self._task.is_complete(self._context, last_result):
                return RuntimeReport(
                    completed=True,
                    reason=CompletionReason.TASK_COMPLETE,
                    control_steps=control_step,
                    last_observation_sequence=(
                        self._context.require_observation().sequence
                    ),
                    last_agent_result=last_result,
                )

            if control_step < self._max_steps:
                # A physical action tool may already have called step(), which
                # refreshes the context. Do not fetch a second frame in that case.
                if self._context.observation_version == version_before_invoke:
                    self.observe()

        return RuntimeReport(
            completed=False,
            reason=CompletionReason.STEP_LIMIT,
            control_steps=self._max_steps,
            last_observation_sequence=self._context.require_observation().sequence,
            last_agent_result=last_result,
        )

    def observe(self) -> Observation:
        """Fetch one observation and make it current for every Task tool."""

        self._require_running()
        observation = self._context.env.get_obs()
        self._context.update_observation(observation)
        logger.debug("observation refreshed sequence=%d", observation.sequence)
        return observation

    def step(self, action: Action, *, refresh_observation: bool = True) -> Result:
        """Execute an action and, by default, immediately refresh shared state.

        A failed Result is still followed by an observation because a rejected or
        partial physical command may have changed the world. Infrastructure errors
        raised by the Env propagate and do not produce a fake Result.
        """

        self._require_running()
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        result = self._context.env.take_action(action)
        if not isinstance(result, Result):
            raise TypeError("env.take_action() must return Result")
        if refresh_observation:
            self.observe()
        return result

    def _require_running(self) -> None:
        if self._state is not RuntimeState.RUNNING:
            raise RuntimeError("runtime operation requires state=running")

    def _close_owned_resources(self) -> list[BaseException]:
        """Attempt Task then Env cleanup independently and mark the owner CLOSED."""

        errors: list[BaseException] = []
        try:
            for owner_name, owner in (
                ("task", self._task),
                ("env", self._context.env),
            ):
                try:
                    # Resolve each close method inside its own error boundary.
                    # A malformed owner must not prevent later owners from being
                    # given their cleanup opportunity.
                    close = owner.close
                    close()
                except BaseException as exc:
                    logger.error(
                        "%s close failed type=%s", owner_name, type(exc).__name__
                    )
                    errors.append(exc)
        finally:
            self._state = RuntimeState.CLOSED
        return errors
