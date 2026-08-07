"""Stable Task contract and dependency ownership."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any, Mapping, Self

from module.agent import AgentTool
from module.tools.manipulation import BaseManipulation, build_manipulation
from module.tools.perception import BasePerceptionTool, build_perception
from proto import Observation

if TYPE_CHECKING:
    from module.runtime import AgentRuntime, RuntimeContext, RuntimeReport


logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Own prompt, Agent allowlist, completion semantics, and Task resources."""

    def __init__(self) -> None:
        self._perception_tools: dict[str, BasePerceptionTool] = {}
        self._manipulation_tools: dict[str, BaseManipulation] = {}
        self._closed = False

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        """Validate Task-specific nested options before acquiring resources."""

        if not isinstance(options, Mapping):
            raise TypeError("task options must be a mapping")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        perception_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Self:
        """Build a concrete Task and its explicitly declared dependencies.

        ``name`` is the stable semantic name used by Task code. A perception
        ``profile`` selects a replaceable implementation. Manipulation entries
        select their implementation directly with ``type`` as specified by the
        developer guide.
        """

        if not isinstance(config, Mapping):
            raise TypeError("task config must be a mapping")
        options = dict(config)
        options.pop("type", None)
        perception_specs = _pop_component_specs(options, "perception_tools")
        manipulation_specs = _pop_component_specs(options, "manipulation_tools")

        cls.validate_config_options(options)
        instance = cls(**options)
        if not isinstance(instance, BaseTask):  # pragma: no cover - defensive.
            raise TypeError("task constructor must return a BaseTask")
        if not all(
            hasattr(instance, attribute)
            for attribute in ("_perception_tools", "_manipulation_tools", "_closed")
        ):
            raise RuntimeError(
                f"{cls.__name__}.__init__() must call BaseTask.__init__()"
            )

        profiles = perception_profiles or {}
        created_perceptions: list[BasePerceptionTool] = []
        created_manipulations: list[BaseManipulation] = []
        try:
            for spec in perception_specs:
                name = _required_name(spec, "perception_tools")
                if name in instance._perception_tools:
                    raise ValueError(f"duplicate perception tool name: {name!r}")
                profile_name = spec.get("profile")
                if not isinstance(profile_name, str) or not profile_name:
                    raise ValueError(
                        f"perception tool {name!r} requires a non-empty profile"
                    )
                try:
                    profile = profiles[profile_name]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown perception profile: {profile_name!r}"
                    ) from exc
                if "type" in spec:
                    raise ValueError(
                        f"perception tool {name!r} cannot override profile type"
                    )
                overrides = {
                    key: value
                    for key, value in spec.items()
                    if key not in {"name", "profile"}
                }
                component_config = {**dict(profile), **overrides}
                perception = build_perception(component_config)
                created_perceptions.append(perception)
                instance._perception_tools[name] = perception

            for spec in manipulation_specs:
                name = _required_name(spec, "manipulation_tools")
                if name in instance._manipulation_tools:
                    raise ValueError(f"duplicate manipulation tool name: {name!r}")
                component_config = {
                    key: value for key, value in spec.items() if key != "name"
                }
                manipulation = build_manipulation(component_config)
                created_manipulations.append(manipulation)
                instance._manipulation_tools[name] = manipulation
        except BaseException as primary_error:
            errors: list[BaseException] = [primary_error]
            for manipulation in reversed(created_manipulations):
                try:
                    manipulation.close()
                except BaseException as exc:
                    logger.error(
                        "task dependency cleanup failed type=%s",
                        type(exc).__name__,
                    )
                    errors.append(exc)
            for perception in reversed(created_perceptions):
                try:
                    perception.close()
                except BaseException as exc:
                    logger.error(
                        "task dependency cleanup failed type=%s",
                        type(exc).__name__,
                    )
                    errors.append(exc)
            if len(errors) > 1:
                raise BaseExceptionGroup(
                    "task dependency construction and cleanup failed", errors
                ) from None
            raise

        return instance

    @abstractmethod
    def build_prompt(self) -> str:
        """Return the system prompt that defines this Task's semantics."""

    @abstractmethod
    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        """Return the complete and exact Agent capability allowlist."""

    @abstractmethod
    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        """Build one small Agent input without copying sensor payloads."""

    @abstractmethod
    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: Mapping[str, Any],
    ) -> bool:
        """Return true only when the result contains verifiable success."""

    def run(self, runtime: AgentRuntime) -> RuntimeReport:
        """Use the shared bounded loop unless a Task needs a custom flow."""

        return runtime.run_default()

    def get_perception_tool(self, name: str) -> BasePerceptionTool:
        """Return a Task-owned perception dependency by semantic name."""

        try:
            return self._perception_tools[name]
        except KeyError as exc:
            raise KeyError(f"task has no perception tool named {name!r}") from exc

    def get_manipulation_tool(self, name: str) -> BaseManipulation:
        """Return a Task-owned manipulation dependency by semantic name."""

        try:
            return self._manipulation_tools[name]
        except KeyError as exc:
            raise KeyError(f"task has no manipulation tool named {name!r}") from exc

    def close(self) -> None:
        """Idempotently close every capability backend created for the Task."""

        if self._closed:
            return
        # Mark the owner before invoking children: even if a backend reports a
        # partial cleanup failure, a repeated Task.close() must not re-run any
        # already-attempted release.
        self._closed = True
        errors: list[BaseException] = []
        for name, manipulation in reversed(tuple(self._manipulation_tools.items())):
            try:
                manipulation.close()
            except BaseException as exc:
                logger.error(
                    "manipulation close failed name=%s type=%s",
                    name,
                    type(exc).__name__,
                )
                errors.append(exc)
        for name, perception in reversed(tuple(self._perception_tools.items())):
            try:
                perception.close()
            except BaseException as exc:
                logger.error(
                    "perception close failed name=%s type=%s",
                    name,
                    type(exc).__name__,
                )
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("multiple task backends failed to close", errors)


def _pop_component_specs(
    options: dict[str, Any], key: str
) -> list[Mapping[str, Any]]:
    raw_specs = options.pop(key, [])
    if not isinstance(raw_specs, list):
        raise TypeError(f"task.{key} must be a list")
    specs: list[Mapping[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"task.{key}[{index}] must be a mapping")
        specs.append(raw_spec)
    return specs


def _required_name(spec: Mapping[str, Any], collection: str) -> str:
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"each task.{collection} entry requires a non-empty name")
    return name
