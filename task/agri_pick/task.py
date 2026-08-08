"""Single-tomato Task exposing exactly five small Agent tools."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping

from module.agent import AgentTool, RecoverableToolError
from module.robotics import ROBOTICS_REGISTRY, build_robotics_bundle
from module.runtime import RuntimeContext
from module.skills.tomato_pick import SKILL_REGISTRY, build_policy
from proto import ArmSide, Observation
from task._base_task import BaseTask

from .factory import (
    PERCEPTION_REGISTRY,
    PLANNER_REGISTRY,
    SAFETY_REGISTRY,
    VERIFIER_REGISTRY,
    build_perception,
    build_planner,
    build_safety,
    build_verifier,
    validate_component_config,
)
from .service import TomatoPickService, TomatoPickServiceError


logger = logging.getLogger(__name__)


class AgriPickTask(BaseTask):
    """Offline C1 vertical slice: detect, plan, pregrasp, pick, verify."""

    def __init__(
        self,
        *,
        robotics: Mapping[str, Any],
        skill: Mapping[str, Any],
        perception: Mapping[str, Any],
        planner: Mapping[str, Any],
        safety: Mapping[str, Any],
        verifier: Mapping[str, Any],
        default_arm_side: str = "right",
        max_skill_duration_s: float = 4.0,
    ) -> None:
        super().__init__()
        options = {
            "robotics": robotics,
            "skill": skill,
            "perception": perception,
            "planner": planner,
            "safety": safety,
            "verifier": verifier,
            "default_arm_side": default_arm_side,
            "max_skill_duration_s": max_skill_duration_s,
        }
        self.validate_config_options(options)
        side = ArmSide(default_arm_side)
        bundle = None
        perception_port = None
        try:
            bundle = build_robotics_bundle(robotics, side=side)
            overrides = bundle.task_component_overrides()
            perception_options = {**dict(perception), **dict(overrides.get("perception", {}))}
            planner_options = {**dict(planner), **dict(overrides.get("planner", {}))}
            safety_options = {**dict(safety), **dict(overrides.get("safety", {}))}
            verifier_options = {**dict(verifier), **dict(overrides.get("verifier", {}))}
            perception_port = build_perception(perception_options, clock=bundle.clock)
            planner_port = build_planner(planner_options, clock=bundle.clock)
            safety_port = build_safety(safety_options, clock=bundle.clock)
            verifier_port = build_verifier(verifier_options, robotics=bundle)
            policy = build_policy(skill)
            self.service = TomatoPickService(
                robotics=bundle,
                perception=perception_port,
                planner=planner_port,
                safety=safety_port,
                verifier=verifier_port,
                policy=policy,
                default_arm_side=side,
                max_skill_duration_s=max_skill_duration_s,
            )
        except BaseException as primary_error:
            errors: list[BaseException] = [primary_error]
            if perception_port is not None:
                try:
                    perception_port.close()
                except BaseException as exc:
                    errors.append(exc)
            if bundle is not None:
                try:
                    bundle.close()
                except BaseException as exc:
                    errors.append(exc)
            if len(errors) > 1:
                raise BaseExceptionGroup("agri-pick construction failed", errors) from None
            raise

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        required = {"robotics", "skill", "perception", "planner", "safety", "verifier"}
        missing = required - set(options)
        if missing:
            raise ValueError(f"missing agri-pick options: {', '.join(sorted(missing))}")
        side_value = options.get("default_arm_side", "right")
        try:
            side = ArmSide(side_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("default_arm_side must be left or right") from exc
        del side
        duration = options.get("max_skill_duration_s", 4.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("max_skill_duration_s must be > 0")
        cls._validate_registered(options["robotics"], "robotics", ROBOTICS_REGISTRY, ("side",))
        cls._validate_registered(options["skill"], "skill", SKILL_REGISTRY, ())
        validate_component_config(
            options["perception"], label="perception", registry=PERCEPTION_REGISTRY, injected_names=("clock",)
        )
        validate_component_config(
            options["planner"], label="planner", registry=PLANNER_REGISTRY, injected_names=("clock",)
        )
        validate_component_config(
            options["safety"], label="safety", registry=SAFETY_REGISTRY, injected_names=("clock",)
        )
        validate_component_config(
            options["verifier"], label="verifier", registry=VERIFIER_REGISTRY, injected_names=("robotics",)
        )

    @staticmethod
    def _validate_registered(
        config: object,
        label: str,
        registry: Mapping[str, Any],
        injected_names: tuple[str, ...],
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError(f"{label} config must be a mapping")
        values = dict(config)
        kind = values.pop("type", None)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"{label}.type must be a non-empty string")
        try:
            builder = registry[kind.strip()]
        except KeyError as exc:
            choices = ", ".join(sorted(registry)) or "<none>"
            raise ValueError(f"unknown {label} type {kind!r}; registered: {choices}") from exc
        injected = {name: object() for name in injected_names}
        try:
            inspect.signature(builder).bind(**injected, **values)
        except TypeError as exc:
            raise TypeError(f"invalid options for {label} type {kind!r}: {exc}") from exc

    def build_prompt(self) -> str:
        return (
            "Pick one ripe tomato using exactly this order when possible: "
            "detect_tomatoes, plan_tomato_grasp, move_to_pregrasp, "
            "execute_tomato_pick, verify_tomato_pick. Tool success is not harvest "
            "success. Completion requires verify_tomato_pick to confirm both fruit "
            "detachment and retained-without-slip evidence. Never invent IDs."
        )

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        def detect_tomatoes() -> Mapping[str, Any]:
            return self.service.detect_tomatoes(context.require_observation())

        def plan_tomato_grasp(target_id: str, arm_side: str = "auto") -> Mapping[str, Any]:
            return self._recover(self.service.plan_tomato_grasp, target_id, arm_side)

        def move_to_pregrasp(grasp_plan_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.move_to_pregrasp, grasp_plan_id)

        def execute_tomato_pick(grasp_plan_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.execute_tomato_pick, grasp_plan_id)

        def verify_tomato_pick(execution_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.verify_tomato_pick, execution_id)

        no_args = {"type": "object", "properties": {}, "additionalProperties": False}
        string_id = lambda name, description: {
            "type": "object",
            "properties": {name: {"type": "string", "description": description}},
            "required": [name],
            "additionalProperties": False,
        }
        return [
            AgentTool(
                name="detect_tomatoes",
                description="Detect ripe tomato target summaries in the latest observation.",
                handler=detect_tomatoes,
                parameters=no_args,
            ),
            AgentTool(
                name="plan_tomato_grasp",
                description="Create a bounded grasp-frame plan for one detected tomato.",
                handler=plan_tomato_grasp,
                parameters={
                    "type": "object",
                    "properties": {
                        "target_id": {"type": "string"},
                        "arm_side": {"type": "string", "enum": ["auto", "left", "right"]},
                    },
                    "required": ["target_id"],
                    "additionalProperties": False,
                },
            ),
            AgentTool(
                name="move_to_pregrasp",
                description="Safety-check the plan and move the selected arm to pregrasp.",
                handler=move_to_pregrasp,
                parameters=string_id("grasp_plan_id", "ID returned by plan_tomato_grasp"),
            ),
            AgentTool(
                name="execute_tomato_pick",
                description="Run the bounded four-phase tomato pick skill once for a prepared plan.",
                handler=execute_tomato_pick,
                parameters=string_id("grasp_plan_id", "Prepared grasp plan ID"),
            ),
            AgentTool(
                name="verify_tomato_pick",
                description="Confirm detached and retained-without-slip evidence for an execution.",
                handler=verify_tomato_pick,
                parameters=string_id("execution_id", "ID returned by execute_tomato_pick"),
            ),
        ]

    @staticmethod
    def _recover(handler: Any, *args: Any) -> Mapping[str, Any]:
        try:
            return handler(*args)
        except TomatoPickServiceError as exc:
            raise RecoverableToolError(str(exc)) from exc

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        status = self.service.status_summary()
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Fresh observation sequence={observation.sequence}. "
                        f"Workflow state={status['workflow_state']}. Continue with the allowed tools."
                    ),
                }
            ]
        }

    def is_complete(self, context: RuntimeContext, agent_result: Mapping[str, Any]) -> bool:
        del context, agent_result
        return self.service.completed

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            self.service.close()
        except BaseException as exc:
            logger.error("agri-pick service close failed type=%s", type(exc).__name__)
            errors.append(exc)
        try:
            super().close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("agri-pick task cleanup failed", errors)


__all__ = ["AgriPickTask"]
