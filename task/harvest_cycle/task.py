"""Task exposing the bounded tools required for one verified harvest cycle."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from module.agent import AgentTool, RecoverableToolError
from module.robotics import build_robotics_bundle
from module.runtime import RuntimeContext
from module.skills import AtomicSkill, build_atomic_skill, validate_atomic_skill_config
from module.skills.harvest_cycle import (
    CLEARANCE_SCENE_REGISTRY,
    DROPPED_PAYLOAD_OBSERVATION_REGISTRY,
    PLACEMENT_VERIFIER_REGISTRY,
    TARGET_OBSERVATION_REGISTRY,
    build_clearance_scene_port,
    build_dropped_payload_observation_port,
    build_placement_verifier_port,
    build_target_observation_port,
    validate_harvest_port_config,
)
from module.skills.system_self_check import (
    CAMERA_HEALTH_REGISTRY,
    EMERGENCY_STOP_HEALTH_REGISTRY,
    VISION_WORKER_HEALTH_REGISTRY,
    build_camera_health_port,
    build_emergency_stop_health_port,
    build_vision_worker_health_port,
    validate_health_port_config,
)
from module.skills.tomato_pick import build_policy
from proto import ArmSide, HarvestContainer, Observation, Pose
from task._base_task import BaseTask
from task.agri_pick.factory import (
    build_perception,
    build_planner,
    build_safety,
    build_verifier,
)
from task.agri_pick.service import TomatoPickService
from task.agri_pick.task import AgriPickTask

from .service import HarvestCycleService, HarvestCycleServiceError


logger = logging.getLogger(__name__)


def _pose(value: object, *, field_name: str) -> Pose:
    if isinstance(value, Pose):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a Pose or mapping")
    try:
        return Pose(**dict(value))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {field_name}: {exc}") from exc


def _container(value: object) -> HarvestContainer:
    if isinstance(value, HarvestContainer):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("container must be a HarvestContainer or mapping")
    options = dict(value)
    options["placement_pose"] = _pose(
        options.get("placement_pose"), field_name="container.placement_pose"
    )
    return HarvestContainer(**options)


class HarvestCycleTask(BaseTask):
    """Offline vertical slice from target observation through container verification."""

    def __init__(
        self,
        *,
        robotics: Mapping[str, Any],
        skill: Mapping[str, Any],
        perception: Mapping[str, Any],
        planner: Mapping[str, Any],
        safety: Mapping[str, Any],
        verifier: Mapping[str, Any],
        atomic_skills: Mapping[str, Mapping[str, Any]],
        target_observation: Mapping[str, Any],
        clearance_scene: Mapping[str, Any],
        dropped_payload_observation: Mapping[str, Any],
        placement_verifier: Mapping[str, Any],
        camera_health: Mapping[str, Any],
        vision_worker_health: Mapping[str, Any],
        emergency_stop_health: Mapping[str, Any],
        container: Mapping[str, Any] | HarvestContainer,
        safe_pose: Mapping[str, Any] | Pose,
        default_arm_side: str = "right",
        max_skill_duration_s: float = 4.0,
        max_atomic_duration_s: float = 8.0,
    ) -> None:
        super().__init__()
        options = {
            "robotics": robotics,
            "skill": skill,
            "perception": perception,
            "planner": planner,
            "safety": safety,
            "verifier": verifier,
            "atomic_skills": atomic_skills,
            "target_observation": target_observation,
            "clearance_scene": clearance_scene,
            "dropped_payload_observation": dropped_payload_observation,
            "placement_verifier": placement_verifier,
            "camera_health": camera_health,
            "vision_worker_health": vision_worker_health,
            "emergency_stop_health": emergency_stop_health,
            "container": container,
            "safe_pose": safe_pose,
            "default_arm_side": default_arm_side,
            "max_skill_duration_s": max_skill_duration_s,
            "max_atomic_duration_s": max_atomic_duration_s,
        }
        self.validate_config_options(options)
        side = ArmSide(default_arm_side)
        prepared_container = _container(container)
        bundle = None
        perception_port = None
        safety_port = None
        pick_service = None
        observation_port = None
        clearance_scene_port = None
        dropped_payload_port = None
        placement_port = None
        camera_health_port = None
        vision_worker_health_port = None
        emergency_stop_health_port = None
        created_atomic: list[AtomicSkill] = []
        try:
            bundle = build_robotics_bundle(robotics, side=side)
            overrides = bundle.task_component_overrides()
            perception_port = build_perception(
                {**dict(perception), **dict(overrides.get("perception", {}))},
                clock=bundle.clock,
            )
            policy = build_policy(skill)
            safety_port = build_safety(
                {**dict(safety), **dict(overrides.get("safety", {}))},
                clock=bundle.clock,
            )
            pick_service = TomatoPickService(
                robotics=bundle,
                perception=perception_port,
                planner=build_planner(
                    {**dict(planner), **dict(overrides.get("planner", {}))},
                    clock=bundle.clock,
                ),
                safety=safety_port,
                verifier=build_verifier(
                    {**dict(verifier), **dict(overrides.get("verifier", {}))},
                    robotics=bundle,
                ),
                policy=policy,
                default_arm_side=side,
                max_skill_duration_s=max_skill_duration_s,
            )
            observation_port = build_target_observation_port(
                target_observation, clock=bundle.clock
            )
            clearance_scene_port = build_clearance_scene_port(
                clearance_scene, clock=bundle.clock
            )
            dropped_payload_port = build_dropped_payload_observation_port(
                dropped_payload_observation, clock=bundle.clock
            )
            placement_port = build_placement_verifier_port(
                placement_verifier, robotics=bundle
            )
            camera_health_port = build_camera_health_port(
                camera_health, clock=bundle.clock
            )
            vision_worker_health_port = build_vision_worker_health_port(
                vision_worker_health, clock=bundle.clock
            )
            emergency_stop_health_port = build_emergency_stop_health_port(
                emergency_stop_health, clock=bundle.clock
            )
            atomic = dict(atomic_skills)
            system_self_check = build_atomic_skill(
                atomic["system_self_check"],
                robotics=bundle,
                camera_health=camera_health_port,
                vision_worker_health=vision_worker_health_port,
                emergency_stop_health=emergency_stop_health_port,
                policy_manifest=pick_service.runner.manifest,
                container=prepared_container,
            )
            camera_health_port = None
            vision_worker_health_port = None
            emergency_stop_health_port = None
            created_atomic.append(system_self_check)
            active_view_reposition = build_atomic_skill(
                atomic["active_view_reposition"],
                robotics=bundle,
            )
            created_atomic.append(active_view_reposition)
            select_harvest_target = build_atomic_skill(
                atomic["select_harvest_target"],
                clock=bundle.clock,
            )
            created_atomic.append(select_harvest_target)
            target_reobserve = build_atomic_skill(
                atomic["target_reobserve"],
                observation_port=observation_port,
                clock=bundle.clock,
            )
            observation_port = None
            created_atomic.append(target_reobserve)
            pregrasp_clearance_check = build_atomic_skill(
                atomic["pregrasp_clearance_check"],
                robotics=bundle,
                scene_port=clearance_scene_port,
            )
            clearance_scene_port = None
            created_atomic.append(pregrasp_clearance_check)
            move_to_pick_ready = build_atomic_skill(
                atomic["move_to_pick_ready"],
                robotics=bundle,
                safety=safety_port,
            )
            created_atomic.append(move_to_pick_ready)
            safe_retract = build_atomic_skill(atomic["safe_retract"], robotics=bundle)
            created_atomic.append(safe_retract)
            payload_retreat = build_atomic_skill(
                atomic["payload_retreat"], robotics=bundle
            )
            created_atomic.append(payload_retreat)
            transport_payload = build_atomic_skill(
                atomic["transport_payload"], robotics=bundle
            )
            created_atomic.append(transport_payload)
            payload_place = build_atomic_skill(atomic["payload_place"], robotics=bundle)
            created_atomic.append(payload_place)
            localize_dropped_payload = build_atomic_skill(
                atomic["localize_dropped_payload"],
                observation_port=dropped_payload_port,
                clock=bundle.clock,
            )
            dropped_payload_port = None
            created_atomic.append(localize_dropped_payload)
            reset_for_next_target = build_atomic_skill(
                atomic["reset_for_next_target"], robotics=bundle
            )
            created_atomic.append(reset_for_next_target)
            self.service = HarvestCycleService(
                pick_service=pick_service,
                system_self_check=system_self_check,
                active_view_reposition=active_view_reposition,
                select_harvest_target=select_harvest_target,
                target_reobserve=target_reobserve,
                pregrasp_clearance_check=pregrasp_clearance_check,
                move_to_pick_ready=move_to_pick_ready,
                safe_retract=safe_retract,
                payload_retreat=payload_retreat,
                transport_payload=transport_payload,
                payload_place=payload_place,
                localize_dropped_payload=localize_dropped_payload,
                reset_for_next_target=reset_for_next_target,
                placement_verifier=placement_port,
                container=prepared_container,
                safe_pose=_pose(safe_pose, field_name="safe_pose"),
                max_atomic_duration_s=max_atomic_duration_s,
            )
            pick_service = None
            placement_port = None
            created_atomic.clear()
        except BaseException as primary_error:
            errors: list[BaseException] = [primary_error]
            for resource in reversed(created_atomic):
                try:
                    resource.close()
                except BaseException as exc:
                    errors.append(exc)
            for resource in (
                placement_port,
                dropped_payload_port,
                clearance_scene_port,
                observation_port,
                emergency_stop_health_port,
                vision_worker_health_port,
                camera_health_port,
                pick_service,
            ):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        errors.append(exc)
            if pick_service is None:
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
                raise BaseExceptionGroup("harvest-cycle construction failed", errors) from None
            raise

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        required = {
            "robotics",
            "skill",
            "perception",
            "planner",
            "safety",
            "verifier",
            "atomic_skills",
            "target_observation",
            "clearance_scene",
            "dropped_payload_observation",
            "placement_verifier",
            "camera_health",
            "vision_worker_health",
            "emergency_stop_health",
            "container",
            "safe_pose",
        }
        missing = required - set(options)
        if missing:
            raise ValueError(f"missing harvest-cycle options: {', '.join(sorted(missing))}")
        AgriPickTask.validate_config_options(
            {
                key: options[key]
                for key in ("robotics", "skill", "perception", "planner", "safety", "verifier")
            }
            | {
                "default_arm_side": options.get("default_arm_side", "right"),
                "max_skill_duration_s": options.get("max_skill_duration_s", 4.0),
            }
        )
        duration = options.get("max_atomic_duration_s", 8.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("max_atomic_duration_s must be > 0")
        atomic = options["atomic_skills"]
        if not isinstance(atomic, Mapping):
            raise TypeError("atomic_skills must be a mapping")
        expected = {
            "system_self_check",
            "active_view_reposition",
            "select_harvest_target",
            "target_reobserve",
            "pregrasp_clearance_check",
            "move_to_pick_ready",
            "safe_retract",
            "payload_retreat",
            "transport_payload",
            "payload_place",
            "localize_dropped_payload",
            "reset_for_next_target",
        }
        if set(atomic) != expected:
            raise ValueError(
                "atomic_skills must contain exactly: " + ", ".join(sorted(expected))
            )
        system_self_check_config = atomic["system_self_check"]
        active_view_config = atomic["active_view_reposition"]
        target_selection_config = atomic["select_harvest_target"]
        target_reobserve_config = atomic["target_reobserve"]
        if not all(
            isinstance(config, Mapping)
            for config in (
                system_self_check_config,
                active_view_config,
                target_selection_config,
                target_reobserve_config,
            )
        ):
            raise TypeError(
                "system_self_check, active_view_reposition, select_harvest_target, "
                "and target_reobserve "
                "configs must be mappings"
            )
        configured_side = options.get("default_arm_side", "right")
        if target_selection_config.get("reachability_arm_side") != configured_side:  # type: ignore[union-attr]
            raise ValueError(
                "select_harvest_target.reachability_arm_side must match "
                "default_arm_side"
            )
        if active_view_config.get("view_arm_side") != configured_side:  # type: ignore[union-attr]
            raise ValueError(
                "active_view_reposition.view_arm_side must match default_arm_side"
            )
        execution_mode = system_self_check_config.get("execution_mode", "offline_mock")
        require_new_sample = target_reobserve_config.get("require_new_sample", True)
        if execution_mode != "offline_mock" and require_new_sample is not True:
            raise ValueError(
                "shadow/hardware harvest requires target_reobserve.require_new_sample=true"
            )
        validate_atomic_skill_config(
            atomic["system_self_check"],
            label="atomic_skills.system_self_check",
            injected_names=(
                "robotics",
                "camera_health",
                "vision_worker_health",
                "emergency_stop_health",
                "policy_manifest",
                "container",
            ),
        )
        validate_atomic_skill_config(
            atomic["active_view_reposition"],
            label="atomic_skills.active_view_reposition",
            injected_names=("robotics",),
        )
        validate_atomic_skill_config(
            atomic["select_harvest_target"],
            label="atomic_skills.select_harvest_target",
            injected_names=("clock",),
        )
        validate_atomic_skill_config(
            atomic["target_reobserve"],
            label="atomic_skills.target_reobserve",
            injected_names=("observation_port", "clock"),
        )
        validate_atomic_skill_config(
            atomic["pregrasp_clearance_check"],
            label="atomic_skills.pregrasp_clearance_check",
            injected_names=("robotics", "scene_port"),
        )
        validate_atomic_skill_config(
            atomic["move_to_pick_ready"],
            label="atomic_skills.move_to_pick_ready",
            injected_names=("robotics", "safety"),
        )
        validate_atomic_skill_config(
            atomic["localize_dropped_payload"],
            label="atomic_skills.localize_dropped_payload",
            injected_names=("observation_port", "clock"),
        )
        for name in (
            "safe_retract",
            "payload_retreat",
            "transport_payload",
            "payload_place",
            "reset_for_next_target",
        ):
            validate_atomic_skill_config(
                atomic[name],
                label=f"atomic_skills.{name}",
                injected_names=("robotics",),
            )
        validate_harvest_port_config(
            options["target_observation"],
            label="target_observation",
            registry=TARGET_OBSERVATION_REGISTRY,
            injected_names=("clock",),
        )
        validate_harvest_port_config(
            options["clearance_scene"],
            label="clearance_scene",
            registry=CLEARANCE_SCENE_REGISTRY,
            injected_names=("clock",),
        )
        validate_harvest_port_config(
            options["dropped_payload_observation"],
            label="dropped_payload_observation",
            registry=DROPPED_PAYLOAD_OBSERVATION_REGISTRY,
            injected_names=("clock",),
        )
        validate_health_port_config(
            options["camera_health"],
            label="camera_health",
            registry=CAMERA_HEALTH_REGISTRY,
        )
        validate_health_port_config(
            options["vision_worker_health"],
            label="vision_worker_health",
            registry=VISION_WORKER_HEALTH_REGISTRY,
        )
        validate_health_port_config(
            options["emergency_stop_health"],
            label="emergency_stop_health",
            registry=EMERGENCY_STOP_HEALTH_REGISTRY,
        )
        validate_harvest_port_config(
            options["placement_verifier"],
            label="placement_verifier",
            registry=PLACEMENT_VERIFIER_REGISTRY,
            injected_names=("robotics",),
        )
        _container(options["container"])
        _pose(options["safe_pose"], field_name="safe_pose")

    def build_prompt(self) -> str:
        return (
            "Complete one tomato harvest cycle using this success order: "
            "check_harvest_system, detect_tomatoes, select_harvest_target, "
            "if selection requests active_view_required call active_view_reposition and "
            "then select_harvest_target again; otherwise continue with "
            "reobserve_tomato, "
            "plan_tomato_grasp, check_pregrasp_clearance, move_to_pick_ready, "
            "execute_tomato_pick, retreat_with_tomato, transport_payload, place_tomato, "
            "verify_harvest_cycle. Completion requires independent pick, retained-retreat, "
            "container placement, release, and empty-hand evidence. Never invent IDs. "
            "If retreat, transport, or placement reports payload_dropped, call "
            "localize_dropped_payload before any recovery motion; never reuse the "
            "hanging-fruit PPO for a dropped fruit. "
            "Only when another cycle is requested, call reset_for_next_target after completion. "
            "After a recoverable motion failure, call recover_harvest_cycle once."
        )

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        def check_harvest_system() -> Mapping[str, Any]:
            return self._recover(self.service.check_harvest_system)

        def detect_tomatoes() -> Mapping[str, Any]:
            return self._recover(
                self.service.detect_tomatoes, context.require_observation()
            )

        def reobserve_tomato(target_id: str) -> Mapping[str, Any]:
            return self._recover(
                self.service.reobserve_tomato,
                context.require_observation(),
                target_id,
            )

        def select_harvest_target() -> Mapping[str, Any]:
            return self._recover(self.service.select_harvest_target)

        def active_view_reposition() -> Mapping[str, Any]:
            return self._recover(
                self.service.active_view_reposition,
                context.require_observation(),
            )

        def plan_tomato_grasp(target_id: str, arm_side: str = "auto") -> Mapping[str, Any]:
            return self._recover(self.service.plan_tomato_grasp, target_id, arm_side)

        def move_to_pick_ready(grasp_plan_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.move_to_pick_ready, grasp_plan_id)

        def check_pregrasp_clearance(grasp_plan_id: str) -> Mapping[str, Any]:
            return self._recover(
                self.service.check_pregrasp_clearance,
                context.require_observation(),
                grasp_plan_id,
            )

        def execute_tomato_pick(grasp_plan_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.execute_tomato_pick, grasp_plan_id)

        def retreat_with_tomato(execution_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.retreat_with_tomato, execution_id)

        def transport_payload(retreat_execution_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.transport_payload, retreat_execution_id)

        def place_tomato(transport_execution_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.place_tomato, transport_execution_id)

        def localize_dropped_payload() -> Mapping[str, Any]:
            return self._recover(self.service.localize_dropped_payload)

        def verify_harvest_cycle(placement_id: str) -> Mapping[str, Any]:
            return self._recover(self.service.verify_harvest_cycle, placement_id)

        def reset_for_next_target() -> Mapping[str, Any]:
            return self._recover(self.service.reset_for_next_target)

        def recover_harvest_cycle(reason: str) -> Mapping[str, Any]:
            return self._recover(self.service.recover_harvest_cycle, reason)

        no_args = {"type": "object", "properties": {}, "additionalProperties": False}

        def string_id(name: str, description: str) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {name: {"type": "string", "description": description}},
                "required": [name],
                "additionalProperties": False,
            }

        return [
            AgentTool(
                "check_harvest_system",
                "Run the read-only camera, robot, tactile, safety, container, and policy admission gate.",
                check_harvest_system,
                no_args,
            ),
            AgentTool(
                "detect_tomatoes",
                "Detect tomato candidate summaries in the current observation.",
                detect_tomatoes,
                no_args,
            ),
            AgentTool(
                "select_harvest_target",
                "Deterministically select one harvestable candidate for re-observation; never authorizes motion.",
                select_harvest_target,
                no_args,
            ),
            AgentTool(
                "active_view_reposition",
                "Use one reviewed simulation/shadow MoveL viewpoint after selection reports active_view_required; accepts no free-form pose.",
                active_view_reposition,
                no_args,
            ),
            AgentTool(
                "reobserve_tomato",
                "Acquire fresh target evidence through the configured observation port.",
                reobserve_tomato,
                string_id("target_id", "Target returned by select_harvest_target"),
            ),
            AgentTool(
                "plan_tomato_grasp",
                "Plan one bounded grasp after target re-observation.",
                plan_tomato_grasp,
                {
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
                "check_pregrasp_clearance",
                "Check the current plan corridor against the configured world-frame scene; sends no motion command.",
                check_pregrasp_clearance,
                string_id("grasp_plan_id", "Plan returned by plan_tomato_grasp"),
            ),
            AgentTool(
                "move_to_pick_ready",
                "Move to the calibrated PPO ready pose and prove training-distribution admission.",
                move_to_pick_ready,
                string_id("grasp_plan_id", "Plan returned by plan_tomato_grasp"),
            ),
            AgentTool(
                "execute_tomato_pick",
                "Run the existing bounded PPO tomato-pick skill.",
                execute_tomato_pick,
                string_id("grasp_plan_id", "Prepared plan ID"),
            ),
            AgentTool(
                "retreat_with_tomato",
                "Verify the pick and retreat while checking payload retention.",
                retreat_with_tomato,
                string_id("execution_id", "Execution returned by execute_tomato_pick"),
            ),
            AgentTool(
                "transport_payload",
                "Lift and move the retained tomato to the configured container pre-place pose while sampling grip retention.",
                transport_payload,
                string_id("retreat_execution_id", "Successful retreat execution ID"),
            ),
            AgentTool(
                "place_tomato",
                "Descend from the transport-bound pre-place pose and release into the configured container.",
                place_tomato,
                string_id("transport_execution_id", "Execution returned by transport_payload"),
            ),
            AgentTool(
                "localize_dropped_payload",
                "After a payload_dropped failure, use fixed-camera evidence to route the fruit without authorizing regrasp motion.",
                localize_dropped_payload,
                no_args,
            ),
            AgentTool(
                "verify_harvest_cycle",
                "Independently verify container placement, release, and empty hand.",
                verify_harvest_cycle,
                string_id("placement_id", "Placement returned by place_tomato"),
            ),
            AgentTool(
                "reset_for_next_target",
                "After a verified cycle, return the empty hand to the configured observation pose and clear current target bindings.",
                reset_for_next_target,
                no_args,
            ),
            AgentTool(
                "recover_harvest_cycle",
                "Move the configured arm to its reviewed safe pose after a failure.",
                recover_harvest_cycle,
                string_id("reason", "Short recovery reason"),
            ),
        ]

    @staticmethod
    def _recover(handler: object, *args: object) -> Mapping[str, Any]:
        try:
            return handler(*args)  # type: ignore[operator, no-any-return]
        except HarvestCycleServiceError as exc:
            raise RecoverableToolError(str(exc)) from exc

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        status = self.service.status_summary()
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Fresh observation sequence={observation.sequence}. "
                        f"Harvest workflow state={status['workflow_state']}. Continue with allowed tools."
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
            logger.error("harvest-cycle service close failed type=%s", type(exc).__name__)
            errors.append(exc)
        try:
            super().close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("harvest-cycle task cleanup failed", errors)


__all__ = ["HarvestCycleTask"]
