"""Application service composing the existing PPO pick with post-pick atoms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from module.skills import AtomicSkill, AtomicSkillResult
from module.skills.harvest_cycle import PlacementVerifierPort
from proto import (
    ActiveViewRepositionRequest,
    DroppedPayloadLocalizationRequest,
    HarvestContainer,
    HarvestCycleState,
    HarvestCycleVerification,
    HarvestPayload,
    Observation,
    PayloadPlaceRequest,
    PayloadRetreatRequest,
    PayloadTransportRequest,
    PlacementReceipt,
    Pose,
    PregraspClearanceRequest,
    ResetForNextTargetRequest,
    SafeRetractRequest,
    SelectHarvestTargetRequest,
    TargetReobserveRequest,
    MoveToPickReadyRequest,
    TomatoDetectionResult,
    TomatoTarget,
)
from task.agri_pick.service import TomatoPickService, TomatoPickServiceError


class HarvestCycleServiceError(RuntimeError):
    """Expected workflow failure safe to surface through an Agent tool."""


class HarvestCycleService:
    """Own one observable detect-to-container harvest transaction."""

    def __init__(
        self,
        *,
        pick_service: TomatoPickService,
        system_self_check: AtomicSkill,
        active_view_reposition: AtomicSkill,
        select_harvest_target: AtomicSkill,
        target_reobserve: AtomicSkill,
        pregrasp_clearance_check: AtomicSkill,
        move_to_pick_ready: AtomicSkill,
        safe_retract: AtomicSkill,
        payload_retreat: AtomicSkill,
        transport_payload: AtomicSkill,
        payload_place: AtomicSkill,
        localize_dropped_payload: AtomicSkill,
        reset_for_next_target: AtomicSkill,
        placement_verifier: PlacementVerifierPort,
        container: HarvestContainer,
        safe_pose: Pose,
        max_atomic_duration_s: float = 8.0,
    ) -> None:
        if not isinstance(pick_service, TomatoPickService):
            raise TypeError("pick_service must be a TomatoPickService")
        for name, skill in {
            "system_self_check": system_self_check,
            "active_view_reposition": active_view_reposition,
            "select_harvest_target": select_harvest_target,
            "target_reobserve": target_reobserve,
            "pregrasp_clearance_check": pregrasp_clearance_check,
            "move_to_pick_ready": move_to_pick_ready,
            "safe_retract": safe_retract,
            "payload_retreat": payload_retreat,
            "transport_payload": transport_payload,
            "payload_place": payload_place,
            "localize_dropped_payload": localize_dropped_payload,
            "reset_for_next_target": reset_for_next_target,
        }.items():
            if not isinstance(skill, AtomicSkill):
                raise TypeError(f"{name} must implement AtomicSkill")
        if not isinstance(placement_verifier, PlacementVerifierPort):
            raise TypeError("placement_verifier must implement PlacementVerifierPort")
        if not isinstance(container, HarvestContainer):
            raise TypeError("container must be a HarvestContainer")
        if not isinstance(safe_pose, Pose):
            raise TypeError("safe_pose must be a Pose")
        if (
            isinstance(max_atomic_duration_s, bool)
            or not isinstance(max_atomic_duration_s, (int, float))
            or max_atomic_duration_s <= 0
        ):
            raise ValueError("max_atomic_duration_s must be > 0")
        self.pick_service = pick_service
        self.system_self_check = system_self_check
        self.active_view_reposition_skill = active_view_reposition
        self.select_harvest_target_skill = select_harvest_target
        self.target_reobserve = target_reobserve
        self.pregrasp_clearance_check_skill = pregrasp_clearance_check
        self.move_to_pick_ready_skill = move_to_pick_ready
        self.safe_retract = safe_retract
        self.payload_retreat = payload_retreat
        self.transport_payload_skill = transport_payload
        self.payload_place = payload_place
        self.localize_dropped_payload_skill = localize_dropped_payload
        self.reset_for_next_target_skill = reset_for_next_target
        self.placement_verifier = placement_verifier
        self.container = container
        self.safe_pose = safe_pose
        self.max_atomic_duration_s = float(max_atomic_duration_s)
        self.state = HarvestCycleState.INIT
        self.atomic_results: dict[str, AtomicSkillResult] = {}
        self.payloads: dict[str, HarvestPayload] = {}
        self._payload_by_retreat: dict[str, str] = {}
        self._payload_by_transport: dict[str, str] = {}
        self._retreat_by_transport: dict[str, str] = {}
        self.receipts: dict[str, PlacementReceipt] = {}
        self._payload_by_placement: dict[str, str] = {}
        self._dropped_payload_id: str | None = None
        self.verifications: dict[str, HarvestCycleVerification] = {}
        self._reobserved_targets: set[str] = set()
        self._clearance_by_plan: dict[str, str] = {}
        self._selected_target_id: str | None = None
        self._pending_active_view_target_id: str | None = None
        self._active_view_reference_pose: Pose | None = None
        self._active_view_viewpoint_index = 0
        self._active_view_selection_revision: str | None = None
        self._active_view_capture_attempt = 0
        self._reobserve_attempts: dict[str, int] = {}
        self._completed_cycle_id: str | None = None
        self._reset_attempt = 0
        self._self_check_attempt = 0
        self._admission_execution_id: str | None = None
        self._closed = False

    @property
    def completed(self) -> bool:
        return (
            self.state is HarvestCycleState.COMPLETE
            and self._completed_cycle_id is not None
            and self.verifications[self._completed_cycle_id].successful
        )

    def check_harvest_system(self) -> dict[str, Any]:
        self._require_open()
        from proto import SystemSelfCheckRequest

        self._self_check_attempt += 1
        execution_id = f"system-self-check-{self._self_check_attempt}"
        result = self.system_self_check.run(
            SystemSelfCheckRequest(
                execution_id=execution_id,
                arm_side=self.pick_service.default_arm_side,
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            self._admission_execution_id = None
            self.state = HarvestCycleState.INIT
            failed = result.output.get("failed_check_names", ())
            failed_text = ", ".join(str(item) for item in failed) or "unknown"
            raise HarvestCycleServiceError(
                f"system self-check rejected admission: {failed_text}"
            )
        self._admission_execution_id = execution_id
        self.state = HarvestCycleState.OBSERVE
        return {**result.as_dict(), "workflow_state": self.state.value}

    def detect_tomatoes(self, observation: Observation) -> dict[str, Any]:
        self._require_open()
        if self._admission_execution_id is None:
            raise HarvestCycleServiceError(
                "call check_harvest_system before detecting harvest targets"
            )
        summary = self._pick(self.pick_service.detect_tomatoes, observation)
        self._reobserved_targets.clear()
        self._clearance_by_plan.clear()
        self._reobserve_attempts.clear()
        self._selected_target_id = None
        self._reset_active_view()
        self.state = HarvestCycleState.SELECT
        return {**summary, "workflow_state": self.state.value}

    def select_harvest_target(self) -> dict[str, Any]:
        self._require_open()
        detection = self.pick_service.last_detection
        if detection is None:
            raise HarvestCycleServiceError(
                "call detect_tomatoes before selecting a harvest target"
            )
        execution_id = (
            f"select-{detection.observation_sequence}-{detection.sample_monotonic_ns}"
        )
        result = self.select_harvest_target_skill.run(
            SelectHarvestTargetRequest(
                execution_id=execution_id,
                detection=detection,
                arm_side=self.pick_service.default_arm_side,
                max_detection_age_s=1.0,
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            self._selected_target_id = None
            self._prepare_active_view_recovery(result)
            raise HarvestCycleServiceError(
                f"harvest target selection failed ({result.failure_code.value}): "
                f"{result.message}; next_action={result.output.get('next_action')}"
            )
        selected = result.output.get("selected_target_id")
        if not isinstance(selected, str) or selected not in self.pick_service.targets:
            self._selected_target_id = None
            self.state = HarvestCycleState.SELECT
            raise HarvestCycleServiceError(
                "harvest target selection returned an unknown target ID"
            )
        if result.output.get("motion_authorized") is not False:
            self._selected_target_id = None
            self.state = HarvestCycleState.SELECT
            raise HarvestCycleServiceError(
                "harvest target selection must not authorize robot motion"
            )
        self._selected_target_id = selected
        self._clearance_by_plan.clear()
        self._reset_active_view()
        self.state = HarvestCycleState.REOBSERVE
        return {**result.as_dict(), "workflow_state": self.state.value}

    def active_view_reposition(self, observation: Observation) -> dict[str, Any]:
        self._require_open()
        if self.state not in {
            HarvestCycleState.ACTIVE_VIEW,
            HarvestCycleState.ACTIVE_VIEW_OBSERVE,
        }:
            raise HarvestCycleServiceError(
                "active_view_reposition is only available after an "
                "active_view_required selection result"
            )
        target_id = self._pending_active_view_target_id
        reference_pose = self._active_view_reference_pose
        selection_revision = self._active_view_selection_revision
        if target_id is None or reference_pose is None or selection_revision is None:
            raise HarvestCycleServiceError("active-view recovery context is incomplete")
        try:
            target = self.pick_service.targets[target_id]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                "active-view target is no longer present in the detection result"
            ) from exc
        viewpoint_count = getattr(
            self.active_view_reposition_skill, "viewpoint_count", None
        )
        if (
            isinstance(viewpoint_count, bool)
            or not isinstance(viewpoint_count, int)
            or viewpoint_count <= 0
        ):
            raise HarvestCycleServiceError(
                "active-view skill did not expose a valid viewpoint budget"
            )
        if self._active_view_viewpoint_index >= viewpoint_count:
            raise HarvestCycleServiceError(
                "active-view viewpoints exhausted without a harvestable observation"
            )
        motion_execution_id = (
            f"active-view-{target_id}-{selection_revision}-"
            f"{self._active_view_viewpoint_index}"
        )
        if self.state is HarvestCycleState.ACTIVE_VIEW:
            motion_result = self.active_view_reposition_skill.run(
                ActiveViewRepositionRequest(
                    execution_id=motion_execution_id,
                    target=target,
                    arm_side=self.pick_service.default_arm_side,
                    reference_pose=reference_pose,
                    viewpoint_index=self._active_view_viewpoint_index,
                    max_duration_s=min(self.max_atomic_duration_s, 2.0),
                )
            )
            self.atomic_results[motion_execution_id] = motion_result
            if not motion_result.success:
                self.state = HarvestCycleState.ACTIVE_VIEW
                raise HarvestCycleServiceError(
                    f"active-view motion failed ({motion_result.failure_code.value}): "
                    f"{motion_result.message}"
                )
            if motion_result.output.get("motion_executed") is not True:
                self.state = HarvestCycleState.ACTIVE_VIEW
                return {
                    **motion_result.as_dict(),
                    "observation_refreshed": False,
                    "workflow_state": self.state.value,
                }
            self.state = HarvestCycleState.ACTIVE_VIEW_OBSERVE
        else:
            motion_result = self.atomic_results.get(motion_execution_id)
            if motion_result is None or not motion_result.success:
                raise HarvestCycleServiceError(
                    "active-view observation retry lost its successful motion result"
                )

        self._active_view_capture_attempt += 1
        observe_execution_id = (
            f"active-view-observe-{target_id}-{selection_revision}-"
            f"{self._active_view_viewpoint_index}-{self._active_view_capture_attempt}"
        )
        observe_result = self.target_reobserve.run(
            TargetReobserveRequest(
                execution_id=observe_execution_id,
                target_id=target_id,
                observation=observation,
                prior_sample_monotonic_ns=target.sample_monotonic_ns,
                force_new_sample=True,
                enforce_visibility_gate=False,
            )
        )
        self.atomic_results[observe_execution_id] = observe_result
        if not observe_result.success:
            self.state = HarvestCycleState.ACTIVE_VIEW_OBSERVE
            raise HarvestCycleServiceError(
                f"active-view observation failed ({observe_result.failure_code.value}): "
                f"{observe_result.message}"
            )
        refreshed_target = self._target_from_result(observe_result)
        previous_occlusion = target.occlusion
        self.pick_service.targets[target_id] = refreshed_target
        self.pick_service.last_detection = TomatoDetectionResult(
            targets=(refreshed_target,),
            observation_sequence=refreshed_target.observation_sequence,
            model="active_view_target_reobserve_v1",
            sample_monotonic_ns=refreshed_target.sample_monotonic_ns,
        )
        self._active_view_viewpoint_index += 1
        self.state = HarvestCycleState.SELECT
        return {
            "success": True,
            "motion": motion_result.as_dict(),
            "observation": observe_result.as_dict(),
            "target_id": target_id,
            "previous_occlusion_ratio": previous_occlusion,
            "current_occlusion_ratio": refreshed_target.occlusion,
            "visibility_improvement": previous_occlusion - refreshed_target.occlusion,
            "observation_refreshed": True,
            "workflow_state": self.state.value,
        }

    def reobserve_tomato(self, observation: Observation, target_id: str) -> dict[str, Any]:
        self._require_open()
        target_id = self._identifier(target_id, "target_id")
        if self._selected_target_id is None:
            raise HarvestCycleServiceError(
                "call select_harvest_target before target re-observation"
            )
        if target_id != self._selected_target_id:
            raise HarvestCycleServiceError(
                f"target {target_id!r} is not the pinned selected target"
            )
        if target_id not in self.pick_service.targets:
            raise HarvestCycleServiceError(
                f"unknown target_id {target_id!r}; call detect_tomatoes first"
            )
        detection = self.pick_service.last_detection
        observation_sequence = observation.sequence if detection is None else detection.observation_sequence
        attempt = self._reobserve_attempts.get(target_id, 0) + 1
        self._reobserve_attempts[target_id] = attempt
        execution_id = f"reobserve-{target_id}-{observation_sequence}-{attempt}"
        result = self.target_reobserve.run(
            TargetReobserveRequest(
                execution_id=execution_id,
                target_id=target_id,
                observation=observation,
                prior_sample_monotonic_ns=self.pick_service.targets[
                    target_id
                ].sample_monotonic_ns,
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            self.state = HarvestCycleState.REOBSERVE
            raise HarvestCycleServiceError(
                f"target re-observation failed ({result.failure_code.value}): {result.message}"
            )
        refreshed_target = self._target_from_result(result)
        self.pick_service.targets[target_id] = refreshed_target
        self.pick_service.last_detection = TomatoDetectionResult(
            targets=(refreshed_target,),
            observation_sequence=refreshed_target.observation_sequence,
            model="atomic_target_reobserve_v1",
            sample_monotonic_ns=refreshed_target.sample_monotonic_ns,
        )
        self._reobserved_targets.add(target_id)
        self._clearance_by_plan.clear()
        self.state = HarvestCycleState.PLAN
        return {**result.as_dict(), "workflow_state": self.state.value}

    def plan_tomato_grasp(self, target_id: str, arm_side: str = "auto") -> dict[str, Any]:
        self._require_open()
        target_id = self._identifier(target_id, "target_id")
        if target_id != self._selected_target_id:
            raise HarvestCycleServiceError(
                f"target {target_id!r} is not the pinned selected target"
            )
        if target_id not in self._reobserved_targets:
            raise HarvestCycleServiceError(
                f"target {target_id!r} must be re-observed before planning"
            )
        summary = self._pick(self.pick_service.plan_tomato_grasp, target_id, arm_side)
        self._clearance_by_plan.clear()
        self.state = HarvestCycleState.CLEARANCE
        return {**summary, "workflow_state": self.state.value}

    def check_pregrasp_clearance(
        self, observation: Observation, grasp_plan_id: str
    ) -> dict[str, Any]:
        self._require_open()
        grasp_plan_id = self._identifier(grasp_plan_id, "grasp_plan_id")
        try:
            plan = self.pick_service.plans[grasp_plan_id]
            target = self.pick_service.targets[plan.target_id]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                f"unknown grasp_plan_id {grasp_plan_id!r}; plan first"
            ) from exc
        execution_id = f"clearance-{grasp_plan_id}-{observation.sequence}"
        result = self.pregrasp_clearance_check_skill.run(
            PregraspClearanceRequest(
                execution_id=execution_id,
                target=target,
                plan=plan,
                observation=observation,
                max_duration_s=min(self.max_atomic_duration_s, 1.0),
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            self._clearance_by_plan.pop(grasp_plan_id, None)
            self.state = HarvestCycleState.CLEARANCE
            raise HarvestCycleServiceError(
                f"pregrasp clearance failed ({result.failure_code.value}): {result.message}"
            )
        if result.output.get("motion_authorized") is not True:
            self.state = HarvestCycleState.CLEARANCE
            raise HarvestCycleServiceError(
                "pregrasp clearance succeeded without motion authorization"
            )
        self._clearance_by_plan[grasp_plan_id] = execution_id
        self.state = HarvestCycleState.PICK_READY
        return {**result.as_dict(), "workflow_state": self.state.value}

    def move_to_pick_ready(self, grasp_plan_id: str) -> dict[str, Any]:
        self._require_open()
        grasp_plan_id = self._identifier(grasp_plan_id, "grasp_plan_id")
        if grasp_plan_id not in self._clearance_by_plan:
            raise HarvestCycleServiceError(
                "call check_pregrasp_clearance for this exact grasp plan before ready motion"
            )
        try:
            plan = self.pick_service.plans[grasp_plan_id]
            target = self.pick_service.targets[plan.target_id]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                f"unknown grasp_plan_id {grasp_plan_id!r}; plan first"
            ) from exc
        result = self.move_to_pick_ready_skill.run(
            MoveToPickReadyRequest(
                execution_id=f"ready-{grasp_plan_id}",
                target=target,
                plan=plan,
                max_duration_s=min(self.max_atomic_duration_s, 2.0),
            )
        )
        self.atomic_results[result.execution_id] = result
        if not result.success:
            raise HarvestCycleServiceError(
                f"PPO ready motion failed ({result.failure_code.value}): {result.message}"
            )
        if bool(result.output.get("shadow_only")):
            self.state = HarvestCycleState.PICK_READY
            return {
                **result.as_dict(),
                "ppo_execution_allowed": False,
                "workflow_state": self.state.value,
            }
        assessment_for = getattr(self.move_to_pick_ready_skill, "assessment_for", None)
        if not callable(assessment_for):
            raise HarvestCycleServiceError(
                "move_to_pick_ready skill did not expose its bound safety assessment"
            )
        self.pick_service.assessments[grasp_plan_id] = assessment_for(grasp_plan_id)
        from task.agri_pick.state import AgriPickState

        self.pick_service.state = AgriPickState.PICK
        self.state = HarvestCycleState.PICK
        return {
            **result.as_dict(),
            "ppo_execution_allowed": True,
            "workflow_state": self.state.value,
        }

    def move_to_pregrasp(self, grasp_plan_id: str) -> dict[str, Any]:
        """Compatibility alias; new harvest orchestration uses move_to_pick_ready."""

        return self.move_to_pick_ready(grasp_plan_id)

    def execute_tomato_pick(self, grasp_plan_id: str) -> dict[str, Any]:
        self._require_open()
        summary = self._pick(self.pick_service.execute_tomato_pick, grasp_plan_id)
        self.state = HarvestCycleState.RETREAT
        return {**summary, "workflow_state": self.state.value}

    def retreat_with_tomato(self, execution_id: str) -> dict[str, Any]:
        self._require_open()
        execution_id = self._identifier(execution_id, "execution_id")
        existing_retreat_id = f"retreat-{execution_id}"
        cached = self.atomic_results.get(existing_retreat_id)
        if cached is not None:
            return {**cached.as_dict(), "workflow_state": self.state.value}
        pick_verification = self._pick(self.pick_service.verify_tomato_pick, execution_id)
        if not bool(pick_verification["success"]):
            self.state = HarvestCycleState.ABORTED
            raise HarvestCycleServiceError(
                "cannot retreat payload because pick detachment/retention was not verified"
            )
        pick_result = self.pick_service.executions[execution_id]
        plan = self.pick_service.plans[pick_result.grasp_plan_id]
        payload = HarvestPayload(
            payload_id=f"payload-{execution_id}",
            target_id=pick_result.target_id,
            pick_execution_id=execution_id,
            arm_side=plan.arm_side,
            detached=True,
            retained=True,
            acquired_monotonic_ns=self.pick_service.robotics.clock.monotonic_ns(),
        )
        result = self.payload_retreat.run(
            PayloadRetreatRequest(
                execution_id=existing_retreat_id,
                payload=payload,
                retreat_pose=plan.retreat_pose,
                max_duration_s=min(self.max_atomic_duration_s, 5.0),
            )
        )
        self.atomic_results[existing_retreat_id] = result
        self.payloads[payload.payload_id] = payload
        self._payload_by_retreat[existing_retreat_id] = payload.payload_id
        if not result.success:
            self._route_payload_failure(result, payload)
            raise HarvestCycleServiceError(
                f"payload retreat failed ({result.failure_code.value}): {result.message}"
            )
        self.state = HarvestCycleState.TRANSPORT
        return {**result.as_dict(), "workflow_state": self.state.value}

    def transport_payload(self, retreat_execution_id: str) -> dict[str, Any]:
        self._require_open()
        retreat_execution_id = self._identifier(
            retreat_execution_id, "retreat_execution_id"
        )
        try:
            payload = self.payloads[self._payload_by_retreat[retreat_execution_id]]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                f"unknown retreat_execution_id {retreat_execution_id!r}; retreat first"
            ) from exc
        transport_execution_id = f"transport-{payload.payload_id}"
        cached = self.atomic_results.get(transport_execution_id)
        if cached is not None and cached.success:
            return {**cached.as_dict(), "workflow_state": self.state.value}
        result = self.transport_payload_skill.run(
            PayloadTransportRequest(
                execution_id=transport_execution_id,
                payload=payload,
                container=self.container,
                max_duration_s=self.max_atomic_duration_s,
            )
        )
        self.atomic_results[transport_execution_id] = result
        if not result.success:
            self._route_payload_failure(result, payload)
            raise HarvestCycleServiceError(
                f"payload transport failed ({result.failure_code.value}): {result.message}"
            )
        if result.output.get("place_authorized") is not True:
            self.state = HarvestCycleState.ABORTED
            raise HarvestCycleServiceError(
                "payload transport succeeded without place authorization"
            )
        self._payload_by_transport[transport_execution_id] = payload.payload_id
        self._retreat_by_transport[transport_execution_id] = retreat_execution_id
        self.state = HarvestCycleState.PLACE
        return {
            **result.as_dict(),
            "transport_execution_id": transport_execution_id,
            "workflow_state": self.state.value,
        }

    def place_tomato(self, transport_execution_id: str) -> dict[str, Any]:
        self._require_open()
        transport_execution_id = self._identifier(
            transport_execution_id, "transport_execution_id"
        )
        try:
            payload = self.payloads[self._payload_by_transport[transport_execution_id]]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                f"unknown transport_execution_id {transport_execution_id!r}; transport first"
            ) from exc
        preplace_pose_for = getattr(self.transport_payload_skill, "preplace_pose_for", None)
        if not callable(preplace_pose_for):
            raise HarvestCycleServiceError(
                "transport_payload skill did not expose its bound pre-place pose"
            )
        preplace_pose = preplace_pose_for(transport_execution_id)
        place_execution_id = f"place-{payload.payload_id}"
        result = self.payload_place.run(
            PayloadPlaceRequest(
                execution_id=place_execution_id,
                payload=payload,
                container=self.container,
                preplace_pose=preplace_pose,
                max_duration_s=self.max_atomic_duration_s,
            )
        )
        self.atomic_results[place_execution_id] = result
        if not result.success:
            self._route_payload_failure(result, payload)
            raise HarvestCycleServiceError(
                f"payload place failed ({result.failure_code.value}): {result.message}"
            )
        output = result.output
        receipt = PlacementReceipt(
            placement_id=str(output["placement_id"]),
            payload_id=payload.payload_id,
            container_id=self.container.container_id,
            release_confirmed=bool(output["release_confirmed"]),
            hand_empty=bool(output["hand_empty"]),
            dropped=bool(output["dropped"]),
            evidence_sources=tuple(output["evidence_sources"]),  # type: ignore[arg-type]
        )
        self.receipts[receipt.placement_id] = receipt
        self._payload_by_placement[receipt.placement_id] = payload.payload_id
        self.state = HarvestCycleState.VERIFY
        return {
            **result.as_dict(),
            "placement_id": receipt.placement_id,
            "workflow_state": self.state.value,
            "admission_execution_id": self._admission_execution_id,
        }

    def localize_dropped_payload(self) -> dict[str, Any]:
        """Locate the most recently dropped payload before any recovery motion."""

        self._require_open()
        payload_id = self._dropped_payload_id
        if (
            self.state is not HarvestCycleState.DROPPED_PAYLOAD_RECOVERY
            or payload_id is None
        ):
            raise HarvestCycleServiceError(
                "localize_dropped_payload requires a detected payload_dropped failure"
            )
        payload = self.payloads[payload_id]
        execution_id = f"localize-dropped-{payload_id}"
        result = self.localize_dropped_payload_skill.run(
            DroppedPayloadLocalizationRequest(
                execution_id=execution_id,
                payload=payload,
                container=self.container,
                max_observation_age_s=min(self.max_atomic_duration_s, 2.0),
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            raise HarvestCycleServiceError(
                f"dropped payload localization failed ({result.failure_code.value}): "
                f"{result.message}"
            )
        disposition = result.output.get("disposition")
        if disposition == "manual_intervention_required":
            self.state = HarvestCycleState.ABORTED
        else:
            self.state = HarvestCycleState.DROPPED_PAYLOAD_RECOVERY
        return {**result.as_dict(), "workflow_state": self.state.value}

    def verify_harvest_cycle(self, placement_id: str) -> dict[str, Any]:
        self._require_open()
        placement_id = self._identifier(placement_id, "placement_id")
        try:
            receipt = self.receipts[placement_id]
            payload = self.payloads[self._payload_by_placement[placement_id]]
        except KeyError as exc:
            raise HarvestCycleServiceError(
                f"unknown placement_id {placement_id!r}; place the payload first"
            ) from exc
        placement = self.placement_verifier.verify(receipt, payload, self.container)
        retreat_id = f"retreat-{payload.pick_execution_id}"
        retreat_result = self.atomic_results[retreat_id]
        transport_id = f"transport-{payload.payload_id}"
        transport_result = self.atomic_results[transport_id]
        pick_result = self.pick_service.executions[payload.pick_execution_id]
        evidence_sources = tuple(
            dict.fromkeys(
                (
                    *pick_result.verification.evidence_sources,
                    *(ref for event in retreat_result.trace for ref in event.evidence_refs),
                    *(ref for event in transport_result.trace for ref in event.evidence_refs),
                    *placement.evidence_sources,
                )
            )
        )
        cycle_id = f"cycle-{payload.pick_execution_id}"
        verification = HarvestCycleVerification(
            cycle_id=cycle_id,
            target_id=payload.target_id,
            pick_verified=pick_result.success and pick_result.verification.successful,
            retreat_verified=retreat_result.success,
            transport_verified=transport_result.success,
            placed_in_container=placement.placed_in_container,
            release_confirmed=placement.release_confirmed,
            hand_empty=placement.hand_empty,
            dropped=placement.dropped,
            evidence_sources=evidence_sources,
            fruit_damage_detected=None,
        )
        self.verifications[cycle_id] = verification
        if verification.successful:
            self._completed_cycle_id = cycle_id
            self.state = HarvestCycleState.COMPLETE
        else:
            self.state = HarvestCycleState.ABORTED
        return {
            "cycle_id": cycle_id,
            "target_id": verification.target_id,
            "success": verification.successful,
            "workflow_state": self.state.value,
            "pick_verified": verification.pick_verified,
            "retreat_verified": verification.retreat_verified,
            "transport_verified": verification.transport_verified,
            "placed_in_container": verification.placed_in_container,
            "release_confirmed": verification.release_confirmed,
            "hand_empty": verification.hand_empty,
            "dropped": verification.dropped,
            "fruit_damage_detected": verification.fruit_damage_detected,
            "evidence_sources": verification.evidence_sources,
        }

    def reset_for_next_target(self) -> dict[str, Any]:
        self._require_open()
        cycle_id = self._completed_cycle_id
        if (
            self.state is not HarvestCycleState.COMPLETE
            or cycle_id is None
            or not self.verifications[cycle_id].successful
        ):
            raise HarvestCycleServiceError(
                "reset_for_next_target requires one successfully verified completed cycle"
            )
        self._reset_attempt += 1
        execution_id = f"reset-{cycle_id}-{self._reset_attempt}"
        self.state = HarvestCycleState.RESET
        result = self.reset_for_next_target_skill.run(
            ResetForNextTargetRequest(
                execution_id=execution_id,
                completed_cycle_id=cycle_id,
                arm_side=self.pick_service.default_arm_side,
                reset_pose=self.safe_pose,
                max_duration_s=min(self.max_atomic_duration_s, 5.0),
            )
        )
        self.atomic_results[execution_id] = result
        if not result.success:
            self.state = HarvestCycleState.COMPLETE
            raise HarvestCycleServiceError(
                f"next-target reset failed ({result.failure_code.value}): {result.message}"
            )
        self.pick_service.reset_transaction()
        self._reobserved_targets.clear()
        self._clearance_by_plan.clear()
        self._selected_target_id = None
        self._reobserve_attempts.clear()
        self._reset_active_view()
        self._completed_cycle_id = None
        self._dropped_payload_id = None
        self.state = HarvestCycleState.OBSERVE
        return {
            **result.as_dict(),
            "reset_cycle_id": cycle_id,
            "workflow_state": self.state.value,
        }

    def recover_harvest_cycle(self, reason: str) -> dict[str, Any]:
        self._require_open()
        reason = self._identifier(reason, "reason")
        execution_id = f"safe-retract-{len(self.atomic_results) + 1}"
        result = self.safe_retract.run(
            SafeRetractRequest(
                execution_id=execution_id,
                arm_side=self.pick_service.default_arm_side,
                safe_pose=self.safe_pose,
                reason=reason,
                max_duration_s=min(self.max_atomic_duration_s, 5.0),
            )
        )
        self.atomic_results[execution_id] = result
        self.state = HarvestCycleState.ABORTED
        if not result.success:
            raise HarvestCycleServiceError(
                f"safe retract failed ({result.failure_code.value}): {result.message}"
            )
        return {**result.as_dict(), "workflow_state": self.state.value}

    def _route_payload_failure(
        self, result: AtomicSkillResult, payload: HarvestPayload
    ) -> None:
        if (
            result.failure_code is not None
            and result.failure_code.value == "payload_dropped"
        ):
            self._dropped_payload_id = payload.payload_id
            self.state = HarvestCycleState.DROPPED_PAYLOAD_RECOVERY
            return
        self.state = HarvestCycleState.ABORTED

    def status_summary(self) -> dict[str, Any]:
        return {
            "workflow_state": self.state.value,
            "admission_execution_id": self._admission_execution_id,
            "detected_target_ids": tuple(self.pick_service.targets),
            "selected_target_id": self._selected_target_id,
            "pending_active_view_target_id": self._pending_active_view_target_id,
            "active_view_viewpoint_index": self._active_view_viewpoint_index,
            "reobserved_target_ids": tuple(sorted(self._reobserved_targets)),
            "plan_ids": tuple(self.pick_service.plans),
            "clearance_plan_ids": tuple(self._clearance_by_plan),
            "pick_execution_ids": tuple(self.pick_service.executions),
            "atomic_execution_ids": tuple(self.atomic_results),
            "payload_ids": tuple(self.payloads),
            "dropped_payload_id": self._dropped_payload_id,
            "transport_execution_ids": tuple(self._payload_by_transport),
            "placement_ids": tuple(self.receipts),
            "completed_cycle_id": self._completed_cycle_id,
            "verified_cycle_ids": tuple(self.verifications),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for resource in (
            self.payload_place,
            self.localize_dropped_payload_skill,
            self.reset_for_next_target_skill,
            self.payload_retreat,
            self.transport_payload_skill,
            self.safe_retract,
            self.target_reobserve,
            self.pregrasp_clearance_check_skill,
            self.select_harvest_target_skill,
            self.active_view_reposition_skill,
            self.move_to_pick_ready_skill,
            self.system_self_check,
            self.placement_verifier,
            self.pick_service,
        ):
            try:
                resource.close()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("harvest-cycle service cleanup failed", errors)

    @staticmethod
    def _identifier(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise HarvestCycleServiceError(f"{field_name} must be a non-empty string")
        return value.strip()

    def _prepare_active_view_recovery(self, result: AtomicSkillResult) -> None:
        if (
            result.failure_code is not None
            and result.failure_code.value == "viewpoint_insufficient"
            and result.output.get("next_action") == "active_view_required"
        ):
            ranked = result.output.get("ranked_candidates", ())
            candidate_id: str | None = None
            for candidate in ranked if isinstance(ranked, (tuple, list)) else ():
                if not isinstance(candidate, dict) and not hasattr(candidate, "items"):
                    continue
                values = dict(candidate)  # type: ignore[arg-type]
                reasons = tuple(values.get("rejection_reasons", ()))
                target_id = values.get("target_id")
                if reasons == ("occlusion_above_maximum",) and isinstance(
                    target_id, str
                ):
                    candidate_id = target_id
                    break
            if candidate_id is not None and candidate_id in self.pick_service.targets:
                if candidate_id != self._pending_active_view_target_id:
                    arm = self.pick_service.robotics.arm.get_state(
                        self.pick_service.default_arm_side
                    )
                    self._active_view_reference_pose = arm.ee_pose
                    self._active_view_viewpoint_index = 0
                    self._active_view_capture_attempt = 0
                self._pending_active_view_target_id = candidate_id
                revision = result.output.get("selection_revision")
                if not isinstance(revision, str) or not revision:
                    raise HarvestCycleServiceError(
                        "active-view selection result omitted selection_revision"
                    )
                self._active_view_selection_revision = revision
                self.state = HarvestCycleState.ACTIVE_VIEW
                return
        self._reset_active_view()
        self.state = HarvestCycleState.SELECT

    def _reset_active_view(self) -> None:
        self._pending_active_view_target_id = None
        self._active_view_reference_pose = None
        self._active_view_viewpoint_index = 0
        self._active_view_selection_revision = None
        self._active_view_capture_attempt = 0

    @staticmethod
    def _target_from_result(result: AtomicSkillResult) -> TomatoTarget:
        target_output = result.output.get("target")
        if not isinstance(target_output, dict) and not hasattr(target_output, "items"):
            raise HarvestCycleServiceError(
                "target re-observation did not return updated target geometry"
            )
        target_values = dict(target_output)  # type: ignore[arg-type]
        pose_values = target_values.get("center_pose")
        if not isinstance(pose_values, dict) and not hasattr(pose_values, "items"):
            raise HarvestCycleServiceError(
                "target re-observation returned invalid center_pose geometry"
            )
        target_values["center_pose"] = Pose(**dict(pose_values))  # type: ignore[arg-type]
        return TomatoTarget(**target_values)

    @staticmethod
    def _pick(handler: Callable[..., dict[str, Any]], *args: object) -> dict[str, Any]:
        try:
            return handler(*args)
        except TomatoPickServiceError as exc:
            raise HarvestCycleServiceError(str(exc)) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("HarvestCycleService is closed")


__all__ = ["HarvestCycleService", "HarvestCycleServiceError"]
