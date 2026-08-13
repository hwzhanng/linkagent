"""DTOs for composing tomato picking into a verified harvest cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from ._base_proto import Observation, _finite_float, _freeze_mapping
from .manipulation import Pose
from .tomato_pick import (
    ArmSide,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoTarget,
)


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive(value: object, *, field_name: str) -> float:
    numeric = _finite_float(value, field_name=field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return numeric


def _optional_non_negative_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0 or None")
    return value


def _arm_side(value: object) -> ArmSide:
    if isinstance(value, ArmSide):
        return value
    try:
        return ArmSide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_side must be left or right") from exc


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    return tuple(_non_empty(item, field_name=f"{field_name} item") for item in items)


class HarvestCycleState(str, Enum):
    INIT = "init"
    OBSERVE = "observe"
    SELECT = "select"
    ACTIVE_VIEW = "active_view"
    ACTIVE_VIEW_OBSERVE = "active_view_observe"
    REOBSERVE = "reobserve"
    PLAN = "plan"
    CLEARANCE = "clearance"
    PICK_READY = "pick_ready"
    PICK = "pick"
    RETREAT = "retreat"
    TRANSPORT = "transport"
    PLACE = "place"
    VERIFY = "verify"
    DROPPED_PAYLOAD_RECOVERY = "dropped_payload_recovery"
    COMPLETE = "complete"
    RESET = "reset"
    ABORTED = "aborted"


class HarvestTargetNextAction(str, Enum):
    REOBSERVE_SELECTED = "reobserve_selected"
    ACTIVE_VIEW_REQUIRED = "active_view_required"
    NO_HARVESTABLE_TARGET = "no_harvestable_target"
    REACHABILITY_CONTEXT_REQUIRED = "reachability_context_required"


class DroppedPayloadDisposition(str, Enum):
    """Deterministic routing after a lost fruit has been localized."""

    IN_CONTAINER = "in_container"
    DEDICATED_RECOVERY_REQUIRED = "dedicated_recovery_required"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


@dataclass(frozen=True, slots=True)
class ActiveViewRepositionRequest:
    execution_id: str
    target: TomatoTarget
    arm_side: ArmSide
    reference_pose: Pose
    viewpoint_index: int
    max_duration_s: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        if not isinstance(self.reference_pose, Pose):
            raise TypeError("reference_pose must be a Pose")
        if self.reference_pose.frame_id != "world":
            raise ValueError("reference_pose.frame_id must be world")
        if (
            isinstance(self.viewpoint_index, bool)
            or not isinstance(self.viewpoint_index, int)
            or self.viewpoint_index < 0
        ):
            raise ValueError("viewpoint_index must be an integer >= 0")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class ActiveViewRepositionEvidence:
    target_id: str
    viewpoint_id: str
    profile_id: str
    arm_side: ArmSide
    requested_pose: Pose
    actual_pose: Pose
    motion_executed: bool
    shadow_only: bool
    simulated_safety_only: bool
    ik_reachable: bool
    position_error_m: float
    orientation_error_rad: float
    command_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("target_id", "viewpoint_id", "profile_id"):
            object.__setattr__(
                self, name, _non_empty(getattr(self, name), field_name=name)
            )
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        if not isinstance(self.requested_pose, Pose) or not isinstance(
            self.actual_pose, Pose
        ):
            raise TypeError("requested_pose and actual_pose must be Pose values")
        for name in (
            "motion_executed",
            "shadow_only",
            "simulated_safety_only",
            "ik_reachable",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("position_error_m", "orientation_error_rad"):
            value = _finite_float(getattr(self, name), field_name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        if self.command_id is not None:
            object.__setattr__(
                self,
                "command_id",
                _non_empty(self.command_id, field_name="command_id"),
            )
        if self.motion_executed and self.command_id is None:
            raise ValueError("executed active-view motion requires command_id")
        if self.shadow_only and self.motion_executed:
            raise ValueError("shadow active-view evidence cannot claim motion")

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "viewpoint_id": self.viewpoint_id,
            "profile_id": self.profile_id,
            "arm_side": self.arm_side.value,
            "requested_pose": self.requested_pose.as_dict(),
            "actual_pose": self.actual_pose.as_dict(),
            "motion_executed": self.motion_executed,
            "shadow_only": self.shadow_only,
            "simulated_safety_only": self.simulated_safety_only,
            "ik_reachable": self.ik_reachable,
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "command_id": self.command_id,
        }


@dataclass(frozen=True, slots=True)
class SelectHarvestTargetRequest:
    execution_id: str
    detection: TomatoDetectionResult
    arm_side: ArmSide
    max_detection_age_s: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.detection, TomatoDetectionResult):
            raise TypeError("detection must be a TomatoDetectionResult")
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        object.__setattr__(
            self,
            "max_detection_age_s",
            _positive(self.max_detection_age_s, field_name="max_detection_age_s"),
        )


@dataclass(frozen=True, slots=True)
class HarvestTargetCandidateScore:
    target_id: str
    eligible: bool
    score: float
    hard_gates: Mapping[str, object]
    score_components: Mapping[str, object]
    rejection_reasons: tuple[str, ...]
    workspace_margin_m: float | None
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _non_empty(self.target_id, field_name="target_id"))
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be a bool")
        score = _finite_float(self.score, field_name="score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        object.__setattr__(self, "score", score)
        gates = _freeze_mapping(self.hard_gates, field_name="hard_gates")
        if not gates or not all(
            isinstance(name, str) and isinstance(value, (bool, type(None)))
            for name, value in gates.items()
        ):
            raise TypeError("hard_gates must map names to bool or None")
        components = _freeze_mapping(
            self.score_components, field_name="score_components"
        )
        for name, value in components.items():
            if not isinstance(name, str):
                raise TypeError("score_components keys must be strings")
            numeric = _finite_float(value, field_name=f"score_components.{name}")
            if not 0.0 <= numeric <= 1.0:
                raise ValueError("score component values must be in [0, 1]")
        reasons = _string_tuple(
            self.rejection_reasons, field_name="rejection_reasons"
        )
        if self.eligible and reasons:
            raise ValueError("eligible candidate cannot have rejection reasons")
        if not self.eligible and not reasons:
            raise ValueError("ineligible candidate requires rejection reasons")
        object.__setattr__(self, "hard_gates", gates)
        object.__setattr__(self, "score_components", components)
        object.__setattr__(self, "rejection_reasons", reasons)
        if self.workspace_margin_m is not None:
            object.__setattr__(
                self,
                "workspace_margin_m",
                _finite_float(self.workspace_margin_m, field_name="workspace_margin_m"),
            )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "eligible": self.eligible,
            "score": self.score,
            "hard_gates": dict(self.hard_gates),
            "score_components": dict(self.score_components),
            "rejection_reasons": list(self.rejection_reasons),
            "workspace_margin_m": self.workspace_margin_m,
            "evidence_sources": list(self.evidence_sources),
        }


@dataclass(frozen=True, slots=True)
class HarvestTargetSelection:
    selection_revision: str
    observation_sequence: int
    sample_monotonic_ns: int
    selected_target_id: str | None
    next_action: HarvestTargetNextAction
    ranked_candidates: tuple[HarvestTargetCandidateScore, ...]
    motion_authorized: bool = False
    requires_pregrasp_clearance_check: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_revision",
            _non_empty(self.selection_revision, field_name="selection_revision"),
        )
        for name in ("observation_sequence", "sample_monotonic_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        try:
            action = (
                self.next_action
                if isinstance(self.next_action, HarvestTargetNextAction)
                else HarvestTargetNextAction(self.next_action)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid harvest target next_action") from exc
        object.__setattr__(self, "next_action", action)
        candidates = tuple(self.ranked_candidates)
        if not all(isinstance(item, HarvestTargetCandidateScore) for item in candidates):
            raise TypeError("ranked_candidates must contain candidate scores")
        ids = tuple(item.target_id for item in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("ranked candidate target IDs must be unique")
        object.__setattr__(self, "ranked_candidates", candidates)
        if self.selected_target_id is not None:
            object.__setattr__(
                self,
                "selected_target_id",
                _non_empty(self.selected_target_id, field_name="selected_target_id"),
            )
        if action is HarvestTargetNextAction.REOBSERVE_SELECTED:
            if self.selected_target_id is None:
                raise ValueError("reobserve_selected requires selected_target_id")
            selected = next(
                (item for item in candidates if item.target_id == self.selected_target_id),
                None,
            )
            if selected is None or not selected.eligible:
                raise ValueError("selected target must be an eligible ranked candidate")
        elif self.selected_target_id is not None:
            raise ValueError("non-selection next_action cannot carry selected_target_id")
        if not isinstance(self.motion_authorized, bool) or not isinstance(
            self.requires_pregrasp_clearance_check, bool
        ):
            raise TypeError("selection authorization flags must be bools")
        if self.motion_authorized:
            raise ValueError("target selection cannot authorize robot motion")

    def as_dict(self) -> dict[str, object]:
        return {
            "selection_revision": self.selection_revision,
            "observation_sequence": self.observation_sequence,
            "sample_monotonic_ns": self.sample_monotonic_ns,
            "selected_target_id": self.selected_target_id,
            "next_action": self.next_action.value,
            "ranked_candidates": [item.as_dict() for item in self.ranked_candidates],
            "motion_authorized": self.motion_authorized,
            "requires_pregrasp_clearance_check": self.requires_pregrasp_clearance_check,
        }


@dataclass(frozen=True, slots=True)
class TargetReobserveRequest:
    execution_id: str
    target_id: str
    observation: Observation
    max_observation_age_s: float = 1.0
    prior_sample_monotonic_ns: int | None = None
    require_new_sample: bool = True
    force_new_sample: bool = False
    enforce_visibility_gate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        object.__setattr__(self, "target_id", _non_empty(self.target_id, field_name="target_id"))
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        object.__setattr__(
            self,
            "max_observation_age_s",
            _positive(self.max_observation_age_s, field_name="max_observation_age_s"),
        )
        object.__setattr__(
            self,
            "prior_sample_monotonic_ns",
            _optional_non_negative_int(
                self.prior_sample_monotonic_ns,
                field_name="prior_sample_monotonic_ns",
            ),
        )
        if not isinstance(self.require_new_sample, bool):
            raise TypeError("require_new_sample must be a bool")
        if not isinstance(self.force_new_sample, bool):
            raise TypeError("force_new_sample must be a bool")
        if not isinstance(self.enforce_visibility_gate, bool):
            raise TypeError("enforce_visibility_gate must be a bool")


@dataclass(frozen=True, slots=True)
class TargetReobserveEvidence:
    target_id: str
    observed: bool
    occlusion_ratio: float
    depth_available: bool
    evidence_sources: tuple[str, ...]
    target: TomatoTarget | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _non_empty(self.target_id, field_name="target_id"))
        if not isinstance(self.observed, bool) or not isinstance(self.depth_available, bool):
            raise TypeError("observed and depth_available must be bools")
        ratio = _finite_float(self.occlusion_ratio, field_name="occlusion_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("occlusion_ratio must be in [0, 1]")
        object.__setattr__(self, "occlusion_ratio", ratio)
        sources = _string_tuple(
            self.evidence_sources,
            field_name="evidence_sources",
        )
        if self.observed and not sources:
            raise ValueError("observed evidence requires at least one evidence source")
        if self.target is not None:
            if not isinstance(self.target, TomatoTarget):
                raise TypeError("target must be a TomatoTarget or None")
            if self.target.target_id != self.target_id:
                raise ValueError("target.target_id must agree with target_id")
        object.__setattr__(self, "evidence_sources", sources)
        object.__setattr__(self, "details", _freeze_mapping(self.details, field_name="details"))


@dataclass(frozen=True, slots=True)
class ClearanceScene:
    """A bounded world-frame point sample used only for pregrasp clearance."""

    scene_id: str
    sample_monotonic_ns: int
    points_world_m: tuple[tuple[float, float, float], ...]
    coverage_ratio: float
    calibration_id: str
    frame_id: str = "world"
    simulated: bool = False
    evidence_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _non_empty(self.scene_id, field_name="scene_id"))
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _optional_non_negative_int(
                self.sample_monotonic_ns, field_name="sample_monotonic_ns"
            ),
        )
        if self.sample_monotonic_ns is None:
            raise ValueError("sample_monotonic_ns must be an integer >= 0")
        if isinstance(self.points_world_m, (str, bytes, bytearray)):
            raise TypeError("points_world_m must be a sequence of xyz points")
        points = tuple(self.points_world_m)
        if len(points) > 20_000:
            raise ValueError("points_world_m must contain at most 20000 points")
        prepared: list[tuple[float, float, float]] = []
        for index, point in enumerate(points):
            if isinstance(point, (str, bytes, bytearray)):
                raise TypeError(f"points_world_m[{index}] must be an xyz sequence")
            values = tuple(point)
            if len(values) != 3:
                raise ValueError(f"points_world_m[{index}] must contain exactly three values")
            prepared.append(
                tuple(
                    _finite_float(value, field_name=f"points_world_m[{index}][{axis}]")
                    for axis, value in enumerate(values)
                )  # type: ignore[arg-type]
            )
        object.__setattr__(self, "points_world_m", tuple(prepared))
        coverage = _finite_float(self.coverage_ratio, field_name="coverage_ratio")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage_ratio must be in [0, 1]")
        object.__setattr__(self, "coverage_ratio", coverage)
        object.__setattr__(
            self, "calibration_id", _non_empty(self.calibration_id, field_name="calibration_id")
        )
        object.__setattr__(self, "frame_id", _non_empty(self.frame_id, field_name="frame_id"))
        if self.frame_id != "world":
            raise ValueError("clearance scene frame_id must be world")
        if not isinstance(self.simulated, bool):
            raise TypeError("simulated must be a bool")
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )


@dataclass(frozen=True, slots=True)
class PregraspClearanceRequest:
    execution_id: str
    target: TomatoTarget
    plan: TomatoGraspPlan
    observation: Observation
    max_duration_s: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        if not isinstance(self.plan, TomatoGraspPlan):
            raise TypeError("plan must be a TomatoGraspPlan")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        if self.target.target_id != self.plan.target_id:
            raise ValueError("target and plan target IDs must agree")
        if self.target.observation_sequence != self.plan.source_observation_sequence:
            raise ValueError("target and plan observation revisions must agree")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class PregraspClearanceEvidence:
    target_id: str
    grasp_plan_id: str
    scene_id: str
    sample_monotonic_ns: int
    point_count: int
    coverage_ratio: float
    checked_segments: tuple[str, ...]
    clearance_required_m: float
    minimum_clearance_m: float
    ignored_target_point_count: int
    ik_reachable: bool
    collision_free: bool
    simulated: bool
    evidence_sources: tuple[str, ...]
    nearest_obstacle_point_world_m: tuple[float, float, float] | None = None
    blocked_segment: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "grasp_plan_id": self.grasp_plan_id,
            "scene_id": self.scene_id,
            "sample_monotonic_ns": self.sample_monotonic_ns,
            "point_count": self.point_count,
            "coverage_ratio": self.coverage_ratio,
            "checked_segments": self.checked_segments,
            "clearance_required_m": self.clearance_required_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "ignored_target_point_count": self.ignored_target_point_count,
            "ik_reachable": self.ik_reachable,
            "collision_free": self.collision_free,
            "simulated": self.simulated,
            "evidence_sources": self.evidence_sources,
            "nearest_obstacle_point_world_m": self.nearest_obstacle_point_world_m,
            "blocked_segment": self.blocked_segment,
        }


@dataclass(frozen=True, slots=True)
class MoveToPickReadyRequest:
    """Bounded request to reach and prove the PPO handoff distribution."""

    execution_id: str
    target: TomatoTarget
    plan: TomatoGraspPlan
    max_duration_s: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        if not isinstance(self.plan, TomatoGraspPlan):
            raise TypeError("plan must be a TomatoGraspPlan")
        if self.target.target_id != self.plan.target_id:
            raise ValueError("target and plan target IDs must agree")
        if self.target.observation_sequence != self.plan.source_observation_sequence:
            raise ValueError("target and plan observation revisions must agree")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class SafeRetractRequest:
    execution_id: str
    arm_side: ArmSide
    safe_pose: Pose
    reason: str
    max_duration_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        if not isinstance(self.safe_pose, Pose):
            raise TypeError("safe_pose must be a Pose")
        object.__setattr__(self, "reason", _non_empty(self.reason, field_name="reason"))
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class ResetForNextTargetRequest:
    execution_id: str
    completed_cycle_id: str
    arm_side: ArmSide
    reset_pose: Pose
    max_duration_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        object.__setattr__(
            self,
            "completed_cycle_id",
            _non_empty(self.completed_cycle_id, field_name="completed_cycle_id"),
        )
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        if not isinstance(self.reset_pose, Pose):
            raise TypeError("reset_pose must be a Pose")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class HarvestPayload:
    payload_id: str
    target_id: str
    pick_execution_id: str
    arm_side: ArmSide
    detached: bool
    retained: bool
    acquired_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_id", _non_empty(self.payload_id, field_name="payload_id"))
        object.__setattr__(self, "target_id", _non_empty(self.target_id, field_name="target_id"))
        object.__setattr__(
            self,
            "pick_execution_id",
            _non_empty(self.pick_execution_id, field_name="pick_execution_id"),
        )
        object.__setattr__(self, "arm_side", _arm_side(self.arm_side))
        if not isinstance(self.detached, bool) or not isinstance(self.retained, bool):
            raise TypeError("detached and retained must be bools")
        if not self.detached or not self.retained:
            raise ValueError("harvest payload requires detached=True and retained=True")
        if (
            isinstance(self.acquired_monotonic_ns, bool)
            or not isinstance(self.acquired_monotonic_ns, int)
            or self.acquired_monotonic_ns < 0
        ):
            raise ValueError("acquired_monotonic_ns must be an integer >= 0")


@dataclass(frozen=True, slots=True)
class PayloadRetreatRequest:
    execution_id: str
    payload: HarvestPayload
    retreat_pose: Pose
    max_duration_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.payload, HarvestPayload):
            raise TypeError("payload must be a HarvestPayload")
        if not isinstance(self.retreat_pose, Pose):
            raise TypeError("retreat_pose must be a Pose")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class PayloadTransportRequest:
    execution_id: str
    payload: HarvestPayload
    container: HarvestContainer
    max_duration_s: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.payload, HarvestPayload):
            raise TypeError("payload must be a HarvestPayload")
        if not isinstance(self.container, HarvestContainer):
            raise TypeError("container must be a HarvestContainer")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class PayloadTransportEvidence:
    transport_id: str
    payload_id: str
    container_id: str
    lift_pose: Pose
    preplace_pose: Pose
    command_ids: tuple[str, ...]
    retention_sample_count: int
    minimum_contact_fingers: int
    minimum_normal_force_n: float
    retained: bool
    dropped: bool
    evidence_sources: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "transport_id": self.transport_id,
            "payload_id": self.payload_id,
            "container_id": self.container_id,
            "lift_pose": self.lift_pose.as_dict(),
            "preplace_pose": self.preplace_pose.as_dict(),
            "command_ids": self.command_ids,
            "retention_sample_count": self.retention_sample_count,
            "minimum_contact_fingers": self.minimum_contact_fingers,
            "minimum_normal_force_n": self.minimum_normal_force_n,
            "retained": self.retained,
            "dropped": self.dropped,
            "evidence_sources": self.evidence_sources,
        }


@dataclass(frozen=True, slots=True)
class HarvestContainer:
    container_id: str
    placement_pose: Pose
    available: bool = True
    capacity_remaining: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "container_id", _non_empty(self.container_id, field_name="container_id")
        )
        if not isinstance(self.placement_pose, Pose):
            raise TypeError("placement_pose must be a Pose")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.capacity_remaining is not None:
            if (
                isinstance(self.capacity_remaining, bool)
                or not isinstance(self.capacity_remaining, int)
                or self.capacity_remaining < 0
            ):
                raise ValueError("capacity_remaining must be an integer >= 0 or None")

    @property
    def accepts_payload(self) -> bool:
        return self.available and self.capacity_remaining != 0


@dataclass(frozen=True, slots=True)
class DroppedPayloadLocalizationRequest:
    """Locate a payload lost after a previously verified pick."""

    execution_id: str
    payload: HarvestPayload
    container: HarvestContainer
    max_observation_age_s: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _non_empty(self.execution_id, field_name="execution_id"),
        )
        if not isinstance(self.payload, HarvestPayload):
            raise TypeError("payload must be a HarvestPayload")
        if not isinstance(self.container, HarvestContainer):
            raise TypeError("container must be a HarvestContainer")
        object.__setattr__(
            self,
            "max_observation_age_s",
            _positive(
                self.max_observation_age_s,
                field_name="max_observation_age_s",
            ),
        )


@dataclass(frozen=True, slots=True)
class DroppedPayloadObservationEvidence:
    """Camera or simulation observation of one lost physical payload."""

    payload_id: str
    observed: bool
    center_pose: Pose | None
    confidence: float
    inside_container: bool
    sample_monotonic_ns: int
    simulated: bool
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "payload_id", _non_empty(self.payload_id, field_name="payload_id")
        )
        for name in ("observed", "inside_container", "simulated"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        confidence = _finite_float(self.confidence, field_name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if (
            isinstance(self.sample_monotonic_ns, bool)
            or not isinstance(self.sample_monotonic_ns, int)
            or self.sample_monotonic_ns < 0
        ):
            raise ValueError("sample_monotonic_ns must be an integer >= 0")
        sources = _string_tuple(
            self.evidence_sources, field_name="evidence_sources"
        )
        object.__setattr__(self, "evidence_sources", sources)
        if self.observed:
            if not isinstance(self.center_pose, Pose):
                raise TypeError("observed payload requires center_pose")
            if self.center_pose.frame_id != "world":
                raise ValueError("dropped payload center_pose must use world frame")
            if confidence <= 0.0 or not sources:
                raise ValueError(
                    "observed payload requires positive confidence and evidence"
                )
        elif self.center_pose is not None or confidence != 0.0 or self.inside_container:
            raise ValueError(
                "unobserved payload cannot claim pose, confidence, or container membership"
            )
        if self.inside_container and not self.observed:
            raise ValueError("inside_container requires an observed payload")

    def as_dict(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "observed": self.observed,
            "center_pose": (
                None if self.center_pose is None else self.center_pose.as_dict()
            ),
            "confidence": self.confidence,
            "inside_container": self.inside_container,
            "sample_monotonic_ns": self.sample_monotonic_ns,
            "simulated": self.simulated,
            "evidence_sources": self.evidence_sources,
        }


@dataclass(frozen=True, slots=True)
class PayloadPlaceRequest:
    execution_id: str
    payload: HarvestPayload
    container: HarvestContainer
    preplace_pose: Pose
    max_duration_s: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        if not isinstance(self.payload, HarvestPayload):
            raise TypeError("payload must be a HarvestPayload")
        if not isinstance(self.container, HarvestContainer):
            raise TypeError("container must be a HarvestContainer")
        if not isinstance(self.preplace_pose, Pose):
            raise TypeError("preplace_pose must be a Pose")
        if self.preplace_pose.frame_id != self.container.placement_pose.frame_id:
            raise ValueError("preplace and placement poses must use the same frame")
        object.__setattr__(
            self,
            "max_duration_s",
            _positive(self.max_duration_s, field_name="max_duration_s"),
        )


@dataclass(frozen=True, slots=True)
class PlacementReceipt:
    placement_id: str
    payload_id: str
    container_id: str
    release_confirmed: bool
    hand_empty: bool
    dropped: bool
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("release_confirmed", "hand_empty", "dropped"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(
            self, "placement_id", _non_empty(self.placement_id, field_name="placement_id")
        )
        object.__setattr__(self, "payload_id", _non_empty(self.payload_id, field_name="payload_id"))
        object.__setattr__(
            self, "container_id", _non_empty(self.container_id, field_name="container_id")
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        if self.successful and not self.evidence_sources:
            raise ValueError("successful placement receipt requires evidence sources")

    @property
    def successful(self) -> bool:
        return self.release_confirmed and self.hand_empty and not self.dropped


@dataclass(frozen=True, slots=True)
class PlacementVerificationEvidence:
    placement_id: str
    container_id: str
    placed_in_container: bool
    release_confirmed: bool
    hand_empty: bool
    dropped: bool
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "placed_in_container",
            "release_confirmed",
            "hand_empty",
            "dropped",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(
            self, "placement_id", _non_empty(self.placement_id, field_name="placement_id")
        )
        object.__setattr__(
            self, "container_id", _non_empty(self.container_id, field_name="container_id")
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        if self.successful and not self.evidence_sources:
            raise ValueError("successful placement verification requires evidence sources")

    @property
    def successful(self) -> bool:
        return (
            self.placed_in_container
            and self.release_confirmed
            and self.hand_empty
            and not self.dropped
        )


@dataclass(frozen=True, slots=True)
class HarvestCycleVerification:
    cycle_id: str
    target_id: str
    pick_verified: bool
    retreat_verified: bool
    transport_verified: bool
    placed_in_container: bool
    release_confirmed: bool
    hand_empty: bool
    dropped: bool
    evidence_sources: tuple[str, ...]
    fruit_damage_detected: bool | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "pick_verified",
            "retreat_verified",
            "transport_verified",
            "placed_in_container",
            "release_confirmed",
            "hand_empty",
            "dropped",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        if self.fruit_damage_detected is not None and not isinstance(
            self.fruit_damage_detected, bool
        ):
            raise TypeError("fruit_damage_detected must be a bool or None")
        object.__setattr__(self, "cycle_id", _non_empty(self.cycle_id, field_name="cycle_id"))
        object.__setattr__(self, "target_id", _non_empty(self.target_id, field_name="target_id"))
        sources = _string_tuple(
            self.evidence_sources,
            field_name="evidence_sources",
        )
        if self.successful and not sources:
            raise ValueError("successful cycle verification requires evidence sources")
        object.__setattr__(self, "evidence_sources", sources)

    @property
    def successful(self) -> bool:
        return (
            self.pick_verified
            and self.retreat_verified
            and self.transport_verified
            and self.placed_in_container
            and self.release_confirmed
            and self.hand_empty
            and not self.dropped
            and self.fruit_damage_detected is not True
        )


__all__ = [
    "ActiveViewRepositionEvidence",
    "ActiveViewRepositionRequest",
    "ClearanceScene",
    "DroppedPayloadDisposition",
    "DroppedPayloadLocalizationRequest",
    "DroppedPayloadObservationEvidence",
    "HarvestTargetCandidateScore",
    "HarvestTargetNextAction",
    "HarvestTargetSelection",
    "HarvestContainer",
    "HarvestCycleState",
    "HarvestCycleVerification",
    "HarvestPayload",
    "PayloadPlaceRequest",
    "PayloadRetreatRequest",
    "PayloadTransportEvidence",
    "PayloadTransportRequest",
    "PregraspClearanceEvidence",
    "PregraspClearanceRequest",
    "PlacementReceipt",
    "PlacementVerificationEvidence",
    "ResetForNextTargetRequest",
    "SafeRetractRequest",
    "SelectHarvestTargetRequest",
    "TargetReobserveEvidence",
    "TargetReobserveRequest",
]
