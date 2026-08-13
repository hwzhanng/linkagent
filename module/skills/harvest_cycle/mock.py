"""Deterministic offline implementations of the first harvest-cycle atoms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from module.robotics import Clock, RoboticsBundle, SystemClock
from module.robotics.control import ControlBusyError
from module.robotics.ports import HandSynergyCommand, LinearMoveCommand, PoseMoveCommand
from proto import (
    ArmSide,
    ClearanceScene,
    DroppedPayloadLocalizationRequest,
    DroppedPayloadObservationEvidence,
    ExecutionState,
    HandSynergy,
    HarvestContainer,
    HarvestPayload,
    PayloadPlaceRequest,
    PayloadRetreatRequest,
    PregraspClearanceRequest,
    PlacementReceipt,
    PlacementVerificationEvidence,
    SafeRetractRequest,
    TargetReobserveEvidence,
    TargetReobserveRequest,
    TomatoTarget,
    Pose,
)

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
)

from .ports import (
    ClearanceScenePort,
    DroppedPayloadObservationPort,
    PlacementVerifierPort,
    TargetObservationPort,
)
from .payload_state import read_payload_state


def _target_rows(request: TargetReobserveRequest) -> tuple[Mapping[str, Any], ...]:
    raw = request.observation.metadata.get("tomatoes", ())
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        raise TypeError("observation.metadata.tomatoes must be a list")
    if not isinstance(raw, Sequence):
        try:
            raw = tuple(raw)
        except TypeError as exc:
            raise TypeError("observation.metadata.tomatoes must be a list") from exc
    rows = tuple(raw)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("observation.metadata.tomatoes entries must be mappings")
    return rows  # type: ignore[return-value]


class MockDroppedPayloadObservationPort:
    """Deterministic lost-fruit observation for workflow and failure tests."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        observed: bool = True,
        center_position_world_m: Sequence[float] = (0.45, -0.20, 0.08),
        confidence: float = 0.95,
        inside_container: bool = False,
        sample_age_s: float = 0.0,
    ) -> None:
        self._clock = SystemClock() if clock is None else clock
        if not isinstance(self._clock, Clock):
            raise TypeError("clock must implement Clock")
        if not isinstance(observed, bool) or not isinstance(inside_container, bool):
            raise TypeError("observed and inside_container must be bools")
        position = tuple(center_position_world_m)
        if len(position) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in position
        ):
            raise ValueError("center_position_world_m must contain three finite values")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("confidence must be in [0, 1]")
        if (
            isinstance(sample_age_s, bool)
            or not isinstance(sample_age_s, (int, float))
            or sample_age_s < 0.0
        ):
            raise ValueError("sample_age_s must be >= 0")
        if not observed and inside_container:
            raise ValueError("unobserved mock payload cannot be inside the container")
        self._observed = observed
        self._position = tuple(float(value) for value in position)
        self._confidence = float(confidence) if observed else 0.0
        self._inside_container = inside_container
        self._sample_age_ns = round(float(sample_age_s) * 1.0e9)
        self._closed = False

    def observe(
        self, request: DroppedPayloadLocalizationRequest
    ) -> DroppedPayloadObservationEvidence:
        if self._closed:
            raise RuntimeError("dropped-payload observation port is closed")
        return DroppedPayloadObservationEvidence(
            payload_id=request.payload.payload_id,
            observed=self._observed,
            center_pose=(
                Pose(self._position, (1.0, 0.0, 0.0, 0.0), "world")
                if self._observed
                else None
            ),
            confidence=self._confidence,
            inside_container=self._inside_container,
            sample_monotonic_ns=max(
                0, self._clock.monotonic_ns() - self._sample_age_ns
            ),
            simulated=True,
            evidence_sources=("mock:fixed-top-rgbd",) if self._observed else (),
        )

    def close(self) -> None:
        self._closed = True


assert isinstance(MockDroppedPayloadObservationPort, type)
assert isinstance(DroppedPayloadObservationPort, type)


class MockClearanceScenePort:
    """Deterministic scene source; default points are safely outside the motion corridor."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        obstacle_points_world_m: Sequence[Sequence[float]] = (
            (0.90, 0.80, 1.30),
            (0.85, 0.75, 1.25),
            (-0.30, -0.70, 0.25),
            (-0.25, 0.70, 1.20),
        ),
        force_missing: bool = False,
        sample_age_s: float = 0.0,
        coverage_ratio: float = 0.95,
        calibration_id: str = "mock-clearance-world-v1",
    ) -> None:
        self._clock = SystemClock() if clock is None else clock
        if not isinstance(self._clock, Clock):
            raise TypeError("clock must implement Clock")
        if not isinstance(force_missing, bool):
            raise TypeError("force_missing must be a bool")
        if isinstance(sample_age_s, bool) or not isinstance(sample_age_s, (int, float)) or sample_age_s < 0:
            raise ValueError("sample_age_s must be >= 0")
        if isinstance(coverage_ratio, bool) or not isinstance(coverage_ratio, (int, float)) or not 0 <= coverage_ratio <= 1:
            raise ValueError("coverage_ratio must be in [0, 1]")
        self._points = tuple(tuple(float(value) for value in point) for point in obstacle_points_world_m)
        self._force_missing = force_missing
        self._sample_age_ns = round(float(sample_age_s) * 1.0e9)
        self._coverage_ratio = float(coverage_ratio)
        self._calibration_id = str(calibration_id)
        self._capture_count = 0
        self._closed = False

    def capture(self, request: PregraspClearanceRequest) -> ClearanceScene:
        if self._closed:
            raise RuntimeError("clearance scene port is closed")
        if self._force_missing:
            raise RuntimeError("mock clearance scene unavailable")
        self._capture_count += 1
        return ClearanceScene(
            scene_id=f"mock-clearance-{request.plan.grasp_plan_id}-{self._capture_count}",
            sample_monotonic_ns=max(0, self._clock.monotonic_ns() - self._sample_age_ns),
            points_world_m=self._points,
            coverage_ratio=self._coverage_ratio,
            calibration_id=self._calibration_id,
            simulated=True,
            evidence_sources=("mock.world_point_cloud",),
        )

    def close(self) -> None:
        self._closed = True


class ObservationClearanceScenePort:
    """Read a future depth-camera world point sample from Observation metadata."""

    def __init__(self, *, clock: Clock | None = None, metadata_key: str = "pregrasp_clearance") -> None:
        self._clock = SystemClock() if clock is None else clock
        if not isinstance(self._clock, Clock):
            raise TypeError("clock must implement Clock")
        if not isinstance(metadata_key, str) or not metadata_key.strip():
            raise ValueError("metadata_key must be a non-empty string")
        self._metadata_key = metadata_key.strip()
        self._closed = False

    def capture(self, request: PregraspClearanceRequest) -> ClearanceScene:
        if self._closed:
            raise RuntimeError("clearance scene port is closed")
        raw = request.observation.metadata.get(self._metadata_key)
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"observation metadata lacks {self._metadata_key!r}")
        values = dict(raw)
        values.setdefault("scene_id", f"observation-{request.observation.sequence}")
        values.setdefault("sample_monotonic_ns", self._clock.monotonic_ns())
        values.setdefault("frame_id", "world")
        values.setdefault("simulated", False)
        return ClearanceScene(**values)

    def close(self) -> None:
        self._closed = True


class MockTargetObservationPort:
    """Read target evidence from an observation without owning a camera."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        force_missing: bool = False,
        force_occlusion_ratio: float | None = None,
        occlusion_sequence: tuple[float, ...] | list[float] | None = None,
        sample_advance_s: float = 0.0,
    ) -> None:
        if not isinstance(force_missing, bool):
            raise TypeError("force_missing must be a bool")
        if force_occlusion_ratio is not None:
            if (
                isinstance(force_occlusion_ratio, bool)
                or not isinstance(force_occlusion_ratio, (int, float))
                or not 0.0 <= float(force_occlusion_ratio) <= 1.0
            ):
                raise ValueError("force_occlusion_ratio must be in [0, 1] or None")
            force_occlusion_ratio = float(force_occlusion_ratio)
        if occlusion_sequence is not None:
            if force_occlusion_ratio is not None:
                raise ValueError(
                    "force_occlusion_ratio and occlusion_sequence are mutually exclusive"
                )
            prepared_occlusion: list[float] = []
            for index, value in enumerate(occlusion_sequence):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(
                        f"occlusion_sequence[{index}] must be in [0, 1]"
                    )
                prepared_occlusion.append(float(value))
            if not prepared_occlusion:
                raise ValueError("occlusion_sequence must not be empty")
            occlusion_values: tuple[float, ...] | None = tuple(prepared_occlusion)
        else:
            occlusion_values = None
        if (
            isinstance(sample_advance_s, bool)
            or not isinstance(sample_advance_s, (int, float))
            or float(sample_advance_s) < 0.0
        ):
            raise ValueError("sample_advance_s must be a number >= 0")
        self._clock = SystemClock() if clock is None else clock
        if not isinstance(self._clock, Clock):
            raise TypeError("clock must implement Clock")
        self._force_missing = force_missing
        self._force_occlusion_ratio = force_occlusion_ratio
        self._occlusion_sequence = occlusion_values
        self._occlusion_index = 0
        self._sample_advance_s = float(sample_advance_s)
        self._closed = False

    def reobserve(self, request: TargetReobserveRequest) -> TargetReobserveEvidence:
        if self._closed:
            raise RuntimeError("target observation port is closed")
        for row in _target_rows(request):
            if not self._force_missing and row.get("target_id") == request.target_id:
                occlusion = row.get("occlusion", 0.0)
                if self._force_occlusion_ratio is not None:
                    occlusion = self._force_occlusion_ratio
                elif self._occlusion_sequence is not None:
                    occlusion = self._occlusion_sequence[
                        min(self._occlusion_index, len(self._occlusion_sequence) - 1)
                    ]
                    self._occlusion_index += 1
                if self._sample_advance_s > 0.0:
                    self._clock.sleep(self._sample_advance_s)
                target: TomatoTarget | None = None
                required = {
                    "center_pose",
                    "semi_axes_m",
                    "ripeness_score",
                    "pick_confidence",
                    "stem_direction_xyz",
                }
                if required <= set(row):
                    options = dict(row)
                    pose = options.get("center_pose")
                    if isinstance(pose, Mapping):
                        options["center_pose"] = Pose(**dict(pose))
                    options["occlusion"] = float(occlusion)
                    options["observation_sequence"] = request.observation.sequence
                    options["sample_monotonic_ns"] = self._clock.monotonic_ns()
                    target = TomatoTarget(**options)
                sources = ["mock.rgb_metadata"]
                depth_available = request.observation.depth_m is not None
                if depth_available:
                    sources.append("mock.depth_m")
                return TargetReobserveEvidence(
                    target_id=request.target_id,
                    observed=True,
                    occlusion_ratio=float(occlusion),
                    depth_available=depth_available,
                    evidence_sources=tuple(sources),
                    target=target,
                    details={"observation_sequence": request.observation.sequence},
                )
        return TargetReobserveEvidence(
            target_id=request.target_id,
            observed=False,
            occlusion_ratio=1.0,
            depth_available=request.observation.depth_m is not None,
            evidence_sources=("mock.rgb_metadata",),
            details={"observation_sequence": request.observation.sequence},
        )

    def close(self) -> None:
        self._closed = True


class MockPlacementVerifierPort:
    """Independent mock evidence reader; a calibrated RGB-D verifier replaces this later."""

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        force_unconfirmed: bool = False,
        force_dropped: bool = False,
    ) -> None:
        if not isinstance(force_unconfirmed, bool) or not isinstance(force_dropped, bool):
            raise TypeError("force_unconfirmed and force_dropped must be bools")
        self._robotics = robotics
        self._force_unconfirmed = force_unconfirmed
        self._force_dropped = force_dropped
        self._closed = False

    def verify(
        self,
        receipt: PlacementReceipt,
        payload: HarvestPayload,
        container: HarvestContainer,
    ) -> PlacementVerificationEvidence:
        if self._closed:
            raise RuntimeError("placement verifier is closed")
        hand_state = self._robotics.hand.get_state(payload.arm_side)
        hand_empty = hand_state.synergy.closure <= 0.05 and not hand_state.fault_codes
        world = self._robotics.verifier_context().get("mock_world")
        world_dropped = bool(getattr(world, "dropped", False))
        dropped = self._force_dropped or world_dropped
        placed = receipt.successful and container.accepts_payload and not self._force_unconfirmed
        return PlacementVerificationEvidence(
            placement_id=receipt.placement_id,
            container_id=container.container_id,
            placed_in_container=placed,
            release_confirmed=receipt.release_confirmed and not self._force_unconfirmed,
            hand_empty=hand_empty,
            dropped=dropped,
            evidence_sources=("mock.container_state", "mock.hand_state"),
        )

    def close(self) -> None:
        self._closed = True


class TargetReobserveSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="target_reobserve",
        version="1.0.0",
        description="Re-observe one selected target through a replaceable RGB-D evidence port.",
        effectful=False,
        required_resources=("target_observation",),
    )

    def __init__(
        self,
        *,
        observation_port: TargetObservationPort,
        clock: Clock,
        max_occlusion_ratio: float = 0.60,
        require_depth: bool = False,
        require_new_sample: bool = True,
    ) -> None:
        super().__init__(clock=clock)
        if not isinstance(observation_port, TargetObservationPort):
            raise TypeError("observation_port must implement TargetObservationPort")
        if (
            isinstance(max_occlusion_ratio, bool)
            or not isinstance(max_occlusion_ratio, (int, float))
            or not 0.0 <= float(max_occlusion_ratio) <= 1.0
        ):
            raise ValueError("max_occlusion_ratio must be in [0, 1]")
        if not isinstance(require_depth, bool):
            raise TypeError("require_depth must be a bool")
        if not isinstance(require_new_sample, bool):
            raise TypeError("require_new_sample must be a bool")
        self._observation_port = observation_port
        self._max_occlusion_ratio = float(max_occlusion_ratio)
        self._require_depth = require_depth
        self._require_new_sample = require_new_sample

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, TargetReobserveRequest):
            raise TypeError("request must be a TargetReobserveRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "observe",
            SkillTraceEventKind.STARTED,
            input_summary={
                "target_id": request.target_id,
                "observation_sequence": request.observation.sequence,
                "rgb": request.observation.rgb,
                "depth_m": request.observation.depth_m,
            },
        )
        require_new_sample = self._require_new_sample or request.force_new_sample
        try:
            evidence = self._observation_port.reobserve(
                replace(request, require_new_sample=require_new_sample)
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            trace.record("observe", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE,
                message=str(exc),
            )
        if not evidence.observed:
            trace.record(
                "observe",
                SkillTraceEventKind.FAILED,
                message="selected target was not found in the new observation",
                evidence_refs=evidence.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.TARGET_STALE,
                message="selected target was not found in the new observation",
                output={"target_id": request.target_id, "observed": False},
            )
        if evidence.target is None and request.prior_sample_monotonic_ns is not None:
            trace.record(
                "freshness_gate",
                SkillTraceEventKind.FAILED,
                message="target re-observation did not return metric target geometry",
                evidence_refs=evidence.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE,
                message="target re-observation did not return metric target geometry",
            )
        target_age_ns: int | None = None
        current_sample = True
        new_sample = True
        if evidence.target is not None:
            now_ns = self._clock.monotonic_ns()
            target_age_ns = now_ns - evidence.target.sample_monotonic_ns
            current_sample = (
                0 <= target_age_ns
                <= request.max_observation_age_s * 1_000_000_000
            )
            new_sample = (
                request.prior_sample_monotonic_ns is None
                or evidence.target.sample_monotonic_ns
                > request.prior_sample_monotonic_ns
            )
        if not current_sample or (require_new_sample and not new_sample):
            trace.record(
                "freshness_gate",
                SkillTraceEventKind.FAILED,
                message="target re-observation did not provide a fresh camera sample",
                output_summary={
                    "sample_monotonic_ns": (
                        None
                        if evidence.target is None
                        else evidence.target.sample_monotonic_ns
                    ),
                    "prior_sample_monotonic_ns": request.prior_sample_monotonic_ns,
                    "target_age_s": (
                        None if target_age_ns is None else target_age_ns / 1.0e9
                    ),
                    "require_new_sample": require_new_sample,
                },
                evidence_refs=evidence.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.TARGET_STALE,
                message="target re-observation did not provide a fresh camera sample",
            )
        if (
            request.enforce_visibility_gate
            and evidence.occlusion_ratio > self._max_occlusion_ratio
        ):
            trace.record(
                "visibility_gate",
                SkillTraceEventKind.FAILED,
                message="target occlusion exceeds the configured limit",
                output_summary={"occlusion_ratio": evidence.occlusion_ratio},
                evidence_refs=evidence.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.TARGET_OCCLUDED,
                message="target occlusion exceeds the configured limit",
                output={
                    "target_id": request.target_id,
                    "occlusion_ratio": evidence.occlusion_ratio,
                },
            )
        if self._require_depth and not evidence.depth_available:
            trace.record(
                "depth_gate",
                SkillTraceEventKind.FAILED,
                message="depth evidence is required but unavailable",
                evidence_refs=evidence.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.DEPTH_UNAVAILABLE,
                message="depth evidence is required but unavailable",
                output={"target_id": request.target_id, "depth_available": False},
            )
        trace.record(
            "observe",
            SkillTraceEventKind.SUCCEEDED,
            output_summary={
                "target_id": evidence.target_id,
                "occlusion_ratio": evidence.occlusion_ratio,
                "depth_available": evidence.depth_available,
            },
            evidence_refs=evidence.evidence_sources,
        )
        target_output: dict[str, object] | None = None
        if evidence.target is not None:
            target_output = {
                **evidence.target.summary(),
                "stem_direction_xyz": evidence.target.stem_direction_xyz,
                "sample_monotonic_ns": evidence.target.sample_monotonic_ns,
            }
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="target re-observed",
            output={
                "target_id": evidence.target_id,
                "observed": True,
                "occlusion_ratio": evidence.occlusion_ratio,
                "depth_available": evidence.depth_available,
                "evidence_sources": evidence.evidence_sources,
                "target": target_output,
            },
        )

    def close(self) -> None:
        if not self._closed:
            self._observation_port.close()
        super().close()


class SafeRetractSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="safe_retract",
        version="1.0.0",
        description="Move one arm to a configured safe pose after a recoverable failure.",
        effectful=True,
        required_resources=("arm_control", "hand_control", "control_lease"),
    )

    def __init__(self, *, robotics: RoboticsBundle) -> None:
        super().__init__(clock=robotics.clock)
        self._robotics = robotics

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, SafeRetractRequest):
            raise TypeError("request must be a SafeRetractRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "retract",
            SkillTraceEventKind.STARTED,
            input_summary={"arm_side": request.arm_side.value, "reason": request.reason},
        )
        lease = None
        try:
            lease = self._robotics.leases.acquire(
                owner_id=request.execution_id,
                side=request.arm_side,
                ttl_s=request.max_duration_s,
            )
            self._robotics.hand.hold(lease, request.arm_side)
            handle = self._robotics.arm.move_pose(
                lease,
                PoseMoveCommand(
                    command_id=f"{request.execution_id}-arm",
                    side=request.arm_side,
                    target_pose=request.safe_pose,
                    speed=0.10,
                    acceleration=0.20,
                    timeout_s=request.max_duration_s,
                ),
            )
            if handle.state is not ExecutionState.SUCCEEDED:
                trace.record("retract", SkillTraceEventKind.FAILED, message="arm command failed")
                return self._result(
                    trace,
                    status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                    failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                    message="safe retract arm command failed",
                )
            trace.record(
                "retract",
                SkillTraceEventKind.SUCCEEDED,
                output_summary={"command_id": handle.command_id},
                evidence_refs=(f"arm_command:{handle.command_id}",),
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                message="arm reached configured safe pose",
                output={"command_id": handle.command_id, "arm_side": request.arm_side.value},
            )
        except ControlBusyError as exc:
            trace.record("retract", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.CONTROL_BUSY,
                message=str(exc),
            )
        except (RuntimeError, ValueError) as exc:
            trace.record("retract", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                message=str(exc),
            )
        finally:
            if lease is not None:
                self._robotics.leases.release(lease)


class PayloadRetreatSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="payload_retreat",
        version="1.0.0",
        description="Retreat a detached tomato while retaining tactile and hand evidence.",
        effectful=True,
        required_resources=("arm_control", "hand_control", "tactile", "control_lease"),
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        force_payload_unstable: bool = False,
        force_drop: bool = False,
        min_contact_fingers: int = 3,
        min_normal_force_n: float = 0.5,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(force_payload_unstable, bool) or not isinstance(force_drop, bool):
            raise TypeError("force_payload_unstable and force_drop must be bools")
        self._robotics = robotics
        self._force_payload_unstable = force_payload_unstable
        self._force_drop = force_drop
        if (
            isinstance(min_contact_fingers, bool)
            or not isinstance(min_contact_fingers, int)
            or not 1 <= min_contact_fingers <= 5
        ):
            raise ValueError("min_contact_fingers must be in [1, 5]")
        if (
            isinstance(min_normal_force_n, bool)
            or not isinstance(min_normal_force_n, (int, float))
            or min_normal_force_n <= 0.0
        ):
            raise ValueError("min_normal_force_n must be > 0")
        self._min_contact_fingers = min_contact_fingers
        self._min_normal_force_n = float(min_normal_force_n)

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, PayloadRetreatRequest):
            raise TypeError("request must be a PayloadRetreatRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "retreat",
            SkillTraceEventKind.STARTED,
            input_summary={"payload_id": request.payload.payload_id},
        )
        initial = self._retention_sample(request.payload.arm_side)
        if not bool(initial["stable"]):
            return self._retention_failure(
                trace, initial, phase="retention_before_motion"
            )
        lease = None
        try:
            lease = self._robotics.leases.acquire(
                owner_id=request.execution_id,
                side=request.payload.arm_side,
                ttl_s=request.max_duration_s,
            )
            self._robotics.hand.hold(lease, request.payload.arm_side)
            handle = self._robotics.arm.move_linear(
                lease,
                LinearMoveCommand(
                    command_id=f"{request.execution_id}-arm",
                    side=request.payload.arm_side,
                    target_pose=request.retreat_pose,
                    speed=0.06,
                    acceleration=0.10,
                    timeout_s=request.max_duration_s,
                ),
            )
            if handle.state is not ExecutionState.SUCCEEDED:
                trace.record("retreat", SkillTraceEventKind.FAILED, message="arm command failed")
                return self._result(
                    trace,
                    status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                    failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                    message="payload retreat arm command failed",
                )
            final = self._retention_sample(request.payload.arm_side)
            if not bool(final["stable"]):
                return self._retention_failure(
                    trace, final, phase="retention_after_motion"
                )
            trace.record(
                "retention_check",
                SkillTraceEventKind.SUCCEEDED,
                output_summary={
                    "contact_fingers": final["contact_fingers"],
                    "dropped": False,
                },
                evidence_refs=("hand_state", "tactile_frame", f"arm_command:{handle.command_id}"),
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                message="payload retained through retreat",
                output={
                    "payload_id": request.payload.payload_id,
                    "command_id": handle.command_id,
                    "contact_fingers": final["contact_fingers"],
                    "retained": True,
                },
            )
        except ControlBusyError as exc:
            trace.record("retreat", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.CONTROL_BUSY,
                message=str(exc),
            )
        except (RuntimeError, ValueError) as exc:
            trace.record("retreat", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                message=str(exc),
            )
        finally:
            if lease is not None:
                self._robotics.leases.release(lease)

    def _retention_sample(self, side: ArmSide) -> dict[str, object]:
        hand = self._robotics.hand.get_state(side)
        tactile = self._robotics.tactile.get_frame(side)
        contact_fingers = sum(
            1
            for finger in tactile.fingers
            if finger.force_valid
            and finger.normal_force_n is not None
            and finger.normal_force_n >= self._min_normal_force_n
        )
        dropped = self._force_drop or bool(
            read_payload_state(self._robotics).get("dropped", False)
        )
        return {
            "stable": (
                not dropped
                and not self._force_payload_unstable
                and not hand.fault_codes
                and hand.synergy.closure >= 0.15
                and contact_fingers >= self._min_contact_fingers
            ),
            "contact_fingers": contact_fingers,
            "dropped": dropped,
            "hand_closure": hand.synergy.closure,
            "tactile_source_sequence": tactile.source_sequence,
        }

    def _retention_failure(
        self,
        trace: SkillTraceRecorder,
        sample: Mapping[str, object],
        *,
        phase: str,
    ) -> AtomicSkillResult:
        dropped = bool(sample["dropped"])
        code = (
            AtomicSkillFailureCode.PAYLOAD_DROPPED
            if dropped
            else AtomicSkillFailureCode.PAYLOAD_UNSTABLE
        )
        trace.record(
            phase,
            SkillTraceEventKind.FAILED,
            output_summary=sample,
            evidence_refs=("hand_state", "tactile_frame"),
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SAFETY_ABORT,
            failure_code=code,
            message=f"payload retention was not confirmed at {phase}",
            output={
                "contact_fingers": sample["contact_fingers"],
                "dropped": dropped,
                "failed_phase": phase,
                "motion_authorized": False,
            },
        )


class PayloadPlaceSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="payload_place",
        version="1.0.0",
        description="Move a retained tomato to a configured container and release it.",
        effectful=True,
        required_resources=("arm_control", "hand_control", "tactile", "control_lease"),
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        max_preplace_error_m: float = 0.015,
        min_contact_fingers: int = 3,
        min_normal_force_n: float = 0.5,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if isinstance(max_preplace_error_m, bool) or not isinstance(max_preplace_error_m, (int, float)) or max_preplace_error_m <= 0:
            raise ValueError("max_preplace_error_m must be > 0")
        self._robotics = robotics
        self._max_preplace_error_m = float(max_preplace_error_m)
        if (
            isinstance(min_contact_fingers, bool)
            or not isinstance(min_contact_fingers, int)
            or not 1 <= min_contact_fingers <= 5
        ):
            raise ValueError("min_contact_fingers must be in [1, 5]")
        if (
            isinstance(min_normal_force_n, bool)
            or not isinstance(min_normal_force_n, (int, float))
            or min_normal_force_n <= 0.0
        ):
            raise ValueError("min_normal_force_n must be > 0")
        self._min_contact_fingers = min_contact_fingers
        self._min_normal_force_n = float(min_normal_force_n)

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, PayloadPlaceRequest):
            raise TypeError("request must be a PayloadPlaceRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "place",
            SkillTraceEventKind.STARTED,
            input_summary={
                "payload_id": request.payload.payload_id,
                "container_id": request.container.container_id,
            },
        )
        if not request.container.accepts_payload:
            trace.record("container_check", SkillTraceEventKind.FAILED, message="container unavailable")
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.CONTAINER_UNAVAILABLE,
                message="configured harvest container is unavailable or full",
            )
        current_pose = self._robotics.arm.get_state(request.payload.arm_side).ee_pose
        preplace_error = math.dist(
            current_pose.position_m, request.preplace_pose.position_m
        )
        if preplace_error > self._max_preplace_error_m:
            trace.record(
                "preplace_check",
                SkillTraceEventKind.FAILED,
                message="payload is not at the transport-bound pre-place pose",
                output_summary={"preplace_error_m": preplace_error},
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.PRECONDITION_FAILED,
                message="payload is not at the transport-bound pre-place pose",
                output={"preplace_error_m": preplace_error},
            )
        lease = None
        try:
            lease = self._robotics.leases.acquire(
                owner_id=request.execution_id,
                side=request.payload.arm_side,
                ttl_s=request.max_duration_s,
            )
            self._robotics.hand.hold(lease, request.payload.arm_side)
            arm_handle = self._robotics.arm.move_linear(
                lease,
                LinearMoveCommand(
                    command_id=f"{request.execution_id}-arm",
                    side=request.payload.arm_side,
                    target_pose=request.container.placement_pose,
                    speed=0.08,
                    acceleration=0.12,
                    timeout_s=request.max_duration_s,
                ),
            )
            if arm_handle.state is not ExecutionState.SUCCEEDED:
                trace.record("place", SkillTraceEventKind.FAILED, message="arm command failed")
                return self._result(
                    trace,
                    status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                    failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                    message="container approach command failed",
                )
            hand_before_release = self._robotics.hand.get_state(request.payload.arm_side)
            tactile = self._robotics.tactile.get_frame(request.payload.arm_side)
            contact_fingers = sum(
                1
                for finger in tactile.fingers
                if finger.force_valid
                and finger.normal_force_n is not None
                and finger.normal_force_n >= self._min_normal_force_n
            )
            dropped = bool(read_payload_state(self._robotics).get("dropped", False))
            if (
                dropped
                or hand_before_release.synergy.closure < 0.15
                or contact_fingers < self._min_contact_fingers
            ):
                trace.record(
                    "pre_release_retention",
                    SkillTraceEventKind.FAILED,
                    output_summary={"contact_fingers": contact_fingers, "dropped": dropped},
                    evidence_refs=("hand_state", "tactile_frame"),
                )
                return self._result(
                    trace,
                    status=AtomicSkillStatus.SAFETY_ABORT,
                    failure_code=(
                        AtomicSkillFailureCode.PAYLOAD_DROPPED
                        if dropped
                        else AtomicSkillFailureCode.PAYLOAD_UNSTABLE
                    ),
                    message="payload retention was lost before release",
                    output={"contact_fingers": contact_fingers, "dropped": dropped},
                )
            hand_handle = self._robotics.hand.command_synergy(
                lease,
                HandSynergyCommand(
                    command_id=f"{request.execution_id}-hand-open",
                    side=request.payload.arm_side,
                    target=HandSynergy(0.0, 0.0, 0.0),
                    timeout_s=min(2.0, request.max_duration_s),
                ),
            )
            if hand_handle.state is not ExecutionState.SUCCEEDED:
                trace.record("release", SkillTraceEventKind.FAILED, message="hand command failed")
                return self._result(
                    trace,
                    status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                    failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                    message="payload release command failed",
                )
            hand = self._robotics.hand.get_state(request.payload.arm_side)
            hand_empty = hand.synergy.closure <= 0.05 and not hand.fault_codes
            if not hand_empty:
                trace.record(
                    "release",
                    SkillTraceEventKind.FAILED,
                    message="hand-empty state was not confirmed after release",
                    output_summary={"hand_empty": False},
                    evidence_refs=("hand_state",),
                )
                return self._result(
                    trace,
                    status=AtomicSkillStatus.RETRYABLE_FAILURE,
                    failure_code=AtomicSkillFailureCode.PLACEMENT_UNCONFIRMED,
                    message="hand-empty state was not confirmed after release",
                    output={"hand_empty": False},
                )
            placement_id = f"placement-{request.payload.payload_id}"
            receipt = PlacementReceipt(
                placement_id=placement_id,
                payload_id=request.payload.payload_id,
                container_id=request.container.container_id,
                release_confirmed=True,
                hand_empty=hand_empty,
                dropped=False,
                evidence_sources=(
                    f"arm_command:{arm_handle.command_id}",
                    f"hand_command:{hand_handle.command_id}",
                    "hand_state",
                ),
            )
            trace.record(
                "release",
                SkillTraceEventKind.SUCCEEDED,
                output_summary={"placement_id": placement_id, "hand_empty": hand_empty},
                evidence_refs=receipt.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                message="payload released into configured container",
                output={
                    "placement_id": placement_id,
                    "payload_id": request.payload.payload_id,
                    "container_id": request.container.container_id,
                    "release_confirmed": True,
                    "hand_empty": hand_empty,
                    "dropped": False,
                    "evidence_sources": receipt.evidence_sources,
                },
            )
        except ControlBusyError as exc:
            trace.record("place", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.CONTROL_BUSY,
                message=str(exc),
            )
        except (RuntimeError, ValueError) as exc:
            trace.record("place", SkillTraceEventKind.FAILED, message=str(exc))
            return self._result(
                trace,
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                failure_code=AtomicSkillFailureCode.COMMAND_FAILED,
                message=str(exc),
            )
        finally:
            if lease is not None:
                self._robotics.leases.release(lease)


__all__ = [
    "MockClearanceScenePort",
    "MockDroppedPayloadObservationPort",
    "MockPlacementVerifierPort",
    "MockTargetObservationPort",
    "ObservationClearanceScenePort",
    "PayloadPlaceSkill",
    "PayloadRetreatSkill",
    "SafeRetractSkill",
    "TargetReobserveSkill",
]
