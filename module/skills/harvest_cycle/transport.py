"""Bounded low-impact transport of a retained tomato to a container pre-place pose."""

from __future__ import annotations

import math

from module.robotics import ControlBusyError, LinearMoveCommand, RoboticsBundle
from proto import (
    ExecutionState,
    PayloadTransportEvidence,
    PayloadTransportRequest,
    Pose,
)

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
    SkillTraceRecorder,
)
from .payload_state import read_payload_state


class TransportPayloadSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="transport_payload",
        version="1.0.0",
        description="Lift and transport a retained tomato to a bounded container pre-place pose.",
        effectful=True,
        required_resources=("arm_control", "hand_control", "tactile", "control_lease"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        vertical_clearance_m: float = 0.12,
        speed: float = 0.06,
        acceleration: float = 0.08,
        max_arrival_error_m: float = 0.01,
        min_contact_fingers: int = 3,
        min_normal_force_n: float = 0.5,
        force_drop_after_segment: int | None = None,
        force_payload_unstable_after_segment: int | None = None,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must be a RoboticsBundle")
        for name, value in {
            "vertical_clearance_m": vertical_clearance_m,
            "speed": speed,
            "acceleration": acceleration,
            "max_arrival_error_m": max_arrival_error_m,
            "min_normal_force_n": min_normal_force_n,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be > 0")
        if isinstance(min_contact_fingers, bool) or not isinstance(min_contact_fingers, int) or not 1 <= min_contact_fingers <= 5:
            raise ValueError("min_contact_fingers must be an integer in [1, 5]")
        for name, value in {
            "force_drop_after_segment": force_drop_after_segment,
            "force_payload_unstable_after_segment": force_payload_unstable_after_segment,
        }.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2}):
                raise ValueError(f"{name} must be 1, 2, or None")
        self._robotics = robotics
        self._vertical_clearance_m = float(vertical_clearance_m)
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._max_arrival_error_m = float(max_arrival_error_m)
        self._min_contact_fingers = min_contact_fingers
        self._min_normal_force_n = float(min_normal_force_n)
        self._force_drop_after_segment = force_drop_after_segment
        self._force_unstable_after_segment = force_payload_unstable_after_segment
        self._preplace_poses: dict[str, Pose] = {}

    def preplace_pose_for(self, transport_id: str) -> Pose:
        try:
            return self._preplace_poses[transport_id]
        except KeyError as exc:
            raise KeyError(f"no successful transport pose for {transport_id!r}") from exc

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, PayloadTransportRequest):
            raise TypeError("request must be a PayloadTransportRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "transport",
            SkillTraceEventKind.STARTED,
            input_summary={
                "payload_id": request.payload.payload_id,
                "container_id": request.container.container_id,
            },
        )
        if not request.container.accepts_payload:
            return self._failed(
                trace,
                AtomicSkillFailureCode.CONTAINER_UNAVAILABLE,
                "configured harvest container is unavailable or full",
            )
        arm = self._robotics.arm.get_state(request.payload.arm_side)
        placement = request.container.placement_pose
        if arm.ee_pose.frame_id != "world" or placement.frame_id != "world":
            return self._failed(
                trace,
                AtomicSkillFailureCode.PRECONDITION_FAILED,
                "transport requires world-frame arm and container poses",
            )
        preplace = Pose(
            (
                placement.position_m[0],
                placement.position_m[1],
                placement.position_m[2] + self._vertical_clearance_m,
            ),
            placement.quaternion_wxyz,
            "world",
        )
        lift = Pose(
            (
                arm.ee_pose.position_m[0],
                arm.ee_pose.position_m[1],
                max(arm.ee_pose.position_m[2], preplace.position_m[2]),
            ),
            arm.ee_pose.quaternion_wxyz,
            "world",
        )

        samples: list[dict[str, object]] = []
        initial = self._retention_sample(sample_index=0)
        samples.append(initial)
        if not bool(initial["stable"]):
            return self._retention_failed(trace, initial, segment="transport_start")

        lease = None
        command_ids: list[str] = []
        try:
            lease = self._robotics.leases.acquire(
                owner_id=request.execution_id,
                side=request.payload.arm_side,
                ttl_s=request.max_duration_s,
            )
            self._robotics.hand.hold(lease, request.payload.arm_side)
            for segment_index, (segment_name, pose) in enumerate(
                (("lift", lift), ("preplace", preplace)), start=1
            ):
                handle = self._robotics.arm.move_linear(
                    lease,
                    LinearMoveCommand(
                        command_id=f"{request.execution_id}-{segment_name}",
                        side=request.payload.arm_side,
                        target_pose=pose,
                        speed=self._speed,
                        acceleration=self._acceleration,
                        timeout_s=request.max_duration_s,
                    ),
                )
                command_ids.append(handle.command_id)
                if handle.state is not ExecutionState.SUCCEEDED:
                    return self._failed(
                        trace,
                        AtomicSkillFailureCode.COMMAND_FAILED,
                        f"payload transport {segment_name} command failed",
                        status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                    )
                sample = self._retention_sample(sample_index=segment_index)
                samples.append(sample)
                trace.record(
                    segment_name,
                    SkillTraceEventKind.CHECKPOINT,
                    output_summary={"command_id": handle.command_id, **sample},
                    evidence_refs=(
                        f"arm_command:{handle.command_id}",
                        "hand_state",
                        "tactile_frame",
                    ),
                )
                if not bool(sample["stable"]):
                    return self._retention_failed(
                        trace,
                        sample,
                        segment=segment_name,
                        command_ids=tuple(command_ids),
                    )
        except ControlBusyError as exc:
            return self._failed(trace, AtomicSkillFailureCode.CONTROL_BUSY, str(exc))
        except (RuntimeError, ValueError) as exc:
            return self._failed(
                trace,
                AtomicSkillFailureCode.COMMAND_FAILED,
                str(exc),
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
            )
        finally:
            if lease is not None:
                self._robotics.leases.release(lease)

        actual = self._robotics.arm.get_state(request.payload.arm_side).ee_pose
        arrival_error = math.dist(actual.position_m, preplace.position_m)
        if arrival_error > self._max_arrival_error_m:
            return self._failed(
                trace,
                AtomicSkillFailureCode.TRANSPORT_POSE_NOT_REACHED,
                "container pre-place pose was not reached",
                output={"arrival_error_m": arrival_error, "command_ids": tuple(command_ids)},
            )
        evidence = PayloadTransportEvidence(
            transport_id=request.execution_id,
            payload_id=request.payload.payload_id,
            container_id=request.container.container_id,
            lift_pose=lift,
            preplace_pose=preplace,
            command_ids=tuple(command_ids),
            retention_sample_count=len(samples),
            minimum_contact_fingers=min(int(sample["contact_fingers"]) for sample in samples),
            minimum_normal_force_n=min(float(sample["minimum_normal_force_n"]) for sample in samples),
            retained=True,
            dropped=False,
            evidence_sources=("hand_state", "tactile_frame", *(f"arm_command:{item}" for item in command_ids)),
        )
        self._preplace_poses[request.execution_id] = preplace
        trace.record(
            "transport",
            SkillTraceEventKind.SUCCEEDED,
            output_summary={**evidence.as_dict(), "arrival_error_m": arrival_error},
            evidence_refs=evidence.evidence_sources,
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="payload retained through transport to container pre-place",
            output={
                "transport_evidence": evidence.as_dict(),
                "arrival_error_m": arrival_error,
                "place_authorized": True,
            },
        )

    def _retention_sample(self, *, sample_index: int) -> dict[str, object]:
        side = self._robotics.side
        world = self._robotics.verifier_context().get("mock_world")
        forced_drop = self._force_drop_after_segment == sample_index
        if forced_drop and world is not None:
            setattr(world, "dropped", True)
            setattr(world, "grasped", False)
        hand = self._robotics.hand.get_state(side)
        tactile = self._robotics.tactile.get_frame(side)
        forces = tuple(
            float(finger.normal_force_n)
            for finger in tactile.fingers
            if finger.force_valid
            and finger.normal_force_n is not None
            and finger.normal_force_n >= self._min_normal_force_n
        )
        dropped = forced_drop or bool(read_payload_state(self._robotics).get("dropped", False))
        forced_unstable = self._force_unstable_after_segment == sample_index
        stable = (
            not dropped
            and not forced_unstable
            and not hand.fault_codes
            and hand.synergy.closure >= 0.15
            and len(forces) >= self._min_contact_fingers
        )
        return {
            "sample_index": sample_index,
            "contact_fingers": len(forces),
            "minimum_normal_force_n": min(forces) if forces else 0.0,
            "hand_closure": hand.synergy.closure,
            "dropped": dropped,
            "stable": stable,
            "tactile_source_sequence": tactile.source_sequence,
        }

    def _retention_failed(
        self,
        trace: SkillTraceRecorder,
        sample: dict[str, object],
        *,
        segment: str,
        command_ids: tuple[str, ...] = (),
    ) -> AtomicSkillResult:
        dropped = bool(sample["dropped"])
        return self._failed(
            trace,
            AtomicSkillFailureCode.PAYLOAD_DROPPED if dropped else AtomicSkillFailureCode.PAYLOAD_UNSTABLE,
            f"payload retention failed at transport segment {segment}",
            status=AtomicSkillStatus.SAFETY_ABORT,
            output={
                "failed_segment": segment,
                "retention_sample": sample,
                "command_ids": command_ids,
                "place_authorized": False,
            },
        )

    def _failed(
        self,
        trace: SkillTraceRecorder,
        code: AtomicSkillFailureCode,
        message: str,
        *,
        status: AtomicSkillStatus = AtomicSkillStatus.RETRYABLE_FAILURE,
        output: dict[str, object] | None = None,
    ) -> AtomicSkillResult:
        trace.record("transport", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=status,
            failure_code=code,
            message=message,
            output={} if output is None else output,
        )


__all__ = ["TransportPayloadSkill"]
