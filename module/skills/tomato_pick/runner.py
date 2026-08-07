"""Skill runner with independent verification, leases, and diagnostic progress."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Protocol, runtime_checkable

from proto import (
    ArmState,
    EvidenceState,
    ExecutionState,
    HandState,
    PickFailureCode,
    PickPhase,
    SafetyAssessment,
    SkillStatus,
    TactileFrame,
    TomatoGraspPlan,
    TomatoDetectionResult,
    TomatoPickSkillRequest,
    TomatoPickSkillResult,
    TomatoPickVerification,
    TomatoTarget,
)

from module.robotics import (
    ArmControlPort,
    Clock,
    ControlBusyError,
    ControlLease,
    ControlLeaseManager,
    FollowPoseCommand,
    HandControlPort,
    HandJointPositionCommand,
    TactilePort,
)

from .contracts import PolicyManifest, TomatoPickAction
from .features import ObservationBuilder, ObservationSnapshot, TactileFeatureExtractor
from .policy import TomatoPickPolicy
from .supervisor import TomatoPickActionSupervisor


@dataclass(frozen=True, slots=True)
class VerifierSession:
    execution_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        object.__setattr__(self, "execution_id", self.execution_id.strip())


@runtime_checkable
class PickVerifier(Protocol):
    def begin(
        self,
        execution_id: str,
        target: TomatoTarget,
        plan: TomatoGraspPlan,
    ) -> VerifierSession: ...

    def update(
        self,
        session: VerifierSession,
        perception: TomatoDetectionResult | None,
        arm: ArmState,
        hand: HandState,
        tactile: TactileFrame,
    ) -> TomatoPickVerification: ...

    def finish(self, session: VerifierSession) -> TomatoPickVerification: ...


@dataclass(frozen=True, slots=True)
class TomatoPickSkillContext:
    target: TomatoTarget
    plan: TomatoGraspPlan
    safety: SafetyAssessment
    perception: TomatoDetectionResult | None
    arm: ArmControlPort
    hand: HandControlPort
    tactile: TactilePort
    leases: ControlLeaseManager
    verifier: PickVerifier
    clock: Clock

    def __post_init__(self) -> None:
        if not isinstance(self.target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        if not isinstance(self.plan, TomatoGraspPlan):
            raise TypeError("plan must be a TomatoGraspPlan")
        if not isinstance(self.safety, SafetyAssessment):
            raise TypeError("safety must be a SafetyAssessment")
        if self.perception is not None and not isinstance(
            self.perception, TomatoDetectionResult
        ):
            raise TypeError("perception must be a TomatoDetectionResult or None")
        if not isinstance(self.arm, ArmControlPort):
            raise TypeError("arm must implement ArmControlPort")
        if not isinstance(self.hand, HandControlPort):
            raise TypeError("hand must implement HandControlPort")
        if not isinstance(self.tactile, TactilePort):
            raise TypeError("tactile must implement TactilePort")
        if not isinstance(self.leases, ControlLeaseManager):
            raise TypeError("leases must be a ControlLeaseManager")
        if not isinstance(self.verifier, PickVerifier):
            raise TypeError("verifier must implement PickVerifier")
        if not isinstance(self.clock, Clock):
            raise TypeError("clock must implement Clock")


@dataclass(slots=True)
class _ActiveExecution:
    request: TomatoPickSkillRequest
    context: TomatoPickSkillContext
    phase: PickPhase
    started_ns: int
    lease: ControlLease | None = None


class TomatoPickSkillRunner:
    """Execute one policy until verified success, failure, cancellation, or timeout."""

    def __init__(
        self,
        policy: TomatoPickPolicy,
        *,
        maximum_normal_force_n: float = 12.0,
    ) -> None:
        if not isinstance(policy, TomatoPickPolicy):
            raise TypeError("policy must implement TomatoPickPolicy")
        if (
            isinstance(maximum_normal_force_n, bool)
            or not isinstance(maximum_normal_force_n, (int, float))
            or not 0.0 < float(maximum_normal_force_n) <= 20.0
        ):
            raise ValueError("maximum_normal_force_n must be in (0,20]")
        self.policy = policy
        self.manifest: PolicyManifest = policy.manifest
        self.maximum_normal_force_n = float(maximum_normal_force_n)
        self.extractor = TactileFeatureExtractor()
        self.observation_builder = ObservationBuilder(self.manifest, self.extractor)
        self.supervisor = TomatoPickActionSupervisor(self.manifest)
        self._results: dict[str, TomatoPickSkillResult] = {}
        self._cancel_requests: dict[str, str] = {}
        self._active: dict[str, _ActiveExecution] = {}
        self._lock = threading.RLock()

    def run(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
    ) -> TomatoPickSkillResult:
        if not isinstance(request, TomatoPickSkillRequest):
            raise TypeError("request must be a TomatoPickSkillRequest")
        if not isinstance(context, TomatoPickSkillContext):
            raise TypeError("context must be a TomatoPickSkillContext")
        with self._lock:
            cached = self._results.get(request.execution_id)
            if cached is not None:
                return cached
            if request.execution_id in self._active:
                raise RuntimeError(f"execution {request.execution_id!r} is already running")
            active = _ActiveExecution(
                request=request,
                context=context,
                phase=PickPhase.CONTACT,
                started_ns=context.clock.monotonic_ns(),
            )
            self._active[request.execution_id] = active

        session: VerifierSession | None = None
        lease: ControlLease | None = None
        result: TomatoPickSkillResult
        try:
            precondition_failure = self._precondition_failure(request, context)
            if precondition_failure is not None:
                result = self._failure_without_session(
                    request,
                    context,
                    phase=PickPhase.CONTACT,
                    code=precondition_failure,
                    status=self._status_for(precondition_failure),
                    started_ns=active.started_ns,
                )
                return self._remember(result)

            session = context.verifier.begin(
                request.execution_id,
                context.target,
                context.plan,
            )
            try:
                lease = context.leases.acquire(
                    owner_id=f"tomato-pick:{request.execution_id}",
                    side=request.arm_side,
                    ttl_s=request.max_duration_s + 1.0,
                )
                active.lease = lease
            except ControlBusyError:
                result = self._failure(
                    request,
                    context,
                    session,
                    phase=PickPhase.CONTACT,
                    code=PickFailureCode.CONTROL_BUSY,
                    status=SkillStatus.SAFETY_ABORT,
                    started_ns=active.started_ns,
                )
                return self._remember(result)

            self.policy.reset()
            self.extractor.reset()
            previous_action = TomatoPickAction.zero()
            peak_normal: list[float | None] = [None] * 5
            peak_tangent: list[float | None] = [None] * 5
            command_index = 0
            failure_by_phase = {
                PickPhase.CONTACT: PickFailureCode.NO_CONTACT,
                PickPhase.GRASP: PickFailureCode.UNSTABLE_GRASP,
                PickPhase.DETACH: PickFailureCode.DETACH_NOT_CONFIRMED,
                PickPhase.HOLD: PickFailureCode.TIMEOUT,
            }
            phase = PickPhase.CONTACT
            while True:
                active.phase = phase
                if request.execution_id in self._cancel_requests:
                    result = self._failure(
                        request,
                        context,
                        session,
                        phase=phase,
                        code=PickFailureCode.CANCELLED,
                        status=SkillStatus.CANCELLED,
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)
                elapsed_ns = context.clock.monotonic_ns() - active.started_ns
                if elapsed_ns > request.max_duration_s * 1e9:
                    # A sub-cycle budget cannot produce meaningful task
                    # evidence.  Otherwise preserve the last progress label as
                    # a useful failure diagnosis without imposing phase clocks.
                    code = (
                        PickFailureCode.TIMEOUT
                        if (
                            command_index == 0
                            or request.max_duration_s
                            < 1.0 / self.manifest.policy_rate_hz
                        )
                        else failure_by_phase[phase]
                    )
                    result = self._failure(
                        request,
                        context,
                        session,
                        phase=phase,
                        code=code,
                        status=(
                            SkillStatus.TIMED_OUT
                            if code is PickFailureCode.TIMEOUT
                            else self._status_for(code)
                        ),
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)

                arm = context.arm.get_state(request.arm_side)
                hand = context.hand.get_state(request.arm_side)
                tactile = context.tactile.get_frame(request.arm_side)
                snapshot = self.observation_builder.build(
                    now_ns=context.clock.monotonic_ns(),
                    target=context.target,
                    plan=context.plan,
                    episode_progress=min(
                        1.0,
                        elapsed_ns / (request.max_duration_s * 1e9),
                    ),
                    arm=arm,
                    hand=hand,
                    tactile=tactile,
                    previous_action=previous_action,
                )
                self._update_peaks(snapshot, peak_normal, peak_tangent)
                health_failure = self._health_failure(snapshot, arm, hand)
                if health_failure is not None:
                    result = self._failure(
                        request,
                        context,
                        session,
                        phase=phase,
                        code=health_failure,
                        status=self._status_for(health_failure),
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)

                verification = context.verifier.update(
                    session,
                    context.perception,
                    arm,
                    hand,
                    tactile,
                )
                evidence_failure = self._evidence_failure(verification, snapshot)
                if evidence_failure is not None:
                    result = self._failure(
                        request,
                        context,
                        session,
                        phase=phase,
                        code=evidence_failure,
                        status=SkillStatus.RETRYABLE_FAILURE,
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)
                if verification.successful:
                    result = self._success(
                        request,
                        context,
                        verification,
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)

                if verification.detached is EvidenceState.CONFIRMED:
                    phase = PickPhase.HOLD
                elif any(
                    force is not None and force > 0.0
                    for force in snapshot.normal_force_n
                ):
                    phase = (
                        PickPhase.DETACH
                        if hand.synergy.closure >= 0.15
                        else PickPhase.GRASP
                    )
                else:
                    phase = PickPhase.CONTACT
                active.phase = phase

                policy_action = self.policy.act(snapshot.observation)
                supervised = self.supervisor.supervise(
                    policy_action,
                    phase=phase,
                    arm=arm,
                    hand=hand,
                    plan=context.plan,
                )
                command_index += 1
                arm_handle = context.arm.follow_pose(
                    lease,
                    FollowPoseCommand(
                        command_id=f"{request.execution_id}-arm-{command_index}",
                        side=request.arm_side,
                        target_pose=supervised.target_pose,
                        timeout_s=0.10,
                    ),
                )
                hand_handle = context.hand.command_joint_positions(
                    lease,
                    HandJointPositionCommand(
                        command_id=f"{request.execution_id}-hand-{command_index}",
                        side=request.arm_side,
                        joint_positions_rad=supervised.target_hand_joint_positions_rad,
                        timeout_s=0.10,
                    ),
                )
                if (
                    arm_handle.state is not ExecutionState.SUCCEEDED
                    or hand_handle.state is not ExecutionState.SUCCEEDED
                ):
                    result = self._failure(
                        request,
                        context,
                        session,
                        phase=phase,
                        code=PickFailureCode.COMMAND_REJECTED,
                        status=SkillStatus.INFRASTRUCTURE_ERROR,
                        started_ns=active.started_ns,
                        peak_normal=peak_normal,
                        peak_tangent=peak_tangent,
                    )
                    return self._remember(result)
                previous_action = supervised.normalized_action
                context.clock.sleep(1.0 / self.manifest.policy_rate_hz)
        except BaseException:
            raise
        finally:
            if lease is not None:
                try:
                    context.hand.hold(lease, request.arm_side)
                except BaseException:
                    pass
                context.leases.release(lease)
            with self._lock:
                self._active.pop(request.execution_id, None)

    def request_cancel(self, execution_id: str, reason: str) -> None:
        """Record an asynchronous cancellation request checked before each command."""

        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            self._cancel_requests[execution_id.strip()] = reason.strip()

    def cancel(self, execution_id: str, reason: str) -> TomatoPickSkillResult:
        """Cancel an active execution; completed executions return their stable result."""

        self.request_cancel(execution_id, reason)
        with self._lock:
            cached = self._results.get(execution_id)
            if cached is not None:
                return cached
            active = self._active.get(execution_id)
            if active is None:
                raise KeyError(f"execution {execution_id!r} is not active")
            return self._failure_without_session(
                active.request,
                active.context,
                phase=active.phase,
                code=PickFailureCode.CANCELLED,
                status=SkillStatus.CANCELLED,
                started_ns=active.started_ns,
            )

    def result(self, execution_id: str) -> TomatoPickSkillResult | None:
        with self._lock:
            return self._results.get(execution_id)

    def _precondition_failure(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
    ) -> PickFailureCode | None:
        now = context.clock.monotonic_ns()
        if request.policy_id != self.manifest.policy_id:
            return PickFailureCode.PRECONDITION_FAILED
        if (
            request.target_id != context.target.target_id
            or request.grasp_plan_id != context.plan.grasp_plan_id
            or request.arm_side is not context.plan.arm_side
        ):
            return PickFailureCode.PRECONDITION_FAILED
        if context.plan.mapping_calibration_id != self.manifest.hand_joint_mapper_id:
            return PickFailureCode.PRECONDITION_FAILED
        if now >= context.plan.expires_monotonic_ns:
            return PickFailureCode.PLAN_STALE
        if now >= context.safety.expires_monotonic_ns:
            return PickFailureCode.PLAN_STALE
        if context.safety.grasp_plan_id != context.plan.grasp_plan_id:
            return PickFailureCode.PRECONDITION_FAILED
        if not context.safety.approved:
            return PickFailureCode.COLLISION_UNSAFE
        arm_state = context.arm.get_state(request.arm_side)
        if arm_state.stamp.source_sequence != context.safety.arm_state_sequence:
            return PickFailureCode.PRECONDITION_FAILED
        return None

    def _health_failure(
        self,
        snapshot: ObservationSnapshot,
        arm: ArmState,
        hand: HandState,
    ) -> PickFailureCode | None:
        if hand.fault_codes or not arm.enabled:
            return PickFailureCode.DEVICE_FAULT
        if not snapshot.target_valid:
            return PickFailureCode.TARGET_STALE
        if not snapshot.arm_valid:
            return PickFailureCode.ARM_STATE_STALE
        if not snapshot.hand_valid:
            return PickFailureCode.HAND_STATE_STALE
        if not all(snapshot.force_valid) or not all(snapshot.matrix_valid):
            return PickFailureCode.TACTILE_STALE
        if snapshot.max_normal_force_n > self.maximum_normal_force_n:
            return PickFailureCode.EXCESS_FORCE
        return None

    @staticmethod
    def _evidence_failure(
        verification: TomatoPickVerification,
        snapshot: ObservationSnapshot,
    ) -> PickFailureCode | None:
        if verification.dropped:
            return PickFailureCode.DROPPED
        if (
            snapshot.slip_detected
            or verification.retained_without_slip is EvidenceState.REJECTED
        ):
            return PickFailureCode.SLIP_DETECTED
        return None

    @staticmethod
    def _update_peaks(
        snapshot: ObservationSnapshot,
        peak_normal: list[float | None],
        peak_tangent: list[float | None],
    ) -> None:
        for index, value in enumerate(snapshot.normal_force_n):
            if value is not None:
                peak_normal[index] = max(peak_normal[index] or 0.0, value)
        for index, value in enumerate(snapshot.tangential_force_n):
            if value is not None:
                peak_tangent[index] = max(peak_tangent[index] or 0.0, value)

    @staticmethod
    def _status_for(code: PickFailureCode) -> SkillStatus:
        if code in {
            PickFailureCode.NO_CONTACT,
            PickFailureCode.UNSTABLE_GRASP,
            PickFailureCode.DETACH_NOT_CONFIRMED,
            PickFailureCode.SLIP_DETECTED,
            PickFailureCode.DROPPED,
        }:
            return SkillStatus.RETRYABLE_FAILURE
        if code is PickFailureCode.TIMEOUT:
            return SkillStatus.TIMED_OUT
        if code is PickFailureCode.CANCELLED:
            return SkillStatus.CANCELLED
        if code in {PickFailureCode.COMMAND_REJECTED, PickFailureCode.INTERNAL_ERROR}:
            return SkillStatus.INFRASTRUCTURE_ERROR
        return SkillStatus.SAFETY_ABORT

    @staticmethod
    def _unknown_verification(
        execution_id: str,
        reason: str,
    ) -> TomatoPickVerification:
        return TomatoPickVerification(
            execution_id=execution_id,
            detached=EvidenceState.UNKNOWN,
            retained_without_slip=EvidenceState.UNKNOWN,
            dropped=False,
            stable_window_s=0.0,
            evidence_sources=(),
            verifier_version="not_started",
            valid=False,
            reason=reason,
        )

    def _failure_without_session(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
        *,
        phase: PickPhase,
        code: PickFailureCode,
        status: SkillStatus,
        started_ns: int,
    ) -> TomatoPickSkillResult:
        return self._build_result(
            request,
            context,
            phase=phase,
            status=status,
            verification=self._unknown_verification(request.execution_id, code.value),
            failure_code=code,
            started_ns=started_ns,
        )

    def _failure(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
        session: VerifierSession,
        *,
        phase: PickPhase,
        code: PickFailureCode,
        status: SkillStatus,
        started_ns: int,
        peak_normal: list[float | None] | None = None,
        peak_tangent: list[float | None] | None = None,
    ) -> TomatoPickSkillResult:
        return self._build_result(
            request,
            context,
            phase=phase,
            status=status,
            verification=context.verifier.finish(session),
            failure_code=code,
            started_ns=started_ns,
            peak_normal=peak_normal,
            peak_tangent=peak_tangent,
        )

    def _success(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
        verification: TomatoPickVerification,
        *,
        started_ns: int,
        peak_normal: list[float | None],
        peak_tangent: list[float | None],
    ) -> TomatoPickSkillResult:
        return self._build_result(
            request,
            context,
            phase=PickPhase.HOLD,
            status=SkillStatus.SUCCEEDED,
            verification=verification,
            failure_code=None,
            started_ns=started_ns,
            peak_normal=peak_normal,
            peak_tangent=peak_tangent,
        )

    def _build_result(
        self,
        request: TomatoPickSkillRequest,
        context: TomatoPickSkillContext,
        *,
        phase: PickPhase,
        status: SkillStatus,
        verification: TomatoPickVerification,
        failure_code: PickFailureCode | None,
        started_ns: int,
        peak_normal: list[float | None] | None = None,
        peak_tangent: list[float | None] | None = None,
    ) -> TomatoPickSkillResult:
        arm = context.arm.get_state(request.arm_side)
        hand = context.hand.get_state(request.arm_side)
        tactile = context.tactile.get_frame(request.arm_side)
        success = status is SkillStatus.SUCCEEDED and verification.successful
        return TomatoPickSkillResult(
            execution_id=request.execution_id,
            target_id=request.target_id,
            grasp_plan_id=request.grasp_plan_id,
            status=status,
            final_phase=phase,
            success=success,
            verification=verification,
            failure_code=failure_code,
            duration_s=max(0.0, (context.clock.monotonic_ns() - started_ns) / 1e9),
            peak_normal_force_n=tuple(peak_normal or [None] * 5),
            peak_tangential_force_n=tuple(peak_tangent or [None] * 5),
            final_arm_sequence=arm.stamp.source_sequence,
            final_hand_sequence=hand.stamp.source_sequence,
            final_tactile_sequence=tactile.source_sequence,
            policy_id=request.policy_id,
        )

    def _remember(self, result: TomatoPickSkillResult) -> TomatoPickSkillResult:
        with self._lock:
            existing = self._results.setdefault(result.execution_id, result)
            return existing


__all__ = [
    "PickVerifier",
    "TomatoPickSkillContext",
    "TomatoPickSkillRunner",
    "VerifierSession",
]
