"""Read-only admission gate for one harvest cycle."""

from __future__ import annotations

from module.robotics import RoboticsBundle
from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
)
from module.skills.tomato_pick.contracts import (
    ACTION_DIM,
    CRITIC_STATE_DIM,
    OBSERVATION_DIM,
    PolicyManifest,
)
from proto import (
    CapabilityCheck,
    CheckState,
    HarvestContainer,
    SystemSelfCheckReport,
    SystemSelfCheckRequest,
)

from .ports import (
    CameraHealthPort,
    EmergencyStopHealthPort,
    VisionWorkerHealthPort,
)


def _age_ok(now_ns: int, sample_ns: int, max_age_s: float) -> bool:
    return 0 <= now_ns - sample_ns <= max_age_s * 1_000_000_000


class SystemSelfCheckSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="system_self_check",
        version="1.0.0",
        description="Read-only admission check for camera, robot, tactile, safety, and policy state.",
        effectful=False,
        required_resources=(
            "camera_health",
            "vision_worker_health",
            "arm_state",
            "hand_state",
            "tactile_state",
            "emergency_stop_health",
            "control_lease",
            "container",
            "policy_manifest",
        ),
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        camera_health: CameraHealthPort,
        vision_worker_health: VisionWorkerHealthPort,
        emergency_stop_health: EmergencyStopHealthPort,
        policy_manifest: PolicyManifest,
        container: HarvestContainer,
        execution_mode: str = "offline_mock",
        expected_camera_calibration_id: str = "orbbec_gemini_pro_mock_calibration_v1",
        expected_camera_device_id: str | None = None,
        expected_tactile_calibration_id: str = "mock_tactile_calibration_v1",
        expected_hand_mapper_id: str = "g20_eigengrasp_5d_v3_thumb_roll_fixed",
        max_camera_age_s: float = 0.5,
        max_vision_worker_age_s: float = 0.5,
        max_estop_age_s: float = 0.5,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(camera_health, CameraHealthPort):
            raise TypeError("camera_health must implement CameraHealthPort")
        if not isinstance(emergency_stop_health, EmergencyStopHealthPort):
            raise TypeError(
                "emergency_stop_health must implement EmergencyStopHealthPort"
            )
        if not isinstance(vision_worker_health, VisionWorkerHealthPort):
            raise TypeError(
                "vision_worker_health must implement VisionWorkerHealthPort"
            )
        if not isinstance(policy_manifest, PolicyManifest):
            raise TypeError("policy_manifest must be a PolicyManifest")
        if not isinstance(container, HarvestContainer):
            raise TypeError("container must be a HarvestContainer")
        if execution_mode not in {"offline_mock", "simulation", "shadow", "hardware"}:
            raise ValueError(
                "execution_mode must be offline_mock, simulation, shadow, or hardware"
            )
        for name, value in {
            "expected_camera_calibration_id": expected_camera_calibration_id,
            "expected_tactile_calibration_id": expected_tactile_calibration_id,
            "expected_hand_mapper_id": expected_hand_mapper_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if expected_camera_device_id is not None and (
            not isinstance(expected_camera_device_id, str)
            or not expected_camera_device_id.strip()
        ):
            raise ValueError(
                "expected_camera_device_id must be a non-empty string or None"
            )
        for name, value in {
            "max_camera_age_s": max_camera_age_s,
            "max_vision_worker_age_s": max_vision_worker_age_s,
            "max_estop_age_s": max_estop_age_s,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be > 0")
        self._robotics = robotics
        self._camera_health = camera_health
        self._vision_worker_health = vision_worker_health
        self._emergency_stop_health = emergency_stop_health
        self._manifest = policy_manifest
        self._container = container
        self._execution_mode = execution_mode
        self._expected_camera_calibration_id = expected_camera_calibration_id.strip()
        self._expected_camera_device_id = (
            None
            if expected_camera_device_id is None
            else expected_camera_device_id.strip()
        )
        self._expected_tactile_calibration_id = expected_tactile_calibration_id.strip()
        self._expected_hand_mapper_id = expected_hand_mapper_id.strip()
        self._max_camera_age_s = float(max_camera_age_s)
        self._max_vision_worker_age_s = float(max_vision_worker_age_s)
        self._max_estop_age_s = float(max_estop_age_s)

    @staticmethod
    def _check(
        name: str,
        passed: bool | None,
        message: str,
        *,
        evidence_sources: tuple[str, ...] = (),
        details: dict[str, object] | None = None,
    ) -> CapabilityCheck:
        state = (
            CheckState.UNKNOWN
            if passed is None
            else CheckState.PASS
            if passed
            else CheckState.FAIL
        )
        return CapabilityCheck(
            name=name,
            state=state,
            critical=True,
            message=message,
            evidence_sources=evidence_sources,
            details={} if details is None else details,
        )

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, SystemSelfCheckRequest):
            raise TypeError("request must be a SystemSelfCheckRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "self_check",
            SkillTraceEventKind.STARTED,
            input_summary={
                "arm_side": request.arm_side.value,
                "execution_mode": self._execution_mode,
            },
        )
        try:
            now_ns = self._robotics.clock.monotonic_ns()
            camera = self._camera_health.snapshot()
            vision_worker = self._vision_worker_health.snapshot()
            estop = self._emergency_stop_health.snapshot()
            arm = self._robotics.arm.get_state(request.arm_side)
            hand = self._robotics.hand.get_state(request.arm_side)
            tactile = self._robotics.tactile.get_frame(request.arm_side)
            checks: list[CapabilityCheck] = []
            checks.append(
                self._check(
                    "camera_sdk",
                    camera.sdk_available,
                    "Orbbec SDK layout is available"
                    if camera.sdk_available
                    else "Orbbec SDK layout is missing or incomplete",
                    evidence_sources=camera.evidence_sources,
                    details={"backend": camera.backend_name},
                )
            )
            camera_fresh = _age_ok(
                now_ns, camera.sample_monotonic_ns, self._max_camera_age_s
            )
            checks.append(
                self._check(
                    "camera_connection",
                    camera.device_connected and camera_fresh,
                    "camera health snapshot is connected and fresh"
                    if camera.device_connected and camera_fresh
                    else "camera is disconnected or its health snapshot is stale",
                    evidence_sources=camera.evidence_sources,
                    details={
                        "device_id": camera.device_id,
                        "simulated": camera.simulated,
                        "fresh": camera_fresh,
                    },
                )
            )
            checks.append(
                self._check(
                    "camera_rgb",
                    camera.rgb_live,
                    "RGB stream is live" if camera.rgb_live else "RGB stream is not live",
                    evidence_sources=camera.evidence_sources,
                )
            )
            checks.append(
                self._check(
                    "camera_depth",
                    camera.depth_live,
                    "depth stream is live"
                    if camera.depth_live
                    else "depth stream is not live",
                    evidence_sources=camera.evidence_sources,
                )
            )
            calibration_ok = (
                camera.calibration_id == self._expected_camera_calibration_id
            )
            checks.append(
                self._check(
                    "camera_calibration",
                    calibration_ok,
                    "camera calibration matches the harvest profile"
                    if calibration_ok
                    else "camera calibration is absent or mismatched",
                    evidence_sources=camera.evidence_sources,
                    details={
                        "actual": camera.calibration_id,
                        "expected": self._expected_camera_calibration_id,
                    },
                )
            )
            identity_ok = (
                self._expected_camera_device_id is None
                and self._execution_mode in {"offline_mock", "simulation"}
            ) or camera.device_id == self._expected_camera_device_id
            checks.append(
                self._check(
                    "camera_identity",
                    identity_ok,
                    "camera identity matches the configured profile"
                    if identity_ok
                    else "camera identity is absent or mismatched",
                    evidence_sources=camera.evidence_sources,
                    details={
                        "actual": camera.device_id,
                        "expected": self._expected_camera_device_id,
                    },
                )
            )
            vision_worker_fresh = _age_ok(
                now_ns,
                vision_worker.sample_monotonic_ns,
                self._max_vision_worker_age_s,
            )
            vision_worker_ok = vision_worker.ready and vision_worker_fresh
            checks.append(
                self._check(
                    "vision_worker",
                    vision_worker_ok,
                    "isolated vision worker is ready and identity-checked"
                    if vision_worker_ok
                    else "isolated vision worker is unavailable, unready, or stale",
                    evidence_sources=vision_worker.evidence_sources,
                    details={
                        "worker_id": vision_worker.worker_id,
                        "backend_id": vision_worker.backend_id,
                        "ready": vision_worker.ready,
                        "fresh": vision_worker_fresh,
                        "simulated": vision_worker.simulated,
                    },
                )
            )
            non_simulated_ok = (
                True
                if self._execution_mode in {"offline_mock", "simulation"}
                else not camera.simulated
                and not vision_worker.simulated
                and not estop.simulated
            )
            checks.append(
                self._check(
                    "execution_mode",
                    non_simulated_ok,
                    "execution mode matches the health backends"
                    if non_simulated_ok
                    else "shadow/hardware mode cannot use simulated health evidence",
                    details={"execution_mode": self._execution_mode},
                )
            )
            arm_fresh = arm.stamp.valid and _age_ok(
                now_ns,
                arm.stamp.sample_monotonic_ns,
                self._manifest.max_state_age_s[1],
            )
            checks.append(
                self._check(
                    "arm_state",
                    arm.enabled and arm_fresh,
                    "arm is enabled and state is fresh"
                    if arm.enabled and arm_fresh
                    else "arm is disabled, invalid, or stale",
                    evidence_sources=(f"arm_state:{arm.stamp.source_sequence}",),
                    details={"enabled": arm.enabled, "fresh": arm_fresh},
                )
            )
            hand_fresh = hand.stamp.valid and _age_ok(
                now_ns,
                hand.stamp.sample_monotonic_ns,
                self._manifest.max_state_age_s[2],
            )
            hand_ok = hand_fresh and not hand.fault_codes
            checks.append(
                self._check(
                    "hand_state",
                    hand_ok,
                    "hand state is fresh and fault-free"
                    if hand_ok
                    else "hand state is stale, invalid, or faulted",
                    evidence_sources=(f"hand_state:{hand.stamp.source_sequence}",),
                    details={"fresh": hand_fresh, "fault_codes": hand.fault_codes},
                )
            )
            tactile_fresh = _age_ok(
                now_ns,
                tactile.received_monotonic_ns,
                self._manifest.max_force_age_s,
            )
            tactile_samples_fresh = all(
                finger.force_sample_monotonic_ns is not None
                and finger.matrix_sample_monotonic_ns is not None
                and _age_ok(
                    now_ns,
                    finger.force_sample_monotonic_ns,
                    self._manifest.max_force_age_s,
                )
                and _age_ok(
                    now_ns,
                    finger.matrix_sample_monotonic_ns,
                    self._manifest.max_force_age_s,
                )
                for finger in tactile.fingers
            )
            tactile_valid = tactile_fresh and tactile_samples_fresh and all(
                finger.force_valid and finger.matrix_valid
                for finger in tactile.fingers
            )
            tactile_calibrated = (
                tactile.calibration_id == self._expected_tactile_calibration_id
            )
            checks.append(
                self._check(
                    "tactile_state",
                    tactile_valid and tactile_calibrated,
                    "five-finger tactile state is fresh, valid, and calibrated"
                    if tactile_valid and tactile_calibrated
                    else "tactile state is stale, invalid, or calibration-mismatched",
                    evidence_sources=(
                        f"tactile_frame:{tactile.source_sequence}",
                        tactile.calibration_id,
                    ),
                    details={
                        "fresh": tactile_fresh,
                        "samples_fresh": tactile_samples_fresh,
                        "calibration_id": tactile.calibration_id,
                        "expected_calibration_id": self._expected_tactile_calibration_id,
                    },
                )
            )
            estop_fresh = _age_ok(
                now_ns, estop.sample_monotonic_ns, self._max_estop_age_s
            )
            estop_ok: bool | None
            if not estop.available or not estop_fresh or estop.healthy is None:
                estop_ok = None
            else:
                estop_ok = estop.healthy
            checks.append(
                self._check(
                    "emergency_stop",
                    estop_ok,
                    "emergency-stop chain is available and healthy"
                    if estop_ok is True
                    else "emergency-stop chain is unhealthy or its state is unknown",
                    evidence_sources=estop.evidence_sources,
                    details={"available": estop.available, "fresh": estop_fresh},
                )
            )
            lease_idle = self._robotics.leases.active(request.arm_side) is None
            checks.append(
                self._check(
                    "control_lease",
                    lease_idle,
                    "arm/hand control lease is idle"
                    if lease_idle
                    else "arm/hand control lease is already owned",
                )
            )
            checks.append(
                self._check(
                    "container",
                    self._container.accepts_payload,
                    "harvest container is available"
                    if self._container.accepts_payload
                    else "harvest container is unavailable or full",
                    details={
                        "container_id": self._container.container_id,
                        "capacity_remaining": self._container.capacity_remaining,
                    },
                )
            )
            policy_ok = (
                self._manifest.hand_joint_mapper_id == self._expected_hand_mapper_id
                and OBSERVATION_DIM == 66
                and CRITIC_STATE_DIM == 83
                and ACTION_DIM == 11
            )
            checks.append(
                self._check(
                    "policy_manifest",
                    policy_ok,
                    "policy and hand-mapper contracts match"
                    if policy_ok
                    else "policy or hand-mapper contract mismatch",
                    details={
                        "policy_id": self._manifest.policy_id,
                        "hand_joint_mapper_id": self._manifest.hand_joint_mapper_id,
                        "expected_hand_mapper_id": self._expected_hand_mapper_id,
                        "actor_observation_dim": OBSERVATION_DIM,
                        "critic_state_dim": CRITIC_STATE_DIM,
                        "action_dim": ACTION_DIM,
                    },
                )
            )
            report = SystemSelfCheckReport(
                execution_id=request.execution_id,
                execution_mode=self._execution_mode,
                checked_monotonic_ns=now_ns,
                checks=tuple(checks),
            )
            for check in report.checks:
                trace.record(
                    check.name,
                    SkillTraceEventKind.CHECKPOINT,
                    message=check.message,
                    output_summary={
                        "state": check.state.value,
                        "critical": check.critical,
                        **dict(check.details),
                    },
                    evidence_refs=check.evidence_sources,
                )
            if not report.admission_allowed:
                trace.record(
                    "admission",
                    SkillTraceEventKind.FAILED,
                    message="system self-check rejected harvest admission",
                    output_summary={
                        "failed_check_names": report.failed_check_names
                    },
                )
                return self._result(
                    trace,
                    status=AtomicSkillStatus.SAFETY_ABORT,
                    failure_code=AtomicSkillFailureCode.SAFETY_REJECTED,
                    message="system self-check rejected harvest admission",
                    output=report.as_dict(),
                )
            trace.record(
                "admission",
                SkillTraceEventKind.SUCCEEDED,
                message="system self-check admitted harvest execution",
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                message="system self-check admitted harvest execution",
                output=report.as_dict(),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            trace.record(
                "self_check",
                SkillTraceEventKind.FAILED,
                message=str(exc),
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                failure_code=AtomicSkillFailureCode.INFRASTRUCTURE_ERROR,
                message=str(exc),
            )

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for resource in (
            self._emergency_stop_health,
            self._vision_worker_health,
            self._camera_health,
        ):
            try:
                resource.close()
            except BaseException as exc:
                errors.append(exc)
        super().close()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("system self-check cleanup failed", errors)


__all__ = ["SystemSelfCheckSkill"]
