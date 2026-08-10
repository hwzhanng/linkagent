"""Localize a dropped tomato and route it without reusing the hanging-fruit PPO."""

from __future__ import annotations

import math
from collections.abc import Sequence

from module.robotics import Clock
from proto import (
    DroppedPayloadDisposition,
    DroppedPayloadLocalizationRequest,
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

from .ports import DroppedPayloadObservationPort


def _workspace(value: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("recovery_workspace_xyzxyz_m must be a six-number sequence")
    items = tuple(value)
    if len(items) != 6:
        raise ValueError("recovery_workspace_xyzxyz_m must contain six values")
    numbers: list[float] = []
    for index, item in enumerate(items):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(
                f"recovery_workspace_xyzxyz_m[{index}] must be finite"
            )
        numbers.append(float(item))
    if any(numbers[index] >= numbers[index + 3] for index in range(3)):
        raise ValueError("recovery workspace minimums must be below maximums")
    return tuple(numbers)  # type: ignore[return-value]


class LocalizeDroppedPayloadSkill(AtomicSkillBase):
    """Resolve where a dropped fruit went; never starts an incompatible regrasp."""

    descriptor = AtomicSkillDescriptor(
        skill_id="localize_dropped_payload",
        version="1.0.0",
        description=(
            "Use fixed-camera evidence to classify a lost fruit as already in the "
            "container, eligible for a dedicated recovery policy, or manual-only."
        ),
        effectful=False,
        required_resources=(
            "fixed_top_rgbd",
            "payload_track",
            "container_geometry",
        ),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        observation_port: DroppedPayloadObservationPort,
        clock: Clock,
        recovery_workspace_xyzxyz_m: Sequence[float],
        minimum_confidence: float = 0.60,
    ) -> None:
        super().__init__(clock=clock)
        if not isinstance(observation_port, DroppedPayloadObservationPort):
            raise TypeError(
                "observation_port must implement DroppedPayloadObservationPort"
            )
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not math.isfinite(float(minimum_confidence))
            or not 0.0 < float(minimum_confidence) <= 1.0
        ):
            raise ValueError("minimum_confidence must be in (0, 1]")
        self._observation_port = observation_port
        self._workspace = _workspace(recovery_workspace_xyzxyz_m)
        self._minimum_confidence = float(minimum_confidence)
        self._closed_port = False

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, DroppedPayloadLocalizationRequest):
            raise TypeError("request must be a DroppedPayloadLocalizationRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "localize",
            SkillTraceEventKind.STARTED,
            input_summary={
                "payload_id": request.payload.payload_id,
                "target_id": request.payload.target_id,
                "container_id": request.container.container_id,
            },
        )
        try:
            observation = self._observation_port.observe(request)
        except (RuntimeError, ValueError) as exc:
            return self._failed(
                trace,
                AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE,
                str(exc),
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                output={"next_action": "retry_localization"},
            )
        if observation.payload_id != request.payload.payload_id:
            return self._failed(
                trace,
                AtomicSkillFailureCode.INVALID_REQUEST,
                "dropped-payload observation does not match the requested payload",
            )
        if not observation.observed or observation.center_pose is None:
            return self._failed(
                trace,
                AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE,
                "dropped payload was not observed by the fixed top camera",
                output={
                    "next_action": "retry_localization_then_manual_intervention",
                    "motion_authorized": False,
                },
            )
        now_ns = self._clock.monotonic_ns()
        if observation.sample_monotonic_ns > now_ns:
            return self._failed(
                trace,
                AtomicSkillFailureCode.INVALID_REQUEST,
                "dropped-payload observation timestamp is in the future",
            )
        age_s = (now_ns - observation.sample_monotonic_ns) / 1.0e9
        if age_s > request.max_observation_age_s:
            return self._failed(
                trace,
                AtomicSkillFailureCode.TARGET_STALE,
                "dropped-payload observation is stale",
                output={"observation_age_s": age_s, "motion_authorized": False},
            )
        if observation.confidence < self._minimum_confidence:
            return self._failed(
                trace,
                AtomicSkillFailureCode.PERCEPTION_UNAVAILABLE,
                "dropped-payload localization confidence is below the configured gate",
                output={
                    "confidence": observation.confidence,
                    "minimum_confidence": self._minimum_confidence,
                    "motion_authorized": False,
                },
            )
        position = observation.center_pose.position_m
        inside_recovery_workspace = all(
            self._workspace[index] <= position[index] <= self._workspace[index + 3]
            for index in range(3)
        )
        if observation.inside_container:
            disposition = DroppedPayloadDisposition.IN_CONTAINER
            next_action = "verify_recovered_placement"
        elif inside_recovery_workspace:
            disposition = DroppedPayloadDisposition.DEDICATED_RECOVERY_REQUIRED
            next_action = "plan_dedicated_dropped_fruit_recovery"
        else:
            disposition = DroppedPayloadDisposition.MANUAL_INTERVENTION_REQUIRED
            next_action = "manual_intervention"
        output = {
            "payload_id": request.payload.payload_id,
            "target_id": request.payload.target_id,
            "container_id": request.container.container_id,
            "disposition": disposition.value,
            "next_action": next_action,
            "motion_authorized": False,
            "inside_recovery_workspace": inside_recovery_workspace,
            "observation_age_s": age_s,
            "observation": observation.as_dict(),
            "frozen_hanging_pick_ppo_reuse_allowed": False,
        }
        trace.record(
            "route",
            SkillTraceEventKind.SUCCEEDED,
            output_summary=output,
            evidence_refs=observation.evidence_sources,
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message=f"dropped payload routed as {disposition.value}",
            output=output,
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
        trace.record("localize", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=status,
            failure_code=code,
            message=message,
            output={} if output is None else output,
        )

    def close(self) -> None:
        if not self._closed_port:
            self._closed_port = True
            self._observation_port.close()
        super().close()


__all__ = ["LocalizeDroppedPayloadSkill"]
