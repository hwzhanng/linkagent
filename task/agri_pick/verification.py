"""Policy-independent tomato-pick evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from module.skills.tomato_pick import VerifierSession
from proto import (
    ArmState,
    EvidenceState,
    HandState,
    TactileFrame,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoPickVerification,
    TomatoTarget,
)


def _finite(value: object, *, field_name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field_name} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PickEvidenceSnapshot:
    """One observation from Isaac, a recorder, or a future real verifier."""

    sample_monotonic_ns: int
    source_sequence: int
    detached: bool | None
    dropped: bool | None
    retained: bool | None
    source_id: str
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("sample_monotonic_ns", "source_sequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        for name in ("detached", "dropped", "retained"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or null")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        object.__setattr__(self, "source_id", self.source_id.strip())
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("reason must be null or a non-empty string")


@runtime_checkable
class PickEvidenceSource(Protocol):
    """Evidence boundary that cannot consume policy output or PPO reward state."""

    @property
    def source_id(self) -> str: ...

    def begin(self, execution_id: str, target_id: str) -> None: ...

    def sample(self, execution_id: str) -> PickEvidenceSnapshot: ...

    def finish(self, execution_id: str) -> None: ...


class InMemoryPickEvidenceSource:
    """Deterministic source used by Isaac bridges and offline tests."""

    def __init__(
        self,
        samples: tuple[PickEvidenceSnapshot, ...] | list[PickEvidenceSnapshot],
        *,
        source_id: str = "in_memory_pick_evidence_v1",
    ) -> None:
        prepared = tuple(samples)
        if not prepared or not all(isinstance(item, PickEvidenceSnapshot) for item in prepared):
            raise TypeError("samples must contain at least one PickEvidenceSnapshot")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be non-empty")
        self._samples = prepared
        self._source_id = source_id.strip()
        self._indices: dict[str, int] = {}

    @property
    def source_id(self) -> str:
        return self._source_id

    def begin(self, execution_id: str, target_id: str) -> None:
        del target_id
        self._indices[execution_id] = 0

    def sample(self, execution_id: str) -> PickEvidenceSnapshot:
        try:
            index = self._indices[execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence execution {execution_id!r}") from exc
        result = self._samples[min(index, len(self._samples) - 1)]
        self._indices[execution_id] = index + 1
        return result

    def finish(self, execution_id: str) -> None:
        self._indices.pop(execution_id, None)


class RecordedPickEvidenceSource(InMemoryPickEvidenceSource):
    """Load immutable evidence captured from Isaac or a recorder."""

    def __init__(self, path: str | Path) -> None:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("recorded pick evidence must use schema_version 1")
        records = payload.get("samples")
        if not isinstance(records, list) or not records:
            raise ValueError("recorded pick evidence requires non-empty samples")
        super().__init__(
            tuple(PickEvidenceSnapshot(**record) for record in records),
            source_id=str(payload.get("source_id", f"recorded:{source.name}")),
        )
        self.path = source


@dataclass(frozen=True, slots=True)
class PickVerifierThresholds:
    """Versioned thresholds; hardware release requires a validated profile."""

    profile_id: str
    tactile_calibration_id: str
    stable_hold_s: float = 1.0
    minimum_contact_fingers: int = 2
    minimum_normal_force_n: float = 0.60
    maximum_normal_force_n: float = 12.0
    maximum_slip_ratio: float = 0.75
    hardware_validated: bool = False

    def __post_init__(self) -> None:
        for name in ("profile_id", "tactile_calibration_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "stable_hold_s", _finite(self.stable_hold_s, field_name="stable_hold_s"))
        if (
            isinstance(self.minimum_contact_fingers, bool)
            or not isinstance(self.minimum_contact_fingers, int)
            or not 1 <= self.minimum_contact_fingers <= 5
        ):
            raise ValueError("minimum_contact_fingers must be in 1..5")
        for name in ("minimum_normal_force_n", "maximum_normal_force_n", "maximum_slip_ratio"):
            object.__setattr__(self, name, _finite(getattr(self, name), field_name=name))
        if self.maximum_normal_force_n <= self.minimum_normal_force_n:
            raise ValueError("maximum_normal_force_n must exceed minimum_normal_force_n")
        if not isinstance(self.hardware_validated, bool):
            raise TypeError("hardware_validated must be bool")

    def require_hardware_validated(self) -> None:
        if not self.hardware_validated:
            raise RuntimeError(
                f"verifier threshold profile {self.profile_id!r} lacks real-data validation"
            )


@dataclass(slots=True)
class _VerifierState:
    last: TomatoPickVerification
    stable_since_ns: int | None = None


class IndependentPickVerifier:
    """Fuse explicit detachment/drop evidence with calibrated tactile stability."""

    VERSION = "independent_pick_verifier_v1"

    def __init__(self, source: PickEvidenceSource, thresholds: PickVerifierThresholds) -> None:
        if not isinstance(source, PickEvidenceSource):
            raise TypeError("source must implement PickEvidenceSource")
        if not isinstance(thresholds, PickVerifierThresholds):
            raise TypeError("thresholds must be PickVerifierThresholds")
        self.source = source
        self.thresholds = thresholds
        self._sessions: dict[str, _VerifierState] = {}

    def begin(self, execution_id: str, target: TomatoTarget, plan: TomatoGraspPlan) -> VerifierSession:
        if target.target_id != plan.target_id:
            raise ValueError("verifier target and plan target differ")
        session = VerifierSession(execution_id)
        self.source.begin(execution_id, target.target_id)
        self._sessions[execution_id] = _VerifierState(
            self._result(
                execution_id,
                EvidenceState.UNKNOWN,
                EvidenceState.UNKNOWN,
                False,
                0.0,
                True,
                "awaiting independent evidence",
            )
        )
        return session

    def update(
        self,
        session: VerifierSession,
        perception: TomatoDetectionResult | None,
        arm: ArmState,
        hand: HandState,
        tactile: TactileFrame,
    ) -> TomatoPickVerification:
        del perception, arm, hand
        try:
            state = self._sessions[session.execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown verifier session {session.execution_id!r}") from exc
        sample = self.source.sample(session.execution_id)
        if not sample.valid:
            state.last = self._result(
                session.execution_id,
                EvidenceState.UNKNOWN,
                EvidenceState.UNKNOWN,
                bool(sample.dropped),
                0.0,
                False,
                sample.reason or "independent evidence invalid",
            )
            return state.last
        if tactile.calibration_id != self.thresholds.tactile_calibration_id:
            state.last = self._result(
                session.execution_id,
                EvidenceState.UNKNOWN,
                EvidenceState.UNKNOWN,
                bool(sample.dropped),
                0.0,
                False,
                "tactile calibration mismatch",
            )
            return state.last
        if not all(
            finger.force_valid
            and finger.normal_force_n is not None
            and finger.tangential_force_n is not None
            for finger in tactile.fingers
        ):
            state.last = self._result(
                session.execution_id,
                EvidenceState.CONFIRMED if sample.detached else EvidenceState.UNKNOWN,
                EvidenceState.UNKNOWN,
                bool(sample.dropped),
                0.0,
                False,
                "tactile force evidence missing or invalid",
            )
            return state.last

        normals = tuple(float(finger.normal_force_n) for finger in tactile.fingers)  # type: ignore[arg-type]
        tangents = tuple(float(finger.tangential_force_n) for finger in tactile.fingers)  # type: ignore[arg-type]
        contacts = sum(value >= self.thresholds.minimum_normal_force_n for value in normals)
        excess = max(normals, default=0.0) > self.thresholds.maximum_normal_force_n
        slip = any(
            tangent / max(normal, 1.0e-6) > self.thresholds.maximum_slip_ratio
            for normal, tangent in zip(normals, tangents)
            if normal >= self.thresholds.minimum_normal_force_n
        )
        dropped = bool(sample.dropped)
        stable = (
            sample.detached is True
            and sample.retained is not False
            and not dropped
            and not excess
            and not slip
            and contacts >= self.thresholds.minimum_contact_fingers
        )
        if stable:
            if state.stable_since_ns is None:
                state.stable_since_ns = sample.sample_monotonic_ns
            stable_ns = max(0, sample.sample_monotonic_ns - state.stable_since_ns)
        else:
            state.stable_since_ns = None
            stable_ns = 0
        stable_s = stable_ns / 1_000_000_000.0
        detached = EvidenceState.CONFIRMED if sample.detached else EvidenceState.UNKNOWN
        if dropped or excess or slip or sample.retained is False:
            retained = EvidenceState.REJECTED
        elif stable_s >= self.thresholds.stable_hold_s:
            retained = EvidenceState.CONFIRMED
        else:
            retained = EvidenceState.UNKNOWN
        if dropped:
            reason = "independent evidence confirms drop"
        elif excess:
            reason = "calibrated tactile force exceeded limit"
        elif slip:
            reason = "calibrated force ratio indicates slip"
        elif sample.detached is not True:
            reason = "detachment not independently confirmed"
        elif contacts < self.thresholds.minimum_contact_fingers:
            reason = "insufficient calibrated contact support"
        elif retained is EvidenceState.UNKNOWN:
            reason = "stable retention window incomplete"
        else:
            reason = None
        state.last = self._result(
            session.execution_id,
            detached,
            retained,
            dropped,
            stable_s,
            True,
            reason,
        )
        return state.last

    def finish(self, session: VerifierSession) -> TomatoPickVerification:
        try:
            state = self._sessions.pop(session.execution_id)
        except KeyError as exc:
            raise KeyError(f"unknown verifier session {session.execution_id!r}") from exc
        self.source.finish(session.execution_id)
        return state.last

    def _result(
        self,
        execution_id: str,
        detached: EvidenceState,
        retained: EvidenceState,
        dropped: bool,
        stable_window_s: float,
        valid: bool,
        reason: str | None,
    ) -> TomatoPickVerification:
        return TomatoPickVerification(
            execution_id=execution_id,
            detached=detached,
            retained_without_slip=retained,
            dropped=dropped,
            stable_window_s=stable_window_s,
            evidence_sources=(self.source.source_id, self.thresholds.tactile_calibration_id),
            verifier_version=f"{self.VERSION}:{self.thresholds.profile_id}",
            valid=valid,
            reason=reason,
        )


__all__ = [
    "IndependentPickVerifier",
    "InMemoryPickEvidenceSource",
    "PickEvidenceSnapshot",
    "PickEvidenceSource",
    "PickVerifierThresholds",
    "RecordedPickEvidenceSource",
]
