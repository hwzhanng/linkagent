"""Reusable lifecycle, idempotency, and trace helpers for atomic skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from module.robotics.clock import Clock, SystemClock

from .contracts import (
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEvent,
    SkillTraceEventKind,
)


class SkillTraceRecorder:
    def __init__(self, *, clock: Clock, execution_id: str, skill_id: str) -> None:
        self._clock = clock
        self.execution_id = execution_id
        self.skill_id = skill_id
        self._started_ns = clock.monotonic_ns()
        self._events: list[SkillTraceEvent] = []

    @property
    def duration_s(self) -> float:
        return max(0.0, (self._clock.monotonic_ns() - self._started_ns) / 1_000_000_000.0)

    @property
    def events(self) -> tuple[SkillTraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        phase: str,
        kind: SkillTraceEventKind,
        *,
        message: str = "",
        input_summary: Mapping[str, object] | None = None,
        output_summary: Mapping[str, object] | None = None,
        evidence_refs: tuple[str, ...] = (),
    ) -> None:
        now_ns = self._clock.monotonic_ns()
        self._events.append(
            SkillTraceEvent(
                sequence=len(self._events),
                phase=phase,
                kind=kind,
                monotonic_ns=now_ns,
                elapsed_ms=max(0.0, (now_ns - self._started_ns) / 1_000_000.0),
                message=message,
                input_summary={} if input_summary is None else input_summary,
                output_summary={} if output_summary is None else output_summary,
                evidence_refs=evidence_refs,
            )
        )


class AtomicSkillBase(ABC):
    """Small base class; implementations retain control of their domain logic."""

    descriptor: AtomicSkillDescriptor

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = SystemClock() if clock is None else clock
        self._terminal_results: dict[str, AtomicSkillResult] = {}
        self._closed = False

    def _execution_id(self, request: object) -> str:
        execution_id = getattr(request, "execution_id", None)
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("request.execution_id must be a non-empty string")
        return execution_id.strip()

    def _cached(self, execution_id: str) -> AtomicSkillResult | None:
        return self._terminal_results.get(execution_id)

    def _recorder(self, execution_id: str) -> SkillTraceRecorder:
        if self._closed:
            raise RuntimeError(f"atomic skill {self.descriptor.skill_id!r} is closed")
        return SkillTraceRecorder(
            clock=self._clock,
            execution_id=execution_id,
            skill_id=self.descriptor.skill_id,
        )

    def _result(
        self,
        recorder: SkillTraceRecorder,
        *,
        status: AtomicSkillStatus,
        failure_code: AtomicSkillFailureCode | None = None,
        message: str = "",
        output: Mapping[str, object] | None = None,
    ) -> AtomicSkillResult:
        result = AtomicSkillResult(
            execution_id=recorder.execution_id,
            skill_id=self.descriptor.skill_id,
            status=status,
            success=status is AtomicSkillStatus.SUCCEEDED,
            duration_s=recorder.duration_s,
            failure_code=failure_code,
            message=message,
            output={} if output is None else output,
            trace=recorder.events,
        )
        self._terminal_results[result.execution_id] = result
        return result

    @abstractmethod
    def run(self, request: object) -> AtomicSkillResult:
        raise NotImplementedError

    def cancel(self, execution_id: str, *, reason: str = "cancelled") -> AtomicSkillResult:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        execution_id = execution_id.strip()
        cached = self._cached(execution_id)
        if cached is not None:
            return cached
        recorder = self._recorder(execution_id)
        recorder.record(
            "cancel",
            SkillTraceEventKind.CANCELLED,
            message=reason,
        )
        return self._result(
            recorder,
            status=AtomicSkillStatus.CANCELLED,
            failure_code=AtomicSkillFailureCode.CANCELLED,
            message=reason,
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["AtomicSkillBase", "SkillTraceRecorder"]
