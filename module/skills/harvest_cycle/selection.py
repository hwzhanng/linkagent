"""Deterministic, motion-free harvest target selection atom."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

from module.robotics import Clock
from proto import (
    ArmSide,
    HarvestTargetCandidateScore,
    HarvestTargetNextAction,
    HarvestTargetSelection,
    SelectHarvestTargetRequest,
    TomatoTarget,
)

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
)


def _probability(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return float(value)


def _positive(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and > 0")
    return float(value)


def _vector3(value: object, *, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a three-number sequence")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a three-number sequence") from exc
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    prepared: list[float] = []
    for index, item in enumerate(values):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{field_name}[{index}] must be finite")
        prepared.append(float(item))
    return prepared[0], prepared[1], prepared[2]


class SelectHarvestTargetSkill(AtomicSkillBase):
    """Choose a candidate for re-observation without authorizing motion."""

    descriptor = AtomicSkillDescriptor(
        skill_id="select_harvest_target",
        version="1.0.0",
        description=(
            "Apply deterministic freshness, harvestability, size, visibility, and "
            "configured workspace gates before ranking a tomato for re-observation."
        ),
        effectful=False,
        required_resources=("tomato_detection", "reachability_profile"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        clock: Clock,
        reachability_profile_id: str,
        reachability_arm_side: ArmSide | str,
        workspace_min_m: Sequence[float] | None,
        workspace_max_m: Sequence[float] | None,
        min_ripeness_score: float = 0.70,
        min_pick_confidence: float = 0.60,
        max_occlusion_ratio: float = 0.40,
        min_axis_m: float = 0.015,
        max_axis_m: float = 0.060,
        reachability_margin_normalization_m: float = 0.15,
        ripeness_weight: float = 0.30,
        confidence_weight: float = 0.25,
        visibility_weight: float = 0.25,
        reachability_weight: float = 0.20,
        max_candidates: int = 20,
    ) -> None:
        super().__init__(clock=clock)
        if not isinstance(reachability_profile_id, str) or not reachability_profile_id.strip():
            raise ValueError("reachability_profile_id must be a non-empty string")
        try:
            profile_arm_side = (
                reachability_arm_side
                if isinstance(reachability_arm_side, ArmSide)
                else ArmSide(reachability_arm_side)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("reachability_arm_side must be left or right") from exc
        if (workspace_min_m is None) != (workspace_max_m is None):
            raise ValueError("workspace bounds must both be provided or both be None")
        minimum = None if workspace_min_m is None else _vector3(
            workspace_min_m, field_name="workspace_min_m"
        )
        maximum = None if workspace_max_m is None else _vector3(
            workspace_max_m, field_name="workspace_max_m"
        )
        if minimum is not None and maximum is not None and any(
            low >= high for low, high in zip(minimum, maximum, strict=True)
        ):
            raise ValueError("workspace_min_m must be strictly below workspace_max_m")
        min_axis = _positive(min_axis_m, field_name="min_axis_m")
        max_axis = _positive(max_axis_m, field_name="max_axis_m")
        if min_axis >= max_axis:
            raise ValueError("min_axis_m must be less than max_axis_m")
        weights = tuple(
            _probability(value, field_name=name)
            for name, value in {
                "ripeness_weight": ripeness_weight,
                "confidence_weight": confidence_weight,
                "visibility_weight": visibility_weight,
                "reachability_weight": reachability_weight,
            }.items()
        )
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("at least one selection weight must be > 0")
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 100:
            raise ValueError("max_candidates must be an integer in [1, 100]")
        self._profile_id = reachability_profile_id.strip()
        self._profile_arm_side = profile_arm_side
        self._workspace_min = minimum
        self._workspace_max = maximum
        self._min_ripeness = _probability(
            min_ripeness_score, field_name="min_ripeness_score"
        )
        self._min_confidence = _probability(
            min_pick_confidence, field_name="min_pick_confidence"
        )
        self._max_occlusion = _probability(
            max_occlusion_ratio, field_name="max_occlusion_ratio"
        )
        self._min_axis = min_axis
        self._max_axis = max_axis
        self._margin_norm = _positive(
            reachability_margin_normalization_m,
            field_name="reachability_margin_normalization_m",
        )
        self._weights = tuple(value / weight_sum for value in weights)
        self._max_candidates = max_candidates

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, SelectHarvestTargetRequest):
            raise TypeError("request must be a SelectHarvestTargetRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        detection = request.detection
        trace.record(
            "candidate_set",
            SkillTraceEventKind.STARTED,
            input_summary={
                "observation_sequence": detection.observation_sequence,
                "sample_monotonic_ns": detection.sample_monotonic_ns,
                "candidate_count": len(detection.targets),
                "model": detection.model,
                "arm_side": request.arm_side.value,
                "reachability_profile_id": self._profile_id,
                "reachability_arm_side": self._profile_arm_side.value,
            },
            evidence_refs=(f"perception:{detection.model}",),
        )
        if request.arm_side is not self._profile_arm_side:
            message = (
                f"reachability profile {self._profile_id!r} is bound to "
                f"{self._profile_arm_side.value} arm, not {request.arm_side.value} arm"
            )
            trace.record(
                "reachability_profile",
                SkillTraceEventKind.FAILED,
                message=message,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.INVALID_REQUEST,
                message=message,
            )
        if len(detection.targets) > self._max_candidates:
            message = (
                f"candidate count {len(detection.targets)} exceeds limit "
                f"{self._max_candidates}"
            )
            trace.record("candidate_set", SkillTraceEventKind.FAILED, message=message)
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.INVALID_REQUEST,
                message=message,
            )
        now_ns = self._clock.monotonic_ns()
        detection_age_ns = now_ns - detection.sample_monotonic_ns
        if not 0 <= detection_age_ns <= request.max_detection_age_s * 1.0e9:
            selection = self._selection(
                request,
                candidates=(),
                selected_target_id=None,
                action=HarvestTargetNextAction.NO_HARVESTABLE_TARGET,
            )
            trace.record(
                "freshness_gate",
                SkillTraceEventKind.FAILED,
                message="tomato candidate set is stale or from a different clock domain",
                output_summary={"detection_age_s": detection_age_ns / 1.0e9},
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.RETRYABLE_FAILURE,
                failure_code=AtomicSkillFailureCode.CANDIDATE_SET_STALE,
                message="tomato candidate set is stale or from a different clock domain",
                output=selection.as_dict(),
            )

        scored = tuple(
            self._score_candidate(target, now_ns, request.max_detection_age_s)
            for target in detection.targets
        )
        ranked = tuple(sorted(scored, key=self._rank_key))
        for candidate in ranked:
            trace.record(
                "candidate_gate",
                SkillTraceEventKind.CHECKPOINT,
                input_summary={"target_id": candidate.target_id},
                output_summary={
                    "eligible": candidate.eligible,
                    "score": candidate.score,
                    "hard_gates": candidate.hard_gates,
                    "rejection_reasons": candidate.rejection_reasons,
                },
                evidence_refs=candidate.evidence_sources,
            )
        selected = next((item for item in ranked if item.eligible), None)
        if selected is not None:
            selection = self._selection(
                request,
                candidates=ranked,
                selected_target_id=selected.target_id,
                action=HarvestTargetNextAction.REOBSERVE_SELECTED,
            )
            trace.record(
                "select",
                SkillTraceEventKind.SUCCEEDED,
                message="selected target for pinned re-observation",
                output_summary={
                    "selected_target_id": selected.target_id,
                    "selection_revision": selection.selection_revision,
                    "score": selected.score,
                    "motion_authorized": False,
                },
                evidence_refs=selected.evidence_sources,
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                output=selection.as_dict(),
            )

        action, failure_code, message = self._failure(ranked)
        selection = self._selection(
            request,
            candidates=ranked,
            selected_target_id=None,
            action=action,
        )
        trace.record(
            "select",
            SkillTraceEventKind.FAILED,
            message=message,
            output_summary={
                "next_action": action.value,
                "selection_revision": selection.selection_revision,
            },
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.RETRYABLE_FAILURE,
            failure_code=failure_code,
            message=message,
            output=selection.as_dict(),
        )

    def _score_candidate(
        self,
        target: TomatoTarget,
        now_ns: int,
        max_age_s: float,
    ) -> HarvestTargetCandidateScore:
        target_age_ns = now_ns - target.sample_monotonic_ns
        fresh = 0 <= target_age_ns <= max_age_s * 1.0e9
        ripe = target.ripeness_score >= self._min_ripeness
        confident = target.pick_confidence >= self._min_confidence
        visible = target.occlusion <= self._max_occlusion
        size_supported = all(
            self._min_axis <= axis <= self._max_axis for axis in target.semi_axes_m
        )
        margin = self._workspace_margin(target)
        reachable = None if margin is None else margin >= 0.0
        hard_gates = {
            "fresh": fresh,
            "ripeness": ripe,
            "pick_confidence": confident,
            "visibility": visible,
            "fruit_size": size_supported,
            "workspace_reachable": reachable,
        }
        reasons: list[str] = []
        for passed, reason in (
            (fresh, "target_stale"),
            (ripe, "ripeness_below_minimum"),
            (confident, "pick_confidence_below_minimum"),
            (visible, "occlusion_above_maximum"),
            (size_supported, "fruit_size_out_of_range"),
        ):
            if not passed:
                reasons.append(reason)
        if reachable is None:
            reasons.append("reachability_unknown")
        elif not reachable:
            reasons.append("outside_reachability_profile")
        visibility = 1.0 - target.occlusion
        margin_component = (
            0.0
            if margin is None
            else min(1.0, max(0.0, margin / self._margin_norm))
        )
        components = {
            "ripeness": target.ripeness_score,
            "pick_confidence": target.pick_confidence,
            "visibility": visibility,
            "reachability_margin": margin_component,
        }
        score = sum(
            weight * component
            for weight, component in zip(
                self._weights, components.values(), strict=True
            )
        )
        evidence = (
            f"perception-target:{target.target_id}",
            f"reachability-profile:{self._profile_id}",
        )
        return HarvestTargetCandidateScore(
            target_id=target.target_id,
            eligible=not reasons,
            score=min(1.0, max(0.0, score)),
            hard_gates=hard_gates,
            score_components=components,
            rejection_reasons=tuple(reasons),
            workspace_margin_m=margin,
            evidence_sources=evidence,
        )

    def _workspace_margin(self, target: TomatoTarget) -> float | None:
        if self._workspace_min is None or self._workspace_max is None:
            return None
        coordinates = target.center_pose.position_m
        return min(
            min(value - low, high - value)
            for value, low, high in zip(
                coordinates,
                self._workspace_min,
                self._workspace_max,
                strict=True,
            )
        )

    @staticmethod
    def _rank_key(candidate: HarvestTargetCandidateScore) -> tuple[object, ...]:
        margin = (
            float("-inf")
            if candidate.workspace_margin_m is None
            else candidate.workspace_margin_m
        )
        return (
            not candidate.eligible,
            -candidate.score,
            -margin,
            candidate.target_id,
        )

    @staticmethod
    def _failure(
        candidates: tuple[HarvestTargetCandidateScore, ...],
    ) -> tuple[HarvestTargetNextAction, AtomicSkillFailureCode, str]:
        if not candidates:
            return (
                HarvestTargetNextAction.NO_HARVESTABLE_TARGET,
                AtomicSkillFailureCode.NO_HARVESTABLE_TARGET,
                "no tomato candidates were detected",
            )
        reason_sets = tuple(set(item.rejection_reasons) for item in candidates)
        if all("reachability_unknown" in reasons for reasons in reason_sets):
            return (
                HarvestTargetNextAction.REACHABILITY_CONTEXT_REQUIRED,
                AtomicSkillFailureCode.REACHABILITY_UNKNOWN,
                "candidate reachability is unknown; a validated profile is required",
            )
        if any(reasons == {"occlusion_above_maximum"} for reasons in reason_sets):
            return (
                HarvestTargetNextAction.ACTIVE_VIEW_REQUIRED,
                AtomicSkillFailureCode.VIEWPOINT_INSUFFICIENT,
                "no current candidate passed visibility gates; active view is required",
            )
        if all("target_stale" in reasons for reasons in reason_sets):
            return (
                HarvestTargetNextAction.NO_HARVESTABLE_TARGET,
                AtomicSkillFailureCode.CANDIDATE_SET_STALE,
                "all tomato candidates are stale",
            )
        return (
            HarvestTargetNextAction.NO_HARVESTABLE_TARGET,
            AtomicSkillFailureCode.NO_HARVESTABLE_TARGET,
            "no tomato candidate passed all harvestability gates",
        )

    def _selection(
        self,
        request: SelectHarvestTargetRequest,
        *,
        candidates: tuple[HarvestTargetCandidateScore, ...],
        selected_target_id: str | None,
        action: HarvestTargetNextAction,
    ) -> HarvestTargetSelection:
        detection = request.detection
        canonical = {
            "policy_version": self.descriptor.version,
            "profile_id": self._profile_id,
            "profile_arm_side": self._profile_arm_side.value,
            "observation_sequence": detection.observation_sequence,
            "sample_monotonic_ns": detection.sample_monotonic_ns,
            "model": detection.model,
            "arm_side": request.arm_side.value,
            "candidates": [item.as_dict() for item in candidates],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return HarvestTargetSelection(
            selection_revision=f"selection-{digest[:24]}",
            observation_sequence=detection.observation_sequence,
            sample_monotonic_ns=detection.sample_monotonic_ns,
            selected_target_id=selected_target_id,
            next_action=action,
            ranked_candidates=candidates,
            motion_authorized=False,
            requires_pregrasp_clearance_check=True,
        )


__all__ = ["SelectHarvestTargetSkill"]
