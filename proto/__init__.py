"""Public, lazily loaded backend-neutral DTO contract."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Action",
    "ArmSide",
    "ArmState",
    "CollisionStatus",
    "CommandResult",
    "EvidenceState",
    "ExecutionHandle",
    "ExecutionState",
    "FingerTactile",
    "GraspCandidate",
    "HandState",
    "HandSynergy",
    "Observation",
    "PickFailureCode",
    "PickPhase",
    "Pose",
    "Result",
    "SafetyAssessment",
    "SkillStatus",
    "StateStamp",
    "TactileFrame",
    "TomatoDetectionResult",
    "TomatoGraspPlan",
    "TomatoPickSkillRequest",
    "TomatoPickSkillResult",
    "TomatoPickVerification",
    "TomatoTarget",
    "VisionInstance",
    "VisionResult",
]


_EXPORTS = {
    "Action": ("proto._base_proto", "Action"),
    "Observation": ("proto._base_proto", "Observation"),
    "Result": ("proto._base_proto", "Result"),
    "ArmSide": ("proto.tomato_pick", "ArmSide"),
    "ArmState": ("proto.tomato_pick", "ArmState"),
    "CollisionStatus": ("proto.tomato_pick", "CollisionStatus"),
    "CommandResult": ("proto.tomato_pick", "CommandResult"),
    "EvidenceState": ("proto.tomato_pick", "EvidenceState"),
    "ExecutionHandle": ("proto.tomato_pick", "ExecutionHandle"),
    "ExecutionState": ("proto.tomato_pick", "ExecutionState"),
    "FingerTactile": ("proto.tomato_pick", "FingerTactile"),
    "GraspCandidate": ("proto.manipulation", "GraspCandidate"),
    "HandState": ("proto.tomato_pick", "HandState"),
    "HandSynergy": ("proto.tomato_pick", "HandSynergy"),
    "PickFailureCode": ("proto.tomato_pick", "PickFailureCode"),
    "PickPhase": ("proto.tomato_pick", "PickPhase"),
    "Pose": ("proto.manipulation", "Pose"),
    "SafetyAssessment": ("proto.tomato_pick", "SafetyAssessment"),
    "SkillStatus": ("proto.tomato_pick", "SkillStatus"),
    "StateStamp": ("proto.tomato_pick", "StateStamp"),
    "TactileFrame": ("proto.tomato_pick", "TactileFrame"),
    "TomatoDetectionResult": ("proto.tomato_pick", "TomatoDetectionResult"),
    "TomatoGraspPlan": ("proto.tomato_pick", "TomatoGraspPlan"),
    "TomatoPickSkillRequest": ("proto.tomato_pick", "TomatoPickSkillRequest"),
    "TomatoPickSkillResult": ("proto.tomato_pick", "TomatoPickSkillResult"),
    "TomatoPickVerification": ("proto.tomato_pick", "TomatoPickVerification"),
    "TomatoTarget": ("proto.tomato_pick", "TomatoTarget"),
    "VisionInstance": ("proto.perception", "VisionInstance"),
    "VisionResult": ("proto.perception", "VisionResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
