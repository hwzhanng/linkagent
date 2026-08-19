"""Public, lazily loaded backend-neutral DTO contract."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActiveViewRepositionEvidence",
    "ActiveViewRepositionRequest",
    "Action",
    "ArmSide",
    "ArmState",
    "CollisionStatus",
    "CameraHealthSnapshot",
    "CameraIntrinsics",
    "ClearanceScene",
    "CapabilityCheck",
    "CheckState",
    "CommandResult",
    "DroppedPayloadDisposition",
    "DroppedPayloadLocalizationRequest",
    "DroppedPayloadObservationEvidence",
    "EvidenceState",
    "ExecutionHandle",
    "ExecutionState",
    "EmergencyStopHealthSnapshot",
    "FingerTactile",
    "GraspCandidate",
    "HarvestContainer",
    "HarvestCycleState",
    "HarvestCycleVerification",
    "HarvestPayload",
    "HarvestTargetCandidateScore",
    "HarvestTargetNextAction",
    "HarvestTargetSelection",
    "HandState",
    "HandSynergy",
    "MoveToPickReadyRequest",
    "Observation",
    "PickFailureCode",
    "PickPhase",
    "PayloadPlaceRequest",
    "PayloadRetreatRequest",
    "PayloadTransportEvidence",
    "PayloadTransportRequest",
    "PlacementReceipt",
    "PlacementVerificationEvidence",
    "Pose",
    "PregraspClearanceEvidence",
    "PregraspClearanceRequest",
    "Result",
    "ResetForNextTargetRequest",
    "SafetyAssessment",
    "SafeRetractRequest",
    "SelectHarvestTargetRequest",
    "SkillStatus",
    "StateStamp",
    "SynchronizedRgbdFrame",
    "SystemSelfCheckReport",
    "SystemSelfCheckRequest",
    "TactileFrame",
    "TargetReobserveEvidence",
    "TargetReobserveRequest",
    "TomatoDetectionResult",
    "TomatoGraspPlan",
    "TomatoGeometryEvidence",
    "TomatoInstance2D",
    "TomatoPickSkillRequest",
    "TomatoPickSkillResult",
    "TomatoPickVerification",
    "TomatoTarget",
    "TomatoVisionResult",
    "VlmTargetSelection",
    "VisionWorkerHealthSnapshot",
    "VisionInstance",
    "VisionResult",
]


_EXPORTS = {
    "ActiveViewRepositionEvidence": (
        "proto.harvest_cycle",
        "ActiveViewRepositionEvidence",
    ),
    "ActiveViewRepositionRequest": (
        "proto.harvest_cycle",
        "ActiveViewRepositionRequest",
    ),
    "Action": ("proto._base_proto", "Action"),
    "Observation": ("proto._base_proto", "Observation"),
    "Result": ("proto._base_proto", "Result"),
    "ResetForNextTargetRequest": (
        "proto.harvest_cycle",
        "ResetForNextTargetRequest",
    ),
    "ArmSide": ("proto.tomato_pick", "ArmSide"),
    "ArmState": ("proto.tomato_pick", "ArmState"),
    "CollisionStatus": ("proto.tomato_pick", "CollisionStatus"),
    "CameraHealthSnapshot": ("proto.system_self_check", "CameraHealthSnapshot"),
    "CameraIntrinsics": ("proto.tomato_vision", "CameraIntrinsics"),
    "ClearanceScene": ("proto.harvest_cycle", "ClearanceScene"),
    "CapabilityCheck": ("proto.system_self_check", "CapabilityCheck"),
    "CheckState": ("proto.system_self_check", "CheckState"),
    "CommandResult": ("proto.tomato_pick", "CommandResult"),
    "DroppedPayloadDisposition": (
        "proto.harvest_cycle",
        "DroppedPayloadDisposition",
    ),
    "DroppedPayloadLocalizationRequest": (
        "proto.harvest_cycle",
        "DroppedPayloadLocalizationRequest",
    ),
    "DroppedPayloadObservationEvidence": (
        "proto.harvest_cycle",
        "DroppedPayloadObservationEvidence",
    ),
    "EvidenceState": ("proto.tomato_pick", "EvidenceState"),
    "ExecutionHandle": ("proto.tomato_pick", "ExecutionHandle"),
    "ExecutionState": ("proto.tomato_pick", "ExecutionState"),
    "EmergencyStopHealthSnapshot": (
        "proto.system_self_check",
        "EmergencyStopHealthSnapshot",
    ),
    "FingerTactile": ("proto.tomato_pick", "FingerTactile"),
    "GraspCandidate": ("proto.manipulation", "GraspCandidate"),
    "HarvestContainer": ("proto.harvest_cycle", "HarvestContainer"),
    "HarvestCycleState": ("proto.harvest_cycle", "HarvestCycleState"),
    "HarvestCycleVerification": ("proto.harvest_cycle", "HarvestCycleVerification"),
    "HarvestPayload": ("proto.harvest_cycle", "HarvestPayload"),
    "HarvestTargetCandidateScore": (
        "proto.harvest_cycle",
        "HarvestTargetCandidateScore",
    ),
    "HarvestTargetNextAction": ("proto.harvest_cycle", "HarvestTargetNextAction"),
    "HarvestTargetSelection": ("proto.harvest_cycle", "HarvestTargetSelection"),
    "HandState": ("proto.tomato_pick", "HandState"),
    "HandSynergy": ("proto.tomato_pick", "HandSynergy"),
    "MoveToPickReadyRequest": ("proto.harvest_cycle", "MoveToPickReadyRequest"),
    "PickFailureCode": ("proto.tomato_pick", "PickFailureCode"),
    "PickPhase": ("proto.tomato_pick", "PickPhase"),
    "PayloadPlaceRequest": ("proto.harvest_cycle", "PayloadPlaceRequest"),
    "PayloadRetreatRequest": ("proto.harvest_cycle", "PayloadRetreatRequest"),
    "PayloadTransportEvidence": (
        "proto.harvest_cycle",
        "PayloadTransportEvidence",
    ),
    "PayloadTransportRequest": (
        "proto.harvest_cycle",
        "PayloadTransportRequest",
    ),
    "PlacementReceipt": ("proto.harvest_cycle", "PlacementReceipt"),
    "PlacementVerificationEvidence": (
        "proto.harvest_cycle",
        "PlacementVerificationEvidence",
    ),
    "Pose": ("proto.manipulation", "Pose"),
    "PregraspClearanceEvidence": (
        "proto.harvest_cycle",
        "PregraspClearanceEvidence",
    ),
    "PregraspClearanceRequest": (
        "proto.harvest_cycle",
        "PregraspClearanceRequest",
    ),
    "SafetyAssessment": ("proto.tomato_pick", "SafetyAssessment"),
    "SafeRetractRequest": ("proto.harvest_cycle", "SafeRetractRequest"),
    "SelectHarvestTargetRequest": (
        "proto.harvest_cycle",
        "SelectHarvestTargetRequest",
    ),
    "SkillStatus": ("proto.tomato_pick", "SkillStatus"),
    "StateStamp": ("proto.tomato_pick", "StateStamp"),
    "SynchronizedRgbdFrame": ("proto.tomato_vision", "SynchronizedRgbdFrame"),
    "SystemSelfCheckReport": ("proto.system_self_check", "SystemSelfCheckReport"),
    "SystemSelfCheckRequest": ("proto.system_self_check", "SystemSelfCheckRequest"),
    "TactileFrame": ("proto.tomato_pick", "TactileFrame"),
    "TargetReobserveEvidence": ("proto.harvest_cycle", "TargetReobserveEvidence"),
    "TargetReobserveRequest": ("proto.harvest_cycle", "TargetReobserveRequest"),
    "TomatoDetectionResult": ("proto.tomato_pick", "TomatoDetectionResult"),
    "TomatoGraspPlan": ("proto.tomato_pick", "TomatoGraspPlan"),
    "TomatoGeometryEvidence": ("proto.tomato_vision", "TomatoGeometryEvidence"),
    "TomatoInstance2D": ("proto.tomato_vision", "TomatoInstance2D"),
    "TomatoPickSkillRequest": ("proto.tomato_pick", "TomatoPickSkillRequest"),
    "TomatoPickSkillResult": ("proto.tomato_pick", "TomatoPickSkillResult"),
    "TomatoPickVerification": ("proto.tomato_pick", "TomatoPickVerification"),
    "TomatoTarget": ("proto.tomato_pick", "TomatoTarget"),
    "TomatoVisionResult": ("proto.tomato_vision", "TomatoVisionResult"),
    "VlmTargetSelection": ("proto.tomato_vision", "VlmTargetSelection"),
    "VisionWorkerHealthSnapshot": (
        "proto.system_self_check",
        "VisionWorkerHealthSnapshot",
    ),
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
