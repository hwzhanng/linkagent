"""Read-only harvest admission skill and health adapters."""

from .camera import MockCameraHealthPort, OrbbecViewerCameraHealthPort
from .emergency_stop import MockEmergencyStopHealthPort
from .factory import (
    CAMERA_HEALTH_REGISTRY,
    EMERGENCY_STOP_HEALTH_REGISTRY,
    VISION_WORKER_HEALTH_REGISTRY,
    build_camera_health_port,
    build_emergency_stop_health_port,
    build_vision_worker_health_port,
    validate_health_port_config,
)
from .ports import CameraHealthPort, EmergencyStopHealthPort, VisionWorkerHealthPort
from .skill import SystemSelfCheckSkill
from .vision_worker import LoopbackVisionWorkerHealthPort, MockVisionWorkerHealthPort

__all__ = [
    "CAMERA_HEALTH_REGISTRY",
    "CameraHealthPort",
    "EMERGENCY_STOP_HEALTH_REGISTRY",
    "EmergencyStopHealthPort",
    "LoopbackVisionWorkerHealthPort",
    "MockCameraHealthPort",
    "MockEmergencyStopHealthPort",
    "MockVisionWorkerHealthPort",
    "OrbbecViewerCameraHealthPort",
    "SystemSelfCheckSkill",
    "VISION_WORKER_HEALTH_REGISTRY",
    "VisionWorkerHealthPort",
    "build_camera_health_port",
    "build_emergency_stop_health_port",
    "build_vision_worker_health_port",
    "validate_health_port_config",
]
