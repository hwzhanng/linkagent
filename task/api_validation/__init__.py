"""Offline API-to-skill validation Task."""

from task.api_validation.task import HardwareCapabilityValidationTask
from task.api_validation.live import HardwareReadOnlyValidationTask

__all__ = ["HardwareCapabilityValidationTask", "HardwareReadOnlyValidationTask"]
