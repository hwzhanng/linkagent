"""Agent capability validation Tasks with explicit hardware admission."""

from task.api_validation.live import HardwareReadOnlyValidationTask
from task.api_validation.task import HardwareCapabilityValidationTask

__all__ = ["HardwareCapabilityValidationTask", "HardwareReadOnlyValidationTask"]
