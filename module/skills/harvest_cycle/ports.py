"""Replaceable perception and verification ports for the harvest-cycle skills."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import (
    ClearanceScene,
    DroppedPayloadLocalizationRequest,
    DroppedPayloadObservationEvidence,
    HarvestContainer,
    HarvestPayload,
    PayloadPlaceRequest,
    PlacementReceipt,
    PlacementVerificationEvidence,
    PregraspClearanceRequest,
    TargetReobserveEvidence,
    TargetReobserveRequest,
)


@runtime_checkable
class ClearanceScenePort(Protocol):
    """Capture a calibrated world-frame scene sample for one clearance request."""

    def capture(self, request: PregraspClearanceRequest) -> ClearanceScene: ...
    def close(self) -> None: ...


@runtime_checkable
class TargetObservationPort(Protocol):
    """Active-view seam for a future calibrated RGB-D camera implementation."""

    def reobserve(self, request: TargetReobserveRequest) -> TargetReobserveEvidence: ...
    def close(self) -> None: ...


@runtime_checkable
class PlacementVerifierPort(Protocol):
    """Independent evidence seam; camera/tactile implementations can replace the mock."""

    def verify(
        self,
        receipt: PlacementReceipt,
        payload: HarvestPayload,
        container: HarvestContainer,
    ) -> PlacementVerificationEvidence: ...

    def close(self) -> None: ...


@runtime_checkable
class DroppedPayloadObservationPort(Protocol):
    """Observe a lost fruit without granting any recovery motion authority."""

    def observe(
        self, request: DroppedPayloadLocalizationRequest
    ) -> DroppedPayloadObservationEvidence: ...

    def close(self) -> None: ...


__all__ = [
    "ClearanceScenePort",
    "DroppedPayloadObservationPort",
    "PlacementVerifierPort",
    "TargetObservationPort",
]
