"""Read-only LinkBot/G20 shadow backend for Agent integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from proto import ArmSide

from .adapters import (
    ArmSdkGateway,
    BotSdkG20Gateway,
    BotSdkLinkBotGateway,
    G20HandTactileAdapter,
    G20SdkGateway,
    LinkBotArmAdapter,
)
from .clock import Clock, SystemClock
from .control import ControlBusyError, ControlLeaseManager
from .deployment import BimanualDeploymentConfig
from .mapping import MockHandSynergyMapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReadOnlyControlLeaseManager(ControlLeaseManager):
    """A structural ControlLeaseManager that can never grant write ownership."""

    def acquire(self, *, owner_id: str, side: ArmSide, ttl_s: float):
        del owner_id, side, ttl_s
        raise ControlBusyError("linkbot_g20_shadow is read-only")


class LinkBotG20ShadowBundle:
    """Same-side bundle exposing live state while rejecting every command lease."""

    def __init__(
        self,
        *,
        side: ArmSide,
        arm_gateway: ArmSdkGateway,
        hand_gateway: G20SdkGateway,
        clock: Clock,
        evidence_source: object | None = None,
    ) -> None:
        self.side = side if isinstance(side, ArmSide) else ArmSide(side)
        self.clock = clock
        self.leases = ReadOnlyControlLeaseManager(clock)
        self.arm = LinkBotArmAdapter(arm_gateway, self.leases, clock)
        self._hand_tactile = G20HandTactileAdapter(
            hand_gateway,
            self.leases,
            MockHandSynergyMapper(),
            clock,
            side=self.side,
        )
        self.hand = self._hand_tactile
        self.tactile = self._hand_tactile
        self._evidence_source = evidence_source
        self.lifecycle_state = "configured"

    def open(self) -> None:
        if self.lifecycle_state == "closed":
            raise RuntimeError("shadow bundle is closed")
        if self.lifecycle_state == "open_read_only":
            return
        self.arm.open(read_only=True)
        try:
            self._hand_tactile.open(read_only=True)
        except BaseException:
            self.arm.close()
            self.lifecycle_state = "closed"
            raise
        self.lifecycle_state = "open_read_only"

    def task_component_overrides(self) -> Mapping[str, Mapping[str, Any]]:
        return {}

    def verifier_context(self) -> Mapping[str, Any]:
        if self._evidence_source is None:
            return {}
        return {"pick_evidence_source": self._evidence_source}

    def close(self) -> None:
        if self.lifecycle_state == "closed":
            return
        self.lifecycle_state = "closed"
        errors: list[BaseException] = []
        for port in (self._hand_tactile, self.arm):
            try:
                port.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("shadow bundle close failed", errors)


def build_linkbot_g20_shadow_bundle(
    *,
    side: ArmSide,
    deployment_config: str,
    connect: bool = True,
) -> LinkBotG20ShadowBundle:
    """Build the registered vendor shadow backend; always read-only."""

    path = Path(deployment_config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    deployment = BimanualDeploymentConfig.from_mapping(raw)
    clock = SystemClock()
    arm_endpoints = {
        endpoint.side: {
            "can_interface": endpoint.can_interface,
            "profile_path": str(PROJECT_ROOT / endpoint.profile_path),
            "allow_can0": endpoint.can_interface == "can0",
        }
        for endpoint in deployment.arms
    }
    hand_endpoint = next(item for item in deployment.hands if item.side is side)
    arm_gateway = BotSdkLinkBotGateway(endpoints=arm_endpoints, clock=clock)
    hand_gateway = BotSdkG20Gateway(
        side=side,
        can_interface=hand_endpoint.can_interface or "",
        clock=clock,
    )
    bundle = LinkBotG20ShadowBundle(
        side=side,
        arm_gateway=arm_gateway,
        hand_gateway=hand_gateway,
        clock=clock,
    )
    if connect:
        bundle.open()
    return bundle


__all__ = [
    "LinkBotG20ShadowBundle",
    "ReadOnlyControlLeaseManager",
    "build_linkbot_g20_shadow_bundle",
]
