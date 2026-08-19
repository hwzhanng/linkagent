"""Writable LinkBot arm and G20 hand bundle using the validated botSDK paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from proto import ArmSide

from .adapters import (
    BotSdkG20Gateway,
    BotSdkLinkBotGateway,
    G20HandTactileAdapter,
    LinkBotArmAdapter,
)
from .clock import Clock, SystemClock
from .control import ControlLeaseManager
from .deployment import BimanualDeploymentConfig
from .g20_calibrated import CalibratedG20Mapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LinkBotG20HardwareBundle:
    """One writable same-side arm/hand view with an explicit enable lifecycle."""

    def __init__(
        self,
        *,
        side: ArmSide,
        arm_gateway: BotSdkLinkBotGateway,
        hand_gateway: BotSdkG20Gateway,
        mapper: CalibratedG20Mapper,
        clock: Clock,
        physical_confirmation: str | None = None,
    ) -> None:
        self.side = side if isinstance(side, ArmSide) else ArmSide(side)
        self.clock = clock
        self.leases = ControlLeaseManager(clock)
        self.arm_gateway = arm_gateway
        self.hand_gateway = hand_gateway
        self.mapper = mapper
        self.arm = LinkBotArmAdapter(arm_gateway, self.leases, clock)
        self._hand_tactile = G20HandTactileAdapter(
            hand_gateway,
            self.leases,
            mapper,
            clock,
            side=self.side,
        )
        self.hand = self._hand_tactile
        self.tactile = self._hand_tactile
        self._physical_confirmation = physical_confirmation
        self._arm_enabled = False
        self.lifecycle_state = "configured"

    def open(self, *, enable_arm: bool = False) -> None:
        if self.lifecycle_state == "closed":
            raise RuntimeError("hardware bundle is closed")
        if self.lifecycle_state == "open":
            if enable_arm and not self._arm_enabled:
                self._enable_arm()
            return
        if not isinstance(enable_arm, bool):
            raise TypeError("enable_arm must be a bool")
        self.arm.open(read_only=False)
        try:
            self._hand_tactile.open(read_only=False)
            if enable_arm:
                self._enable_arm()
        except BaseException:
            self.close()
            raise
        self.lifecycle_state = "open"

    def _enable_arm(self) -> None:
        if not isinstance(self._physical_confirmation, str) or not self._physical_confirmation:
            raise ValueError(
                "physical_confirmation must be supplied at runtime to enable the arm"
            )
        self.arm_gateway.set_enabled(
            self.side,
            True,
            self._physical_confirmation,
        )
        self._arm_enabled = True

    def task_component_overrides(self) -> Mapping[str, Mapping[str, Any]]:
        return {}

    def verifier_context(self) -> Mapping[str, Any]:
        return {}

    def close(self) -> None:
        if self.lifecycle_state == "closed":
            return
        self.lifecycle_state = "closed"
        errors: list[BaseException] = []
        if self._arm_enabled:
            try:
                self.arm_gateway.set_enabled(self.side, False)
            except BaseException as exc:
                errors.append(exc)
            self._arm_enabled = False
        for port in (self._hand_tactile, self.arm):
            try:
                port.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("hardware bundle close failed", errors)


def build_linkbot_g20_hardware_bundle(
    *,
    side: ArmSide,
    deployment_config: str,
    motion_calibration: str,
    connect: bool = False,
    enable_arm: bool = False,
    physical_confirmation: str | None = None,
) -> LinkBotG20HardwareBundle:
    """Build one physical-side bundle; construction alone never opens CAN."""

    side = side if isinstance(side, ArmSide) else ArmSide(side)
    if enable_arm and not connect:
        raise ValueError("enable_arm requires connect=true")
    deployment_path = _project_path(deployment_config)
    deployment = BimanualDeploymentConfig.from_mapping(
        json.loads(deployment_path.read_text(encoding="utf-8"))
    )
    arm_endpoint = next(item for item in deployment.arms if item.side is side)
    hand_endpoint = next(item for item in deployment.hands if item.side is side)
    missing: list[str] = []
    if arm_endpoint.can_interface is None:
        missing.append(f"arms.{side.value}.can_interface")
    if not arm_endpoint.validated:
        missing.append(f"arms.{side.value}.validated")
    if hand_endpoint.can_interface is None:
        missing.append(f"hands.{side.value}.can_interface")
    if hand_endpoint.can_id is None:
        missing.append(f"hands.{side.value}.can_id")
    if not hand_endpoint.validated:
        missing.append(f"hands.{side.value}.validated")
    if connect and not deployment.topology_validated:
        missing.append("topology_validated")
    if missing:
        raise RuntimeError(
            f"{side.value} hardware endpoint is incomplete: " + ", ".join(missing)
        )

    clock = SystemClock()
    mapper = CalibratedG20Mapper.from_file(_project_path(motion_calibration))
    endpoint = {
        "can_interface": arm_endpoint.can_interface,
        "profile_path": str(_project_path(arm_endpoint.profile_path)),
        "allow_can0": arm_endpoint.can_interface == "can0",
        "physical_confirmation": physical_confirmation,
    }
    arm_gateway = BotSdkLinkBotGateway(
        endpoints={side: endpoint},
        clock=clock,
        hardware_write_authorized=True,
    )
    hand_gateway = BotSdkG20Gateway(
        side=side,
        can_interface=hand_endpoint.can_interface or "",
        clock=clock,
        hardware_write_authorized=True,
        continuous_speed_raw=204,
        torque_limit_raw=153,
    )
    bundle = LinkBotG20HardwareBundle(
        side=side,
        arm_gateway=arm_gateway,
        hand_gateway=hand_gateway,
        mapper=mapper,
        clock=clock,
        physical_confirmation=physical_confirmation,
    )
    if connect:
        bundle.open(enable_arm=enable_arm)
    return bundle


def _project_path(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("repository path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "LinkBotG20HardwareBundle",
    "build_linkbot_g20_hardware_bundle",
]
