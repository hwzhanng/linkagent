"""Dual-arm/dual-hand lifecycle, snapshots, leases, and whole-system stop."""

from __future__ import annotations

from dataclasses import dataclass

from proto import ArmSide, ArmState, CommandResult, HandState, TactileFrame

from .adapters import G20HandTactileAdapter, LinkBotArmAdapter
from .clock import Clock
from .control import ControlLease, ControlLeaseManager


@dataclass(frozen=True, slots=True)
class BimanualRobotSnapshot:
    left_arm: ArmState
    right_arm: ArmState
    left_hand: HandState
    right_hand: HandState
    left_tactile: TactileFrame
    right_tactile: TactileFrame
    captured_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class EmergencyStopReport:
    arm_results: tuple[CommandResult, CommandResult]
    hand_results: tuple[CommandResult, CommandResult]


class BimanualRobotInfra:
    """Composition-owned real-adapter shell; no hardware access before open()."""

    def __init__(
        self,
        *,
        arm: LinkBotArmAdapter,
        left_hand_tactile: G20HandTactileAdapter,
        right_hand_tactile: G20HandTactileAdapter,
        leases: ControlLeaseManager,
        clock: Clock,
    ) -> None:
        self.arm = arm
        self.left_hand = left_hand_tactile
        self.right_hand = right_hand_tactile
        self.leases = leases
        self.clock = clock
        self._opened = False
        self._closed = False
        self._read_only = True

    def open(self, *, read_only: bool) -> None:
        if self._closed:
            raise RuntimeError("BimanualRobotInfra is closed")
        if self._opened:
            if self._read_only != read_only:
                raise RuntimeError("robot infra cannot change mode after open")
            return
        attempted: list[object] = []
        try:
            for component in (self.arm, self.left_hand, self.right_hand):
                attempted.append(component)
                component.open(read_only=read_only)
        except BaseException as primary:
            self._closed = True
            cleanup_errors: list[BaseException] = []
            for component in reversed(attempted):
                try:
                    component.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "bimanual robot open and rollback failed",
                    [primary, *cleanup_errors],
                )
            raise
        self._opened = True
        self._read_only = read_only

    def acquire(self, *, owner_id: str, side: ArmSide, ttl_s: float) -> ControlLease:
        if not self._opened or self._closed:
            raise RuntimeError("BimanualRobotInfra must be open")
        if self._read_only:
            raise RuntimeError("read-only robot infra cannot grant control leases")
        return self.leases.acquire(owner_id=owner_id, side=side, ttl_s=ttl_s)

    def hand_for(self, side: ArmSide) -> G20HandTactileAdapter:
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        return self.left_hand if side is ArmSide.LEFT else self.right_hand

    def tactile_for(self, side: ArmSide) -> G20HandTactileAdapter:
        return self.hand_for(side)

    def snapshot(self) -> BimanualRobotSnapshot:
        if not self._opened or self._closed:
            raise RuntimeError("BimanualRobotInfra must be open")
        return BimanualRobotSnapshot(
            left_arm=self.arm.get_state(ArmSide.LEFT),
            right_arm=self.arm.get_state(ArmSide.RIGHT),
            left_hand=self.left_hand.get_state(ArmSide.LEFT),
            right_hand=self.right_hand.get_state(ArmSide.RIGHT),
            left_tactile=self.left_hand.get_frame(ArmSide.LEFT),
            right_tactile=self.right_hand.get_frame(ArmSide.RIGHT),
            captured_monotonic_ns=self.clock.monotonic_ns(),
        )

    def emergency_stop_all(self, reason: str) -> EmergencyStopReport:
        if not self._opened or self._closed:
            raise RuntimeError("BimanualRobotInfra must be open")
        if self._read_only:
            raise RuntimeError("read-only robot infra cannot send emergency-stop commands")
        arm_results: list[CommandResult] = []
        hand_results: list[CommandResult] = []
        errors: list[BaseException] = []
        for side in (ArmSide.LEFT, ArmSide.RIGHT):
            try:
                arm_results.append(self.arm.emergency_stop(side, reason))
            except BaseException as exc:
                errors.append(RuntimeError(
                    f"{side.value} arm emergency stop failed: {exc}"
                ))
        for side, hand in (
            (ArmSide.LEFT, self.left_hand),
            (ArmSide.RIGHT, self.right_hand),
        ):
            try:
                hand_results.append(hand.emergency_stop(side, reason))
            except BaseException as exc:
                errors.append(RuntimeError(
                    f"{side.value} hand emergency stop failed: {exc}"
                ))
        if errors:
            raise BaseExceptionGroup("whole-system emergency stop incomplete", errors)
        return EmergencyStopReport(
            (arm_results[0], arm_results[1]),
            (hand_results[0], hand_results[1]),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for port in (self.right_hand, self.left_hand, self.arm):
            try:
                port.close()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("bimanual robot close failed", errors)


__all__ = [
    "BimanualRobotInfra",
    "BimanualRobotSnapshot",
    "EmergencyStopReport",
]
