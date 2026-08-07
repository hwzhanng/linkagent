"""Neutral Port implementations over injected LinkBot/G20 SDK gateways."""

from __future__ import annotations

import math

from proto import (
    ArmSide,
    ArmState,
    CommandResult,
    ExecutionHandle,
    FingerTactile,
    HandState,
    HandSynergy,
    Pose,
    StateStamp,
    TactileFrame,
)

from ..clock import Clock
from ..control import ControlLease, ControlLeaseManager
from ..mapping import HandSynergyMapper
from ..ports import (
    FKResult,
    FollowPoseCommand,
    HandJointPositionCommand,
    HandSession,
    HandSynergyCommand,
    IKResult,
    JointMoveCommand,
    LinearMoveCommand,
    PoseMoveCommand,
)
from .contracts import ArmSdkGateway, ArmSdkState, G20SdkGateway, SdkPoseRpy


def rpy_to_quaternion_wxyz(rpy_rad: tuple[float, ...]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quaternion_wxyz_to_rpy(quaternion: tuple[float, ...]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def _to_sdk_pose(pose: Pose) -> SdkPoseRpy:
    return SdkPoseRpy(
        pose.position_m,
        quaternion_wxyz_to_rpy(pose.quaternion_wxyz),
        pose.frame_id,
    )


def _to_pose(pose: SdkPoseRpy) -> Pose:
    return Pose(pose.position_m, rpy_to_quaternion_wxyz(pose.rpy_rad), pose.frame_id)


class LinkBotArmAdapter:
    """Translate frozen ArmControlPort calls to an injected SDK gateway."""

    def __init__(self, gateway: ArmSdkGateway, leases: ControlLeaseManager, clock: Clock) -> None:
        if not isinstance(gateway, ArmSdkGateway):
            raise TypeError("gateway must implement ArmSdkGateway")
        self._gateway = gateway
        self._leases = leases
        self._clock = clock
        self._opened = False
        self._read_only = True
        self._closed = False
        self._previous: dict[ArmSide, ArmSdkState] = {}
        self._submitted: dict[str, tuple[object, ExecutionHandle]] = {}

    def open(self, *, read_only: bool) -> None:
        if self._closed:
            raise RuntimeError("LinkBotArmAdapter is closed")
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be bool")
        if self._opened:
            if self._read_only != read_only:
                raise RuntimeError("arm adapter cannot change mode after open")
            return
        try:
            self._gateway.open(read_only=read_only)
        except BaseException as primary:
            self._closed = True
            try:
                self._gateway.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "arm gateway open and rollback failed", [primary, cleanup]
                )
            raise
        self._read_only = read_only
        self._opened = True

    def _require_open(self, *, writable: bool = False) -> None:
        if self._closed:
            raise RuntimeError("LinkBotArmAdapter is closed")
        if not self._opened:
            raise RuntimeError("LinkBotArmAdapter must be explicitly opened")
        if writable and self._read_only:
            raise RuntimeError("LinkBotArmAdapter is read-only")

    def get_state(self, side: ArmSide) -> ArmState:
        self._require_open()
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        raw = self._gateway.read_state(side)
        if raw.side is not side:
            raise RuntimeError("arm gateway returned the wrong side")
        velocities = raw.joint_velocities_rad_s
        velocity_source = "measured"
        if velocities is None:
            velocity_source = "estimated"
            previous = self._previous.get(side)
            if previous is None or raw.sample_monotonic_ns <= previous.sample_monotonic_ns:
                velocities = (0.0,) * 7
            else:
                dt = (raw.sample_monotonic_ns - previous.sample_monotonic_ns) / 1_000_000_000
                velocities = tuple(
                    (current - old) / dt
                    for current, old in zip(raw.joint_positions_rad, previous.joint_positions_rad)
                )
        self._previous[side] = raw
        return ArmState(
            side=side,
            joint_positions_rad=raw.joint_positions_rad,
            joint_velocities_rad_s=velocities,
            ee_pose=_to_pose(raw.ee_pose_rpy),
            ee_twist=raw.ee_twist or (0.0,) * 6,
            stamp=StateStamp(
                raw.source_sequence,
                raw.sample_monotonic_ns,
                raw.received_monotonic_ns,
                raw.valid,
                raw.error_code,
            ),
            enabled=raw.enabled,
            run_state=raw.run_state,
            velocity_source=velocity_source,
        )

    def fk(self, side: ArmSide, joint_positions_rad: tuple[float, ...]) -> FKResult:
        self._require_open()
        success, pose, error = self._gateway.fk(side, tuple(joint_positions_rad))
        return FKResult(success, _to_pose(pose) if pose is not None else None, error)

    def ik(self, side: ArmSide, target_pose: Pose, seed_joint_positions_rad: tuple[float, ...]) -> IKResult:
        self._require_open()
        success, joints, error = self._gateway.ik(side, _to_sdk_pose(target_pose), tuple(seed_joint_positions_rad))
        return IKResult(success, joints, error)

    def _submit(self, lease: ControlLease, command: object, submit) -> ExecutionHandle:
        self._require_open(writable=True)
        self._leases.validate(lease, side=command.side)
        fingerprint = (lease.owner_id, command)
        existing = self._submitted.get(command.command_id)
        if existing is not None:
            if existing[0] != fingerprint:
                raise ValueError("command_id was reused with different command content")
            return existing[1]
        handle = submit()
        if handle.command_id != command.command_id or handle.owner_id != lease.owner_id:
            raise RuntimeError("arm gateway returned a mismatched execution handle")
        self._submitted[command.command_id] = (fingerprint, handle)
        return handle

    def move_joint(self, lease: ControlLease, command: JointMoveCommand) -> ExecutionHandle:
        return self._submit(lease, command, lambda: self._gateway.submit_move_j(
            owner_id=lease.owner_id, command_id=command.command_id, side=command.side,
            joints_rad=command.joint_positions_rad, speed=command.speed,
            acceleration=command.acceleration, timeout_s=command.timeout_s,
        ))

    def move_pose(self, lease: ControlLease, command: PoseMoveCommand) -> ExecutionHandle:
        return self._submit(lease, command, lambda: self._gateway.submit_move_jp(
            owner_id=lease.owner_id, command_id=command.command_id, side=command.side,
            pose=_to_sdk_pose(command.target_pose), speed=command.speed,
            acceleration=command.acceleration, timeout_s=command.timeout_s,
        ))

    def move_linear(self, lease: ControlLease, command: LinearMoveCommand) -> ExecutionHandle:
        return self._submit(lease, command, lambda: self._gateway.submit_move_l(
            owner_id=lease.owner_id, command_id=command.command_id, side=command.side,
            pose=_to_sdk_pose(command.target_pose), speed=command.speed,
            acceleration=command.acceleration, timeout_s=command.timeout_s,
        ))

    def follow_pose(self, lease: ControlLease, command: FollowPoseCommand) -> ExecutionHandle:
        return self._submit(lease, command, lambda: self._gateway.submit_follow_p(
            owner_id=lease.owner_id, command_id=command.command_id, side=command.side,
            pose=_to_sdk_pose(command.target_pose), timeout_s=command.timeout_s,
        ))

    def get_result(self, command_id: str) -> CommandResult:
        self._require_open()
        return self._gateway.get_result(command_id)

    def cancel(self, lease: ControlLease, command_id: str) -> CommandResult:
        self._require_open(writable=True)
        self._leases.validate(lease, side=lease.side)
        return self._gateway.cancel(owner_id=lease.owner_id, command_id=command_id)

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        self._require_open(writable=True)
        try:
            return self._gateway.emergency_stop(side, reason)
        finally:
            # A transport failure must not leave software control ownership alive.
            self._leases.revoke(side)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._gateway.close()


class G20HandTactileAdapter:
    """One physical G20 implementing the hand and tactile Ports for one side."""

    def __init__(
        self,
        gateway: G20SdkGateway,
        leases: ControlLeaseManager,
        mapper: HandSynergyMapper,
        clock: Clock,
        *,
        side: ArmSide,
    ) -> None:
        if not isinstance(gateway, G20SdkGateway):
            raise TypeError("gateway must implement G20SdkGateway")
        if not isinstance(mapper, HandSynergyMapper):
            raise TypeError("mapper must implement HandSynergyMapper")
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        self._gateway = gateway
        self._leases = leases
        self._mapper = mapper
        self._clock = clock
        self._side = side
        self._opened = False
        self._read_only = True
        self._closed = False
        self._targets: dict[ArmSide, HandSynergy] = {}
        self._joint_targets: dict[ArmSide, tuple[float, ...]] = {}
        self._submitted: dict[str, tuple[object, ExecutionHandle]] = {}

    def open(self, *, read_only: bool) -> HandSession:
        if self._closed:
            raise RuntimeError("G20HandTactileAdapter is closed")
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be bool")
        if self._opened:
            if self._read_only != read_only:
                raise RuntimeError("G20 adapter cannot change mode after open")
        else:
            try:
                self._gateway.open(self._side, read_only=read_only)
            except BaseException as primary:
                self._closed = True
                try:
                    self._gateway.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "G20 gateway open and rollback failed", [primary, cleanup]
                    )
                raise
            self._opened = True
            self._read_only = read_only
        return HandSession(self._side, read_only, self._clock.monotonic_ns())

    def _require_open(self, *, writable: bool = False) -> None:
        if self._closed:
            raise RuntimeError("G20HandTactileAdapter is closed")
        if not self._opened:
            raise RuntimeError("G20HandTactileAdapter must be explicitly opened")
        if writable and self._read_only:
            raise RuntimeError("G20HandTactileAdapter is read-only")

    def get_state(self, side: ArmSide) -> HandState:
        self._require_open()
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        self._require_side(side)
        raw = self._gateway.read_state(side)
        if raw.side is not side:
            raise RuntimeError("G20 gateway returned the wrong side")
        synergy = self._mapper.from_g20_state(side, raw.logical_raw)
        target = self._targets.get(side, synergy)
        joint_positions = self._mapper.from_g20_joint_positions(side, raw.logical_raw)
        joint_target = self._joint_targets.get(side, joint_positions)
        return HandState(
            side=side,
            synergy=synergy,
            target_synergy=target,
            tracking_error=tuple(wanted - actual for wanted, actual in zip(target.as_tuple(), synergy.as_tuple())),
            stamp=StateStamp(
                raw.source_sequence, raw.sample_monotonic_ns,
                raw.received_monotonic_ns, raw.valid, raw.error_code,
            ),
            logical_raw=raw.logical_raw,
            temperatures_c=raw.temperatures_c,
            fault_codes=tuple(code for code in raw.fault_codes if code != 0),
            joint_positions_rad=joint_positions,
            joint_velocities_rad_s=(0.0,) * 16,
            target_joint_positions_rad=joint_target,
        )

    def get_frame(self, side: ArmSide) -> TactileFrame:
        self._require_open()
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        self._require_side(side)
        raw = self._gateway.read_tactile(side)
        if raw.side is not side:
            raise RuntimeError("G20 tactile gateway returned the wrong side")
        fingers = tuple(
            FingerTactile(
                normal_force_n=item.normal_force_n,
                tangential_force_n=item.tangential_force_n,
                tangential_direction_raw=item.tangential_direction_raw,
                matrix_weight_raw=item.matrix_weight_raw,
                approach_raw=item.approach_raw,
                force_sample_monotonic_ns=item.force_sample_monotonic_ns,
                matrix_sample_monotonic_ns=item.matrix_sample_monotonic_ns,
                force_valid=item.force_valid,
                matrix_valid=item.matrix_valid,
            )
            for item in raw.fingers
        )
        return TactileFrame(
            side, fingers, raw.calibration_id, raw.feature_extractor_id,
            raw.received_monotonic_ns, raw.source_sequence,
        )

    def command_synergy(self, lease: ControlLease, command: HandSynergyCommand) -> ExecutionHandle:
        self._require_open(writable=True)
        self._require_side(command.side)
        self._leases.validate(lease, side=command.side)
        fingerprint = (lease.owner_id, command)
        existing = self._submitted.get(command.command_id)
        if existing is not None:
            if existing[0] != fingerprint:
                raise ValueError("command_id was reused with different command content")
            return existing[1]
        logical = self._mapper.to_g20_logical(command.side, command.target)
        if len(logical) != 20:
            raise RuntimeError("G20 mapper must return exactly 20 values")
        handle = self._gateway.submit_logical(
            owner_id=lease.owner_id,
            command_id=command.command_id,
            side=command.side,
            logical_raw=logical,
            timeout_s=command.timeout_s,
        )
        if handle.command_id != command.command_id or handle.owner_id != lease.owner_id:
            raise RuntimeError("G20 gateway returned a mismatched execution handle")
        self._targets[command.side] = command.target
        self._submitted[command.command_id] = (fingerprint, handle)
        return handle

    def command_joint_positions(
        self,
        lease: ControlLease,
        command: HandJointPositionCommand,
    ) -> ExecutionHandle:
        self._require_open(writable=True)
        self._require_side(command.side)
        self._leases.validate(lease, side=command.side)
        fingerprint = (lease.owner_id, command)
        existing = self._submitted.get(command.command_id)
        if existing is not None:
            if existing[0] != fingerprint:
                raise ValueError("command_id was reused with different command content")
            return existing[1]
        logical = self._mapper.to_g20_joint_positions(
            command.side, command.joint_positions_rad
        )
        handle = self._gateway.submit_logical(
            owner_id=lease.owner_id,
            command_id=command.command_id,
            side=command.side,
            logical_raw=logical,
            timeout_s=command.timeout_s,
        )
        if handle.command_id != command.command_id or handle.owner_id != lease.owner_id:
            raise RuntimeError("G20 gateway returned a mismatched execution handle")
        self._joint_targets[command.side] = command.joint_positions_rad
        self._submitted[command.command_id] = (fingerprint, handle)
        return handle

    def hold(self, lease: ControlLease, side: ArmSide) -> CommandResult:
        self._require_open(writable=True)
        self._require_side(side)
        self._leases.validate(lease, side=side)
        return self._gateway.hold(owner_id=lease.owner_id, side=side)

    def stop(self, lease: ControlLease, side: ArmSide, reason: str) -> CommandResult:
        self._require_open(writable=True)
        self._require_side(side)
        self._leases.validate(lease, side=side)
        return self._gateway.stop(owner_id=lease.owner_id, side=side, reason=reason)

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        self._require_open(writable=True)
        self._require_side(side)
        try:
            return self._gateway.emergency_stop(side, reason)
        finally:
            self._leases.revoke(side)

    def _require_side(self, side: ArmSide) -> None:
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        if side is not self._side:
            raise ValueError(
                f"G20 adapter is configured for {self._side.value}, not {side.value}"
            )

    def get_result(self, command_id: str) -> CommandResult:
        self._require_open()
        return self._gateway.get_result(command_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._gateway.close()


__all__ = [
    "G20HandTactileAdapter",
    "LinkBotArmAdapter",
    "quaternion_wxyz_to_rpy",
    "rpy_to_quaternion_wxyz",
]
