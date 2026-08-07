"""Deterministic LinkBot/G20-shaped device mocks; no ROS, CAN, or wall waits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from proto import (
    ArmSide,
    ArmState,
    CommandResult,
    ExecutionHandle,
    ExecutionState,
    FingerTactile,
    HandState,
    HandSynergy,
    Pose,
    StateStamp,
    TactileFrame,
)

from .clock import ManualClock
from .control import ControlLease, ControlLeaseManager
from .mapping import MockHandSynergyMapper
from .ports import (
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


class MockPickOutcome(str, Enum):
    """One primary deterministic failure injection per scenario."""

    SUCCESS = "success"
    NO_CONTACT = "no_contact"
    UNSTABLE_GRASP = "unstable_grasp"
    DETACH_FAILURE = "detach_failure"
    SLIP = "slip"
    DROPPED = "dropped"
    EXCESS_FORCE = "excess_force"
    ARM_STATE_STALE = "arm_state_stale"
    HAND_STATE_STALE = "hand_state_stale"
    TACTILE_STALE = "tactile_stale"
    COMMAND_REJECTED = "command_rejected"
    DEVICE_FAULT = "device_fault"
    TARGET_STALE = "target_stale"
    PLAN_STALE = "plan_stale"
    COLLISION_UNSAFE = "collision_unsafe"


@dataclass(frozen=True, slots=True)
class MockPickScenario:
    outcome: MockPickOutcome = MockPickOutcome.SUCCESS
    contact_ticks: int = 2
    grasp_ticks: int = 2
    detach_ticks: int = 2
    stable_hold_ticks: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, MockPickOutcome):
            try:
                object.__setattr__(self, "outcome", MockPickOutcome(self.outcome))
            except (TypeError, ValueError) as exc:
                choices = ", ".join(item.value for item in MockPickOutcome)
                raise ValueError(f"outcome must be one of: {choices}") from exc
        for field_name in (
            "contact_ticks",
            "grasp_ticks",
            "detach_ticks",
            "stable_hold_ticks",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be an integer > 0")

    @classmethod
    def from_value(cls, value: "MockPickScenario | Mapping[str, Any] | None") -> "MockPickScenario":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("scenario must be a mapping or MockPickScenario")
        return cls(**dict(value))


class MockPickWorld:
    """Shared deterministic physical state behind the three device ports."""

    def __init__(
        self,
        *,
        side: ArmSide,
        scenario: MockPickScenario,
        clock: ManualClock,
        mapper: MockHandSynergyMapper,
    ) -> None:
        if not isinstance(side, ArmSide):
            side = ArmSide(side)
        self.side = side
        self.scenario = scenario
        self.clock = clock
        self.mapper = mapper
        self.arm_sequence = 0
        self.hand_sequence = 0
        self.tactile_sequence = 0
        self.joint_positions_rad = (0.0,) * 7
        self.joint_velocities_rad_s = (0.0,) * 7
        self.ee_pose = Pose((0.0, 0.0, 0.70), (1.0, 0.0, 0.0, 0.0), "world")
        self.hand_synergy = HandSynergy(0.0, 0.0, 0.0)
        self.hand_target = self.hand_synergy
        self.hand_joint_positions_rad = self.mapper.to_sim_independent(
            self.side, self.hand_synergy
        )
        self.hand_target_joint_positions_rad = self.hand_joint_positions_rad
        self.enabled = True
        self.cancelled = False
        self.contact_attempts = 0
        self.grasp_attempts = 0
        self.detach_attempts = 0
        self.hold_checks = 0
        self.contact = False
        self.grasped = False
        self.detached = False
        self.slip = False
        self.dropped = False

    def arm_state(self) -> ArmState:
        now = self.clock.monotonic_ns()
        sample = now
        if self.scenario.outcome is MockPickOutcome.ARM_STATE_STALE:
            sample = max(0, now - 1_000_000_000)
        return ArmState(
            side=self.side,
            joint_positions_rad=self.joint_positions_rad,
            joint_velocities_rad_s=self.joint_velocities_rad_s,
            ee_pose=self.ee_pose,
            ee_twist=(0.0,) * 6,
            stamp=StateStamp(self.arm_sequence, sample, now),
            enabled=self.enabled,
            run_state="idle" if self.enabled else "disabled",
            velocity_source="measured",
        )

    def hand_state(self) -> HandState:
        now = self.clock.monotonic_ns()
        sample = now
        if self.scenario.outcome is MockPickOutcome.HAND_STATE_STALE:
            sample = max(0, now - 1_000_000_000)
        current = self.hand_synergy.as_tuple()
        target = self.hand_target.as_tuple()
        return HandState(
            side=self.side,
            synergy=self.hand_synergy,
            target_synergy=self.hand_target,
            tracking_error=tuple(wanted - actual for wanted, actual in zip(target, current)),
            stamp=StateStamp(self.hand_sequence, sample, now),
            logical_raw=self.mapper.to_g20_logical(self.side, self.hand_synergy),
            temperatures_c=(30.0,) * 5,
            fault_codes=(1,) if self.scenario.outcome is MockPickOutcome.DEVICE_FAULT else (),
            joint_positions_rad=self.hand_joint_positions_rad,
            joint_velocities_rad_s=(0.0,) * 16,
            target_joint_positions_rad=self.hand_target_joint_positions_rad,
        )

    def tactile_frame(self) -> TactileFrame:
        now = self.clock.monotonic_ns()
        force_sample = now
        matrix_sample = now
        if self.scenario.outcome is MockPickOutcome.TACTILE_STALE:
            force_sample = max(0, now - 1_000_000_000)
            matrix_sample = force_sample

        contact_indices: set[int]
        if not self.contact:
            contact_indices = set()
        elif self.scenario.outcome is MockPickOutcome.UNSTABLE_GRASP:
            contact_indices = {1, 2}
        else:
            contact_indices = {0, 1, 2, 3, 4}

        fingers: list[FingerTactile] = []
        for index in range(5):
            active = index in contact_indices
            normal = 0.0
            tangent = 0.0
            if active:
                normal = 4.0 if self.grasped else 1.5
                tangent = 0.35
            if active and self.scenario.outcome is MockPickOutcome.EXCESS_FORCE:
                normal = 18.0
            if active and self.slip:
                tangent = 8.0
            matrix = [[0 for _ in range(6)] for _ in range(12)]
            if active:
                matrix[6][2] = 160
                matrix[6][3] = 120
                matrix[7][2] = 80
            fingers.append(
                FingerTactile(
                    normal_force_n=normal,
                    tangential_force_n=tangent,
                    tangential_direction_raw=float(index * 40),
                    matrix_weight_raw=tuple(tuple(row) for row in matrix),
                    approach_raw=1.0 if active else 0.0,
                    force_sample_monotonic_ns=force_sample,
                    matrix_sample_monotonic_ns=matrix_sample,
                    force_valid=True,
                    matrix_valid=True,
                )
            )
        return TactileFrame(
            side=self.side,
            fingers=tuple(fingers),
            calibration_id="mock_tactile_calibration_v1",
            feature_extractor_id="g20_tactile_features_v1",
            received_monotonic_ns=now,
            source_sequence=self.tactile_sequence,
        )

    def set_joint_positions(self, joints: tuple[float, ...]) -> None:
        self.joint_positions_rad = joints
        self.joint_velocities_rad_s = (0.0,) * 7
        self.arm_sequence += 1

    def set_pose(self, pose: Pose) -> None:
        self.ee_pose = pose
        self.arm_sequence += 1

    def follow_pose(self, pose: Pose) -> None:
        old_z = self.ee_pose.position_m[2]
        self.ee_pose = pose
        self.arm_sequence += 1
        delta_z = pose.position_m[2] - old_z
        if not self.contact and delta_z < -1e-9:
            self.contact_attempts += 1
            if (
                self.scenario.outcome is not MockPickOutcome.NO_CONTACT
                and self.contact_attempts >= self.scenario.contact_ticks
            ):
                self.contact = True
                self.tactile_sequence += 1
        elif self.grasped and delta_z > 1e-9:
            self.detach_attempts += 1
            if (
                self.scenario.outcome is not MockPickOutcome.DETACH_FAILURE
                and self.detach_attempts >= self.scenario.detach_ticks
            ):
                self.detached = True

    def set_synergy(self, target: HandSynergy) -> None:
        self.hand_target = target
        self.hand_synergy = target
        self.hand_target_joint_positions_rad = self.mapper.to_sim_independent(
            self.side, target
        )
        self.hand_joint_positions_rad = self.hand_target_joint_positions_rad
        self.hand_sequence += 1
        if self.contact and target.closure >= 0.15:
            self.grasp_attempts += 1
            if (
                self.scenario.outcome is not MockPickOutcome.UNSTABLE_GRASP
                and self.grasp_attempts >= self.scenario.grasp_ticks
            ):
                self.grasped = True
        self.tactile_sequence += 1

    def set_hand_joint_positions(self, target: tuple[float, ...]) -> None:
        self.hand_target_joint_positions_rad = tuple(target)
        self.hand_joint_positions_rad = tuple(target)
        flexion_indices = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15)
        closure = max(
            0.0,
            min(1.0, sum(max(target[index], 0.0) for index in flexion_indices) / 10.0),
        )
        self.hand_target = HandSynergy(closure, 0.0, 0.0)
        self.hand_synergy = self.hand_target
        self.hand_sequence += 1
        if self.contact and closure >= 0.15:
            self.grasp_attempts += 1
            if (
                self.scenario.outcome is not MockPickOutcome.UNSTABLE_GRASP
                and self.grasp_attempts >= self.scenario.grasp_ticks
            ):
                self.grasped = True
        self.tactile_sequence += 1

    def update_verifier(self) -> None:
        if not self.detached:
            return
        self.hold_checks += 1
        if self.scenario.outcome is MockPickOutcome.SLIP:
            self.slip = True
            self.tactile_sequence += 1
        if self.scenario.outcome is MockPickOutcome.DROPPED:
            self.dropped = True
            self.grasped = False
            self.tactile_sequence += 1

    @property
    def retained_stably(self) -> bool:
        return (
            self.detached
            and self.grasped
            and not self.slip
            and not self.dropped
            and self.hold_checks >= self.scenario.stable_hold_ticks
        )


class _MockCommandStore:
    def __init__(self) -> None:
        self._results: dict[str, CommandResult] = {}
        self._owners: dict[str, str] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _existing_handle(self, command_id: str, now_ns: int) -> ExecutionHandle | None:
        result = self._results.get(command_id)
        if result is None:
            return None
        return ExecutionHandle(
            command_id=command_id,
            owner_id=self._owners[command_id],
            state=result.state,
            submitted_monotonic_ns=now_ns,
        )

    def _finish(
        self,
        *,
        command_id: str,
        owner_id: str,
        now_ns: int,
        success: bool,
        message: str,
        error_code: str | None = None,
    ) -> ExecutionHandle:
        state = ExecutionState.SUCCEEDED if success else ExecutionState.FAILED
        self._owners[command_id] = owner_id
        self._results[command_id] = CommandResult(
            command_id=command_id,
            state=state,
            message=message,
            error_code=error_code,
            started_monotonic_ns=now_ns,
            finished_monotonic_ns=now_ns,
        )
        return ExecutionHandle(command_id, owner_id, state, now_ns)

    def get_result(self, command_id: str) -> CommandResult:
        self._require_open()
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError("command_id must be a non-empty string")
        try:
            return self._results[command_id.strip()]
        except KeyError as exc:
            raise KeyError(f"unknown command_id {command_id!r}") from exc

    def close(self) -> None:
        self._closed = True


class MockArmControlPort(_MockCommandStore):
    def __init__(self, world: MockPickWorld, leases: ControlLeaseManager) -> None:
        super().__init__()
        self._world = world
        self._leases = leases

    def _check(self, lease: ControlLease, side: ArmSide) -> None:
        self._require_open()
        if side is not self._world.side:
            raise ValueError("command side does not match mock arm")
        self._leases.validate(lease, side=side)

    def get_state(self, side: ArmSide) -> ArmState:
        self._require_open()
        if side is not self._world.side:
            raise ValueError("requested side does not match mock arm")
        return self._world.arm_state()

    def fk(self, side: ArmSide, joint_positions_rad: tuple[float, ...]) -> FKResult:
        self._require_open()
        if side is not self._world.side or len(tuple(joint_positions_rad)) != 7:
            return FKResult(False, None, "invalid_joint_request")
        return FKResult(True, self._world.ee_pose)

    def ik(self, side: ArmSide, target_pose: Pose, seed_joint_positions_rad: tuple[float, ...]) -> IKResult:
        self._require_open()
        if (
            side is not self._world.side
            or not isinstance(target_pose, Pose)
            or len(tuple(seed_joint_positions_rad)) != 7
        ):
            return IKResult(False, (), "invalid_ik_request")
        return IKResult(True, tuple(seed_joint_positions_rad))

    def _execute(self, lease: ControlLease, command: object, update: Any) -> ExecutionHandle:
        side = command.side
        command_id = command.command_id
        self._check(lease, side)
        now = self._world.clock.monotonic_ns()
        existing = self._existing_handle(command_id, now)
        if existing is not None:
            return existing
        if self._world.scenario.outcome is MockPickOutcome.COMMAND_REJECTED:
            return self._finish(
                command_id=command_id,
                owner_id=lease.owner_id,
                now_ns=now,
                success=False,
                message="mock controller rejected command",
                error_code="command_rejected",
            )
        update()
        return self._finish(
            command_id=command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="mock command completed",
        )

    def move_joint(self, lease: ControlLease, command: JointMoveCommand) -> ExecutionHandle:
        return self._execute(
            lease,
            command,
            lambda: self._world.set_joint_positions(command.joint_positions_rad),
        )

    def move_pose(self, lease: ControlLease, command: PoseMoveCommand) -> ExecutionHandle:
        return self._execute(
            lease, command, lambda: self._world.set_pose(command.target_pose)
        )

    def move_linear(self, lease: ControlLease, command: LinearMoveCommand) -> ExecutionHandle:
        return self._execute(
            lease, command, lambda: self._world.set_pose(command.target_pose)
        )

    def follow_pose(self, lease: ControlLease, command: FollowPoseCommand) -> ExecutionHandle:
        return self._execute(
            lease, command, lambda: self._world.follow_pose(command.target_pose)
        )

    def cancel(self, lease: ControlLease, command_id: str) -> CommandResult:
        self._check(lease, lease.side)
        existing = self.get_result(command_id)
        if existing.state in {
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        }:
            return existing
        now = self._world.clock.monotonic_ns()
        result = CommandResult(
            command_id=command_id,
            state=ExecutionState.CANCELLED,
            message="mock command cancelled",
            started_monotonic_ns=existing.started_monotonic_ns,
            finished_monotonic_ns=now,
        )
        self._results[command_id] = result
        self._world.cancelled = True
        return result

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        self._require_open()
        if side is not self._world.side:
            raise ValueError("requested side does not match mock arm")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        self._world.enabled = False
        self._world.cancelled = True
        self._leases.revoke(side)
        now = self._world.clock.monotonic_ns()
        return CommandResult(
            command_id=f"emergency-stop-{side.value}-{now}",
            state=ExecutionState.SUCCEEDED,
            message="mock emergency stop applied",
            started_monotonic_ns=now,
            finished_monotonic_ns=now,
        )


class MockHandControlPort(_MockCommandStore):
    def __init__(self, world: MockPickWorld, leases: ControlLeaseManager) -> None:
        super().__init__()
        self._world = world
        self._leases = leases
        self._session: HandSession | None = None

    def open(self, *, read_only: bool) -> HandSession:
        self._require_open()
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool")
        session = HandSession(
            self._world.side,
            read_only,
            self._world.clock.monotonic_ns(),
        )
        self._session = session
        return session

    def _require_session(self, *, writable: bool) -> None:
        self._require_open()
        if self._session is None:
            raise RuntimeError("mock hand must be explicitly opened")
        if writable and self._session.read_only:
            raise RuntimeError("mock hand session is read-only")

    def get_state(self, side: ArmSide) -> HandState:
        self._require_session(writable=False)
        if side is not self._world.side:
            raise ValueError("requested side does not match mock hand")
        return self._world.hand_state()

    def command_synergy(self, lease: ControlLease, command: HandSynergyCommand) -> ExecutionHandle:
        self._require_session(writable=True)
        if command.side is not self._world.side:
            raise ValueError("command side does not match mock hand")
        self._leases.validate(lease, side=command.side)
        now = self._world.clock.monotonic_ns()
        existing = self._existing_handle(command.command_id, now)
        if existing is not None:
            return existing
        if self._world.scenario.outcome is MockPickOutcome.COMMAND_REJECTED:
            return self._finish(
                command_id=command.command_id,
                owner_id=lease.owner_id,
                now_ns=now,
                success=False,
                message="mock hand rejected command",
                error_code="command_rejected",
            )
        self._world.set_synergy(command.target)
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="mock hand command completed",
        )

    def command_joint_positions(
        self,
        lease: ControlLease,
        command: HandJointPositionCommand,
    ) -> ExecutionHandle:
        self._require_session(writable=True)
        if command.side is not self._world.side:
            raise ValueError("command side does not match mock hand")
        self._leases.validate(lease, side=command.side)
        now = self._world.clock.monotonic_ns()
        existing = self._existing_handle(command.command_id, now)
        if existing is not None:
            return existing
        if self._world.scenario.outcome is MockPickOutcome.COMMAND_REJECTED:
            return self._finish(
                command_id=command.command_id,
                owner_id=lease.owner_id,
                now_ns=now,
                success=False,
                message="mock hand rejected command",
                error_code="command_rejected",
            )
        self._world.set_hand_joint_positions(command.joint_positions_rad)
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="mock hand joint command completed",
        )

    def hold(self, lease: ControlLease, side: ArmSide) -> CommandResult:
        self._require_session(writable=True)
        self._leases.validate(lease, side=side)
        now = self._world.clock.monotonic_ns()
        command_id = f"hand-hold-{side.value}-{now}"
        self._finish(
            command_id=command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="mock hand holding",
        )
        return self._results[command_id]

    def stop(self, lease: ControlLease, side: ArmSide, reason: str) -> CommandResult:
        self._require_session(writable=True)
        self._leases.validate(lease, side=side)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return self.hold(lease, side)


class MockTactilePort:
    def __init__(self, world: MockPickWorld) -> None:
        self._world = world
        self._closed = False

    def get_frame(self, side: ArmSide) -> TactileFrame:
        if self._closed:
            raise RuntimeError("MockTactilePort is closed")
        if side is not self._world.side:
            raise ValueError("requested side does not match mock tactile port")
        return self._world.tactile_frame()

    def close(self) -> None:
        self._closed = True


@dataclass(slots=True)
class MockRoboticsBundle:
    """Composition-owned mocks sharing one world and one lease manager."""

    world: MockPickWorld
    clock: ManualClock
    leases: ControlLeaseManager
    mapper: MockHandSynergyMapper
    arm: MockArmControlPort
    hand: MockHandControlPort
    tactile: MockTactilePort
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for port in (self.tactile, self.hand, self.arm):
            try:
                port.close()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("mock robotics close failed", errors)


def build_mock_robotics_bundle(
    *,
    side: ArmSide,
    scenario: MockPickScenario | Mapping[str, Any] | None = None,
) -> MockRoboticsBundle:
    """Build all same-side device mocks without any external side effect."""

    if not isinstance(side, ArmSide):
        side = ArmSide(side)
    prepared_scenario = MockPickScenario.from_value(scenario)
    clock = ManualClock()
    mapper = MockHandSynergyMapper()
    leases = ControlLeaseManager(clock)
    world = MockPickWorld(
        side=side,
        scenario=prepared_scenario,
        clock=clock,
        mapper=mapper,
    )
    arm = MockArmControlPort(world, leases)
    hand = MockHandControlPort(world, leases)
    tactile = MockTactilePort(world)
    hand.open(read_only=False)
    return MockRoboticsBundle(world, clock, leases, mapper, arm, hand, tactile)


__all__ = [
    "MockArmControlPort",
    "MockHandControlPort",
    "MockPickOutcome",
    "MockPickScenario",
    "MockPickWorld",
    "MockRoboticsBundle",
    "MockTactilePort",
    "build_mock_robotics_bundle",
]
