"""Fault-injectable SDK gateways driving the normal production adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from proto import ArmSide, CommandResult, ExecutionHandle, ExecutionState, Pose

from .adapters import (
    ArmSdkState,
    G20FingerSample,
    G20HandTactileAdapter,
    G20SdkState,
    G20TactileSample,
    LinkBotArmAdapter,
    SdkPoseRpy,
)
from .adapters.linkbot_g20 import quaternion_wxyz_to_rpy, rpy_to_quaternion_wxyz
from .clock import ManualClock
from .control import ControlLeaseManager
from .mapping import MockHandSynergyMapper
from .mock import MockPickOutcome, MockPickScenario, MockPickWorld


@dataclass(frozen=True, slots=True)
class FakeGatewayFaults:
    reject_commands: bool = False
    command_timeout: bool = False
    fail_open: bool = False

    @classmethod
    def from_value(cls, value: "FakeGatewayFaults | Mapping[str, Any] | None") -> "FakeGatewayFaults":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("gateway_faults must be a mapping")
        return cls(**dict(value))


def _command_result(command_id: str, state: ExecutionState, now: int, message: str, error: str | None = None) -> CommandResult:
    return CommandResult(
        command_id, state, message, error,
        started_monotonic_ns=now, finished_monotonic_ns=now,
    )


class FakeArmSdkGateway:
    def __init__(self, world: MockPickWorld, faults: FakeGatewayFaults) -> None:
        self.world, self.faults = world, faults
        self.results: dict[str, CommandResult] = {}
        self.operations: list[str] = []
        self.opened = False
        self.read_only = True
        self.closed = False

    def open(self, *, read_only: bool) -> None:
        if self.faults.fail_open:
            raise RuntimeError("injected arm gateway open failure")
        self.read_only = read_only
        self.opened = True

    def read_state(self, side: ArmSide) -> ArmSdkState:
        state = self.world.arm_state()
        return ArmSdkState(
            side, state.joint_positions_rad,
            SdkPoseRpy(state.ee_pose.position_m, quaternion_wxyz_to_rpy(state.ee_pose.quaternion_wxyz), state.ee_pose.frame_id),
            state.stamp.source_sequence, state.stamp.sample_monotonic_ns,
            state.stamp.received_monotonic_ns,
            joint_velocities_rad_s=state.joint_velocities_rad_s,
            ee_twist=state.ee_twist, enabled=state.enabled,
            run_state=state.run_state, valid=state.stamp.valid,
            error_code=state.stamp.error_code,
        )

    def fk(self, side: ArmSide, joints_rad: tuple[float, ...]):
        del side, joints_rad
        pose = self.world.ee_pose
        return True, SdkPoseRpy(pose.position_m, quaternion_wxyz_to_rpy(pose.quaternion_wxyz), pose.frame_id), None

    def ik(self, side: ArmSide, pose: SdkPoseRpy, seed_rad: tuple[float, ...]):
        del side, pose
        return True, tuple(seed_rad), None

    def _submit(self, name: str, *, owner_id: str, command_id: str, timeout_s: float, update) -> ExecutionHandle:
        if self.read_only:
            raise RuntimeError("fake arm gateway is read-only")
        self.operations.append(name)
        if self.faults.command_timeout:
            self.world.clock.advance_s(timeout_s + 0.01)
            state, message, error = ExecutionState.FAILED, "injected gateway timeout", "timeout"
        elif self.faults.reject_commands or self.world.scenario.outcome is MockPickOutcome.COMMAND_REJECTED:
            state, message, error = ExecutionState.FAILED, "injected command rejection", "rejected"
        else:
            update()
            state, message, error = ExecutionState.SUCCEEDED, "fake gateway command complete", None
        now = self.world.clock.monotonic_ns()
        self.results[command_id] = _command_result(command_id, state, now, message, error)
        return ExecutionHandle(command_id, owner_id, state, now)

    def submit_move_j(self, *, owner_id: str, command_id: str, side: ArmSide, joints_rad: tuple[float, ...], speed: float, acceleration: float, timeout_s: float):
        del side, speed, acceleration
        return self._submit("move_j", owner_id=owner_id, command_id=command_id, timeout_s=timeout_s, update=lambda: self.world.set_joint_positions(joints_rad))

    def _pose(self, value: SdkPoseRpy) -> Pose:
        return Pose(value.position_m, rpy_to_quaternion_wxyz(value.rpy_rad), value.frame_id)

    def submit_move_jp(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, speed: float, acceleration: float, timeout_s: float):
        del side, speed, acceleration
        return self._submit("move_jp", owner_id=owner_id, command_id=command_id, timeout_s=timeout_s, update=lambda: self.world.set_pose(self._pose(pose)))

    def submit_move_l(self, **kwargs):
        return self.submit_move_jp(**kwargs)

    def submit_follow_p(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, timeout_s: float):
        del side
        return self._submit("follow_p", owner_id=owner_id, command_id=command_id, timeout_s=timeout_s, update=lambda: self.world.follow_pose(self._pose(pose)))

    def get_result(self, command_id: str) -> CommandResult:
        return self.results[command_id]

    def cancel(self, *, owner_id: str, command_id: str) -> CommandResult:
        del owner_id
        self.operations.append("cancel")
        self.world.cancelled = True
        now = self.world.clock.monotonic_ns()
        self.results[command_id] = _command_result(command_id, ExecutionState.CANCELLED, now, "cancelled")
        return self.results[command_id]

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        del side
        self.world.enabled = False
        return _command_result("fake-arm-estop", ExecutionState.SUCCEEDED, self.world.clock.monotonic_ns(), reason)

    def close(self) -> None:
        self.closed = True


class FakeG20SdkGateway:
    def __init__(self, world: MockPickWorld, mapper: MockHandSynergyMapper, faults: FakeGatewayFaults) -> None:
        self.world, self.mapper, self.faults = world, mapper, faults
        self.results: dict[str, CommandResult] = {}
        self.operations: list[str] = []
        self.opened = False
        self.read_only = True
        self.closed = False

    def open(self, side: ArmSide, *, read_only: bool) -> None:
        if self.faults.fail_open:
            raise RuntimeError("injected G20 gateway open failure")
        if side is not self.world.side:
            raise RuntimeError("fake G20 side mismatch")
        self.read_only = read_only
        self.opened = True

    def read_state(self, side: ArmSide) -> G20SdkState:
        state = self.world.hand_state()
        logical = state.logical_raw or self.mapper.to_g20_logical(side, state.synergy)
        faults = tuple(state.fault_codes) + (0,) * (20 - len(state.fault_codes))
        return G20SdkState(
            side, logical, (30.0,) * 20, faults[:20],
            state.stamp.source_sequence, state.stamp.sample_monotonic_ns,
            state.stamp.received_monotonic_ns, state.stamp.valid,
            state.stamp.error_code,
        )

    def read_tactile(self, side: ArmSide) -> G20TactileSample:
        frame = self.world.tactile_frame()
        fingers = tuple(
            G20FingerSample(
                finger.normal_force_n, finger.tangential_force_n,
                finger.tangential_direction_raw, finger.matrix_weight_raw,
                finger.approach_raw, finger.force_sample_monotonic_ns,
                finger.matrix_sample_monotonic_ns,
                finger.force_valid, finger.matrix_valid,
            )
            for finger in frame.fingers
        )
        return G20TactileSample(
            side, fingers, frame.calibration_id, frame.feature_extractor_id,
            frame.received_monotonic_ns, frame.source_sequence,
        )

    def submit_logical(self, *, owner_id: str, command_id: str, side: ArmSide, logical_raw: tuple[int, ...], timeout_s: float):
        if self.read_only:
            raise RuntimeError("fake G20 gateway is read-only")
        self.operations.append("logical")
        if self.faults.command_timeout:
            self.world.clock.advance_s(timeout_s + 0.01)
            state, error = ExecutionState.FAILED, "timeout"
        elif self.faults.reject_commands:
            state, error = ExecutionState.FAILED, "rejected"
        else:
            self.world.set_hand_joint_positions(self.mapper.from_g20_joint_positions(side, logical_raw))
            state, error = ExecutionState.SUCCEEDED, None
        now = self.world.clock.monotonic_ns()
        self.results[command_id] = _command_result(command_id, state, now, "fake G20 command", error)
        return ExecutionHandle(command_id, owner_id, state, now)

    def hold(self, *, owner_id: str, side: ArmSide) -> CommandResult:
        del owner_id, side
        return _command_result("fake-g20-hold", ExecutionState.SUCCEEDED, self.world.clock.monotonic_ns(), "hold")

    def stop(self, *, owner_id: str, side: ArmSide, reason: str) -> CommandResult:
        del owner_id, side
        return _command_result("fake-g20-stop", ExecutionState.SUCCEEDED, self.world.clock.monotonic_ns(), reason)

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        del side
        return _command_result("fake-g20-estop", ExecutionState.SUCCEEDED, self.world.clock.monotonic_ns(), reason)

    def get_result(self, command_id: str) -> CommandResult:
        return self.results[command_id]

    def close(self) -> None:
        self.closed = True


class FakeGatewayRoboticsBundle:
    def __init__(self, *, side: ArmSide, scenario: MockPickScenario, faults: FakeGatewayFaults) -> None:
        self.side = side
        self.clock = ManualClock()
        self.leases = ControlLeaseManager(self.clock)
        self.mapper = MockHandSynergyMapper()
        self.world = MockPickWorld(side=side, scenario=scenario, clock=self.clock, mapper=self.mapper)
        self.arm_gateway = FakeArmSdkGateway(self.world, faults)
        self.hand_gateway = FakeG20SdkGateway(self.world, self.mapper, faults)
        self.arm = LinkBotArmAdapter(self.arm_gateway, self.leases, self.clock)
        self._hand_tactile = G20HandTactileAdapter(self.hand_gateway, self.leases, self.mapper, self.clock, side=side)
        self.hand = self._hand_tactile
        self.tactile = self._hand_tactile
        self.arm.open(read_only=False)
        self._hand_tactile.open(read_only=False)
        self._closed = False

    def task_component_overrides(self) -> Mapping[str, Mapping[str, Any]]:
        outcome = self.world.scenario.outcome
        return {
            "perception": {"force_stale": True} if outcome is MockPickOutcome.TARGET_STALE else {},
            "planner": {"force_stale": True} if outcome is MockPickOutcome.PLAN_STALE else {},
            "safety": {"force_unsafe": True} if outcome is MockPickOutcome.COLLISION_UNSAFE else {},
            "verifier": {},
        }

    def verifier_context(self) -> Mapping[str, Any]:
        return {"mock_world": self.world}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hand_tactile.close()
        self.arm.close()


def build_fake_gateway_robotics_bundle(
    *, side: ArmSide,
    scenario: MockPickScenario | Mapping[str, Any] | None = None,
    gateway_faults: FakeGatewayFaults | Mapping[str, Any] | None = None,
) -> FakeGatewayRoboticsBundle:
    return FakeGatewayRoboticsBundle(
        side=side if isinstance(side, ArmSide) else ArmSide(side),
        scenario=MockPickScenario.from_value(scenario),
        faults=FakeGatewayFaults.from_value(gateway_faults),
    )


__all__ = [
    "FakeArmSdkGateway",
    "FakeG20SdkGateway",
    "FakeGatewayFaults",
    "FakeGatewayRoboticsBundle",
    "build_fake_gateway_robotics_bundle",
]
