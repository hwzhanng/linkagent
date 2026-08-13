"""Robot-port adapter that executes atomic skills against one Isaac scene."""

from __future__ import annotations

import math
from typing import Any

import torch

from module.robotics import Clock, ControlLeaseManager, RoboticsBundle
from module.robotics.control import ControlLease
from module.robotics.ports import (
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
from module.skills.tomato_pick.hand_synergy import HAND_LOWER_RAD, HAND_UPPER_RAD
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


def _quat_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _normalized_quat_lerp(
    start: tuple[float, ...], finish: tuple[float, ...], fraction: float
) -> tuple[float, float, float, float]:
    sign = -1.0 if sum(a * b for a, b in zip(start, finish, strict=True)) < 0.0 else 1.0
    values = tuple(
        (1.0 - fraction) * a + fraction * sign * b
        for a, b in zip(start, finish, strict=True)
    )
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)  # type: ignore[return-value]


class IsaacSimulationClock:
    """Monotonic clock advanced by actual PhysX substeps, never wall sleeps."""

    def __init__(self, *, start_ns: int = 1_000_000_000) -> None:
        self._now_ns = start_ns
        self._bundle: IsaacHarvestRoboticsBundle | None = None

    def attach(self, bundle: "IsaacHarvestRoboticsBundle") -> None:
        self._bundle = bundle

    def monotonic_ns(self) -> int:
        return self._now_ns

    def sleep(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("seconds must be a real number")
        if seconds < 0.0:
            raise ValueError("seconds must be >= 0")
        if seconds == 0.0:
            return
        if self._bundle is None:
            raise RuntimeError("simulation clock is not attached")
        self._bundle.sleep(float(seconds))

    def _advance_one_substep(self, dt_s: float) -> None:
        self._now_ns += round(dt_s * 1.0e9)


class _CommandStore:
    def __init__(self) -> None:
        self._results: dict[str, CommandResult] = {}
        self._owners: dict[str, str] = {}
        self._closed = False

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

    def _existing(self, command_id: str, now_ns: int) -> ExecutionHandle | None:
        result = self._results.get(command_id)
        if result is None:
            return None
        return ExecutionHandle(command_id, self._owners[command_id], result.state, now_ns)

    def get_result(self, command_id: str) -> CommandResult:
        if self._closed:
            raise RuntimeError("simulation command port is closed")
        try:
            return self._results[command_id]
        except KeyError as exc:
            raise KeyError(f"unknown command_id {command_id!r}") from exc

    def close(self) -> None:
        self._closed = True


class IsaacArmControlPort(_CommandStore):
    def __init__(self, bundle: "IsaacHarvestRoboticsBundle") -> None:
        super().__init__()
        self._bundle = bundle

    def _check(self, lease: ControlLease, side: ArmSide) -> None:
        if side is not self._bundle.side:
            raise ValueError("command side does not match Isaac arm")
        self._bundle.leases.validate(lease, side=side)

    def get_state(self, side: ArmSide) -> ArmState:
        if side is not self._bundle.side:
            raise ValueError("requested side does not match Isaac arm")
        env = self._bundle.env
        pose = env._robot.data.body_pose_w.torch[0, env._ee_body_id]
        position = tuple(float(value) for value in pose[:3])
        quaternion = tuple(float(value) for value in pose[3:7])
        joints = env._robot.data.joint_pos.torch[0, env._arm_joint_ids]
        velocities = env._robot.data.joint_vel.torch[0, env._arm_joint_ids]
        linear = env._robot.data.body_lin_vel_w.torch[0, env._ee_body_id]
        angular = env._robot.data.body_ang_vel_w.torch[0, env._ee_body_id]
        now = self._bundle.clock.monotonic_ns()
        return ArmState(
            side=side,
            joint_positions_rad=tuple(float(value) for value in joints),
            joint_velocities_rad_s=tuple(float(value) for value in velocities),
            ee_pose=Pose(position, quaternion, "world"),
            ee_twist=tuple(float(value) for value in torch.cat((linear, angular))),
            stamp=StateStamp(self._bundle.sequence, now, now),
            enabled=self._bundle.enabled,
            run_state="idle" if self._bundle.enabled else "disabled",
            velocity_source="measured",
        )

    def fk(self, side: ArmSide, joint_positions_rad: tuple[float, ...]) -> FKResult:
        if side is not self._bundle.side or len(tuple(joint_positions_rad)) != 7:
            return FKResult(False, None, "invalid_joint_request")
        current = self.get_state(side)
        if max(abs(a - b) for a, b in zip(current.joint_positions_rad, joint_positions_rad)) > 1.0e-4:
            return FKResult(False, None, "offline_fk_requires_current_isaac_state")
        return FKResult(True, current.ee_pose)

    def ik(
        self,
        side: ArmSide,
        target_pose: Pose,
        seed_joint_positions_rad: tuple[float, ...],
    ) -> IKResult:
        if (
            side is not self._bundle.side
            or not isinstance(target_pose, Pose)
            or target_pose.frame_id != "world"
            or len(tuple(seed_joint_positions_rad)) != 7
        ):
            return IKResult(False, (), "invalid_ik_request")
        reference = self._bundle.reference_pose
        distance = math.dist(reference.position_m, target_pose.position_m)
        if distance > self._bundle.maximum_cartesian_radius_m:
            return IKResult(False, (), "outside_acceptance_workspace")
        if not 1.45 <= target_pose.position_m[2] <= 2.30:
            return IKResult(False, (), "outside_acceptance_height")
        # The arrival is proven by the same iterative DLS controller used by
        # the environment; this admission result is not reported as collision proof.
        return IKResult(True, tuple(seed_joint_positions_rad))

    def _pose_command(
        self, lease: ControlLease, command: PoseMoveCommand, *, linear: bool
    ) -> ExecutionHandle:
        self._check(lease, command.side)
        now = self._bundle.clock.monotonic_ns()
        existing = self._existing(command.command_id, now)
        if existing is not None:
            return existing
        success, error = self._bundle.move_cartesian(
            command.target_pose,
            speed_m_s=command.speed,
            timeout_s=command.timeout_s,
            linear=linear,
        )
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=self._bundle.clock.monotonic_ns(),
            success=success,
            message="Isaac Cartesian command reached target" if success else "Isaac Cartesian command missed target",
            error_code=None if success else error,
        )

    def move_pose(self, lease: ControlLease, command: PoseMoveCommand) -> ExecutionHandle:
        return self._pose_command(lease, command, linear=False)

    def move_linear(self, lease: ControlLease, command: LinearMoveCommand) -> ExecutionHandle:
        return self._pose_command(lease, command, linear=True)

    def follow_pose(self, lease: ControlLease, command: FollowPoseCommand) -> ExecutionHandle:
        self._check(lease, command.side)
        now = self._bundle.clock.monotonic_ns()
        existing = self._existing(command.command_id, now)
        if existing is not None:
            return existing
        self._bundle.set_pose_target(command.target_pose)
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="Isaac follow-pose target queued",
        )

    def move_joint(self, lease: ControlLease, command: JointMoveCommand) -> ExecutionHandle:
        self._check(lease, command.side)
        now = self._bundle.clock.monotonic_ns()
        success = self._bundle.move_arm_joints(
            command.joint_positions_rad, timeout_s=command.timeout_s
        )
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=self._bundle.clock.monotonic_ns(),
            success=success,
            message="Isaac joint command completed" if success else "Isaac joint command timed out",
            error_code=None if success else "joint_target_not_reached",
        )

    def cancel(self, lease: ControlLease, command_id: str) -> CommandResult:
        self._check(lease, lease.side)
        previous = self.get_result(command_id)
        if previous.state in {ExecutionState.SUCCEEDED, ExecutionState.FAILED, ExecutionState.CANCELLED}:
            return previous
        now = self._bundle.clock.monotonic_ns()
        result = CommandResult(
            command_id=command_id,
            state=ExecutionState.CANCELLED,
            message="Isaac command cancelled",
            started_monotonic_ns=previous.started_monotonic_ns,
            finished_monotonic_ns=now,
        )
        self._results[command_id] = result
        return result

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        if side is not self._bundle.side or not isinstance(reason, str) or not reason.strip():
            raise ValueError("valid side and non-empty reason are required")
        self._bundle.enabled = False
        self._bundle.leases.revoke(side)
        now = self._bundle.clock.monotonic_ns()
        return CommandResult(
            command_id=f"isaac-estop-{now}",
            state=ExecutionState.SUCCEEDED,
            message="Isaac controller disabled",
            started_monotonic_ns=now,
            finished_monotonic_ns=now,
        )


class IsaacHandControlPort(_CommandStore):
    def __init__(self, bundle: "IsaacHarvestRoboticsBundle") -> None:
        super().__init__()
        self._bundle = bundle
        self._session: HandSession | None = None

    def open(self, *, read_only: bool) -> HandSession:
        self._session = HandSession(
            self._bundle.side, read_only, self._bundle.clock.monotonic_ns()
        )
        return self._session

    def _writable(self, lease: ControlLease, side: ArmSide) -> None:
        if self._session is None or self._session.read_only:
            raise RuntimeError("Isaac hand requires a writable session")
        self._bundle.leases.validate(lease, side=side)

    @staticmethod
    def _summary(values: tuple[float, ...]) -> HandSynergy:
        flexion = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15)
        units = tuple(
            (values[index] - HAND_LOWER_RAD[index])
            / max(1.0e-6, HAND_UPPER_RAD[index] - HAND_LOWER_RAD[index])
            for index in flexion
        )
        closure = max(0.0, min(1.0, sum(units) / len(units)))
        thumb = (values[11] - HAND_LOWER_RAD[11]) / (
            HAND_UPPER_RAD[11] - HAND_LOWER_RAD[11]
        )
        return HandSynergy(closure, max(-1.0, min(1.0, 2.0 * thumb - 1.0)), 0.0)

    def get_state(self, side: ArmSide) -> HandState:
        if self._session is None:
            raise RuntimeError("Isaac hand must be explicitly opened")
        if side is not self._bundle.side:
            raise ValueError("requested side does not match Isaac hand")
        env = self._bundle.env
        actual_t = env._robot.data.joint_pos.torch[0, env._hand_joint_ids]
        velocity_t = env._robot.data.joint_vel.torch[0, env._hand_joint_ids]
        target_t = env._joint_targets[0, env._hand_joint_ids]
        actual = tuple(float(value) for value in actual_t)
        target = tuple(float(value) for value in target_t)
        actual_summary = self._summary(actual)
        target_summary = self._summary(target)
        now = self._bundle.clock.monotonic_ns()
        return HandState(
            side=side,
            synergy=actual_summary,
            target_synergy=target_summary,
            tracking_error=tuple(
                wanted - current
                for wanted, current in zip(target_summary.as_tuple(), actual_summary.as_tuple(), strict=True)
            ),
            stamp=StateStamp(self._bundle.sequence, now, now),
            logical_raw=None,
            temperatures_c=(30.0,) * 5,
            fault_codes=(),
            joint_positions_rad=actual,
            joint_velocities_rad_s=tuple(float(value) for value in velocity_t),
            target_joint_positions_rad=target,
        )

    def command_synergy(
        self, lease: ControlLease, command: HandSynergyCommand
    ) -> ExecutionHandle:
        self._writable(lease, command.side)
        now = self._bundle.clock.monotonic_ns()
        existing = self._existing(command.command_id, now)
        if existing is not None:
            return existing
        closure = command.target.closure
        targets = tuple(
            lower + closure * (upper - lower)
            for lower, upper in zip(HAND_LOWER_RAD, HAND_UPPER_RAD, strict=True)
        )
        success = self._bundle.move_hand_joints(targets, timeout_s=command.timeout_s)
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=self._bundle.clock.monotonic_ns(),
            success=success,
            message="Isaac hand synergy reached" if success else "Isaac hand synergy timed out",
            error_code=None if success else "hand_target_not_reached",
        )

    def command_joint_positions(
        self, lease: ControlLease, command: HandJointPositionCommand
    ) -> ExecutionHandle:
        self._writable(lease, command.side)
        now = self._bundle.clock.monotonic_ns()
        existing = self._existing(command.command_id, now)
        if existing is not None:
            return existing
        self._bundle.set_hand_target(command.joint_positions_rad)
        # Follow-pose is submitted immediately before this command by the PPO
        # runner. One 20 Hz physical control tick applies both targets.
        self._bundle.advance(1.0 / 20.0)
        self._bundle._sleep_credit_s += 1.0 / 20.0
        return self._finish(
            command_id=command.command_id,
            owner_id=lease.owner_id,
            now_ns=self._bundle.clock.monotonic_ns(),
            success=True,
            message="Isaac hand joint target applied",
        )

    def hold(self, lease: ControlLease, side: ArmSide) -> CommandResult:
        self._writable(lease, side)
        now = self._bundle.clock.monotonic_ns()
        command_id = f"isaac-hand-hold-{now}"
        self._finish(
            command_id=command_id,
            owner_id=lease.owner_id,
            now_ns=now,
            success=True,
            message="Isaac hand target held",
        )
        return self._results[command_id]

    def stop(self, lease: ControlLease, side: ArmSide, reason: str) -> CommandResult:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty")
        return self.hold(lease, side)


class IsaacTactilePort:
    def __init__(self, bundle: "IsaacHarvestRoboticsBundle") -> None:
        self._bundle = bundle
        self._closed = False

    def get_frame(self, side: ArmSide) -> TactileFrame:
        if self._closed:
            raise RuntimeError("Isaac tactile port is closed")
        if side is not self._bundle.side:
            raise ValueError("requested side does not match Isaac tactile")
        forces = self._bundle.env._physical_finger_force()[0]
        now = self._bundle.clock.monotonic_ns()
        fingers: list[FingerTactile] = []
        for force_value in forces:
            force = min(20.0, max(0.0, float(force_value)))
            matrix = [[0 for _ in range(6)] for _ in range(12)]
            matrix[6][2] = min(255, round(force / 20.0 * 255.0))
            fingers.append(
                FingerTactile(
                    normal_force_n=force,
                    tangential_force_n=0.0,
                    tangential_direction_raw=0.0,
                    matrix_weight_raw=tuple(tuple(row) for row in matrix),
                    approach_raw=1.0 if force >= 0.6 else 0.0,
                    force_sample_monotonic_ns=now,
                    matrix_sample_monotonic_ns=now,
                    force_valid=True,
                    matrix_valid=True,
                )
            )
        return TactileFrame(
            side=side,
            fingers=tuple(fingers),
            calibration_id="isaac_g20_contact_v1",
            feature_extractor_id="g20_pressure_5d_v1",
            received_monotonic_ns=now,
            source_sequence=self._bundle.sequence,
        )

    def close(self) -> None:
        self._closed = True


class IsaacHarvestRoboticsBundle:
    """One right-side robot view backed by the live CUDA PhysX environment."""

    side = ArmSide.RIGHT

    def __init__(
        self,
        env: Any,
        *,
        maximum_cartesian_radius_m: float = 0.55,
        position_tolerance_m: float = 0.006,
        orientation_tolerance_rad: float = 0.05,
    ) -> None:
        self.env = env
        self.maximum_cartesian_radius_m = float(maximum_cartesian_radius_m)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.clock = IsaacSimulationClock()
        self.leases = ControlLeaseManager(self.clock)
        self.sequence = 0
        self.enabled = True
        self._closed = False
        # The PPO runner submits arm and hand targets, then sleeps for one
        # policy period.  The hand command already advances one physical tick;
        # this credit prevents the subsequent clock sleep from advancing the
        # same control period twice.
        self._sleep_credit_s = 0.0
        self.arm = IsaacArmControlPort(self)
        self.hand = IsaacHandControlPort(self)
        self.tactile = IsaacTactilePort(self)
        self.hand.open(read_only=False)
        self.clock.attach(self)
        self.reference_pose = self.arm.get_state(self.side).ee_pose

    def task_component_overrides(self) -> dict[str, dict[str, object]]:
        return {}

    def verifier_context(self) -> dict[str, object]:
        return {
            "isaac_env": self.env,
            "simulation_backend": "isaac_lab_cuda_physx",
            "payload_state": self.payload_state,
        }

    def payload_state(self) -> dict[str, object]:
        self.refresh_outcomes()
        return {
            "detached": bool(self.env._stem_detached[0]),
            "retained": bool(self.env._retained[0]),
            "dropped": bool(self.env._dropped[0]),
            "success": bool(self.env._success[0]),
            "hold_steps": int(self.env._hold_streak[0]),
        }

    def refresh_outcomes(self) -> None:
        self.env._get_dones()

    def set_pose_target(self, pose: Pose) -> None:
        if pose.frame_id != "world":
            raise ValueError("Isaac acceptance uses world-frame Cartesian targets")
        self.env._ee_command_pose_w[0, :3] = torch.tensor(
            pose.position_m, device=self.env.device
        )
        self.env._ee_command_pose_w[0, 3:7] = torch.tensor(
            pose.quaternion_wxyz, device=self.env.device
        )

    def set_hand_target(self, targets: tuple[float, ...]) -> None:
        self.env._joint_targets[0, self.env._hand_joint_ids] = torch.tensor(
            targets, device=self.env.device
        )
        env_ids = torch.tensor([0], device=self.env.device, dtype=torch.long)
        self.env._update_mimic_targets(env_ids)

    def advance(self, seconds: float) -> None:
        dt = float(self.env.sim.get_physics_dt())
        substeps = max(1, round(float(seconds) / dt))
        self.env._stem_break_event.zero_()
        for _ in range(substeps):
            self.env._apply_action()
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(dt)
            self.sequence += 1
            self.clock._advance_one_substep(dt)

    def sleep(self, seconds: float) -> None:
        credited = min(self._sleep_credit_s, seconds)
        self._sleep_credit_s -= credited
        remaining = seconds - credited
        if remaining > 1.0e-9:
            self.advance(remaining)

    def move_cartesian(
        self,
        target: Pose,
        *,
        speed_m_s: float,
        timeout_s: float,
        linear: bool,
    ) -> tuple[bool, str | None]:
        del linear
        start = self.arm.get_state(self.side).ee_pose
        distance = math.dist(start.position_m, target.position_m)
        orientation = _quat_error(start.quaternion_wxyz, target.quaternion_wxyz)
        duration = max(distance / max(speed_m_s, 1.0e-3), orientation / 0.8, 0.05)
        duration = min(float(timeout_s), duration)
        ticks = max(1, math.ceil(duration * 20.0))
        for index in range(1, ticks + 1):
            fraction = index / ticks
            pose = Pose(
                tuple(
                    (1.0 - fraction) * a + fraction * b
                    for a, b in zip(start.position_m, target.position_m, strict=True)
                ),
                _normalized_quat_lerp(
                    start.quaternion_wxyz, target.quaternion_wxyz, fraction
                ),
                "world",
            )
            self.set_pose_target(pose)
            self.advance(1.0 / 20.0)
        # Let the DLS target tracker settle inside the declared command budget.
        remaining_ticks = max(0, math.floor(float(timeout_s) * 20.0) - ticks)
        for _ in range(remaining_ticks + 1):
            actual = self.arm.get_state(self.side).ee_pose
            if (
                math.dist(actual.position_m, target.position_m)
                <= self.position_tolerance_m
                and _quat_error(actual.quaternion_wxyz, target.quaternion_wxyz)
                <= self.orientation_tolerance_rad
            ):
                return True, None
            self.set_pose_target(target)
            self.advance(1.0 / 20.0)
        return False, "cartesian_target_not_reached"

    def move_hand_joints(self, targets: tuple[float, ...], *, timeout_s: float) -> bool:
        self.set_hand_target(targets)
        for _ in range(max(1, math.ceil(float(timeout_s) * 20.0))):
            self.advance(1.0 / 20.0)
            actual = self.env._robot.data.joint_pos.torch[0, self.env._hand_joint_ids]
            wanted = torch.tensor(targets, device=self.env.device)
            if float((actual - wanted).abs().max()) <= 0.08:
                return True
        return False

    def move_arm_joints(self, targets: tuple[float, ...], *, timeout_s: float) -> bool:
        wanted = torch.tensor(targets, device=self.env.device)
        for _ in range(max(1, math.ceil(float(timeout_s) * 20.0))):
            self.env._joint_targets[0, self.env._arm_joint_ids] = wanted
            self.advance(1.0 / 20.0)
            actual = self.env._robot.data.joint_pos.torch[0, self.env._arm_joint_ids]
            if float((actual - wanted).abs().max()) <= 0.02:
                return True
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.arm.close()
        self.hand.close()
        self.tactile.close()


assert isinstance(IsaacHarvestRoboticsBundle, type)

__all__ = ["IsaacHarvestRoboticsBundle", "IsaacSimulationClock"]
