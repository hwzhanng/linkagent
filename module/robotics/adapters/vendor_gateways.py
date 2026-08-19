"""Lazy botSDK gateways; importing and construction never open CAN."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from proto import ArmSide, CommandResult, ExecutionHandle, ExecutionState

from ..clock import Clock
from .contracts import ArmSdkState, G20FingerSample, G20SdkState, G20TactileSample, SdkPoseRpy


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _result(command_id: str, state: ExecutionState, now: int, message: str) -> CommandResult:
    return CommandResult(
        command_id, state, message,
        started_monotonic_ns=now, finished_monotonic_ns=now,
    )


class BotSdkLinkBotGateway:
    """Arm gateway over botSDK's ArmCanRuntime/AdvancedArmController."""

    def __init__(
        self,
        *,
        endpoints: Mapping[ArmSide, Mapping[str, object]],
        clock: Clock,
        hardware_write_authorized: bool = False,
        runtime_factory: Callable[..., Any] | None = None,
        controller_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoints = {
            (side if isinstance(side, ArmSide) else ArmSide(side)): dict(config)
            for side, config in endpoints.items()
        }
        if not self.endpoints:
            raise ValueError("at least one LinkBot arm endpoint is required")
        self.clock = clock
        self.hardware_write_authorized = bool(hardware_write_authorized)
        self._runtime_factory = runtime_factory
        self._controller_factory = controller_factory
        self._runtimes: dict[ArmSide, Any] = {}
        self._controllers: dict[ArmSide, Any] = {}
        self._results: dict[str, CommandResult] = {}
        self._opened = False
        self._closed = False
        self._read_only = True

    def _factories(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        if self._runtime_factory is not None and self._controller_factory is not None:
            return self._runtime_factory, self._controller_factory
        root = PROJECT_ROOT / "botSDK/lbot_arm_api-master"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        service = importlib.import_module("tools.local_arm_can.service")
        controller = importlib.import_module("tools.local_arm_can.advanced_controller")
        return service.ArmCanRuntime, controller.AdvancedArmController

    def open(self, *, read_only: bool) -> None:
        if self._closed:
            raise RuntimeError("LinkBot gateway is closed")
        if self._opened:
            if self._read_only != read_only:
                raise RuntimeError("LinkBot gateway mode is immutable")
            return
        if not read_only and not self.hardware_write_authorized:
            raise PermissionError("LinkBot write mode lacks explicit authorization")
        runtime_factory, controller_factory = self._factories()
        attempted: list[Any] = []
        try:
            for side, config in self.endpoints.items():
                interface = config.get("can_interface")
                if not isinstance(interface, str) or not interface:
                    raise ValueError(f"{side.value} arm CAN interface is missing")
                runtime = runtime_factory(
                    interface, arm=side.value, allow_tx=not read_only,
                    allow_can0=bool(config.get("allow_can0", False)),
                    poll_telemetry=not read_only,
                )
                attempted.append(runtime)
                runtime.start()
                self._runtimes[side] = runtime
                self._controllers[side] = controller_factory(
                    runtime, config_path=config.get("profile_path")
                )
        except BaseException:
            for runtime in reversed(attempted):
                try:
                    runtime.close()
                except BaseException:
                    pass
            self._closed = True
            raise
        self._read_only = read_only
        self._opened = True

    def _require(self, side: ArmSide, *, writable: bool = False) -> tuple[Any, Any]:
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        if side not in self.endpoints:
            raise ValueError(f"{side.value} arm is not configured in this gateway")
        if not self._opened or self._closed:
            raise RuntimeError("LinkBot gateway is not open")
        if writable and self._read_only:
            raise RuntimeError("LinkBot gateway is read-only")
        return self._runtimes[side], self._controllers[side]

    def set_enabled(
        self,
        side: ArmSide,
        enabled: bool,
        confirmation: str | None = None,
    ) -> None:
        """Use the validated HAL enable/disable sequence explicitly."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        runtime, controller = self._require(side, writable=True)
        runtime.set_enabled(enabled, confirmation if enabled else None)
        controller.set_enabled_state(enabled)

    def read_state(self, side: ArmSide) -> ArmSdkState:
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        runtime, controller = self._require(side)
        snapshot = runtime.snapshot()
        rows = tuple(snapshot.get("joints", ()))
        positions = tuple(row.get("position") for row in rows)
        if len(positions) != 7 or any(value is None for value in positions):
            raise RuntimeError(f"{side.value} arm feedback is incomplete")
        raw_velocities = tuple(row.get("velocity") for row in rows)
        velocities = (
            None if any(value is None for value in raw_velocities)
            else tuple(float(value) for value in raw_velocities)
        )
        pose = controller.forward(tuple(float(value) for value in positions))
        now = self.clock.monotonic_ns()
        ages = [row.get("state_age") for row in rows if row.get("state_age") is not None]
        sample = now - round(max(ages, default=0.0) * 1e9)
        return ArmSdkState(
            side, tuple(float(value) for value in positions),
            SdkPoseRpy(tuple(pose.position), tuple(pose.rpy), "world"),
            int(runtime.sequence()), max(0, sample), now,
            joint_velocities_rad_s=velocities,
            enabled=bool(snapshot.get("software_enabled", False)),
            run_state=str(controller.run_state),
            valid=bool(snapshot.get("all_joints_healthy", False)),
            error_code=snapshot.get("last_error"),
        )

    def fk(self, side: ArmSide, joints_rad: tuple[float, ...]):
        _, controller = self._require(side)
        try:
            pose = controller.forward(joints_rad)
            return True, SdkPoseRpy(tuple(pose.position), tuple(pose.rpy), "world"), None
        except BaseException as exc:
            return False, None, f"{type(exc).__name__}: {exc}"

    def ik(self, side: ArmSide, pose: SdkPoseRpy, seed_rad: tuple[float, ...]):
        _, controller = self._require(side)
        try:
            solution = controller.inverse(pose.position_m, pose.rpy_rad, seed_rad)
            return (
                (True, tuple(solution.joints), None)
                if solution.success else (False, (), "IK_FAILED")
            )
        except BaseException as exc:
            return False, (), f"{type(exc).__name__}: {exc}"

    def _submit(self, owner_id: str, command_id: str, side: ArmSide, call: Callable[[Any, str], bool]) -> ExecutionHandle:
        _, controller = self._require(side, writable=True)
        now = self.clock.monotonic_ns()
        try:
            token = self.endpoints[side].get("physical_confirmation")
            if not isinstance(token, str) or not token:
                raise ValueError("physical confirmation is missing for arm motion")
            accepted = bool(call(controller, token))
            state = ExecutionState.SUCCEEDED if accepted else ExecutionState.FAILED
            message = "vendor command completed" if accepted else "vendor command rejected"
        except BaseException as exc:
            state, message = ExecutionState.FAILED, f"{type(exc).__name__}: {exc}"
        self._results[command_id] = _result(command_id, state, now, message)
        return ExecutionHandle(command_id, owner_id, state, now)

    def submit_move_j(self, *, owner_id: str, command_id: str, side: ArmSide, joints_rad: tuple[float, ...], speed: float, acceleration: float, timeout_s: float):
        del timeout_s
        return self._submit(owner_id, command_id, side, lambda c, t: c.movej(joints_rad, speed, acceleration, True, t))

    def submit_move_jp(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, speed: float, acceleration: float, timeout_s: float):
        del timeout_s
        vendor_pose = SimpleNamespace(position=pose.position_m, rpy=pose.rpy_rad)
        return self._submit(owner_id, command_id, side, lambda c, t: c.movejp(vendor_pose, speed, acceleration, True, t))

    def submit_move_l(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, speed: float, acceleration: float, timeout_s: float):
        del timeout_s
        vendor_pose = SimpleNamespace(position=pose.position_m, rpy=pose.rpy_rad)
        return self._submit(owner_id, command_id, side, lambda c, t: c.movel(vendor_pose, speed, acceleration, True, t))

    def submit_follow_p(self, *, owner_id: str, command_id: str, side: ArmSide, pose: SdkPoseRpy, timeout_s: float):
        del timeout_s
        vendor_pose = SimpleNamespace(position=pose.position_m, rpy=pose.rpy_rad)
        return self._submit(owner_id, command_id, side, lambda c, t: c.followp(vendor_pose, t))

    def get_result(self, command_id: str) -> CommandResult:
        return self._results[command_id]

    def cancel(self, *, owner_id: str, command_id: str) -> CommandResult:
        del owner_id
        for controller in self._controllers.values():
            controller.cancel()
        now = self.clock.monotonic_ns()
        self._results[command_id] = _result(command_id, ExecutionState.CANCELLED, now, "cancel requested")
        return self._results[command_id]

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        runtime, controller = self._require(side, writable=True)
        controller.cancel()
        runtime.set_enabled(False, None)
        return _result(f"arm-estop-{side.value}", ExecutionState.SUCCEEDED, self.clock.monotonic_ns(), reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for runtime in reversed(tuple(self._runtimes.values())):
            try:
                runtime.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("LinkBot gateway close failed", errors)


class BotSdkG20Gateway:
    """G20 gateway; matrix data remains force-invalid until real calibration."""

    def __init__(
        self, *, side: ArmSide, can_interface: str, clock: Clock,
        calibration_file: str | Path | None = None,
        hardware_write_authorized: bool = False,
        continuous_speed_raw: int = 204,
        torque_limit_raw: int = 153,
        api_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.side = side if isinstance(side, ArmSide) else ArmSide(side)
        self.can_interface = can_interface
        self.clock = clock
        self.hardware_write_authorized = bool(hardware_write_authorized)
        for name, value in {
            "continuous_speed_raw": continuous_speed_raw,
            "torque_limit_raw": torque_limit_raw,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer in 0..255")
        self.continuous_speed_raw = continuous_speed_raw
        self.torque_limit_raw = torque_limit_raw
        self._api_factory = api_factory
        self._api: Any = None
        self._opened = False
        self._closed = False
        self._read_only = True
        self._sequence = 0
        self._results: dict[str, CommandResult] = {}
        self._calibration = None
        if calibration_file is not None:
            path = Path(calibration_file).expanduser().resolve()
            self._calibration = json.loads(path.read_text(encoding="utf-8"))
            if self._calibration.get("schema_version") != 1:
                raise ValueError("invalid tactile calibration schema")

    def _factory(self) -> Callable[..., Any]:
        if self._api_factory is not None:
            return self._api_factory
        root = PROJECT_ROOT / "botSDK/linkerhand-ros2-sdk-main/linker_hand_ros2_sdk/linker_hand_ros2_sdk"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return importlib.import_module("LinkerHand.linker_hand_api").LinkerHandApi

    def open(self, side: ArmSide, *, read_only: bool) -> None:
        side = side if isinstance(side, ArmSide) else ArmSide(side)
        if side is not self.side:
            raise ValueError("wrong G20 side")
        if self._closed:
            raise RuntimeError("G20 gateway is closed")
        if self._opened:
            if self._read_only != read_only:
                raise RuntimeError("G20 gateway mode is immutable")
            return
        if not read_only and not self.hardware_write_authorized:
            raise PermissionError("G20 write mode lacks explicit authorization")
        self._api = self._factory()(hand_type=side.value, hand_joint="G20", can=self.can_interface)
        if not read_only:
            # Values are the connected-G20 motion-calibration settings that
            # passed 100/100 hot steady-state trials.  The torque API sets a
            # maximum limit; it is not measured output torque.
            self._api.set_speed([self.continuous_speed_raw] * 5)
            self._api.set_torque([self.torque_limit_raw] * 5)
        self._read_only, self._opened = read_only, True

    def _require(self, *, writable: bool = False) -> Any:
        if not self._opened or self._closed:
            raise RuntimeError("G20 gateway is not open")
        if writable and self._read_only:
            raise RuntimeError("G20 gateway is read-only")
        return self._api

    @staticmethod
    def _twenty(values: object, default: int | float) -> tuple[Any, ...]:
        raw = tuple(values) if values is not None else ()  # type: ignore[arg-type]
        return (raw + (default,) * 20)[:20]

    def read_state(self, side: ArmSide) -> G20SdkState:
        if side is not self.side:
            raise ValueError("wrong G20 side")
        api = self._require()
        now = self.clock.monotonic_ns()
        self._sequence += 1
        return G20SdkState(
            side,
            tuple(int(v) for v in self._twenty(api.get_state(), 255)),
            tuple(float(v) for v in self._twenty(api.get_temperature(), 0.0)),
            tuple(int(v) for v in self._twenty(api.get_fault(), 0)),
            self._sequence, now, now,
        )

    def read_tactile(self, side: ArmSide) -> G20TactileSample:
        if side is not self.side:
            raise ValueError("wrong G20 side")
        matrices = tuple(self._require().get_matrix_touch_v2())
        if len(matrices) != 5:
            raise RuntimeError("G20 returned wrong tactile finger count")
        now = self.clock.monotonic_ns()
        self._sequence += 1
        fingers = tuple(
            G20FingerSample(
                None, None, None,
                tuple(tuple(int(v) for v in row) for row in matrix),
                None, None, now, False, True,
            )
            for matrix in matrices
        )
        calibration_id = (
            str(self._calibration.get("calibration_id"))
            if self._calibration else "g20_unvalidated_raw_matrix"
        )
        return G20TactileSample(
            side, fingers, calibration_id,
            "g20_raw_matrix_v1_force_unavailable", now, self._sequence,
        )

    def submit_logical(self, *, owner_id: str, command_id: str, side: ArmSide, logical_raw: tuple[int, ...], timeout_s: float):
        del timeout_s
        if side is not self.side:
            raise ValueError("wrong G20 side")
        now = self.clock.monotonic_ns()
        self._require(writable=True).finger_move(list(logical_raw))
        self._results[command_id] = _result(command_id, ExecutionState.SUCCEEDED, now, "G20 command submitted")
        return ExecutionHandle(command_id, owner_id, ExecutionState.SUCCEEDED, now)

    def hold(self, *, owner_id: str, side: ArmSide) -> CommandResult:
        del owner_id
        self._require(writable=True)
        return _result(f"g20-hold-{side.value}", ExecutionState.SUCCEEDED, self.clock.monotonic_ns(), "holding last target")

    def stop(self, *, owner_id: str, side: ArmSide, reason: str) -> CommandResult:
        del owner_id
        self._require(writable=True)
        return _result(f"g20-stop-{side.value}", ExecutionState.FAILED, self.clock.monotonic_ns(), f"no verified vendor stop primitive: {reason}")

    def emergency_stop(self, side: ArmSide, reason: str) -> CommandResult:
        self._require(writable=True)
        return _result(f"g20-estop-{side.value}", ExecutionState.FAILED, self.clock.monotonic_ns(), f"no verified vendor e-stop primitive: {reason}")

    def get_result(self, command_id: str) -> CommandResult:
        return self._results[command_id]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._api is not None:
            hand = getattr(self._api, "hand", None)
            close_interface = getattr(hand, "close_can_interface", None)
            if callable(close_interface):
                close_interface()
            else:
                self._api.close_can()


__all__ = ["BotSdkG20Gateway", "BotSdkLinkBotGateway"]
