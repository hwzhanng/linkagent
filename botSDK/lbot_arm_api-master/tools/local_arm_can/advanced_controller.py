"""High-level FK/IK and trajectory layer above :mod:`service` CAN HAL.

Construction is motionless.  Physical movement still requires all of the
low-level runtime's existing gates: explicit TX authorization, seven fresh
joint states, software enable, software limits, and the per-arm area token.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Iterable

from .kinematics import ArmKinematics, IKResult, Pose
from .service import ArmCanRuntime
from .trajectory import JointSample, sample_movej, sample_movel, sample_smooth_movej


RUN_IDLE = 0
RUN_EMERGENCY = 11
RUN_EMERGENCY_DISABLED = 12
RUN_DISABLED = 22
RUN_FOLLOW = 31
RUN_MOVEJ = 32
RUN_MOVEJP = 33
RUN_MOVEL = 34
MAX_SAMPLE_STEP = 0.08
MAX_SCHEDULE_OVERRUN = 0.05
CAN_COMMAND_PERIOD = 0.02


class MotionBusyError(RuntimeError):
    pass


class AdvancedArmController:
    def __init__(
        self,
        runtime: ArmCanRuntime,
        *,
        config_path: str | Path | None = None,
        base_offset: tuple[float, ...] = (0.0,) * 6,
        flange_offset: tuple[float, ...] = (0.0,) * 6,
    ) -> None:
        self.runtime = runtime
        self.config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).with_name(f"a7_v2_{runtime.arm}.json")
        )
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = json.load(stream)
        parameters = self.config["motorevo_parameters"]
        self.evo_parameters = [
            float(parameters[name])
            for name in ("velocity", "kp", "kd", "kp_i", "kd_i", "ki_i")
        ]
        self.kinematics = ArmKinematics(
            runtime.arm,
            config_path=self.config_path,
            base_offset=base_offset,
            flange_offset=flange_offset,
        )
        self._state_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._run_state = RUN_DISABLED
        self._last_motion_error: str | None = None

    @property
    def run_state(self) -> int:
        with self._state_lock:
            return self._run_state

    @property
    def last_motion_error(self) -> str | None:
        with self._state_lock:
            return self._last_motion_error

    def _set_state(self, state: int, error: str | None = None) -> None:
        with self._state_lock:
            self._run_state = state
            self._last_motion_error = error

    def current_joints(self) -> tuple[float, ...]:
        snapshot = self.runtime.snapshot()
        if not snapshot["all_joints_healthy"]:
            raise RuntimeError(
                f"only {snapshot['healthy_joint_count']}/7 joint states are fresh"
            )
        values = tuple(row["position"] for row in snapshot["joints"])
        if len(values) != 7 or any(value is None or not math.isfinite(value) for value in values):
            raise RuntimeError("current joint state is incomplete")
        return tuple(float(value) for value in values)

    def forward(self, joints: Iterable[float]) -> Pose:
        return self.kinematics.forward(tuple(joints))

    def inverse(
        self,
        position: Iterable[float],
        rpy: Iterable[float],
        joints: Iterable[float] | None = None,
    ) -> IKResult:
        # The vendor uses current feedback whenever request.joints is not
        # exactly seven elements.  Preserve that behavior safely.
        requested_seed = tuple(joints) if joints is not None else ()
        seed = requested_seed if len(requested_seed) == 7 else self.current_joints()
        return self.kinematics.inverse(tuple(position), tuple(rpy), seed)

    def cancel(self) -> None:
        self._cancel.set()

    def set_enabled_state(self, enabled: bool) -> None:
        if not enabled:
            self.cancel()
        self._set_state(RUN_IDLE if enabled else RUN_DISABLED)

    def _validate_samples(
        self, samples: list[JointSample], initial: tuple[float, ...]
    ) -> None:
        if not samples:
            raise ValueError("trajectory contains no samples")
        previous = initial
        for sample in samples:
            self.kinematics.validate_joints(sample.positions, enforce_limits=True)
            if max(
                abs(current - prior)
                for current, prior in zip(sample.positions, previous, strict=True)
            ) > MAX_SAMPLE_STEP:
                raise ValueError(
                    f"trajectory sample exceeds {MAX_SAMPLE_STEP:.3f} rad step limit"
                )
            previous = sample.positions

    def _execute_samples(
        self,
        samples: list[JointSample],
        confirmation: str,
        state: int,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        self._validate_samples(samples, self.current_joints())
        if not self._motion_lock.acquire(blocking=False):
            raise MotionBusyError("another trajectory is already running")
        self._cancel.clear()
        self._set_state(state)
        started = time.monotonic()
        last_command_time: float | None = None
        try:
            for index, sample in enumerate(samples):
                if self._cancel.is_set():
                    raise RuntimeError("trajectory cancelled")
                target_time = started + sample.time
                delay = target_time - time.monotonic()
                if delay < -MAX_SCHEDULE_OVERRUN:
                    raise RuntimeError(
                        f"trajectory scheduler overrun {-delay:.3f}s exceeds "
                        f"{MAX_SCHEDULE_OVERRUN:.3f}s"
                    )
                if delay > 0.0 and self._cancel.wait(delay):
                    raise RuntimeError("trajectory cancelled")
                # The recovered high-level planner samples at 100 Hz, while
                # the proven direct-CAN commissioning path runs at 50 Hz.
                # Sending all seven motor targets at 100 Hz caused a classic
                # joint to intermittently stop reporting for >250 ms on real
                # hardware.  Preserve the 100 Hz time law, but dispatch CAN
                # targets at 50 Hz and always send the exact final sample.
                final_sample = index == len(samples) - 1
                due = (
                    last_command_time is None
                    or sample.time - last_command_time >= CAN_COMMAND_PERIOD - 1.0e-9
                )
                if due or final_sample:
                    self.runtime.command_positions(
                        list(sample.positions), confirmation, self.evo_parameters
                    )
                    last_command_time = sample.time
                    if on_sample is not None:
                        on_sample(sample)
            self._set_state(RUN_IDLE)
            return True
        except Exception as error:
            snapshot = self.runtime.snapshot()
            failure_state = RUN_DISABLED if not snapshot["software_enabled"] else RUN_IDLE
            self._set_state(failure_state, f"{type(error).__name__}: {error}")
            raise
        finally:
            self._motion_lock.release()

    def _dispatch(
        self,
        samples: list[JointSample],
        confirmation: str,
        state: int,
        block: bool,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        snapshot = self.runtime.snapshot()
        if not snapshot["tx_enabled"]:
            raise PermissionError("advanced controller is read-only; TX was not authorized")
        if not snapshot["software_enabled"]:
            raise PermissionError("arm is not software-enabled")
        if not snapshot["all_joints_healthy"]:
            raise RuntimeError("motion rejected while any joint feedback is stale or missing")
        self.runtime.require_physical_confirmation(confirmation)
        if block:
            return self._execute_samples(samples, confirmation, state, on_sample)
        if self._worker is not None and self._worker.is_alive():
            raise MotionBusyError("another trajectory is already running")

        def worker() -> None:
            try:
                self._execute_samples(samples, confirmation, state, on_sample)
            except Exception:
                # Error text is retained in last_motion_error for ROS/HTTP status.
                pass

        self._worker = threading.Thread(
            target=worker,
            name=f"{self.runtime.arm}-arm-trajectory",
            daemon=True,
        )
        self._worker.start()
        return True

    def movej(
        self,
        joints: Iterable[float],
        speed: float,
        acceleration: float,
        block: bool,
        confirmation: str,
        *,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        target = tuple(joints)
        self.kinematics.validate_joints(target, enforce_limits=True)
        samples = sample_movej(self.current_joints(), target, speed, acceleration)
        return self._dispatch(samples, confirmation, RUN_MOVEJ, block, on_sample)

    def smooth_movej(
        self,
        joints: Iterable[float],
        duration: float,
        block: bool,
        confirmation: str,
        *,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        target = tuple(joints)
        self.kinematics.validate_joints(target, enforce_limits=True)
        samples = sample_smooth_movej(self.current_joints(), target, duration)
        return self._dispatch(samples, confirmation, RUN_MOVEJ, block, on_sample)

    def movejp(
        self,
        pose: Pose,
        speed: float,
        acceleration: float,
        block: bool,
        confirmation: str,
        *,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        start = self.current_joints()
        solution = self.kinematics.inverse(pose.position, pose.rpy, start)
        if not solution.success:
            raise RuntimeError(
                f"MoveJP IK failed: position_error={solution.position_error:.3g}, "
                f"rotation_error={solution.rotation_error:.3g}"
            )
        samples = sample_movej(start, solution.joints, speed, acceleration)
        return self._dispatch(samples, confirmation, RUN_MOVEJP, block, on_sample)

    def movel(
        self,
        pose: Pose,
        speed: float,
        acceleration: float,
        block: bool,
        confirmation: str,
        *,
        on_sample: Callable[[JointSample], None] | None = None,
    ) -> bool:
        samples = sample_movel(
            self.kinematics,
            self.current_joints(),
            pose,
            speed,
            acceleration,
        )
        return self._dispatch(samples, confirmation, RUN_MOVEL, block, on_sample)

    def followp(self, pose: Pose, confirmation: str) -> bool:
        start = self.current_joints()
        solution = self.kinematics.inverse(pose.position, pose.rpy, start)
        if not solution.success:
            return False
        sample = JointSample(0.0, solution.joints, (0.0,) * 7, (0.0,) * 7)
        return self._execute_samples([sample], confirmation, RUN_FOLLOW)
