#!/usr/bin/env python3
"""Safety-gated commissioning client for the local dual-arm CAN services."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
ARM_ENDPOINTS = {
    "left": ("http://127.0.0.1:8767", "LEFT_ARM_AREA_CLEAR"),
    "right": ("http://127.0.0.1:8766", "RIGHT_ARM_AREA_CLEAR"),
}
ARM_ORDER = ("left", "right")
HARD_MOTION_CAP_DEG = 20.0
CONTROL_PERIOD = 0.02


class CommissionError(RuntimeError):
    pass


class ArmClient:
    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.base_url, self.confirmation = ARM_ENDPOINTS[arm]
        with (HERE / f"a7_v2_{arm}.json").open(encoding="utf-8") as config_file:
            self.config = json.load(config_file)
        parameters = self.config["motorevo_parameters"]
        self.evo_parameters = [
            parameters[name]
            for name in ("velocity", "kp", "kd", "kp_i", "kd_i", "ki_i")
        ]

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if payload is None:
            request = urllib.request.Request(self.base_url + path, method="GET")
        else:
            request = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise CommissionError(f"{self.arm} {path}: HTTP {error.code}: {detail}") from error
        except OSError as error:
            raise CommissionError(f"{self.arm} service unavailable: {error}") from error

    def state(self) -> dict[str, Any]:
        return self._request("/state")

    @staticmethod
    def positions(state: dict[str, Any]) -> list[float]:
        values = [row.get("position") for row in state["joints"]]
        if len(values) != 7 or any(value is None for value in values):
            raise CommissionError("joint state does not contain seven positions")
        return [float(value) for value in values]

    def require_ready(self, *, enabled: bool | None = None) -> dict[str, Any]:
        state = self.state()
        if not state["all_joints_healthy"] or state["healthy_joint_count"] != 7:
            raise CommissionError(
                f"{self.arm} rejected: only {state['healthy_joint_count']}/7 joints are healthy"
            )
        if state["last_error"]:
            raise CommissionError(f"{self.arm} runtime error: {state['last_error']}")
        if enabled is not None and bool(state["software_enabled"]) != enabled:
            required = "enabled" if enabled else "disabled"
            raise CommissionError(f"{self.arm} must be {required}")
        return state

    def enable(self) -> dict[str, Any]:
        result = self._request("/api/enable", {"confirmation": self.confirmation})
        if not result.get("ok") or not result["state"]["software_enabled"]:
            raise CommissionError(f"{self.arm} enable failed: {result}")
        return result["state"]

    def disable(self) -> dict[str, Any]:
        result = self._request("/api/disable", {})
        if not result.get("ok"):
            raise CommissionError(f"{self.arm} disable failed: {result}")
        return result["state"]

    def validate_target(self, target: list[float]) -> None:
        if len(target) != 7 or not all(math.isfinite(value) for value in target):
            raise CommissionError("target must contain seven finite radians")
        limits = self.config["joint_limits"]["current"]
        for index, (value, lower, upper) in enumerate(
            zip(target, limits["lower"], limits["upper"], strict=True)
        ):
            if not float(lower) <= value <= float(upper):
                raise CommissionError(
                    f"{self.arm} J{index + 1} target {value:.4f} is outside "
                    f"[{float(lower):.4f}, {float(upper):.4f}]"
                )

    def command(self, target: list[float]) -> dict[str, Any]:
        self.validate_target(target)
        result = self._request(
            "/api/joint_positions",
            {
                "positions": target,
                "evo_parameters": self.evo_parameters,
                "confirmation": self.confirmation,
            },
        )
        if not result.get("ok") or not result["state"]["all_joints_healthy"]:
            raise CommissionError(f"{self.arm} command rejected: {result}")
        return result["state"]

    def smooth_move(self, target: list[float], speed_rad_s: float, *, minimum: float = 0.8) -> int:
        state = self.require_ready(enabled=True)
        start = self.positions(state)
        self.validate_target(target)
        maximum_delta = max(abs(a - b) for a, b in zip(start, target, strict=True))
        if maximum_delta > math.radians(HARD_MOTION_CAP_DEG):
            raise CommissionError(
                f"{self.arm} motion {math.degrees(maximum_delta):.3f} deg exceeds "
                f"the hard {HARD_MOTION_CAP_DEG:.0f} deg cap"
            )
        duration = max(minimum, maximum_delta / speed_rad_s)
        steps = max(2, math.ceil(duration / CONTROL_PERIOD))
        started = time.monotonic()
        for index in range(1, steps + 1):
            u = index / steps
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            point = [a + (b - a) * blend for a, b in zip(start, target, strict=True)]
            self.command(point)
            delay = started + index * duration / steps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return steps


def selected_arms(value: str) -> list[str]:
    return list(ARM_ORDER) if value == "both" else [value]


def require_physical_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_area_clear or not args.confirm_emergency_reset:
        raise CommissionError(
            "motion requires both --confirm-area-clear and --confirm-emergency-reset"
        )


def status(arms: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        client = ArmClient(arm)
        state = client.state()
        positions = client.positions(state)
        result[arm] = {
            "interface": state["interface"],
            "healthy_joint_count": state["healthy_joint_count"],
            "all_joints_healthy": state["all_joints_healthy"],
            "software_enabled": state["software_enabled"],
            "max_abs_position_deg": math.degrees(max(abs(value) for value in positions)),
            "last_error": state["last_error"],
        }
    return result


def disable(arms: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        state = ArmClient(arm).disable()
        result[arm] = {
            "software_enabled": state["software_enabled"],
            "healthy_joint_count": state["healthy_joint_count"],
        }
    return result


def disable_best_effort(arms: list[str]) -> None:
    for arm in arms:
        try:
            ArmClient(arm).disable()
        except Exception:
            pass


def enable_home(args: argparse.Namespace, arms: list[str]) -> dict[str, Any]:
    require_physical_confirmation(args)
    cap = float(args.max_degrees)
    if not 0 < cap <= HARD_MOTION_CAP_DEG:
        raise CommissionError(f"--max-degrees must be in (0, {HARD_MOTION_CAP_DEG}]")
    speed = math.radians(float(args.speed_deg_s))
    if speed <= 0:
        raise CommissionError("--speed-deg-s must be positive")
    result: dict[str, Any] = {}
    for arm in arms:
        client = ArmClient(arm)
        initial = client.require_ready()
        initial_positions = client.positions(initial)
        maximum_delta = max(abs(value) for value in initial_positions)
        if maximum_delta > math.radians(cap):
            raise CommissionError(
                f"{arm} home delta {math.degrees(maximum_delta):.3f} deg exceeds {cap:.3f} deg"
            )
        client.enable()
        steps = client.smooth_move([0.0] * 7, speed, minimum=2.0)
        time.sleep(0.4)
        final = client.require_ready(enabled=True)
        error = max(abs(value) for value in client.positions(final))
        result[arm] = {
            "enabled": True,
            "healthy_joint_count": 7,
            "planned_delta_deg": math.degrees(maximum_delta),
            "command_points": steps,
            "final_max_zero_error_deg": math.degrees(error),
        }
    return result


def smoke_test(args: argparse.Namespace, arms: list[str]) -> dict[str, Any]:
    require_physical_confirmation(args)
    amplitude_deg = float(args.amplitude_deg)
    if not 0 < amplitude_deg <= HARD_MOTION_CAP_DEG:
        raise CommissionError(f"--amplitude-deg must be in (0, {HARD_MOTION_CAP_DEG}]")
    speed = math.radians(float(args.speed_deg_s))
    tolerance = math.radians(float(args.tracking_tolerance_deg))
    if speed <= 0 or tolerance <= 0:
        raise CommissionError("speed and tracking tolerance must be positive")
    amplitude = math.radians(amplitude_deg)
    result: dict[str, Any] = {}
    for arm in arms:
        client = ArmClient(arm)
        initial = client.require_ready(enabled=True)
        if max(abs(value) for value in client.positions(initial)) > math.radians(1.0):
            raise CommissionError(f"{arm} must be at zero before smoke-test")
        joint_rows: list[dict[str, Any]] = []
        for joint in range(7):
            target = [0.0] * 7
            target[joint] = amplitude
            client.smooth_move(target, speed)
            time.sleep(0.25)
            peak_state = client.require_ready(enabled=True)
            peak = client.positions(peak_state)
            peak_error = abs(peak[joint] - amplitude)
            if peak_error > tolerance:
                raise CommissionError(
                    f"{arm} J{joint + 1} tracking error "
                    f"{math.degrees(peak_error):.3f} deg exceeds tolerance"
                )
            client.smooth_move([0.0] * 7, speed)
            time.sleep(0.25)
            home_state = client.require_ready(enabled=True)
            home_error = max(abs(value) for value in client.positions(home_state))
            if home_error > tolerance:
                raise CommissionError(
                    f"{arm} J{joint + 1} return error "
                    f"{math.degrees(home_error):.3f} deg exceeds tolerance"
                )
            joint_rows.append(
                {
                    "joint": joint + 1,
                    "command_deg": amplitude_deg,
                    "measured_peak_deg": math.degrees(peak[joint]),
                    "peak_error_deg": math.degrees(peak_error),
                    "return_max_error_deg": math.degrees(home_error),
                }
            )
        result[arm] = {"enabled": True, "healthy_joint_count": 7, "joints": joint_rows}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("status", "disable"):
        command = subparsers.add_parser(name)
        command.add_argument("--arm", choices=("left", "right", "both"), default="both")
        command.add_argument("--output")

    for name in ("enable-home", "smoke-test"):
        command = subparsers.add_parser(name)
        command.add_argument("--arm", choices=("left", "right", "both"), default="both")
        command.add_argument("--confirm-area-clear", action="store_true")
        command.add_argument("--confirm-emergency-reset", action="store_true")
        command.add_argument("--speed-deg-s", type=float, default=5.0 if name == "enable-home" else 2.5)
        command.add_argument("--output")
        if name == "enable-home":
            command.add_argument("--max-degrees", type=float, default=20.0)
        else:
            command.add_argument("--amplitude-deg", type=float, default=3.0)
            command.add_argument("--tracking-tolerance-deg", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    arms = selected_arms(args.arm)
    try:
        if args.command == "status":
            result = status(arms)
        elif args.command == "disable":
            result = disable(arms)
        elif args.command == "enable-home":
            result = enable_home(args, arms)
        elif args.command == "smoke-test":
            result = smoke_test(args, arms)
        else:
            raise AssertionError(args.command)
    except Exception as error:
        if args.command in ("enable-home", "smoke-test"):
            disable_best_effort(arms)
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    payload = {"ok": True, "command": args.command, "result": result}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
