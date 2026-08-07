#!/usr/bin/env python3
"""Read-only LinkerHand host and SDK preflight checks."""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SDK = WORKSPACE.parent / "linkerhand-ros2-sdk-main"
API_RELATIVE_PATH = Path(
    "linker_hand_ros2_sdk/linker_hand_ros2_sdk/LinkerHand/linker_hand_api.py"
)
EXPECTED_METHODS = {
    "finger_move",
    "get_embedded_version",
    "get_serial_number",
    "get_state",
    "set_speed",
    "set_torque",
}


def status(label: str, ok: bool, detail: str, *, required: bool = True) -> None:
    marker = "OK" if ok else ("MISSING" if required else "OPTION")
    print(f"[{marker:<7}] {label}: {detail}")


def distribution_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def command_output(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def inspect_api(api_path: Path) -> tuple[bool, str]:
    if not api_path.is_file():
        return False, f"not found: {api_path}"
    try:
        tree = ast.parse(api_path.read_text(encoding="utf-8"), filename=str(api_path))
    except (OSError, SyntaxError) as exc:
        return False, f"cannot parse: {exc}"

    methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LinkerHandApi":
            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            break
    missing = sorted(EXPECTED_METHODS - methods)
    if missing:
        return False, "API methods absent: " + ", ".join(missing)
    return True, str(api_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk",
        type=Path,
        default=DEFAULT_SDK,
        help=f"SDK root (default: {DEFAULT_SDK})",
    )
    args = parser.parse_args()
    sdk_root = args.sdk.expanduser().resolve()

    print("LinkerHand preflight (read-only; no bus frames are sent)\n")
    status("Python", sys.version_info >= (3, 8), sys.version.split()[0])
    expected_prefix = WORKSPACE / ".conda/envs/linkerhand"
    active_prefix = os.environ.get("CONDA_PREFIX")
    status(
        "Conda environment",
        active_prefix == str(expected_prefix),
        active_prefix or "not active; source scripts/activate_env.sh",
    )

    api_ok, api_detail = inspect_api(sdk_root / API_RELATIVE_PATH)
    status("SDK API", api_ok, api_detail)

    ros2 = shutil.which("ros2")
    ros_setups = sorted(glob.glob("/opt/ros/*/setup.bash"))
    ros_detail = ros2 or (", ".join(ros_setups) if ros_setups else "not in PATH; no /opt/ros setup found")
    status("ROS 2 (optional)", bool(ros2 or ros_setups), ros_detail, required=False)

    for module, package in (
        ("numpy", "NumPy"),
        ("can", "python-can"),
        ("yaml", "PyYAML"),
        ("serial", "pyserial"),
        ("minimalmodbus", "minimalmodbus"),
        ("pymodbus", "pymodbus"),
    ):
        available = importlib.util.find_spec(module) is not None
        status(package, available, "importable" if available else f"module '{module}' not found")

    candle_available = distribution_available("python-can-candle")
    status(
        "python-can-candle",
        candle_available,
        "installed" if candle_available else "distribution not found",
    )

    rclpy_available = importlib.util.find_spec("rclpy") is not None
    status(
        "ROS 2 Python client (optional)",
        rclpy_available,
        "importable" if rclpy_available else "module 'rclpy' not found",
        required=False,
    )

    for command in ("ip", "cansend", "candump", "lsusb"):
        path = shutil.which(command)
        status(command, path is not None, path or "not found")

    ip = shutil.which("ip")
    if ip:
        _, can_output = command_output([ip, "-brief", "link", "show", "type", "can"])
        can_lines = [line for line in can_output.splitlines() if line.strip()]
    else:
        can_lines = []
    can_up = any(
        len(fields := line.split()) >= 2 and fields[1].upper() == "UP"
        for line in can_lines
    )
    status(
        "CAN interface UP",
        can_up,
        "; ".join(can_lines) or "none detected",
    )

    serial_ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    status("Serial interface", bool(serial_ports), ", ".join(serial_ports) or "none detected")

    lsusb = shutil.which("lsusb")
    if lsusb:
        _, usb_output = command_output([lsusb])
        usb_devices = [line for line in usb_output.splitlines() if line.strip()]
        print(f"\n[INFO   ] USB devices ({len(usb_devices)}):")
        for device in usb_devices:
            print(f"         {device}")

    print("\nNo SDK module was imported and no CAN/RS485 data was transmitted.")
    return 0 if api_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
