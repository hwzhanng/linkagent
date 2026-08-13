"""P4 live read-only admission check for the connected arms and RGB-D camera.

This script deliberately does not open hand gateways, enable a drive, send a
target, or send a motion frame.  The arm vendor runtime may emit bounded CAN
telemetry/state-query frames so that feedback freshness can be proven; those
frames are counted separately from motion writes.  Missing/stale feedback is
recorded as a blocking result instead of manufacturing a pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

from module.robotics import (
    ControlBusyError,
    ReadOnlyControlLeaseManager,
    SystemClock,
)
from proto import ArmSide, ArmState


ROOT = Path(__file__).resolve().parents[1]
VIEWER_STATUS_URL = "http://127.0.0.1:8080/api/status"
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
from tools.local_arm_can.advanced_controller import AdvancedArmController  # noqa: E402
from tools.local_arm_can.service import ArmCanRuntime  # noqa: E402


def _json_command(command: list[str]) -> Any:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except Exception as exc:  # diagnostics must never abort cleanup
        return f"{type(exc).__name__}: {exc}"


def _can_snapshot(interface: str) -> dict[str, Any]:
    raw = _json_command(["ip", "-details", "-statistics", "link", "show", interface])
    rx_match = re.search(r"\n\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s*\n\s*TX:", raw)
    return {
        "interface": interface,
        "raw_summary": raw,
        "up": "<NOARP,UP," in raw or "state UP" in raw,
        "error_active": "ERROR-ACTIVE" in raw,
        "rx_bytes": int(rx_match.group(1)) if rx_match else None,
        "rx_packets": int(rx_match.group(2)) if rx_match else None,
        "rx_packets_nonzero": bool(rx_match and int(rx_match.group(2)) > 0),
    }


def _viewer_status() -> dict[str, Any]:
    request = Request(VIEWER_STATUS_URL, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        streams = payload.get("streams", {})
        device = payload.get("device", {})
        return {
            "reachable": True,
            "viewer_url": "http://127.0.0.1:8080/",
            "rgb_stream_url": "http://127.0.0.1:8080/stream/rgb.mjpg",
            "depth_stream_url": "http://127.0.0.1:8080/stream/depth.mjpg",
            "state": payload.get("state"),
            "device": {
                "name": device.get("name"),
                "serial": device.get("serial"),
                "firmware": device.get("firmware"),
                "vid": device.get("vid"),
                "pid": device.get("pid"),
                "sdk_version": device.get("sdk_version"),
            },
            "rgb": streams.get("rgb", {}),
            "depth": streams.get("depth", {}),
            "rgb_depth_live": bool(
                streams.get("rgb", {}).get("live") is True
                and streams.get("depth", {}).get("live") is True
            ),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "viewer_url": "http://127.0.0.1:8080/",
            "rgb_stream_url": "http://127.0.0.1:8080/stream/rgb.mjpg",
            "depth_stream_url": "http://127.0.0.1:8080/stream/depth.mjpg",
            "state": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "rgb_depth_live": False,
        }


def _arm_state_record(state: ArmState) -> dict[str, Any]:
    return {
        "side": state.side.value,
        "joint_positions_rad": list(state.joint_positions_rad),
        "joint_velocities_rad_s": list(state.joint_velocities_rad_s),
        "enabled": state.enabled,
        "run_state": state.run_state,
        "velocity_source": state.velocity_source,
        "ee_pose": state.ee_pose.as_dict(),
        "state_stamp": {
            "source_sequence": state.stamp.source_sequence,
            "sample_monotonic_ns": state.stamp.sample_monotonic_ns,
            "received_monotonic_ns": state.stamp.received_monotonic_ns,
            "valid": state.stamp.valid,
            "error_code": state.stamp.error_code,
        },
    }


def main() -> int:
    run_id = "p4-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deployment = json.loads(
        (ROOT / "config" / "robot_bimanual_offline.yaml").read_text(encoding="utf-8")
    )
    can = {
        side: _can_snapshot(deployment["arms"][side]["can_interface"])
        for side in ("left", "right")
    }
    camera = _viewer_status()

    runtimes: dict[str, ArmCanRuntime] = {}
    controllers: dict[str, AdvancedArmController] = {}
    shadow_leases = ReadOnlyControlLeaseManager(SystemClock())
    arm_records: dict[str, Any] = {}
    try:
        # State-query and temperature frames are the only TX permitted in P4.
        # No enable, mode, position, trajectory, or emergency-stop frame is
        # sent. This is telemetry-only, not a writable arm session.
        for side, interface, allow_can0 in (
            ("left", "can0", True),
            ("right", "can1", False),
        ):
            runtime = ArmCanRuntime(
                interface,
                arm=side,
                allow_tx=True,
                allow_can0=allow_can0,
                poll_telemetry=True,
            )
            runtime.start()
            runtimes[side] = runtime
            controllers[side] = AdvancedArmController(
                runtime,
                config_path=ROOT / deployment["arms"][side]["profile_path"],
            )
        time.sleep(2.0)
        for side in ("left", "right"):
            runtime = runtimes[side]
            try:
                snapshot = runtime.snapshot()
                rows = snapshot["joints"]
                joints = tuple(float(row["position"]) for row in rows)
                pose = controllers[side].forward(joints)
                arm_records[side] = {
                    "connected": True,
                    "fresh_complete": bool(snapshot["all_joints_healthy"]),
                    "state": {
                        "interface": snapshot["interface"],
                        "arm": snapshot["arm"],
                        "healthy_joint_count": snapshot["healthy_joint_count"],
                        "all_joints_healthy": snapshot["all_joints_healthy"],
                        "software_enabled": snapshot["software_enabled"],
                        "last_error": snapshot["last_error"],
                        "rx_frames": snapshot["rx_frames"],
                        "tx_frames": snapshot["tx_frames"],
                        "joints": rows,
                        "fk_pose": {
                            "position_m": list(pose.position),
                            "rpy_rad": list(pose.rpy),
                            "frame_id": "world",
                        },
                    },
                }
            except Exception as exc:
                arm_records[side] = {
                    "connected": True,
                    "fresh_complete": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        # Prove the same read-only lease boundary used by the shadow bundle.
        # This check has no hardware side effect and cannot grant write access.
        lease_rejected = True
        try:
            shadow_leases.acquire(owner_id="p4-audit", side=ArmSide.LEFT, ttl_s=1.0)
            lease_rejected = False
        except ControlBusyError:
            pass
    finally:
        for runtime in runtimes.values():
            runtime.close()

    hands = {
        "left": {"connected": False, "checked": False, "reason": "hand not connected by user"},
        "right": {"connected": False, "checked": False, "reason": "hand not connected by user"},
    }
    blocking_reasons: list[str] = []
    if not camera.get("rgb_depth_live"):
        blocking_reasons.append("RGB and Depth are not both live")
    for side in ("left", "right"):
        if not arm_records.get(side, {}).get("fresh_complete"):
            blocking_reasons.append(f"{side} arm has no complete fresh feedback")

    manifest = {
        "run_id": run_id,
        "stage": "P4",
        "device_connections": {"arms": True, "hands": False, "camera": camera.get("reachable", False)},
        "control_enabled": False,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "telemetry_query_frames_sent": sum(
            int(record.get("state", {}).get("tx_frames", 0))
            for record in arm_records.values()
        ),
        "camera": camera,
        "can": can,
        "arms": arm_records,
        "hands": hands,
        "read_only_lease_rejected": lease_rejected,
        "passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "passed": manifest["passed"], "blocking_reasons": blocking_reasons, "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
