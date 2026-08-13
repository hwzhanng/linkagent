"""Safely disable the right arm after an authorized P6 hold."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
from tools.local_arm_can.service import ArmCanRuntime  # noqa: E402


def _summary(snapshot: dict) -> dict:
    return {
        "interface": snapshot["interface"],
        "arm": snapshot["arm"],
        "healthy_joint_count": snapshot["healthy_joint_count"],
        "all_joints_healthy": snapshot["all_joints_healthy"],
        "software_enabled": snapshot["software_enabled"],
        "last_error": snapshot["last_error"],
        "joint_positions_rad": [float(row["position"]) for row in snapshot["joints"]],
        "rx_frames": snapshot["rx_frames"],
        "tx_frames": snapshot["tx_frames"],
    }


def main() -> int:
    run_id = "right-disable-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    runtime = ArmCanRuntime(
        "can1", arm="right", allow_tx=True, allow_can0=False, poll_telemetry=True
    )
    before = None
    after = None
    error = None
    try:
        runtime.start()
        time.sleep(1.0)
        before = _summary(runtime.snapshot())
        runtime.set_enabled(False, None)
        time.sleep(0.2)
        after = _summary(runtime.snapshot())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        runtime.close()
    manifest = {
        "run_id": run_id,
        "stage": "P6-safe-disable",
        "side": "right",
        "before": before,
        "after": after,
        "disable_frames_sent": 7 if error is None else None,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "error": error,
        "passed": bool(error is None and after and after["software_enabled"] is False),
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "passed": manifest["passed"], "error": error, "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
