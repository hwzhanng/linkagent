"""Half-hour health monitor for the local Stage-0 curriculum run."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "artifacts/stage0_curriculum"
CHECKPOINT = RUN_DIR / "latest.pt"
REPORT = RUN_DIR / "monitor.jsonl"
MANUAL_PAUSE = RUN_DIR / "MANUAL_PAUSE"
SERVICE = "linkbot-stage0-curriculum.service"


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def read_first(pattern: str) -> str | None:
    paths = sorted(Path("/sys/class/power_supply").glob(pattern))
    if not paths:
        return None
    try:
        return paths[0].read_text().strip()
    except OSError:
        return None


def service_active() -> bool:
    return command("systemctl", "--user", "is-active", "--quiet", SERVICE).returncode == 0


def restart_service() -> bool:
    return command("systemctl", "--user", "restart", SERVICE).returncode == 0


def stop_service() -> bool:
    return command("systemctl", "--user", "stop", SERVICE).returncode == 0


def gpu_state() -> dict[str, float | None]:
    result = command(
        "nvidia-smi",
        "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        temperature, memory, utilization = (float(item.strip()) for item in result.stdout.splitlines()[0].split(","))
        state = {"temperature_c": temperature, "memory_used_mb": memory, "utilization_percent": utilization}
    except (IndexError, ValueError):
        state = {"temperature_c": None, "memory_used_mb": None, "utilization_percent": None}
    cuda_test = command(
        "/home/miobarnacle/anaconda3/envs/isaacsim/bin/python",
        "-c",
        "import torch; assert torch.cuda.is_available(); torch.ones(1, device='cuda')",
    )
    state["cuda_available"] = cuda_test.returncode == 0
    return state


def checkpoint_state() -> dict:
    if not CHECKPOINT.exists():
        return {"exists": False}
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metrics = state.get("metrics", {})
    selected = {
        key: metrics.get(key)
        for key in (
            "curriculum/level",
            "curriculum/xy_jitter_mm",
            "eval/success_rate",
            "eval/edge_success_rate",
            "eval/mean_contact_load",
            "eval/max_contact_load",
            "eval/overload_failure_rate",
            "eval/undercontact_failure_rate",
            "train/action_std_mean",
            "train/learning_rate",
            "train/elapsed_hours",
        )
    }
    return {
        "exists": True,
        "age_seconds": time.time() - CHECKPOINT.stat().st_mtime,
        "update": state.get("update"),
        "total_steps": state.get("total_steps"),
        "reason": state.get("reason"),
        "wandb_run_id": state.get("wandb_run_id"),
        "metrics": selected,
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    active = service_active()
    gpu = gpu_state()
    checkpoint = checkpoint_state()
    ac_online = read_first("A*/online")
    battery_capacity = read_first("BAT*/capacity")
    decision = "healthy"

    low_power = ac_online == "0" and battery_capacity is not None and int(battery_capacity) <= 25
    too_hot = gpu["temperature_c"] is not None and gpu["temperature_c"] >= 87.0
    cuda_unavailable = not gpu.get("cuda_available", False)
    stalled = active and checkpoint.get("age_seconds", 0.0) > 1200.0

    if MANUAL_PAUSE.exists():
        if active:
            stop_service()
        decision = "manual_pause"
    elif low_power:
        if active:
            stop_service()
        decision = "paused_low_power"
    elif too_hot:
        if active:
            stop_service()
        decision = "paused_thermal"
    elif cuda_unavailable:
        if active:
            stop_service()
        decision = "waiting_cuda_recovery"
    elif stalled:
        decision = "restarted_stalled" if restart_service() else "restart_stalled_failed"
    elif not active and checkpoint.get("reason") not in ("completed",):
        decision = "restarted_inactive" if restart_service() else "restart_inactive_failed"

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "service_active_before": active,
        "service_active_after": service_active(),
        "ac_online": ac_online,
        "battery_capacity": battery_capacity,
        "gpu": gpu,
        "checkpoint": checkpoint,
        "decision": decision,
    }
    with REPORT.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
    print(json.dumps(record, ensure_ascii=False, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
