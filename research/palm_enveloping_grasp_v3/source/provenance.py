"""Content-addressed provenance utilities for v3 experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path, *, suffixes: tuple[str, ...] = (".py", ".json", ".md")) -> str:
    base = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in suffixes):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def running_simulation_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,args="], text=True, capture_output=True, check=True, timeout=10
    )
    needles = ("isaac-sim", "isaacsim", "train_v2.py", "evaluate_v2.py", "ppo_tomato")
    own_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, elapsed, command = fields
        if int(pid) == own_pid or any(
            marker in command
            for marker in ("provenance.py", "build_freeze_manifest", "run_v3_baseline")
        ):
            continue
        lowered = command.lower()
        if any(needle in lowered for needle in needles):
            matches.append({"pid": int(pid), "elapsed_s": int(elapsed), "command": command})
    return matches


def machine_snapshot(path_for_disk: str | Path) -> dict[str, Any]:
    disk = shutil.disk_usage(Path(path_for_disk).expanduser().resolve())
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "gpu": _command_output([
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]),
        "simulation_processes": running_simulation_processes(),
    }


def artifact_records(paths: Iterable[str | Path], *, relative_to: str | Path | None = None) -> list[dict[str, Any]]:
    base = Path(relative_to).expanduser().resolve() if relative_to is not None else None
    records = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        label = path.relative_to(base).as_posix() if base is not None and path.is_relative_to(base) else str(path)
        if not path.exists():
            records.append({"path": label, "exists": False})
        elif path.is_file():
            records.append({"path": label, "exists": True, "kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        else:
            records.append({"path": label, "exists": True, "kind": "tree", "sha256": sha256_tree(path)})
    return records


def atomic_json_write(path: str | Path, value: Any) -> None:
    """Write a generated artifact atomically without overwriting silently."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
