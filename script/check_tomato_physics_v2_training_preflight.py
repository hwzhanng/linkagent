"""Read-only preflight at the boundary before integrated Physics-v2 work."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.config import load_isaac_runtime_profile
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def python_version(executable: Path) -> str | None:
    if not executable.is_file():
        return None
    try:
        result = subprocess.run(
            [str(executable), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def wandb_credential_present() -> bool:
    try:
        credentials = netrc.netrc()
        return credentials.authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_physics_v2_profile()
    frozen = load_isaac_runtime_profile()
    expected = frozen["primary"]
    configured = args.candidate_root
    if configured is None and os.environ.get("LINKAGENT_ISAACLAB_ROOT"):
        configured = Path(os.environ["LINKAGENT_ISAACLAB_ROOT"])
    candidate = configured.resolve() if configured is not None else None
    expected_python = candidate / "env_isaaclab/bin/python" if candidate else None
    expected_launcher = candidate / "isaaclab.sh" if candidate else None
    candidate_python_version = python_version(expected_python) if expected_python else None

    local_packages = {
        name: package_version(name)
        for name in ("isaaclab", "isaaclab-rl", "isaaclab-tasks", "isaaclab-physx", "torch", "wandb")
    }
    checks = {
        "exact_runtime_root_configured": candidate is not None,
        "managed_python_present": bool(expected_python and expected_python.is_file()),
        "isaaclab_launcher_present": bool(expected_launcher and expected_launcher.is_file()),
        "managed_python_is_3_12": bool(candidate_python_version and candidate_python_version.startswith("3.12.")),
        "wandb_credential_present_without_exposing_secret": wandb_credential_present(),
        "physics_v2_pure_gates_complete": profile["stage_gate"]["current_stage"] >= 7,
        "formal_training_still_fail_closed": profile["implementation_boundary"]["training_allowed"] is False,
    }
    exact_runtime_ready = all(
        checks[key]
        for key in (
            "exact_runtime_root_configured",
            "managed_python_present",
            "isaaclab_launcher_present",
            "managed_python_is_3_12",
        )
    )
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_training_boundary_preflight",
        "ready_for_integrated_gate": exact_runtime_ready,
        "ready_for_training": False,
        "checks": checks,
        "expected_runtime": {
            "source_tag": expected["source_tag"],
            "python": expected["python"],
            "isaac_sim": expected["isaac_sim"],
            "isaaclab_physx": expected["isaaclab_physx"],
        },
        "configured_candidate_root": str(candidate) if candidate else None,
        "candidate_python_version": candidate_python_version,
        "detected_current_interpreter": {
            "executable": sys.executable,
            "python": ".".join(map(str, sys.version_info[:3])),
            "packages": local_packages,
        },
        "decision_required": (
            "Authorize the fail-closed 256-environment, four-update numerical PPO probe."
            if exact_runtime_ready
            else (
                "Provide/install the frozen Isaac Lab v3.0.0-beta2.patch1 managed runtime, "
                "or explicitly authorize a separate compatibility port."
            )
        ),
        "prohibited_interpretation": (
            "Passing integrated gates does not authorize formal PPO training."
            if exact_runtime_ready
            else (
                "A non-frozen Isaac/Python environment may not be used to claim "
                "Physics-v2 integrated or training qualification."
            )
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if exact_runtime_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
