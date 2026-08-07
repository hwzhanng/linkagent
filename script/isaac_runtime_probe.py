"""Report Isaac/PPO module versions from the interpreter chosen by Isaac Lab."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata, util
import json
import platform
import sys
from typing import Any, Sequence


REQUIRED_MODULES = (
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_physx",
    "gymnasium",
    "torch",
    "rsl_rl",
)


def _distribution_version(module_name: str) -> str | None:
    candidates = {
        "isaaclab": ("isaaclab",),
        "isaaclab_tasks": ("isaaclab-tasks", "isaaclab_tasks"),
        "isaaclab_rl": ("isaaclab-rl", "isaaclab_rl"),
        "rsl_rl": ("rsl-rl-lib", "rsl_rl"),
    }.get(module_name, (module_name,))
    for name in candidates:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def probe(*, with_gpu: bool) -> tuple[int, dict[str, Any]]:
    """Probe module discoverability and optionally Torch CUDA capability."""

    modules = {
        name: {
            "present": util.find_spec(name) is not None,
            "version": _distribution_version(name),
        }
        for name in REQUIRED_MODULES
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "modules": modules,
        "all_required_modules_present": all(item["present"] for item in modules.values()),
        "gpu_probe_requested": with_gpu,
        "torch_cuda_available": None,
        "torch_cuda_device_count": None,
    }
    if with_gpu and modules["torch"]["present"]:
        torch = importlib.import_module("torch")
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_device_count"] = int(torch.cuda.device_count())
    passed = report["all_required_modules_present"] and (
        not with_gpu or report["torch_cuda_available"] is True
    )
    report["passed"] = passed
    return (0 if passed else 2), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-gpu", action="store_true")
    args = parser.parse_args(argv)
    code, report = probe(with_gpu=args.with_gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
