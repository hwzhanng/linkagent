from __future__ import annotations

from pathlib import Path

from script.run_tomato_physics_v2_cpu_qualification import qualification_commands


def test_cpu_qualification_allowlist_excludes_gpu_and_hardware_entrypoints(tmp_path: Path) -> None:
    commands = qualification_commands(Path("/test/python"), tmp_path)
    assert len(commands) == 10
    names = [name for name, _ in commands]
    assert names[:7] == [f"step_{index}" for index in range(1, 8)]
    flattened = " ".join(token for _, command in commands for token in command).lower()
    forbidden = (
        "train_tomato",
        "evaluate_tomato",
        "check_tomato_physics_v2_integrated",
        "probe_tomato_physics_v2_ppo",
        "monitor_tomato",
        "nvidia-smi",
        "lbot",
        "linkerhand",
    )
    assert all(token not in flattened for token in forbidden)
