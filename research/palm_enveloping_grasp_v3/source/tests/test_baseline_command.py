from __future__ import annotations

from pathlib import Path

from simulation.ppo_tomato_grasp_v3.run_v3_baseline import build_command
from simulation.ppo_tomato_grasp_v3.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_command_contains_audited_geometry() -> None:
    scenario, _ = load_scenario(ROOT / "scenarios/v3_large90_nominal.json")
    command = build_command(
        scenario,
        python=Path("/isaac/python"),
        output=Path("/tmp/report.json"),
        seed=7,
        num_envs=8,
        steps=700,
    )
    joined = " ".join(command)
    assert "--axis-range-m 0.045" in joined
    assert "--tomato-contact-offset 0.0003" in joined
    assert "--proxy-contact-offset 0.0001" in joined
    assert "--residual-thumb-contact-gate-steps 5" in joined
    assert any(item.startswith("--residual-target-action=-0.125534") for item in command)
    assert "--strict-require-measured-distal-contact" in command
    assert "--disable-virtual-distal-contact" not in command
    assert "--headless" in command


def test_disabling_virtual_shaping_is_explicit() -> None:
    scenario, _ = load_scenario(ROOT / "scenarios/v3_large90_nominal.json")
    scenario["contact_geometry"]["virtual_distal_contact_for_shaping"] = False
    command = build_command(
        scenario,
        python=Path("/isaac/python"),
        output=Path("/tmp/report.json"),
        seed=7,
        num_envs=1,
        steps=10,
    )
    assert "--disable-virtual-distal-contact" in command
