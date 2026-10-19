"""CPU evidence for the single 2026 RoTO2 object-pressure Jacobian track.

No Isaac, no simulation, no PPO duplication. Tests target the one algorithmic
change: measured sparse pressure differences -> conservative action descent.
"""
from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from simulation.roto2_pressure.pressure_track import (
    ACTIONS, CHANNELS, PressureJacobianTrack, hotspot_cost,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = (ROOT / "botSDK/urdf/01_vendor_original/"
        "02_extracted_packages/L20_8_r_s1/L20_8_r_s1.urdf")


def test_writes_no_new_ppo_trainer():
    # The research branch must import 2026 upstream RoTO2 PPO, not our old PPO.
    project = ROOT / "simulation/roto2_pressure"
    assert not (project / "ppo.py").exists()
    train_source = (ROOT / "script/train_roto2_pressure.py").read_text()
    assert "from common_utils import make_env, train_one_seed" in train_source
    assert "from multimodal_rl.tools.writer import Writer" in train_source
    tree = ast.parse(train_source)
    assert not any(isinstance(x, ast.ClassDef) for x in tree.body)


def test_vendor_urdf_is_one_L20_not_bimanual():
    root = ET.parse(URDF).getroot()
    nonfixed = {x.attrib["name"] for x in root.findall("joint")
                if x.attrib.get("type") != "fixed"}
    assert len(nonfixed) == 21
    assert {"index_joint1", "thumb_joint2", "thumb_joint4"} <= nonfixed
    assert not any(n.startswith(("left_", "right_")) for n in nonfixed)
    from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS
    assert len(ACTIVE_JOINTS) == ACTIONS == 16
    assert len(MIMICS) == 5


def test_sparse_pressure_risk_ignores_invalid_panels():
    p = torch.full((2, CHANNELS), .85)
    mask = torch.ones_like(p)
    base = hotspot_cost(p, mask)
    p[:, :4] = 40.0
    mask[:, :4] = 0.0
    assert torch.allclose(hotspot_cost(p, mask), base)


def test_online_secant_gradient_actually_decreases_object_hotspot():
    torch.manual_seed(10)
    n = 8
    track = PressureJacobianTrack(n, "cpu", enabled=True, min_excitations=8)
    mask = torch.ones(n, CHANNELS)
    # Analytic local contact physics: object patch load rises under finger 0
    # flexion, while spin responds ONLY to finger 1. This is not a model fit
    # taken on faith; the secant update must discover the correct direction.
    truth = torch.zeros(CHANNELS, ACTIONS)
    truth[:8, 0] = .65
    truth[8:12, 2] = .28
    q = torch.zeros(n, ACTIONS)
    for i in range(350):
        q = torch.empty(n, ACTIONS).uniform_(-0.9, .9)
        actual_p = (torch.ones(n, CHANNELS) * 1.20 + q @ truth.T).clamp(0, 4)
        actual_spin = q[:, 1] * .8 + .5
        track.observe(q, actual_p, mask, actual_spin)
    assert track.ready.all()
    assert track.J[:, :8, 0].mean().item() > .12
    assert track.spin_J[:, 1].mean().item() > .20

    current = torch.ones(n, CHANNELS) * 1.20
    proposal = torch.zeros(n, ACTIONS)
    corrected = track.correct(proposal, current, mask, torch.ones(n, dtype=torch.bool))
    assert track.last_accepted.float().mean().item() > 0.5
    assert corrected[:, 0].mean().item() < -0.01
    # Physical truth, not just model prediction, must show a pressure reduction.
    after = (current + corrected @ truth.T).clamp(0, 4)
    assert hotspot_cost(after, mask).mean() < hotspot_cost(current, mask).mean()
    assert ((corrected[:, 1] * .8).abs() < .035).all()


def test_stress_gradient_inactive_for_no_grip_or_disabled_track():
    n = 2
    track = PressureJacobianTrack(n, "cpu", min_excitations=0)
    track.J[:, :8, 0] = .65
    p = torch.ones(n, CHANNELS) * 1.5
    a = torch.zeros(n, ACTIONS)
    mask = torch.ones_like(p)
    assert torch.equal(track.correct(a, p, mask, torch.zeros(n, dtype=torch.bool)), a)
    track.enabled = False
    assert torch.equal(track.correct(a, p, mask, torch.ones(n, dtype=torch.bool)), a)


def test_readme_and_agent_cfg_include_ablation_variants():
    s = (ROOT / "script/train_roto2_pressure.py").read_text()
    assert '"plain", "scalar", "jacobian"' in s
    import yaml
    agent = yaml.safe_load((ROOT / "simulation/roto2_pressure/agent.yaml").read_text())
    assert agent["observations"]["obs_list"] == ["prop", "tactile"]
    assert agent["observations"]["obs_stack"] == 4
    assert agent["agent"]["learning_rate"] > 0
