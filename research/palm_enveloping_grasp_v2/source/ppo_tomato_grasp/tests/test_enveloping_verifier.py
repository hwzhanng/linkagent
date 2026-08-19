import math

import torch

from simulation.ppo_tomato_grasp.enveloping_verifier import (
    EnvelopingVerifierConfig,
    verify_enveloping_trace,
)


def _trace(*, pinch: bool = False, transient: bool = False, proximal_only: bool = False):
    steps, envs = 32, 1
    center = torch.zeros((steps, envs, 3), dtype=torch.float32)
    normal = torch.zeros_like(center)
    normal[..., 2] = 1.0
    palm_force = torch.full((steps, envs), 0.8)
    digit_force = torch.full((steps, envs, 5), 0.45)
    valid = torch.zeros((steps, envs, 5), dtype=torch.bool)
    # Palm support precedes any three-finger contact by seven steps.
    valid[7:12, :, :3] = True
    valid[12:, :, :] = True
    if transient:
        valid[12:, :, :] = False
        valid[12, :, :] = True

    if pinch:
        angles = torch.tensor((-0.15, -0.08, 0.0, 0.08, 0.15))
    elif proximal_only:
        angles = torch.tensor((-1.0, -0.75, -0.5, -0.25, 0.0))
    else:
        # Thumb at pi and four remaining contacts spread over the opposite arc.
        angles = torch.tensor((math.pi, -0.95, -0.1, 0.75, 1.55))
    points = torch.zeros((steps, envs, 5, 3), dtype=torch.float32)
    points[..., 0] = torch.cos(angles)
    points[..., 1] = torch.sin(angles)
    joints = torch.zeros((steps, envs, 16), dtype=torch.float32)
    ramp = torch.linspace(0.0, 0.25, steps).view(steps, 1, 1)
    joints[:, :, [1, 2, 3, 4]] = ramp
    if not proximal_only:
        joints[:, :, [12, 13, 14, 15]] = torch.where(
            torch.arange(steps).view(steps, 1, 1) >= 8,
            torch.full((steps, 1, 1), 0.08),
            torch.zeros((steps, 1, 1)),
        )
    return dict(
        palm_normal_w=normal,
        fruit_center_w=center,
        palm_force_n=palm_force,
        digit_force_n=digit_force,
        contact_points_w=points,
        contact_valid=valid,
        joint_pos_rad=joints,
    )


def test_valid_palm_supported_enclosure_passes() -> None:
    result = verify_enveloping_trace(**_trace())
    assert bool(result["strict_envelop_success"][0])
    assert int(result["palm_first_margin_steps"][0]) >= 3
    assert bool(result["distal_joint_participation"][0].all())


def test_pinch_topology_is_rejected() -> None:
    result = verify_enveloping_trace(**_trace(pinch=True))
    assert not bool(result["strict_envelop_success"][0])
    assert "under_enclosure" in result["failure_reasons"][0]


def test_transient_five_contact_is_rejected() -> None:
    result = verify_enveloping_trace(**_trace(transient=True))
    assert not bool(result["strict_envelop_success"][0])
    assert "envelope_hold_short" in result["failure_reasons"][0]


def test_proximal_only_closure_is_rejected() -> None:
    result = verify_enveloping_trace(
        **_trace(proximal_only=True),
        cfg=EnvelopingVerifierConfig(angular_coverage_min_deg=160.0),
    )
    assert not bool(result["strict_envelop_success"][0])
    assert bool(result["proximal_only_closure"][0])
    assert "proximal_only_closure" in result["failure_reasons"][0]


def test_postbreak_hold_is_required_when_enabled() -> None:
    trace = _trace()
    result = verify_enveloping_trace(
        **trace,
        cfg=EnvelopingVerifierConfig(require_postbreak_hold=True, postbreak_hold_steps=8),
        postbreak_start_step=torch.tensor([24]),
    )
    assert bool(result["strict_envelop_success"][0])
