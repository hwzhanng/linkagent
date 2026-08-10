from __future__ import annotations

from copy import deepcopy

import pytest

from script.audit_tomato_physics_v2_reward_counterfactuals import build_reward_audit
from simulation.tomato_pick.curriculum_v2 import load_curriculum_v2, validate_curriculum_v2


def test_inactive_curriculum_is_ordered_general_and_fail_closed() -> None:
    value = load_curriculum_v2()
    stages = value["stages"]
    assert [stage["index"] for stage in stages] == list(range(len(stages)))
    fractions = [stage["randomization_fraction"] for stage in stages]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert value["grasp_contract"]["maximum_supported_fingertips"] == 5
    assert value["activation_guard"]["requires_explicit_authorization"] is True


def test_curriculum_rejects_tennis_pose_prior_or_silent_activation() -> None:
    value = load_curriculum_v2()
    tennis = deepcopy(value)
    tennis["grasp_contract"]["tennis_ball_demo_used_as_prior"] = True
    with pytest.raises(ValueError, match="tennis demo"):
        validate_curriculum_v2(tennis)
    active = deepcopy(value)
    active["status"] = "runtime_active"
    with pytest.raises(ValueError, match="inactive draft"):
        validate_curriculum_v2(active)


def test_reward_counterfactual_matrix_passes() -> None:
    report = build_reward_audit()
    assert report["passed"] is True
    assert len(report["counterfactuals"]) >= 10
    assert report["runtime_activation_authorized"] is False
