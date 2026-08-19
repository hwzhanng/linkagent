from simulation.ppo_tomato_grasp_v2.config_profiles import PROFILE_DEFINITIONS, profile_definition
from simulation.ppo_tomato_grasp_v2.contract import (
    ACTION_ORDER,
    OBSERVATION_BLOCKS,
    V2_ACTION_DIM,
    V2_ACTOR_OBSERVATION_DIM,
)
from simulation.ppo_tomato_grasp_v2.normalization import normalize_semi_axes
import torch


def test_v2_layout_is_contiguous():
    assert len(ACTION_ORDER) == V2_ACTION_DIM
    cursor = 0
    for block in OBSERVATION_BLOCKS.values():
        assert block.start == cursor
        cursor = block.stop
    assert cursor == V2_ACTOR_OBSERVATION_DIM


def test_profiles_are_named_and_band1_isotropic():
    assert {
        "v1_nominal_reliable",
        "v2_band1_match",
        "v2_band2_shape",
        "v2_band3_sensor_actuator",
        "v2_band4_stress",
    }.issubset(PROFILE_DEFINITIONS)
    assert profile_definition("v2_band1_match")["isotropic_axes"] is True
    assert profile_definition("v2_band2_shape")["isotropic_axes"] is False


def test_profile_definition_is_a_copy():
    profile = profile_definition("v2_band1_match")
    profile["position_jitter"] = (9.0, 9.0, 9.0)
    assert profile_definition("v2_band1_match")["position_jitter"] != (9.0, 9.0, 9.0)


def test_v2_axis_normalizer_keeps_30_to_50mm_distinct():
    values = torch.tensor([[0.030, 0.035, 0.040], [0.045, 0.050, 0.055]])
    normalized = normalize_semi_axes(values)
    assert torch.all(normalized[0] < normalized[1])
    assert torch.all(normalized[0] != 1.0)
    assert torch.allclose(normalized[0], torch.tensor([-1.0, -0.5, 0.0]))
    assert torch.allclose(normalized[1], torch.tensor([0.5, 1.0, 1.0]))
