"""CPU合成布局测试：只验证原始接触缓冲解码，不冒充物理证据。"""
from types import SimpleNamespace
import pytest
import torch
from research.fixed_fruit_grasp_v1 import physical_contacts as P


def sensor(counts, starts, distances, paths):
    data = [None, None, None, torch.tensor(distances), torch.tensor(counts), torch.tensor(starts)]
    return SimpleNamespace(contact_view=SimpleNamespace(get_contact_data=lambda dt: data),
                           body_physx_view=SimpleNamespace(prim_paths=paths),
                           cfg=SimpleNamespace(max_contact_data_count_per_prim=4))


def test_named_mapping_and_unused_nan_are_preserved(monkeypatch):
    monkeypatch.setattr(P.wp, 'to_torch', lambda x: x)
    s = sensor([2, 1], [0, 2], [0.002, -0.003, -0.001, float('nan')],
               ['/World/envs/env_1/Robot/base_link', '/World/envs/env_0/Robot/base_link'])
    depth, count = P.penetration_from_sensors([s], 2, 'cpu')
    assert depth.tolist() == pytest.approx([0.001, 0.003])
    assert count.tolist() == [1, 2]


def test_saturation_refuses_missing_contacts(monkeypatch):
    monkeypatch.setattr(P.wp, 'to_torch', lambda x: x)
    s = sensor([4], [0], [0., 0., 0., 0.], ['/World/envs/env_0/Robot/base_link'])
    with pytest.raises(RuntimeError, match='容量'):
        P.penetration_from_sensors([s], 1, 'cpu')


def test_no_contacts_ignores_undefined_start(monkeypatch):
    monkeypatch.setattr(P.wp, 'to_torch', lambda x: x)
    s = sensor([0], [-1], [float('nan')], ['/World/envs/env_0/Robot/base_link'])
    depth, count = P.penetration_from_sensors([s], 1, 'cpu')
    assert depth.item() == 0 and count.item() == 0


def test_self_collision_guard_uses_real_separation_despite_large_aabb_overlap(monkeypatch):
    from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv
    monkeypatch.setattr(P.wp, 'to_torch', lambda x: x)
    env = object.__new__(FixedFruitGraspEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(sim=SimpleNamespace(device='cpu'))
    env.sim = SimpleNamespace(device='cpu')
    env.scene = SimpleNamespace(num_envs=2)
    env._all_env_ids = torch.tensor([0, 1])
    env._effective_self_collision_pairs = (('thumb_link4', 'index_link3'),)
    env._hand_obb_pair_clearance_for = lambda *args: torch.tensor([-0.040, -0.003])
    env._self_penetration_sensors = [sensor([1, 1], [0, 1], [-0.002, -0.012],
        ['/World/envs/env_0/Robot/thumb_link4', '/World/envs/env_1/Robot/thumb_link4'])]
    env.extras = {}
    clearance = env._self_collision_clearance()
    assert clearance.tolist() == pytest.approx([-0.002, -0.012])
    assert (clearance < -0.010).tolist() == [False, True]
    assert env.extras['physical_self_penetration']['points_count'].tolist() == [1, 1]
