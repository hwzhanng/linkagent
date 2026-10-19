"""检验轮换手指、漏步、故障及漂移不能伪造连续三指成功。"""
from types import SimpleNamespace
import pytest
import torch
from research.firm_grasp_v1.physical_guard import CombinationHoldGuard


def state():
    return SimpleNamespace(fruit_in_hand_m=torch.zeros((1,3)),
        released=torch.ones(1,dtype=torch.bool), relative_speed_m_s=torch.zeros(1))


def observe(g, s, k, combo=0, fault=False, support=False):
    q=torch.zeros((1,6),dtype=torch.bool);q[0,combo]=True
    g.observe(state=s,qualified=q,hard_fault_any=torch.tensor([fault]),
              support_zero=torch.tensor([not support]),counter=k,dt=1/120)
    if k%6==0:g.complete_policy_interval()


def test_finger_replacement_cannot_accumulate_window():
    g=CombinationHoldGuard(1,device='cpu',initial_counter=0)
    for k in range(1,241):observe(g,state(),k,combo=k%2)
    assert int(g.peak.max())==1


def test_fixed_combination_requires_every_substep_and_fault_resets():
    g=CombinationHoldGuard(1,device='cpu',initial_counter=0)
    for k in range(1,121):observe(g,state(),k)
    assert int(g.streak[0,0])==120
    observe(g,state(),121,fault=True)
    assert int(g.streak.max())==0


def test_drift_and_speed_cannot_pass_even_with_three_contact_flags():
    g=CombinationHoldGuard(1,device='cpu',initial_counter=0)
    observe(g,state(),1)
    s=state();s.fruit_in_hand_m[0,0]=.02
    observe(g,s,2)
    assert int(g.streak.max())==0
    s=state();s.relative_speed_m_s[0]=.051
    observe(g,s,3)
    assert int(g.streak.max())==0


def test_missing_substep_fails_closed():
    g=CombinationHoldGuard(1,device='cpu',initial_counter=0)
    with pytest.raises(RuntimeError):observe(g,state(),2)


def test_release_flag_does_not_override_nonzero_support():
    g=CombinationHoldGuard(1,device='cpu',initial_counter=0)
    for k in range(1,121):observe(g,state(),k,support=True)
    assert int(g.peak.max())==0


def test_reset_after_inference_sampling_preserves_other_env_state():
    g=CombinationHoldGuard(2,device='cpu',initial_counter=0)
    s=SimpleNamespace(fruit_in_hand_m=torch.zeros((2,3)),
        released=torch.ones(2,dtype=torch.bool),relative_speed_m_s=torch.zeros(2))
    g.observe(state=s,qualified=torch.ones((2,6),dtype=torch.bool),
        hard_fault_any=torch.zeros(2,dtype=torch.bool),support_zero=torch.ones(2,dtype=torch.bool),counter=1,dt=1/120)
    g.reset(torch.tensor([0]))
    assert int(g.streak[0].max())==0 and int(g.streak[1].max())==1
