"""Anti-exploit regression tests, CPU-only and independent of Isaac Sim."""
import math
import torch
from .reward_geometry import pose_heading, heading_delta, pad_contact, pad_frames, reward_v2
from .core import fingertip_taxels


def test_pose_rotation():
    angles = torch.linspace(0, 4*math.pi, 1001)
    q = torch.zeros(1001, 4); q[:,0] = (angles/2).cos(); q[:,3] = (angles/2).sin()
    h = pose_heading(q)
    delta = heading_delta(h[:-1], h[1:])
    assert abs(delta.sum().item()-4*math.pi)<1e-5
    assert torch.allclose(pose_heading(-q), h, atol=1e-6)
    assert heading_delta(torch.tensor([3.13]), torch.tensor([-3.13])).item() > 0
    # Arbitrary reported angular velocity cannot alter a fixed material pose.
    assert heading_delta(h[:1], h[:1]).item() == 0


def test_contacts():
    c,u,v,n=pad_frames('cpu');f=torch.ones(1,5)
    xy,force,quality,valid=pad_contact(c[None],f)
    assert valid.all() and torch.allclose(force,f)
    # Real force on the side/back of a sensor housing must yield no taxels.
    for displaced in (c+.015*u, c-.01*n, c+.02*v):
        xy,force,_,valid=pad_contact(displaced[None],f)
        assert not valid.any()
        assert fingertip_taxels(force,xy).count_nonzero()==0
    assert pad_contact(c[None],torch.zeros_like(f))[1].count_nonzero()==0


def test_reward_loopholes():
    z=torch.zeros(1);actions=torch.zeros(1,16);f=torch.ones(1,5);quality=f
    def reward(delta,progress,forces=f):
        return reward_v2(torch.tensor([delta]),torch.ones(1)*.6,forces,quality,
                         torch.tensor([progress]),z,z,actions,actions,torch.ones(1,dtype=torch.bool),.05)
    r,terms,_=reward(.03,.03)
    assert terms['rotation'].item()>0
    assert reward(0.,0.)[0].item()<0  # safe holding is not a successful task
    assert reward(.03,.03,torch.zeros_like(f))[1]['rotation'].item()==0
    assert reward(-.03,0.,torch.zeros_like(f))[1]['reverse'].item()<0
    assert reward(.03,0.)[1]['rotation'].item()==0 # revisiting a previous peak
    # Oscillating .03 -> 0 -> .03 cannot farm repeated progress.
    assert reward(.03,0.)[0].item()+reward(-.03,0.)[0].item()<0

def test_approach_guidance():
    from .reward_geometry import pad_approach_quality
    c,u,v,n = pad_frames('cpu')
    q=torch.zeros(1,5,4);q[...,0]=1
    fq=torch.tensor([[1.,0,0,0]])
    fruit=torch.zeros(1,3);axes=torch.ones(1,3)*.03
    ideal=(-.03*n-c)[None]
    near=pad_approach_quality(ideal,q,fruit,fq,axes)
    distant=pad_approach_quality(ideal-.05*n[None],q,fruit,fq,axes)
    backside=pad_approach_quality((.03*n-c)[None],q,fruit,fq,axes)
    assert torch.allclose(near,torch.ones_like(near),atol=1e-5)
    assert (near>distant).all() and (near>backside*10).all()

def test_contact_curriculum():
    z=torch.zeros(1);a=torch.zeros(1,16);f=torch.ones(1,5)
    def call(forces):
        return reward_v2(z,torch.ones(1)*.6,forces,forces,z,z,z,a,a,
                         torch.zeros(1,dtype=torch.bool),.05,torch.ones_like(f),"contact")
    none=call(torch.zeros_like(f));one=call(torch.tensor([[1.,0,0,0,0]]));full=call(f)
    assert full[0].item()>one[0].item()>none[0].item()
    assert none[1]['rotation'].item()==0
    # A contact curriculum may reward a stationary valid grasp, but the full
    # rotation objective must not (covered independently above).
    assert full[0].item()>0

if __name__=='__main__':
    test_pose_rotation();test_contacts();test_reward_loopholes();test_approach_guidance();test_contact_curriculum()
    print('PASS: pose winding/sign, side/back rejection, stall/release/reversal/oscillation exploits')
