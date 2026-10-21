"""Meaningful task-contract checks before any new PPO run."""
import torch
from .reward_baseline import baseline_reward, support_masks


def test_contract():
    z = torch.zeros(1); q=torch.zeros(1,16); yes=torch.ones(1,dtype=torch.bool)
    limits=torch.stack((q,q+1),-1)
    def run(delta=.02,pads=None,speed=0.,height=0.,displacement=0.,valid=True,qp=None,lim=None):
        return baseline_reward(torch.tensor([delta],dtype=torch.float32),z+.4,
            torch.tensor([[0.,1.,1.,1.,0.]]) if pads is None else torch.tensor([pads]),
            z+displacement,z+speed,z+height,z,q if qp is None else qp,q,
            limits if lim is None else lim,q,q,q,q,yes & valid,yes,.05)
    # No thumb is needed; stable release with no pad remains eligible.
    assert run()[1]['rotation'] > 0
    assert run(pads=[0.,0.,0.,0.,0.])[1]['rotation'] > 0
    # Airborne/moving, displaced, or falling no-contact objects cannot earn motion.
    for kwargs in ({'speed':.2},{'height':.02},{'displacement':.03}):
        assert run(pads=[0.,0.,0.,0.,0.],**kwargs)[1]['rotation']==0
    assert run(valid=False)[1]['rotation']==0
    # A forward/backward loop has no positive undiscounted motion reward,
    # including fast reverse that would exploit symmetric velocity clipping.
    assert 4*run(.02)[1]['rotation']+run(-.08)[1]['rotation']<=1e-6
    assert run()[0] > run(0)[0]
    assert run(0,pads=[1.,1.,1.,1.,1.])[0]<=0
    # Equal fractions of different joint travel incur equal pose cost.
    small=q.clone();small[0,0]=.1
    large=q.clone();large[0,1]=.5
    lim=limits.clone();lim[0,0,1]=.2
    assert torch.allclose(run(qp=small,lim=lim)[1]['pose_deviation_normalized'],
                          run(qp=large,lim=lim)[1]['pose_deviation_normalized'])
    print('PASS: no-thumb support, stable release, airborne rejection, motion dominance, reverse loop, normalized joints')

if __name__=='__main__':test_contract()
