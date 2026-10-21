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

def test_approach_contract():
    z = torch.zeros(1); q = torch.zeros(1, 16)
    yes = torch.ones(1, dtype=torch.bool)
    limits = torch.stack((q, q + 1), -1)
    def run(pads, quality, delta=.02, displacement=.03, weight=.025):
        return baseline_reward(z + delta, z + .4, torch.tensor([pads]),
            z + displacement, z, z, z, q, q, limits, q, q, q, q,
            yes, yes, .05, pose_weight=.3,
            approach_quality=torch.tensor([quality]), approach_weight=weight,
            approach_pad_count=4)
    full = run([1., 1., 1., 1., 0.], [1., 1., 1., 1., -2.])
    for released in range(4):
        pads = [1., 1., 1., 1., 0.]; pads[released] = 0.
        quality = [1., 1., 1., 1., -2.]; quality[released] = -2.
        moving, terms, support = run(pads, quality)
        assert support.all()  # Three contacts suffice even outside stable-center criteria.
        assert torch.equal(terms['rotation'], full[1]['rotation'])
        assert torch.allclose(terms['pad_approach_recovery'], z - .01875)
        stationary = run(pads, quality, delta=0.)[0]
        assert stationary <= 0 and moving > stationary
    # Even two-pad release remains legal under the existing stable-object path.
    two = run([0., 1., 1., 0., 0.], [-2., 1., 1., -2., -2.], displacement=0.)
    assert two[2].all() and two[1]['rotation'] > 0
    # Off means no dependence on the candidate geometry, not merely a small cost.
    off = run([1., 1., 1., 1., 0.], [float('nan')] * 5, weight=0.)
    assert torch.isfinite(off[0]).all() and 'pad_approach_recovery' not in off[1]
    print('PASS: soft fourth-pad recovery permits release, preserves support and rotation, off ignores geometry')

if __name__=='__main__':
    test_contract()
    test_approach_contract()
