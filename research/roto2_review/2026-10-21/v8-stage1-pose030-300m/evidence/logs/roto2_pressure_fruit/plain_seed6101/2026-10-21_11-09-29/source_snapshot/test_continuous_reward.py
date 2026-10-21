"""Behavioral counterexamples for the v7 reward; no simulator dependency."""
import torch
from .reward_geometry import continuous_rotation_reward


def test_continuous_reward():
    full = torch.tensor([[1., 1., 1., 0., 0.]])
    two = torch.tensor([[1., 0., 1., 0., 0.]])
    def call(delta=0., pads=full, raw=None, q=None, goal=.30, valid=True, ready=True):
        z = torch.zeros(1)
        a = torch.zeros(1, 16)
        return continuous_rotation_reward(torch.tensor([delta]), torch.tensor([goal]),
            pads, pads, pads if raw is None else raw, z, z, a, a,
            torch.ones_like(pads), a if q is None else q, a,
            torch.tensor([valid]), torch.tensor([ready]), .05)
    assert call()[0] < 0                     # no holding bonus
    assert call(.0125)[0] > 0               # supported 0.25 rad/s is worthwhile
    assert call(.1, two)[0] < 0             # even capped two-pad motion cannot win
    assert call(.1, torch.zeros_like(full))[1]['rotation'] == 0
    assert call(.1, valid=False)[1]['rotation'] == 0
    assert call(.1, ready=False)[1]['rotation'] == 0
    assert call(-.01, torch.zeros_like(full))[1]['rotation'] < 0
    assert call(.01, two, raw=full)[0] < call(.01, two)[0]
    near_limit = torch.zeros(1, 16); near_limit[:, 6] = .25
    recovered = near_limit * .2
    assert call(q=near_limit)[0] < call(q=recovered)[0]
    assert torch.allclose(call(.01)[0], call(-.01, goal=-.3)[0])
    # An unproductive release/recontact has no event bonus.
    assert call(pads=two)[0] + call()[0] < 0
    # A complete forward/backward excursion cannot farm undiscounted motion.
    assert call(.01)[1]['rotation'] + call(-.01)[1]['rotation'] <= 1e-7
    # A useful cycle with 0.5s recovery is preferable to the same duration idle.
    productive = 20 * call(.0125)[0] + 10 * call(pads=two, q=near_limit)[0]
    idle = 30 * call()[0]
    assert productive > idle
    # Conversely, push once then keep two-pad contact is worse than restoring it.
    stuck = 20 * call(.0125)[0] + 100 * call(pads=two, q=near_limit)[0]
    restored = 20 * call(.0125)[0] + 10 * call(pads=two, q=near_limit)[0] + 90 * call()[0]
    assert restored > stuck


if __name__ == '__main__':
    test_continuous_reward()
    print('PASS: static, useful rotation, two-pad shortcut, outside-pad force, pose recovery, '
          'direction, invalid path, settling, release farming, oscillation and useful gait')
