"""Counterexample tests for dense rotation and one-time regrasp credit."""
import math
import torch
from .reward_geometry import staged_rotation_reward


def test_curriculum_invariants():
    full = torch.tensor([[1., 1., 1., 0., 0.]])
    two = torch.tensor([[1., 1., 0., 0., 0.]])
    none = torch.zeros_like(full)
    def call(delta=0., pads=full, angle=0., paid=0, valid=True, ready=True):
        z = torch.zeros(1); a = torch.zeros(1,16)
        return staged_rotation_reward(torch.tensor([delta]), torch.tensor([.25]),
            pads, pads, z, z, a, a, torch.ones_like(pads), torch.ones(1)*.7,
            torch.tensor([angle]), torch.tensor([paid]), torch.tensor([valid]),
            torch.tensor([ready]), .05, 'regrasp')
    assert call()[0].item() < 0  # stationary holding cannot earn positive return
    assert call(.01, none)[1]['rotation'].item() == 0
    assert call(.01, two)[1]['rotation'].item() > 0  # a finger may reposition
    forward = call(.01, two)[1]['rotation']
    reverse = call(-.01, none)[1]['rotation']
    assert (forward+reverse).item() <= 1e-7  # release/backtrack cannot farm motion
    assert (call(.1, full)[1]['rotation']+call(-.1, none)[1]['rotation']).item() < 0
    assert call(.01, full, valid=False)[1]['rotation'].item() == 0
    assert call(.01, full, ready=False)[1]['rotation'].item() == 0
    angle = math.radians(11)
    assert call(.01, two, angle)[1]['regrasp_milestone'].item() == 0
    first = call(0., full, angle)
    assert first[1]['regrasp_milestone'].item() > 0
    paid = int(first[3][0])
    assert call(0., full, angle, paid)[1]['regrasp_milestone'].item() == 0
    assert call(0., full, 0., paid)[1]['regrasp_milestone'].item() == 0
    assert call(0., full, angle, paid)[1]['regrasp_milestone'].item() == 0
    assert call(0., full, angle, valid=False)[1]['regrasp_milestone'].item() == 0
    assert call(0., full, angle, ready=False)[1]['regrasp_milestone'].item() == 0


if __name__ == '__main__':
    test_curriculum_invariants()
    print('PASS: no-contact, stationary, reversal, oscillation, settling, invalid pose, and repeated milestone exploits')
