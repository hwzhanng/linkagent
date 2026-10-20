"""Reward/sensor geometry only; no policy or PPO changes.

Pad frames are engineering priors fitted to the vendor distal sensor-cap STL,
not hardware calibration. Coordinates are in each distal link frame (metres).
"""
import torch
from .core import quat_rotate

REWARD_VERSION = 'pose_pad_v2'


def pose_heading(quat_wxyz):
    # Heading of the material X axis. A full revolution must actually change
    # orientation; spinning velocity reports and roll/pitch loops are not turns.
    x = torch.zeros_like(quat_wxyz[..., 1:]); x[..., 0] = 1
    axis = quat_rotate(quat_wxyz, x)
    return torch.atan2(axis[..., 1], axis[..., 0])


def heading_delta(previous, current):
    return torch.atan2(torch.sin(current - previous), torch.cos(current - previous))


def pad_frames(device, dtype=torch.float32):
    normal = torch.tensor([[0., .49, .87]] + [[.86, 0., .51]] * 4, device=device, dtype=dtype)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    back = torch.tensor([[-.007996, .024621, .000033]] + [[-.010314, .005054, .021898]] * 4,
                        device=device, dtype=dtype)
    center = back + .0035 * normal
    u = torch.tensor([[1., 0., 0.]] + [[0., 1., 0.]] * 4, device=device, dtype=dtype)
    v = torch.cross(normal, u, dim=-1)
    return center, u, v, normal


def pad_contact(local_points, normal_forces):
    center, u, v, normal = pad_frames(local_points.device, local_points.dtype)
    relative = local_points - center
    xy = torch.stack(((relative*u).sum(-1), (relative*v).sum(-1)), -1)
    depth = (relative*normal).sum(-1)
    # Never clamp a point outside the active rectangle into a taxel.
    inside = (xy[..., 0].abs() <= .0048) & (xy[..., 1].abs() <= .0072)
    front = (depth >= -.0015) & (depth <= .0040)
    valid = inside & front & (normal_forces > .025)
    # Smooth bounded guidance toward the active face; no invented contact force.
    outside_xy = torch.relu(xy.abs() - xy.new_tensor([.0048, .0072]))
    outside_z = torch.relu(-.0015 - depth) + torch.relu(depth - .004)
    distance2 = outside_xy.square().sum(-1) + outside_z.square()
    quality = torch.exp(-distance2 / (2*.004**2)) * (normal_forces / .3).clamp(0, 1)
    return xy, normal_forces * valid, quality, valid


def reward_v2(delta, goal, pad_forces, pad_quality, new_progress, translation,
              tilt, actions, previous_actions, previous_support, dt):
    directed = goal.sign() * delta
    thumb = pad_forces[:, 0] > .15
    others = (pad_forces[:, 1:] > .15).sum(-1) >= 2
    supported = thumb & others & (tilt < .5)
    # Only growth of the episode's actual orientation high-water mark earns
    # progress. Backtracking cannot repeatedly farm the same orientation arc.
    # Negative direction is penalized even when contact is deliberately lost.
    positive = new_progress * supported.float()
    seek = .5 * pad_quality[:, 0] + .5 * pad_quality[:, 1:].topk(2, dim=-1).values.mean(-1)
    terms = {
        'rotation': 12.0 * positive,
        'reverse': 2.0 * directed.clamp(max=0),
        'pad_guidance': .04 * seek,
        'missing_pad_support': -.04 * (~supported).float(),
        'lost_pad_support': -.08 * (previous_support & ~supported).float(),
        'stall': -.03 * (new_progress < .001).float(),
        'centering': -.04 * (translation/.05).square().clamp(max=9),
        'tilt': -.06 * tilt.square(),
        'action_smoothness': -.006 * (actions-previous_actions).square().mean(-1),
        'time': torch.full_like(delta, -.025),
    }
    return torch.stack(list(terms.values()), -1).sum(-1), terms, supported
