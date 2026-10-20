"""Reward/sensor geometry only; no policy or PPO changes.

Pad frames are engineering priors fitted to the vendor distal sensor-cap STL,
not hardware calibration. Coordinates are in each distal link frame (metres).
"""
import torch
from .core import quat_rotate

REWARD_VERSION = 'pose_pad_v6'


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
              tilt, actions, previous_actions, previous_support, dt, approach_quality=None, stage="rotation"):
    directed = goal.sign() * delta
    thumb = pad_forces[:, 0] > .15
    others = (pad_forces[:, 1:] > .15).sum(-1) >= 2
    supported = thumb & others & (tilt < .5)
    # Only growth of the episode's actual orientation high-water mark earns
    # progress. Backtracking cannot repeatedly farm the same orientation arc.
    # Negative direction is penalized even when contact is deliberately lost.
    positive = new_progress * supported.float()
    seek = .5 * pad_quality[:, 0] + .5 * pad_quality[:, 1:].topk(2, dim=-1).values.mean(-1)
    approach = seek if approach_quality is None else (.5 * approach_quality[:,0] + .5 * approach_quality[:,1:].topk(2,dim=-1).values.mean(-1))
    terms = {
        'rotation': 12.0 * positive,
        'reverse': 2.0 * directed.clamp(max=0),
        'pad_guidance': .15 * (seek - 1),
        'pad_approach': .15 * (approach - 1),
        'missing_pad_support': -.04 * (~supported).float(),
        'lost_pad_support': -.08 * (previous_support & ~supported).float(),
        'stall': -.03 * (new_progress < .001).float(),
        'centering': -.04 * (translation/.05).square().clamp(max=9),
        'tilt': -.06 * tilt.square(),
        'action_smoothness': -.006 * (actions-previous_actions).square().mean(-1),
        'time': torch.full_like(delta, -.025),
    }
    if stage == "contact":
        # Contact-acquisition curriculum. Holding real active pads is rewarded
        # here; deployment/rotation acceptance still requires genuine turns.
        coverage = .5*thumb.float() + .5*((pad_forces[:,1:]>.15).sum(-1).float()/2).clamp(max=1)
        terms["pad_guidance"] = .15 * seek
        terms["active_pad_coverage"] = .25 * coverage
        terms["complete_pad_grasp"] = .25 * supported.float()
        terms["pad_approach"] = .10 * (approach - 1)
        terms["missing_pad_support"] = -.02 * (~supported).float()
        terms["stall"] = torch.zeros_like(delta)
        terms["reverse"] = .2 * directed.clamp(max=0)
        terms["time"] = torch.full_like(delta, -.05)
    return torch.stack(list(terms.values()), -1).sum(-1), terms, supported


def pad_approach_quality(tip_pos, tip_quat, fruit_pos, fruit_quat, axes):
    """Dense reward-only approach guidance, including before contact exists.

    Uses simulator geometry exclusively in the reward; never actor inputs.
    Desired pad front points toward fruit center and stays near its surface.
    """
    from .core import quat_inv_rotate
    center, _, _, normal = pad_frames(tip_pos.device, tip_pos.dtype)
    center_w = tip_pos + quat_rotate(tip_quat, center.expand_as(tip_pos))
    normal_w = quat_rotate(tip_quat, normal.expand_as(tip_pos))
    to_fruit = fruit_pos[:,None] - center_w
    distance = torch.linalg.vector_norm(to_fruit, dim=-1).clamp_min(1e-6)
    direction = to_fruit / distance[...,None]
    alignment = (normal_w*direction).sum(-1).clamp(-1,1)
    direction_local = quat_inv_rotate(fruit_quat[:,None].expand(-1,5,-1), direction)
    radius = ((direction_local / axes[:,None]).square().sum(-1)).rsqrt()
    gap = distance - radius
    # Linear geometric cost retains guidance when pads initially point away
    # or lie several centimetres from the fruit (the exponential was flat).
    target_error = torch.linalg.vector_norm(center_w + normal_w*radius[...,None] - fruit_pos[:,None],dim=-1)
    quality = 1 - (target_error/.06).clamp(max=3)
    return quality


def valid_rotation_step(previous_upright, current_tilt, delta):
    """Reject heading singularities/large pose jumps as rewarded progress."""
    return previous_upright & (current_tilt < .5) & (delta.abs() < .5) & torch.isfinite(delta)


def staged_rotation_reward(delta, goal, pad_forces, pad_quality, translation,
                           tilt, actions, previous_actions, approach_quality,
                           support_average, reward_angle, milestones_paid,
                           eligible_path, ready, dt, stage):
    """Reward-only curriculum: tactile driving, regrasp, and signed pose motion.

    Positive/negative motion use equal per-radian weights. Thus returning over
    an already traversed arc provides dense feedback, but oscillations have
    nonpositive total motion reward. No positive reward without real contacts.
    Milestone rewards require restored thumb+two-pad support and pay only once.
    The strict independent full-turn acceptance criterion is unchanged.
    """
    pads = pad_forces > .15
    full = pads[:, 0] & (pads[:, 1:].sum(-1) >= 2) & (tilt < .5)
    driving = (pads.sum(-1) >= 2) & (tilt < .5) & (translation < .06)
    directed = goal.sign() * delta
    valid = ready & eligible_path
    # Cap positive credit at twice commanded speed; never cap reverse cost.
    positive = torch.minimum(directed.clamp_min(0), 2*goal.abs()*dt)
    motion = 20 * (positive * (valid & driving).float()
                   + directed.clamp(max=0) * ready.float())
    seek = .5*pad_quality[:, 0] + .5*pad_quality[:, 1:].topk(2,dim=-1).values.mean(-1)
    approach = .5*approach_quality[:, 0] + .5*approach_quality[:, 1:].topk(2,dim=-1).values.mean(-1)
    target_support = .50 if stage == 'regrasp' else .70
    degrees = [5., 10., 20., 40., 60., 90.] if stage == 'regrasp' else [90., 180., 270., 360.]
    thresholds = delta.new_tensor(degrees) * (torch.pi / 180)
    attained = (reward_angle[:, None] >= thresholds).sum(-1)
    new_count = (attained - milestones_paid).clamp_min(0)
    pay = (valid & full & (translation < .06)).long() * new_count
    terms = {
        'rotation': motion,
        'regrasp_milestone': 2.0*pay.float(),
        'pad_guidance': .04*(seek-1),
        'pad_approach': .04*(approach-1),
        'missing_driving_contacts': -.10*(~driving).float(),
        'support_duty_deficit': -.12*(target_support-support_average).clamp_min(0),
        'centering': -.04*(translation/.05).square().clamp(max=9),
        'tilt': -.10*tilt.square(),
        'action_smoothness': -.006*(actions-previous_actions).square().mean(-1),
        'time': torch.full_like(delta, -.025),
        'invalid_rotation_path': -.20*(ready & ~eligible_path).float(),
    }
    return torch.stack(list(terms.values()), -1).sum(-1), terms, full, milestones_paid+pay
