"""v8 task-only baseline. No policy, actuator, observation, or PPO changes.

The capped velocity is reconstructed from actual material-axis pose increments.
No thumb privilege; stability may support a temporary pad release. Old v7
reward and acceptance remain available and must be reported as comparison.
"""
import torch


def support_masks(pad_forces, displacement, speed, height_drop, tilt):
    three_pad = (pad_forces > .15).sum(-1) >= 3
    stable = ((displacement < .025) & (speed < .12) & (height_drop < .015)
              & (tilt < .5) & torch.isfinite(speed) & torch.isfinite(displacement))
    support = (three_pad | stable) & (tilt < .5)
    return support, three_pad, stable


def baseline_reward(delta, goal, pad_forces, displacement, speed, height_drop,
                    tilt, joint_pos, reference_pos, joint_limits, joint_velocity,
                    applied_torque, actions, previous_actions, valid, ready, dt,
                    pose_weight=.01, approach_quality=None, approach_weight=0.,
                    approach_pad_count=4, motion_weight=1., centering_weight=.02):
    support, three_pad, stable = support_masks(pad_forces, displacement, speed, height_drop, tilt)
    directed_speed = goal.sign() * delta / dt
    # Positive cap only: a fast reverse cannot erase its cost through clipping.
    forward = torch.minimum(directed_speed.clamp_min(0), 2*goal.abs())
    motion = forward * (valid & ready & support).float() + directed_speed.clamp(max=0)*ready.float()
    normalized_pose = (joint_pos-reference_pos)/(joint_limits[..., 1]-joint_limits[..., 0]).clamp_min(.01)
    terms = {
        'rotation': motion_weight * motion,
        'pose_deviation_normalized': -pose_weight*normalized_pose.square().mean(-1),
        'mechanical_power': -.002*(applied_torque*joint_velocity).abs().sum(-1),
        'pad_bonus': .005*(pad_forces > .15).float().mean(-1),
        'linear_speed': -.05*speed,
        'centering': -centering_weight*(displacement/.05).square().clamp(max=9),
        'tilt': -.05*tilt.square(),
        'action_smoothness': -.001*(actions-previous_actions).square().mean(-1),
        'time': torch.full_like(delta, -.005),
        'invalid_rotation_path': -.20*(ready & ~valid).float(),
    }
    if approach_weight:
        if approach_quality is None:
            raise ValueError("Nonzero approach weight requires actual pad geometry")
        # A soft recovery cost, never a support/action gate. Four near pads
        # encourage redundancy; releasing one still leaves three-pad support.
        closest = approach_quality.topk(approach_pad_count, dim=-1).values
        terms['pad_approach_recovery'] = -approach_weight * (1 - closest).mean(-1)
    return sum(terms.values()), terms, support
