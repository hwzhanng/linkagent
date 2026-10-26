"""Causal, current-state simulator features for value diagnostics only.

No sensor reads, RNG draws, physics writes or actor-observation mutation.
Angular progress uses material-pose differences, not the unreliable reported
angular velocity. Values are scaled using fixed physical units (not test data).
"""
import torch

FEATURE_GROUPS = (
    ('center_error', 3), ('quaternion_xyzw', 4), ('linear_velocity', 3),
    ('previous_pose_rate', 1), ('target_lead', 16), ('grasp_reference_error', 16),
    ('body_contact_forces', 6), ('pad_contact_forces', 5), ('dense_pressure', 2),
    ('axes', 3), ('mass', 1), ('elapsed_fraction', 1), ('reward_path_valid', 1),
    ('overpressure_streak', 1), ('previous_upright', 1),
)
FEATURE_NAMES = [f'{name}_{i}' for name,n in FEATURE_GROUPS for i in range(n)]
CRITIC_FEATURE_DIM = len(FEATURE_NAMES)
assert CRITIC_FEATURE_DIM == 64
BOOKKEEPING_START = 60

@torch.no_grad()
def critic_features(raw):
    pose=raw._fruit.data.root_pose_w.torch
    quat=pose[:,3:7]
    if not raw._native_xyzw:
        quat=quat[:,[1,2,3,0]]
    q=raw._robot.data.joint_pos.torch[:,raw._active_ids]
    limits=raw._active_limits
    extent=(limits[...,1]-limits[...,0]).clamp_min(.01)
    values=(
        (pose[:,:3]-raw._center_reference_w)/.1,
        quat,
        raw._fruit.data.root_lin_vel_w.torch/.5,
        (raw._pose_delta/raw.step_dt)[:,None],
        (raw._q_target-q)/raw.cfg.maximum_joint_target_lead_rad,
        (raw._grasp_reference-q)/extent,
        raw._forces/8.,raw._pad_forces/8.,
        torch.stack((raw._dense.amax(-1),raw._dense.mean(-1)),-1)/18000.,
        raw._axes_m/.032,
        raw._fruit.data.body_mass.torch.reshape(raw.num_envs,-1).sum(-1,keepdim=True)/.15,
        (raw.episode_length_buf.float()/raw.max_episode_length)[:,None],
        raw._reward_path_valid.float()[:,None],
        (raw._overpressure_streak.float()/raw.cfg.overpressure_termination_steps)[:,None],
        raw._previous_upright.float()[:,None],
    )
    result=torch.cat(values,-1).detach().clone()
    assert result.shape==(raw.num_envs,CRITIC_FEATURE_DIM)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite current-state critic diagnostic feature')
    return result.clamp(-20,20)
