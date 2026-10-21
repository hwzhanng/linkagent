"""Bounded real-physics action-response probe, never used as a controller."""
import json
from pathlib import Path
import numpy as np
import torch
from .core import ACTIVE_JOINTS


@torch.no_grad()
def trace_policy(trainer, wrapped, checkpoint, output):
    """First-episode reward timing and actuator saturation diagnostics only."""
    agent = trainer.agent
    modules = torch.load(checkpoint, map_location=wrapped.device, weights_only=True)
    for key in ('encoder', 'policy'):
        getattr(agent, key).load_state_dict(modules[key]);getattr(agent, key).eval()
    if agent.encoder.state_preprocessor is not None:
        agent.encoder.state_preprocessor.load_state_dict(modules['state_preprocessor'])
    states, _ = wrapped.reset(hard=True);raw = wrapped._unwrapped
    pending = torch.ones(raw.num_envs,device=raw.device,dtype=torch.bool)
    data = {k:[] for k in ('joint','target','action','reward','active','done','pad','pose','forces','contact_local',
                           'tip_position_w','tip_quaternion_xyzw','approach_quality',
                           'peak_pressure_pa','terminated','truncated')}
    term_data = {}
    from .core import quat_inv_rotate
    limits=raw._active_limits.cpu().numpy().copy()
    grasp_reference_initial=raw._grasp_reference.cpu().numpy().copy()
    center_reference_initial=raw._center_reference_w.cpu().numpy().copy()
    fruit_axes_m=raw._axes_m.cpu().numpy().copy()
    for t in range(int(raw.max_episode_length+2)):
        if not pending.any(): break
        action,_,_=agent.policy.act(agent.encoder(states),deterministic=True)
        data['joint'].append(raw._robot.data.joint_pos.torch[:,raw._active_ids].cpu().numpy().copy())
        data['target'].append(raw._q_target.cpu().numpy().copy())
        data['action'].append(action.cpu().numpy().copy())
        data['active'].append(pending.cpu().numpy().copy())
        states,reward,term,trunc,info=wrapped.step(action)
        e=info['step'];done=(term|trunc).reshape(-1)
        data['reward'].append(reward.reshape(-1).cpu().numpy().copy())
        data['done'].append(done.cpu().numpy().copy())
        data['terminated'].append(term.reshape(-1).cpu().numpy().copy())
        data['truncated'].append(trunc.reshape(-1).cpu().numpy().copy())
        data['pad'].append(e['pad_contact'].cpu().numpy().copy())
        data['pose'].append(e['fruit_pose_native'].cpu().numpy().copy())
        data['forces'].append(raw._forces[:,:5].cpu().numpy().copy())
        ids=raw._contact_body_ids[:5]
        tip_pos=raw._robot.data.body_pos_w.torch[:,ids]
        tip_quat=raw._robot.data.body_quat_w.torch[:,ids]
        data['tip_position_w'].append(tip_pos.cpu().numpy().copy())
        data['tip_quaternion_xyzw'].append(tip_quat.cpu().numpy().copy())
        data['approach_quality'].append(e['pad_approach_quality'].cpu().numpy().copy())
        data['peak_pressure_pa'].append(e['peak_pressure_pa'].cpu().numpy().copy())
        local=quat_inv_rotate(raw._wxyz(raw._robot.data.body_quat_w.torch[:,ids]),
                              raw._contact_positions[:,:5]-raw._robot.data.body_pos_w.torch[:,ids])
        data['contact_local'].append(local.cpu().numpy().copy())
        for k,v in e['reward_terms'].items():term_data.setdefault(k,[]).append(v.cpu().numpy().copy())
        pending &= ~done
    data={k:np.array(v) for k,v in data.items()}
    rewards=data['reward']*data['active'];discount=.99**np.arange(len(rewards))
    near=np.minimum(data['joint']-limits[None,:,:,0],limits[None,:,:,1]-data['joint'])<.02
    near_fraction=(near*data['active'][...,None]).sum((0,1))/data['active'].sum()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),**data,limits=limits,
                        grasp_reference_initial=grasp_reference_initial,
                        center_reference_initial=center_reference_initial,
                        fruit_axes_m=fruit_axes_m,
                        **{'term_'+k:np.array(v) for k,v in term_data.items()})
    result={'checkpoint':str(checkpoint),'episodes':raw.num_envs,'dt':raw.step_dt,
            'discount':.99,'mean_undiscounted_return':float(rewards.sum(0).mean()),
            'mean_discounted_return':float((rewards*discount[:,None]).sum(0).mean()),
            'near_limit_fraction_by_joint':dict(zip(ACTIVE_JOINTS,near_fraction.tolist())),
            'policy_std':float(agent.policy.log_std_parameter.exp().mean()),
            'geometry_trace_note':'World-frame tip positions and native xyzw quaternions. Approach quality, peak pressure, pad flags and fruit pose come from pre-reset step snapshots. Exclude done steps for raw tip, force and contact-position buffers, which automatic resets may replace. Joint and target arrays are pre-action.'}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def probe_rotation_actions(trainer, wrapped, checkpoint, output):
    modules = torch.load(checkpoint, map_location=wrapped.device, weights_only=True)
    agent = trainer.agent
    for key in ("encoder", "policy"):
        getattr(agent, key).load_state_dict(modules[key]); getattr(agent, key).eval()
    if agent.encoder.state_preprocessor is not None:
        agent.encoder.state_preprocessor.load_state_dict(modules["state_preprocessor"])
    raw = wrapped._unwrapped
    n = raw.num_envs
    # 32 single-joint +/- pulses, plus an unperturbed policy control group.
    group = torch.arange(n, device=raw.device) % 33
    pulse = torch.zeros_like(raw._action)
    for j in range(32):
        pulse[group == j, j // 2] = .35 if j % 2 == 0 else -.35
    states, _ = wrapped.reset(hard=True)
    initial = raw._fruit.data.root_pose_w.torch.clone().cpu().numpy()
    pending = torch.ones(n, device=raw.device, dtype=torch.bool)
    rows = {k: [] for k in ("pose", "pad", "alive", "action", "delta", "pressure")}
    for t in range(140):
        action, _, _ = agent.policy.act(agent.encoder(states), deterministic=True)
        if 80 <= t < 100:
            action = action + pulse
        states, _, term, trunc, info = wrapped.step(action)
        e = info["step"]
        rows["pose"].append(e["fruit_pose_native"].cpu().numpy())
        rows["pad"].append(e["pad_contact"].cpu().numpy())
        rows["alive"].append(pending.cpu().numpy().copy())
        rows["action"].append(action.cpu().numpy())
        rows["delta"].append(e["pose_delta_rad"].cpu().numpy())
        rows["pressure"].append(e["peak_pressure_pa"].cpu().numpy())
        pending &= ~(term | trunc).reshape(-1)
    data = {k: np.array(v) for k, v in rows.items()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), **data, initial_pose=initial,
                        pulse=pulse.cpu().numpy(), group=group.cpu().numpy())
    results = []
    for j in range(33):
        ids = (group.cpu().numpy() == j) & data['alive'][79]
        if not ids.any():
            continue
        degrees = np.rad2deg((data['delta'][80:100, ids] * data['alive'][80:100, ids]).sum(0))
        pad = data['pad'][80:100, ids]
        full = pad[..., 0] & (pad[..., 1:].sum(-1) >= 2)
        results.append({'group': j, 'joint': ACTIVE_JOINTS[j//2] if j < 32 else 'control',
                        'pulse': (.35 if j%2 == 0 else -.35) if j < 32 else 0.,
                        'eligible_environments': int(ids.sum()),
                        'mean_pulse_world_z_degrees': float(degrees.mean()),
                        'mean_full_pad_support': float(full.mean()),
                        'mean_two_pad_support': float((pad.sum(-1) >= 2).mean()),
                        'alive_after_pulse': int(data['alive'][100, ids].sum())})
    result = {'checkpoint': str(checkpoint), 'hand_posture': raw.cfg.hand_posture,
              'probe_only': True, 'warmup_control_steps':80, 'pulse_control_steps':20,
              'results': results}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
