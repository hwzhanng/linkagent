"""Bounded real-physics action-response probe, never used as a controller."""
import json
from pathlib import Path
import numpy as np
import torch
from .core import ACTIVE_JOINTS


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
