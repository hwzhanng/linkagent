"""Comparable rotation, tactile-contact and pressure metrics from first episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from simulation.roto2_pressure.reward_geometry import pad_contact
from simulation.roto2_pressure.reward_tactile import tactile_constraint_costs, contact_motion_discount


def summarize(folder, seed):
    evaluation = json.loads((folder / f'evaluation_seed{seed}.json').read_text())
    meta = json.loads((folder / f'action_trace_seed{seed}.json').read_text())
    arrays = np.load(folder / f'action_trace_seed{seed}.npz')
    active = arrays['active'] & ~arrays['done']
    if 'distal_forces_n' in arrays:
        forces = torch.from_numpy(arrays['distal_forces_n'])
        pads = torch.from_numpy(arrays['pad_forces_n'])
        quality = torch.from_numpy(arrays['pad_quality'])
        source = 'pre-reset measured step snapshots'
    else:
        forces = torch.from_numpy(arrays['forces'])
        _, pads, quality, _ = pad_contact(torch.from_numpy(arrays['contact_local']), forces)
        source = 'reconstructed from measured local contacts; terminal/reset steps excluded'
    assert np.array_equal((pads.numpy() > .15)[active], arrays['pad'][active])
    valid_quality = quality.clamp(0, 1) * (pads > .15)
    top3 = valid_quality.topk(3, dim=-1).values.mean(-1).numpy()
    outside_force = (forces - pads).clamp_min(0).sum(-1).numpy()
    pressure = arrays['peak_pressure_pa']
    thresholds = meta.get('failure_thresholds', {})
    pressure_limit = thresholds.get('maximum_pressure_proxy_pa', 18000.)
    force_limit = thresholds.get('maximum_contact_force_n', 8.)
    dt = meta['dt']
    times = (np.arange(len(active)) + 1) * dt
    windows = {}
    for name, cutoff in [('after_settling', 1.), ('after10s', 10.)]:
        mask = active & (times[:, None] > cutoff)
        n = int(mask.sum())
        windows[name] = {'active_nonterminal_environment_steps': n}
        if n:
            windows[name].update(
                mean_valid_pad_count=float(arrays['pad'].sum(-1)[mask].mean()),
                three_pad_contact_fraction=float((arrays['pad'].sum(-1)[mask] >= 3).mean()),
                mean_top3_measured_pad_quality=float(top3[mask].mean()),
                mean_total_offpad_distal_force_n=float(outside_force[mask].mean()),
                mean_peak_pressure_pa=float(pressure[mask].mean()),
                p95_peak_pressure_pa=float(np.quantile(pressure[mask], .95)),
                pressure_above_half_limit_fraction=float((pressure[mask] > pressure_limit / 2).mean()),
                pressure_above_limit_fraction=float((pressure[mask] > pressure_limit).mean()))
    terms = tactile_constraint_costs(
        quality, pads, forces, torch.from_numpy(pressure),
        quality_weight=meta.get('baseline_pad_quality_weight', 0.),
        offpad_weight=meta.get('baseline_offpad_weight', 0.),
        pressure_weight=meta.get('baseline_pressure_weight', 0.),
        pressure_limit_pa=pressure_limit, force_limit_n=force_limit)
    errors = {}
    floor = meta.get('baseline_contact_motion_floor', 1.)
    if floor < 1.:
        terms['tactile_motion_discount'] = contact_motion_discount(
            torch.from_numpy(arrays['term_rotation']), quality, pads, floor=floor)
    for key, value in terms.items():
        error = float(np.abs(value.numpy() - arrays['term_' + key])[active].max())
        assert error < 1e-5, (key, error)
        errors[key] = error
    return {
        'seed': seed, 'episodes': evaluation['episodes'], 'checkpoint': evaluation['checkpoint'],
        'safe_full_turns': sum(r['success'] for r in evaluation['per_episode']),
        'physical_failures': sum(r['failed'] for r in evaluation['per_episode']),
        'mean_total_degrees': evaluation['mean_directed_rotations'] * 360,
        'mean_after10s_degrees': evaluation['mean_after_10s_directed_turns'] * 360,
        'windows': windows, 'contact_source': source, 'tactile_reward_reconstruction_errors': errors,
        'baseline_contact_motion_floor': floor,
        'mean_net_rotation_reward_per_initial_episode': float((
            (arrays['term_rotation'] + arrays['term_tactile_motion_discount']
             if 'term_tactile_motion_discount' in arrays else arrays['term_rotation'])
            * arrays['active']).sum(0).mean()),
        'scope': 'First episodes; rotation includes failures in original episode denominator. Contact/pressure uses active nonterminal time-step denominators, excludes initial settling. Distal-link force only, not all palm/proximal contacts. Pressure is a simulation proxy, not calibrated tissue stress.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=5101)
    args = parser.parse_args()
    result = summarize(args.folder, args.seed)
    (args.folder / f'TRADEOFF_METRICS_seed{args.seed}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
