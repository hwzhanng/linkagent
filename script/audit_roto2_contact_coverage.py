"""Offline first-episode tactile coverage; descriptive metrics, not a success gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def longest_seconds(flags, dt):
    """Longest contiguous true interval per environment; gaps always break bouts."""
    current = np.zeros(flags.shape[1], dtype=np.int64)
    longest = current.copy()
    for row in flags:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest * dt


def audit(folder, seed):
    paths = [folder / f'{prefix}_seed{seed}.{suffix}' for prefix, suffix in
             [('evaluation', 'json'), ('evaluation', 'npz'),
              ('action_trace', 'json'), ('action_trace', 'npz')]]
    evaluation = json.loads(paths[0].read_text())
    meta = json.loads(paths[2].read_text())
    with np.load(paths[1]) as e, np.load(paths[3]) as tr:
        dt = float(meta['dt'])
        active = tr['active'].astype(bool) & ~tr['done'].astype(bool)
        assert np.array_equal(e['active'], tr['active'])
        assert np.max(np.abs(e['poses'][active] - tr['pose'][active])) < 1e-6
        count = tr['pad'].sum(-1)
        assert count.shape == active.shape
        poses = np.concatenate([e['initial_pose'][None], e['poses']])
        x, y, z, w = (poses[..., i] for i in range(3, 7))
        heading = np.arctan2(2 * (x*y + w*z), 1 - 2 * (y*y + z*z))
        diff = np.diff(heading, axis=0)
        degrees = np.rad2deg(np.arctan2(np.sin(diff), np.cos(diff))) * e['target_direction'][None]
        expected = np.array([r['directed_pose_turns'] * 360 for r in evaluation['per_episode']])
        assert np.max(np.abs((degrees * e['active']).sum(0) - expected)) < .01
        late = active & (((np.arange(len(active)) + 1) * dt)[:, None] > 10)
        n = active.shape[1]
        assert n == evaluation['episodes']
        windows = {}
        for label, mask in [('after1s', active & (((np.arange(len(active)) + 1)*dt)[:, None] > 1)),
                            ('after10s', late)]:
            denominator = int(mask.sum())
            window = dict(active_nonterminal_steps=denominator, episodes_reaching_window=int(mask.any(0).sum()),
                          mean_active_seconds_per_initial_episode=float(mask.sum()*dt/n), counts={})
            for threshold in [1, 2, 3]:
                covered = mask & (count >= threshold)
                longest = longest_seconds(covered, dt)
                seconds = covered.sum(0) * dt
                window['counts'][str(threshold)] = dict(
                    covered_steps=int(covered.sum()),
                    fraction_of_active_steps=float(covered.sum()/denominator) if denominator else None,
                    mean_contact_seconds_per_initial_episode=float(seconds.mean()),
                    episodes_with_any_contact=int(covered.any(0).sum()),
                    episodes_with_contiguous_contact_at_least_half_second=int((longest >= .5-1e-8).sum()),
                    longest_contact_seconds_per_episode=longest.tolist(),
                    net_degrees_with_contact_per_initial_episode=float(degrees[covered].sum()/n),
                    net_degrees_without_contact_per_initial_episode=float(degrees[mask & ~covered].sum()/n),
                )
            windows[label] = window
        return dict(seed=seed, episodes=n, windows=windows,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    definitions={'effective_contact': 'Existing measured pad flag, force >0.15 N; no threshold change.',
                                 'time': 'Post-step time, excludes auto-reset/terminal steps and stops at first episode.',
                                 'angle': 'Directed quaternion heading increments at contact-bearing post-step samples; correlation, not causal attribution.',
                                 'bout': 'Consecutive qualifying samples times dt; 0.5s is descriptive, not an acceptance threshold.',
                                 'denominators': 'Occupancy uses active nonterminal steps. Seconds/angle use all initial episodes, including failures.'},
                    stage2_acceptance_changed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    result = audit(args.folder, args.seed)
    output = args.folder / f'CONTACT_COVERAGE_seed{args.seed}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
