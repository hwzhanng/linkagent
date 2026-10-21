"""Reconstruct failure predicates from pre-reset first-episode measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(trace, thresholds, dt):
    active = trace['active'].astype(bool)
    terminated = trace['terminated'].astype(bool)
    truncated = trace['truncated'].astype(bool)
    done = trace['done'].astype(bool)
    assert np.array_equal(done, terminated | truncated)
    assert np.all(active[0])
    assert np.array_equal(active[1:], active[:-1] & ~done[:-1])
    assert np.all((active & done).sum(0) == 1), 'Incomplete first episodes'
    position = trace['pose'][..., :3]
    displacement = np.linalg.norm(position - trace['center_reference_initial'][None], axis=-1)
    pressure = trace['peak_pressure_pa']
    force = trace['maximum_distal_contact_force_n']
    assert np.isfinite(pressure[active]).all() and np.isfinite(force[active]).all()
    above = pressure > thresholds['maximum_pressure_proxy_pa']
    streak = np.zeros(active.shape[1], dtype=int)
    sustained = np.zeros_like(active)
    for t in range(len(active)):
        streak = np.where(active[t] & above[t], streak + 1, 0)
        sustained[t] = streak >= thresholds['overpressure_termination_steps']
    causes = {
        'center_displacement': displacement > thresholds['maximum_fruit_center_displacement_m'],
        'distal_contact_force': force > thresholds['maximum_contact_force_n'],
        'sustained_overpressure': sustained,
        'world_height': position[..., 2] < thresholds['minimum_world_z_m'],
        'nonfinite_position': ~np.isfinite(position).all(-1),
    }
    reconstructed = np.logical_or.reduce(list(causes.values()))
    mismatch = active & (reconstructed != terminated)
    assert not mismatch.any(), f'Failure reconstruction mismatch: {np.argwhere(mismatch).tolist()}'
    rows = []
    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    for env in range(active.shape[1]):
        t = int(np.flatnonzero(active[:, env] & done[:, env])[0])
        rows.append({
            'env': env, 'step': t + 1, 'time_s': (t + 1) * dt,
            'terminated': bool(terminated[t, env]), 'truncated': bool(truncated[t, env]),
            'causes': [name for name, values in causes.items() if values[t, env]],
            'center_displacement_m': finite_or_none(displacement[t, env]),
            'maximum_distal_contact_force_n': float(force[t, env]),
            'peak_pressure_pa': float(pressure[t, env]),
            'world_z_m': finite_or_none(position[t, env, 2]),
        })
    return {
        'failure_predicates_match_all_active_steps': True,
        'episodes': len(rows), 'active_environment_steps': int(active.sum()),
        'physical_failures': sum(r['terminated'] for r in rows),
        'time_limit_events': sum(r['truncated'] for r in rows),
        'cause_counts_nonexclusive': {name: int((values & active & terminated).sum())
                                      for name, values in causes.items()},
        'thresholds': thresholds, 'per_episode': rows,
        'scope': 'First episodes only; overlapping failure causes retained. Raw post-reset contact buffers are not used.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True, help='Trace metadata JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.trace.read_text())
    npz = args.trace.with_suffix('.npz')
    with np.load(npz) as trace:
        result = audit(trace, metadata['failure_thresholds'], metadata['dt'])
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.trace, npz)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_episode'}, indent=2))


if __name__ == '__main__':
    main()
