"""Training-seed-level pressure comparison with explicit rotation comparability."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def correction_metrics(path):
    with np.load(path) as d:
        active = d['active'] & ~d['done']
        count = int(active.sum())
        if not count:
            return dict(active_steps=0, accepted_steps=0, changed_steps=0,
                        supported_steps=0, ready_steps=0)
        delta = d['executed_action'] - d['action'].clip(-1, 1)
        accepted = d['jacobian_accepted'][active]
        norm = np.linalg.norm(delta[active], axis=-1)
        assert np.isfinite(norm).all()
        assert np.all(norm[~accepted] < 1e-6), 'Unexpected action change outside correction acceptance'
        return dict(active_steps=count, accepted_steps=int(accepted.sum()),
                    changed_steps=int((norm > 1e-6).sum()),
                    supported_steps=int(d['correction_support_pre'][active].sum()),
                    ready_steps=int(d['jacobian_ready_pre'][active].sum()))


def summarize_run(folder, seeds):
    evaluations = [json.loads((folder / f'seed{s}/evaluation_seed{s}.json').read_text()) for s in seeds]
    metrics = [json.loads((folder / f'seed{s}/TRADEOFF_METRICS_seed{s}.json').read_text()) for s in seeds]
    corrections = [correction_metrics(folder / f'seed{s}/action_trace_seed{s}.npz') for s in seeds]
    episodes = sum(e['episodes'] for e in evaluations)
    assert all(e['simulation_physx_rollout_completed'] and e['pose_metric_max_error_turns'] < 1e-4 for e in evaluations)
    row = dict(episodes=episodes,
               safe_full_turn_fraction=sum(m['safe_full_turns'] for m in metrics) / episodes,
               physical_failure_fraction=sum(m['physical_failures'] for m in metrics) / episodes,
               mean_after10s_degrees=float(np.mean([m['mean_after10s_degrees'] for m in metrics])),
               mean_episode_peak_pressure_pa=float(np.mean([e['mean_peak_pressure_proxy_pa'] for e in evaluations])),
               rotation_eligible=all(m['mean_after10s_degrees'] > 90 and m['safe_full_turns'] > 0 for m in metrics))
    windows = [m['windows']['after10s'] for m in metrics]
    counts = [w['active_nonterminal_environment_steps'] for w in windows]
    row['late_active_steps'] = sum(counts)
    for key in ['mean_valid_pad_count', 'mean_top3_measured_pad_quality', 'mean_peak_pressure_pa',
                'pressure_above_half_limit_fraction', 'pressure_above_limit_fraction']:
        row['late_' + key] = (sum(w[key] * n for w, n in zip(windows, counts) if n) / sum(counts)
                              if sum(counts) else None)
    for key in corrections[0]:
        row['correction_' + key] = sum(c[key] for c in corrections)
    row['correction_acceptance_fraction'] = (row['correction_accepted_steps'] / row['correction_active_steps']
                                            if row['correction_active_steps'] else None)
    return row


def compare(rows, spec):
    metrics = ['mean_after10s_degrees', 'safe_full_turn_fraction', 'physical_failure_fraction',
               'mean_episode_peak_pressure_pa', 'late_mean_peak_pressure_pa',
               'late_mean_valid_pad_count', 'late_mean_top3_measured_pad_quality',
               'correction_acceptance_fraction']
    summaries = {}
    for variant in spec['variants']:
        group = [r for r in rows if r['variant'] == variant]
        summaries[variant] = dict(training_seeds=[r['seed'] for r in group], metrics={})
        for key in metrics:
            values = [r[key] for r in group if r[key] is not None]
            summaries[variant]['metrics'][key] = dict(
                n_training_seeds=len(values), values=values,
                mean=float(np.mean(values)) if values else None,
                sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    paired = []
    by_key = {(r['variant'], r['seed']): r for r in rows}
    for seed in spec['training_seeds']:
        for baseline, treatment in [('plain', 'scalar'), ('plain', 'jacobian'), ('scalar', 'jacobian')]:
            if (baseline, seed) not in by_key or (treatment, seed) not in by_key:
                continue
            a, b = by_key[baseline, seed], by_key[treatment, seed]
            comparable = (a['rotation_eligible'] and b['rotation_eligible']
                          and abs(b['mean_after10s_degrees'] - a['mean_after10s_degrees'])
                          <= spec['rotation_comparability_relative_tolerance'] * abs(a['mean_after10s_degrees'])
                          and abs(b['safe_full_turn_fraction'] - a['safe_full_turn_fraction'])
                          <= spec['safe_full_turn_comparability_absolute_tolerance'])
            paired.append(dict(seed=seed, baseline=baseline, treatment=treatment,
                               rotation_comparable=comparable,
                               treatment_minus_baseline={key: b[key] - a[key] if a[key] is not None and b[key] is not None else None
                                                         for key in metrics}))
    return dict(stage3_accepted=False, scientific_and_visual_review_pending=True,
                complete=len(rows) == len(spec['variants']) * len(spec['training_seeds']),
                rows=rows, summaries=summaries, paired_differences=paired,
                scope='Independent continuation seeds from a shared selected warm start, not independent from-scratch replications. Training seed is the statistical unit; evaluation episodes are not additional training replicates. Late pressure is conditional on survival; report physical failures and all-episode peak pressure alongside it. Correction acceptance is a model prediction, not measured benefit. Pressure remains an uncalibrated simulation proxy. No automatic innovation success claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.results / 'STATUS.json').read_text())
    spec = json.loads(args.spec.read_text())
    rows = []
    for name, run in status['runs'].items():
        if run['status'] != 'completed':
            continue
        assert run['actual_additional_samples'] == status['actual_samples_per_run']
        assert run['initial_state_fingerprint'] == status['initial_state_fingerprint']
        row = summarize_run(args.results / name, spec['evaluation_seeds'])
        row.update(seed=run['seed'], variant=run['variant'], additional_samples=run['actual_additional_samples'])
        if run['variant'] != 'jacobian':
            assert row['correction_accepted_steps'] == row['correction_changed_steps'] == 0
        rows.append(row)
    assert rows, 'No completed runs'
    report = compare(rows, spec)
    (args.results / 'COMPARISON.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.results / 'seed_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for variant in spec['variants']:
        group = [r for r in rows if r['variant'] == variant]
        for ax, key, label in zip(axes, ['mean_episode_peak_pressure_pa', 'late_mean_peak_pressure_pa'],
                                 ['Mean episode peak pressure (kPa)', 'Mean late pressure (kPa; surviving steps)']):
            values = [r for r in group if r[key] is not None]
            ax.scatter([r['mean_after10s_degrees'] for r in values], [r[key] / 1000 for r in values], label=variant)
            for r in values:
                ax.annotate(str(r['seed']), (r['mean_after10s_degrees'], r[key] / 1000), fontsize=8)
            ax.set(xlabel='Mean net rotation after 10 s (degrees)', ylabel=label)
    axes[0].legend()
    fig.suptitle(f'Stage3: {len(rows)}/{len(spec["variants"]) * len(spec["training_seeds"])} training runs')
    fig.savefig(args.results / 'pressure_rotation.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(completed_runs=len(rows), complete=report['complete'])))


if __name__ == '__main__':
    main()
