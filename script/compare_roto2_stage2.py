"""Compare completed paired stage2 arms without automatically accepting a winner."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.results / 'STATUS.json').read_text())
    spec = json.loads(args.spec.read_text())
    rows = []
    for arm in spec['arms']:
        name = arm['name']
        run = status['arms'].get(name)
        if not run or run['status'] != 'completed':
            continue
        assert run['actual_additional_samples'] == status['actual_samples_per_arm']
        assert run['initial_state_fingerprint'] == status['initial_state_fingerprint']
        metrics = [json.loads((args.results / name / f'seed{seed}' /
                              f'TRADEOFF_METRICS_seed{seed}.json').read_text())
                   for seed in spec['evaluation_seeds']]
        assert all(m['episodes'] == spec['evaluation_episodes_per_seed'] for m in metrics)
        late = [m['windows']['after10s'] for m in metrics]
        counts = np.array([m['active_nonterminal_environment_steps'] for m in late])
        row = dict(arm, additional_samples=run['actual_additional_samples'],
                   episodes=sum(m['episodes'] for m in metrics),
                   safe_full_turns=sum(m['safe_full_turns'] for m in metrics),
                   physical_failures=sum(m['physical_failures'] for m in metrics),
                   mean_after10s_degrees=float(np.mean([m['mean_after10s_degrees'] for m in metrics])),
                   rotation_eligible=all(m['mean_after10s_degrees'] > 90 and m['safe_full_turns'] > 0 for m in metrics),
                   active_late_steps=int(counts.sum()))
        for key in ['mean_valid_pad_count', 'three_pad_contact_fraction',
                    'mean_top3_measured_pad_quality', 'mean_total_offpad_distal_force_n',
                    'mean_peak_pressure_pa', 'pressure_above_half_limit_fraction']:
            row['late_' + key] = (float(sum(m.get(key, 0.) * n for m, n in zip(late, counts)) / counts.sum())
                                 if counts.sum() else None)
        available_p95 = [m['p95_peak_pressure_pa'] for m in late if m['active_nonterminal_environment_steps']]
        row['mean_seed_late_pressure_p95_pa'] = float(np.mean(available_p95)) if available_p95 else None
        rows.append(row)
    if not rows:
        raise ValueError('No completed comparison arms yet')
    control = next(r for r in rows if r['name'] == 'control')
    for row in rows:
        row['contact_improvement_over_control'] = (
            row['active_late_steps'] > 0 and control['active_late_steps'] > 0
            and row['late_mean_valid_pad_count'] > control['late_mean_valid_pad_count']
            and row['late_mean_top3_measured_pad_quality'] > control['late_mean_top3_measured_pad_quality'])
        row['nondominated'] = row['rotation_eligible'] and not any(
            all([other['mean_after10s_degrees'] >= row['mean_after10s_degrees'],
                 other['late_mean_valid_pad_count'] >= row['late_mean_valid_pad_count'],
                 other['late_mean_peak_pressure_pa'] <= row['late_mean_peak_pressure_pa'],
                 other['physical_failures'] <= row['physical_failures']])
            and any([other['mean_after10s_degrees'] > row['mean_after10s_degrees'],
                     other['late_mean_valid_pad_count'] > row['late_mean_valid_pad_count'],
                     other['late_mean_peak_pressure_pa'] < row['late_mean_peak_pressure_pa'],
                     other['physical_failures'] < row['physical_failures']])
            for other in rows if other['rotation_eligible'])
    report = {'stage2_accepted': False, 'visual_review_and_selection_pending': True,
              'complete_comparison': len(rows) == len(spec['arms']), 'rows': rows,
              'candidate_names': [r['name'] for r in rows if r['rotation_eligible']
                                  and r['contact_improvement_over_control'] and r['nondominated']],
              'scope': 'Same initial state and additional training budget. Two evaluation seeds, one training seed: screening only, not a multi-training-seed generalization claim. Contact and pressure are pooled active nonterminal late-step means; P95 is the mean of seed-specific P95s. Candidate status does not establish robust tactile regrasp.'}
    (args.results / 'COMPARISON.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.results / 'tradeoff.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plotted = [r for r in rows if r['active_late_steps']]
    scatter = ax.scatter([r['mean_after10s_degrees'] for r in plotted],
                         [r['late_mean_valid_pad_count'] for r in plotted],
                         c=[r['late_mean_peak_pressure_pa'] / 1000 for r in plotted],
                         s=[50 + 350 * r['safe_full_turns'] / r['episodes'] for r in plotted],
                         cmap='viridis_r', edgecolors='black')
    for row in plotted:
        ax.annotate(row['name'], (row['mean_after10s_degrees'], row['late_mean_valid_pad_count']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.axvline(90, linestyle='--', color='gray')
    ax.set(xlabel='Mean net rotation after 10 s (degrees)', ylabel='Mean effective pad count after 10 s',
           title=f'Stage2: {len(rows)}/{len(spec["arms"])} arms; size = safe full-turn fraction')
    if plotted:
        fig.colorbar(scatter, ax=ax, label='Mean late pressure proxy (kPa)')
    missing = [r['name'] for r in rows if not r['active_late_steps']]
    if missing:
        ax.text(.02, .98, 'No late active samples: ' + ', '.join(missing),
                transform=ax.transAxes, va='top', fontsize=8)
    fig.savefig(args.results / 'tradeoff.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'completed_arms': len(rows), 'candidate_names': report['candidate_names']}))


if __name__ == '__main__':
    main()
