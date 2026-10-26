"""Paired, equal-budget pressure-variant fine-tuning from accepted Stage2.

Only orchestration: no controller, task, network or PPO implementation changes.
The selection file is written after actual Stage2 contact/video assessment.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_roto2_stage2 import ROOT, RUNTIME, command, sha, write


def training_command(cfg, spec, variant, seed, checkpoint):
    paired = dict(spec, training_seed=seed,
                  additional_samples_per_arm=spec['additional_samples_per_run'])
    cmd = command(cfg, paired, {}, checkpoint)
    cmd[cmd.index('--pressure-variant') + 1] = variant
    return cmd


def resume_command(row, spec, source_result):
    """Resume saved state without resetting exploration or extending the budget."""
    old = Path(row['run'])
    cfg = json.loads((old / 'run_config.json').read_text())
    progress = json.loads((old / 'progress.json').read_text())
    checkpoint = old / 'checkpoints/last_agent.pt'
    assert checkpoint.is_file() and Path(progress['checkpoint']).resolve() == checkpoint.resolve()
    assert cfg['num_envs'] == spec['num_envs']
    assert cfg['rollout_steps_effective'] == spec['rollout_steps']
    n = cfg['num_envs'] - cfg['num_eval_envs']
    target = int(spec['additional_samples_per_run'] / n) * n
    used = progress['cumulative_training_transitions'] - source_result['cumulative_training_transitions']
    remaining = target - used
    assert 0 < remaining <= target and remaining % n == 0
    cmd = list(row['command'])
    cmd[cmd.index('--resume-checkpoint') + 1] = str(checkpoint)
    if '--resume-action-std' in cmd:
        index = cmd.index('--resume-action-std')
        del cmd[index:index + 2]
    # Midpoint avoids float rounding below an integer environment step.
    cmd[cmd.index('--million-steps') + 1] = str((remaining + n / 2) / 1e6)
    assert int(float(cmd[cmd.index('--million-steps') + 1]) * 1e6 / n) * n == remaining
    record = dict(previous_run=str(old), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), progress=progress,
                  saved_additional_samples=used, remaining_samples=remaining,
                  target_additional_samples=target,
                  discarded_partial_rollout_samples=(progress['training_transitions'] // n % spec['rollout_steps']) * n,
                  limitation='Model, optimizer and scalers restored; physics, RNG and partial rollout are reset. Not bitwise uninterrupted training.',
                  created_utc=datetime.now(timezone.utc).isoformat())
    return cmd, record


def validate_selection(selection, spec):
    assert selection['stage2_accepted'] and selection['visual_review_completed']
    assert selection['measured_contact_improvement_verified']
    run = Path(selection['selected_run']).resolve()
    result = json.loads((run / 'training_result.json').read_text())
    cfg = json.loads((run / 'run_config.json').read_text())
    checkpoint = Path(result['final_checkpoint']).resolve()
    assert result['training_completed'] and result['finite_parameters']
    assert checkpoint.parent.parent == run
    assert sha(checkpoint) == selection['checkpoint_sha256']
    assert sha(Path(cfg['reset_grasps_file'])) == cfg['reset_grasps_sha256']
    assert cfg['pressure_variant'] == 'plain' and cfg['reward_stage'] == 'baseline'
    assert set(spec['variants']) == {'plain', 'scalar', 'jacobian'}
    assert len(set(spec['training_seeds'])) >= 3
    assert spec['additional_samples_per_run'] >= 300_000_000
    for path, digest in spec['frozen_source_sha256'].items():
        assert sha(ROOT / path) == digest, f'Frozen source changed: {path}'
    return run, result, cfg, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--resume-incomplete', action='store_true',
                        help='Explicitly resume interrupted rows; preserve prior artifacts')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    selection = json.loads(args.selection.read_text())
    run, source_result, cfg, checkpoint = validate_selection(selection, spec)
    schedule = [(seed, variant) for i, seed in enumerate(spec['training_seeds'])
                for variant in spec['variants'][i % 3:] + spec['variants'][:i % 3]]
    if args.preview:
        print(json.dumps([training_command(cfg, spec, variant, seed, checkpoint)
                          for seed, variant in schedule], indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'common_source'
    (source / 'checkpoints').mkdir(parents=True, exist_ok=True)
    common = source / 'checkpoints/final_agent.pt'
    if not common.exists():
        shutil.copy2(checkpoint, common)
        for name in ('run_config.json', 'training_result.json'):
            shutil.copy2(run / name, source / name)
        shutil.copy2(args.selection, source / 'SELECTION.json')
    assert sha(common) == sha(checkpoint)
    path = output / 'STATUS.json'
    status = (json.loads(path.read_text()) if path.exists() else
              dict(status='running', source_sha256=sha(common), spec_sha256=sha(args.spec),
                   selection_sha256=sha(args.selection), runs={}))
    assert status['spec_sha256'] == sha(args.spec)
    assert status['source_sha256'] == sha(common)
    assert status['selection_sha256'] == sha(args.selection)
    try:
        for seed, variant in schedule:
            # Do not silently mix task/controller implementations across variants.
            validate_selection(selection, spec)
            name = f'{variant}_seed{seed}'
            folder = output / name
            folder.mkdir(exist_ok=True)
            row = status['runs'].get(name)
            if row and row['status'] == 'completed':
                assert sha(Path(row['final_checkpoint'])) == row['final_checkpoint_sha256']
                for evaluation_seed in spec['evaluation_seeds']:
                    assert json.loads((folder / f'seed{evaluation_seed}/DIAGNOSTIC_PIPELINE.json').read_text())['status'] == 'completed'
                continue
            existing_result = None
            if row:
                if not args.resume_incomplete:
                    raise RuntimeError(f'{name} incomplete: use --resume-incomplete explicitly')
                if row.get('pid') and Path(f"/proc/{row['pid']}").exists():
                    raise RuntimeError(f'{name} still has a live PID; refusing duplicate training')
                trained = Path(row['run'])
                row.setdefault('original_initial_checkpoint', str(trained / 'checkpoints/initial_agent.pt'))
                if (trained / 'training_result.json').exists():
                    existing_result = json.loads((trained / 'training_result.json').read_text())
                    assert existing_result['training_completed'] and existing_result['finite_parameters']
                else:
                    cmd, recovery = resume_command(row, spec, source_result)
                    recovery['previous_command'] = row['command']
                    row.setdefault('recoveries', []).append(recovery)
                    row.update(status='training', command=cmd)
                    archive = folder / f"recovery{len(row['recoveries'])}"
                    archive.mkdir(exist_ok=False)
                    for filename in ('progress.json', 'run_config.json'):
                        shutil.copy2(trained / filename, archive / filename)
                    write(archive / 'RECOVERY.json', recovery)
            else:
                cmd = training_command(cfg, spec, variant, seed, common)
                row = dict(status='training', seed=seed, variant=variant, command=cmd)
                status['runs'][name] = row
            parent = ROOT / f'logs/roto2_pressure_fruit/{variant}_seed{seed}'
            before = set(parent.iterdir()) if parent.exists() else set()
            status['status'] = 'running'
            status.pop('error', None)
            write(path, status)
            log_name = f"train_resume{len(row['recoveries'])}.log" if row.get('recoveries') else 'train.log'
            if existing_result is None:
                with (folder / log_name).open('x') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    row['pid'] = proc.pid
                    write(path, status)
                    while proc.poll() is None:
                        candidates = set(parent.iterdir()) - before if parent.exists() else set()
                        if candidates:
                            assert len(candidates) == 1
                            row['run'] = str(next(iter(candidates)))
                            write(path, status)
                        time.sleep(15)
                    row['returncode'] = proc.returncode
                    write(path, status)
                    if proc.returncode:
                        raise RuntimeError(f'{name} exited {proc.returncode}')
            trained = Path(row['run'])
            result = json.loads((trained / 'training_result.json').read_text())
            assert result['training_completed'] and result['finite_parameters']
            samples = result['cumulative_training_transitions'] - source_result['cumulative_training_transitions']
            assert abs(samples - spec['additional_samples_per_run']) < spec['num_envs'] * spec['rollout_steps']
            initial = row.get('original_initial_checkpoint', str(trained / 'checkpoints/initial_agent.pt'))
            fingerprint = subprocess.check_output(
                [str(RUNTIME), 'script/run_roto2_stage2.py', '--fingerprint',
                 initial], cwd=ROOT, text=True).strip()
            if 'initial_state_fingerprint' in status:
                assert fingerprint == status['initial_state_fingerprint']
                assert samples == status['actual_samples_per_run']
            else:
                status.update(initial_state_fingerprint=fingerprint, actual_samples_per_run=samples)
            final = Path(result['final_checkpoint'])
            row.update(status='evaluating', actual_additional_samples=samples,
                       final_checkpoint=str(final), final_checkpoint_sha256=sha(final),
                       initial_state_fingerprint=fingerprint)
            write(path, status)
            for evaluation_seed in spec['evaluation_seeds']:
                target = folder / f'seed{evaluation_seed}'
                target.mkdir(exist_ok=True)
                if (target / 'DIAGNOSTIC_PIPELINE.json').exists():
                    pipeline = json.loads((target / 'DIAGNOSTIC_PIPELINE.json').read_text())
                    if pipeline['status'] == 'completed':
                        assert sha(target / 'agent.pt') == sha(final)
                        continue
                    raise RuntimeError(f'{target} has an incomplete evaluation; preserve and recover explicitly')
                shutil.copy2(final, target / 'agent.pt')
                write(target / 'PROVENANCE.json', dict(result, checkpoint=str(final), snapshot_sha256=sha(final)))
                with (target / 'pipeline.log').open('w') as log:
                    subprocess.run([sys.executable, 'script/evaluate_roto2_snapshot.py',
                                    '--run', str(trained), '--snapshot', str(target),
                                    '--training-pid', '0', '--seed', str(evaluation_seed)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            row['status'] = 'completed'
            write(path, status)
            with (output / 'comparison.log').open('a') as log:
                subprocess.run([str(RUNTIME), 'script/compare_roto2_stage3.py',
                                '--results', str(output), '--spec', str(args.spec.resolve())],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['status'] = 'completed_pending_scientific_and_visual_review'
        write(path, status)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(path, status)
        raise


if __name__ == '__main__':
    main()
