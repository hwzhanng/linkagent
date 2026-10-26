"""One full contact-guided continuation after the complete Stage2 screen.

Requires an explicitly selected rotating arm; never auto-accepts Stage2.
Reuses the PPO trainer and physical task unchanged, with reward-only credit.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_roto2_stage2 import ROOT, RUNTIME, command, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--source-arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    batch, output = args.batch.resolve(), args.output.resolve()
    batch_status = json.loads((batch / 'STATUS.json').read_text())
    comparison = json.loads((batch / 'COMPARISON.json').read_text())
    assert comparison['complete_comparison'] and len(comparison['rows']) == 7
    assert all(r['status'] == 'completed' for r in batch_status['arms'].values())
    row = batch_status['arms'][args.source_arm]
    screened = next(r for r in comparison['rows'] if r['name'] == args.source_arm)
    assert screened['rotation_eligible'], 'Source must retain actual sustained rotation'
    source_run = Path(row['run'])
    source_result = json.loads((source_run / 'training_result.json').read_text())
    cfg = json.loads((source_run / 'run_config.json').read_text())
    checkpoint = Path(row['final_checkpoint'])
    assert source_result['training_completed'] and source_result['finite_parameters']
    assert checkpoint.resolve() == Path(source_result['final_checkpoint']).resolve()
    assert sha(checkpoint) == row['final_checkpoint_sha256']
    assert sha(Path(cfg['reset_grasps_file'])) == cfg['reset_grasps_sha256']
    spec = dict(num_envs=8192, training_seed=7401, additional_samples_per_arm=300_000_000,
                rollout_steps=16, common_resume_action_std=.5)
    overrides = {'baseline_contact_motion_floor': .25}
    cmd = command(cfg, spec, overrides, checkpoint)
    if args.preview:
        print(json.dumps(cmd, indent=2))
        return
    if (output / 'STATUS.json').exists():
        raise RuntimeError('Existing attempt preserved. Explicitly resume its checkpoint; do not silently restart.')
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'source'
    (source / 'checkpoints').mkdir(parents=True, exist_ok=True)
    common = source / 'checkpoints/final_agent.pt'
    shutil.copy2(checkpoint, common)
    for name in ['run_config.json', 'training_result.json']:
        shutil.copy2(source_run / name, source / name)
    assert sha(common) == row['final_checkpoint_sha256']
    cmd = command(cfg, spec, overrides, common)
    prior = source_result['cumulative_training_transitions']
    milestones = [prior + x for x in [100_000_000, 200_000_000]]
    plan = dict(source_arm=args.source_arm, source_run=str(source_run), source_sha256=sha(common),
                source_screen=screened, training=spec, reward_overrides=overrides,
                unchanged='All other selected reward coefficients, physical task, success definitions, network, PPO, observations and actions.',
                interim_seed=8201, final_seeds=[9201, 9202], milestone_cumulative_samples=milestones,
                github_sync=False)
    write(output / 'PLAN.json', plan)
    status = dict(status='training', command=cmd, helpers=[], stage2_accepted=False)
    status_file = output / 'STATUS.json'
    write(status_file, status)
    helpers, handles = [], []

    def helper(cmd, log):
        handle = log.open('w')
        handles.append(handle)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        helpers.append(proc)
        status['helpers'].append(dict(pid=proc.pid, command=cmd, log=str(log)))
        write(status_file, status)

    try:
        parent = ROOT / 'logs/roto2_pressure_fruit/plain_seed7401'
        before = set(parent.iterdir()) if parent.exists() else set()
        with (output / 'train.log').open('w') as log:
            train = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status['training_pid'] = train.pid
            write(status_file, status)
            while train.poll() is None:
                candidates = set(parent.iterdir()) - before if parent.exists() else set()
                if candidates and 'run' not in status:
                    assert len(candidates) == 1
                    run = next(iter(candidates))
                    status['run'] = str(run)
                    helper([sys.executable, 'script/snapshot_roto2_milestones.py',
                            '--run', str(run), '--output', str(output), '--pid', str(train.pid),
                            '--milestones', *map(str, milestones)], output / 'snapshots.log')
                    for milestone in milestones:
                        target = output / f'milestone_{milestone}'
                        helper([sys.executable, 'script/evaluate_roto2_snapshot.py',
                                '--run', str(run), '--snapshot', str(target),
                                '--training-pid', str(train.pid), '--seed', '8201'],
                               output / f'diagnostic_{milestone}.log')
                time.sleep(15)
            status['training_returncode'] = train.returncode
            write(status_file, status)
            if train.returncode:
                raise RuntimeError(f'Training exited {train.returncode}')
        run = Path(status['run'])
        result = json.loads((run / 'training_result.json').read_text())
        assert result['training_completed'] and result['finite_parameters']
        assert abs(result['cumulative_training_transitions'] - prior - 300_000_000) < 8192 * 16
        assert result['ppo_updates'] >= 2200
        for proc in helpers:
            code = proc.wait()
            if code:
                raise RuntimeError(f'Intermediate diagnostic PID{proc.pid} exited {code}')
        final = Path(result['final_checkpoint'])
        status.update(status='evaluating_final', training_result=result, final_checkpoint_sha256=sha(final))
        write(status_file, status)
        for seed in plan['final_seeds']:
            target = output / f'final_seed{seed}'
            target.mkdir(exist_ok=True)
            shutil.copy2(final, target / 'agent.pt')
            write(target / 'PROVENANCE.json', dict(result, checkpoint=str(final), snapshot_sha256=sha(final)))
            with (target / 'pipeline.log').open('w') as log:
                subprocess.run([sys.executable, 'script/evaluate_roto2_snapshot.py',
                                '--run', str(run), '--snapshot', str(target),
                                '--training-pid', '0', '--seed', str(seed)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['status'] = 'completed_pending_contact_assessment_and_videos'
        write(status_file, status)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(status_file, status)
        # Stop only this attempt's helper observers, not any unrelated workload.
        for proc in helpers:
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
