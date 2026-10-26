"""Run the preregistered equal-budget tactile tradeoff after stage1 acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python'
WEIGHTS = ('baseline_pose_weight', 'baseline_approach_weight', 'baseline_approach_pad_count',
           'baseline_motion_weight', 'baseline_centering_weight', 'baseline_failure_penalty',
           'baseline_pad_quality_weight', 'baseline_offpad_weight', 'baseline_pressure_weight',
           'baseline_contact_motion_floor')
DEFAULTS = (.01, 0., 4, 1., .02, 60., 0., 0., 0., 1.)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_fingerprint(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()

    def walk(value):
        if torch.is_tensor(value):
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(repr(key).encode())
                walk(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        else:
            digest.update(repr(value).encode())
    walk(state)
    return digest.hexdigest()


def command(cfg, spec, arm, checkpoint):
    cmd = ['bash', 'simulation/.runtime/run_roto2.sh', 'train',
           '--num-envs', str(spec['num_envs']), '--seed', str(spec['training_seed']),
           '--million-steps', str(spec['additional_samples_per_arm'] / 1e6),
           '--rollout-steps', str(spec['rollout_steps']), '--pressure-variant', 'plain',
           '--domain-profile', cfg['domain_profile'], '--hand-posture', cfg['hand_posture'],
           '--reward-stage', 'baseline', '--goal-direction', cfg['goal_direction'],
           '--reset-grasps', cfg['reset_grasps_file'], '--validate-contracts',
           '--resume-checkpoint', str(checkpoint),
           '--resume-action-std', str(spec['common_resume_action_std'])]
    # Preserve the stage1 exploration loss coefficient in every comparison arm.
    cmd += ['--entropy-weight', str(cfg['entropy_loss_scale_effective'])]
    for key, default in zip(WEIGHTS, DEFAULTS):
        cmd += ['--' + key.replace('_', '-'), str(arm.get(key, cfg.get(key, default)))]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprint', type=Path)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--stage1-run', type=Path)
    parser.add_argument('--stage1-validation', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.fingerprint:
        print(state_fingerprint(args.fingerprint))
        return
    if not all((args.spec, args.stage1_run, args.stage1_validation, args.output)):
        parser.error('spec, stage1-run, stage1-validation and output are required')
    spec = json.loads(args.spec.read_text())
    run, validation, output = (p.resolve() for p in (
        args.stage1_run, args.stage1_validation, args.output))
    gate = json.loads((validation / 'QUANTITATIVE_GATE.json').read_text())
    result = json.loads((run / 'training_result.json').read_text())
    cfg = json.loads((run / 'run_config.json').read_text())
    verified = json.loads((validation / 'VALIDATION_PIPELINE.json').read_text())
    assert gate['stage1_accepted'] and gate['both_independent_seeds_pass']
    assert not gate['visual_review_pending']
    assert result['training_completed'] and result['finite_parameters']
    prior = result['cumulative_training_transitions']
    assert prior >= 299_000_000
    checkpoint = Path(result['final_checkpoint']).resolve()
    assert checkpoint.parent.parent == run and sha(checkpoint) == verified['checkpoint_sha256']
    assert cfg['pressure_variant'] == spec['pressure_variant'] == 'plain'
    assert sha(Path(cfg['reset_grasps_file'])) == cfg['reset_grasps_sha256']
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'common_source'
    (source / 'checkpoints').mkdir(parents=True, exist_ok=True)
    common = source / 'checkpoints/final_agent.pt'
    if not common.exists():
        shutil.copy2(checkpoint, common)
        shutil.copy2(run / 'run_config.json', source / 'run_config.json')
        shutil.copy2(run / 'training_result.json', source / 'training_result.json')
    assert sha(common) == sha(checkpoint)
    status_file = output / 'STATUS.json'
    status = (json.loads(status_file.read_text()) if status_file.exists() else
              {'status': 'running', 'source_sha256': sha(common), 'spec_sha256': sha(args.spec), 'arms': {}})
    assert status['source_sha256'] == sha(common) and status['spec_sha256'] == sha(args.spec)
    try:
        for arm in spec['arms']:
            name = arm['name']
            folder = output / name
            folder.mkdir(exist_ok=True)
            row = status['arms'].get(name)
            if row and row['status'] != 'completed':
                raise RuntimeError(f'{name} is an interrupted/incomplete arm; preserve it and resume its checkpoint explicitly')
            if row:
                assert sha(Path(row['final_checkpoint'])) == row['final_checkpoint_sha256']
                for seed in spec['evaluation_seeds']:
                    pipeline = json.loads((folder / f'seed{seed}/DIAGNOSTIC_PIPELINE.json').read_text())
                    assert pipeline['status'] == 'completed'
                continue
            cmd = command(cfg, spec, arm, common)
            parent = ROOT / f'logs/roto2_pressure_fruit/plain_seed{spec["training_seed"]}'
            before = set(parent.iterdir()) if parent.exists() else set()
            row = {'status': 'training', 'configuration': arm, 'command': cmd}
            status['arms'][name] = row
            write(status_file, status)
            with (folder / 'train.log').open('w') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                write(status_file, status)
                while process.poll() is None:
                    candidates = set(parent.iterdir()) - before if parent.exists() else set()
                    if candidates:
                        assert len(candidates) == 1, candidates
                        row['run'] = str(next(iter(candidates)))
                        write(status_file, status)
                    time.sleep(15)
                row['returncode'] = process.returncode
                write(status_file, status)
                if process.returncode:
                    raise RuntimeError(f'{name} training exited {process.returncode}')
            arm_run = Path(row['run'])
            completed = json.loads((arm_run / 'training_result.json').read_text())
            assert completed['training_completed'] and completed['finite_parameters']
            samples = completed['cumulative_training_transitions'] - prior
            assert abs(samples - spec['additional_samples_per_arm']) < spec['num_envs'] * spec['rollout_steps']
            fingerprint = subprocess.check_output(
                [str(RUNTIME), str(Path(__file__).resolve()), '--fingerprint',
                 str(arm_run / 'checkpoints/initial_agent.pt')], text=True).strip()
            if 'initial_state_fingerprint' in status:
                assert fingerprint == status['initial_state_fingerprint']
                assert samples == status['actual_samples_per_arm']
            else:
                status.update(initial_state_fingerprint=fingerprint, actual_samples_per_arm=samples)
            final = Path(completed['final_checkpoint'])
            row.update(status='evaluating', actual_additional_samples=samples,
                       final_checkpoint=str(final), final_checkpoint_sha256=sha(final),
                       initial_state_fingerprint=fingerprint)
            write(status_file, status)
            for seed in spec['evaluation_seeds']:
                target = folder / f'seed{seed}'
                target.mkdir(exist_ok=True)
                shutil.copy2(final, target / 'agent.pt')
                provenance = dict(completed, checkpoint=str(final), snapshot_sha256=sha(final))
                write(target / 'PROVENANCE.json', provenance)
                with (target / 'pipeline.log').open('w') as log:
                    subprocess.run([sys.executable, 'script/evaluate_roto2_snapshot.py',
                                    '--run', str(arm_run), '--snapshot', str(target),
                                    '--training-pid', '0', '--seed', str(seed)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            row['status'] = 'completed'
            write(status_file, status)
            with (output / 'comparison.log').open('a') as log:
                subprocess.run([str(RUNTIME), 'script/compare_roto2_stage2.py',
                                '--results', str(output), '--spec', str(args.spec.resolve())],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['status'] = 'completed_pending_comparison_and_visual_review'
        write(status_file, status)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(status_file, status)
        raise


if __name__ == '__main__':
    main()
