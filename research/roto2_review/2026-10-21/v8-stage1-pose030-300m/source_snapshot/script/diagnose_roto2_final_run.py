"""After final evaluation/video jobs, collect matching action and failure evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--wait-pid', type=int)
    a = p.parse_args()
    run, validation = a.run.resolve(), a.validation.resolve()
    output = validation / 'action_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    status = {'status': 'waiting_for_final_validation', 'run': str(run), 'commands': []}

    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'PIPELINE.json').write_text(json.dumps(status, indent=2) + '\n')

    def execute(command, name):
        row = {'command': [str(x) for x in command], 'log': str(output / (name + '.log'))}
        status['commands'].append(row)
        save()
        with Path(row['log']).open('w') as log:
            result = subprocess.run(row['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row['returncode'] = result.returncode
        save()
        result.check_returncode()

    save()
    try:
        if a.wait_pid:
            while Path(f'/proc/{a.wait_pid}').exists():
                time.sleep(30)
        parent = json.loads((validation / 'VALIDATION_PIPELINE.json').read_text())
        assert parent['status'] == 'completed_pending_visual_review', parent['status']
        assert Path(parent['training_run']) == run
        cfg = json.loads((run / 'run_config.json').read_text())
        result = json.loads((run / 'training_result.json').read_text())
        checkpoint = Path(result['final_checkpoint'])
        assert result['training_completed'] and result['finite_parameters']
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == parent['checkpoint_sha256']
        evaluation = validation / 'final_seed6101.json'
        trace = output / 'action_trace_seed6101.json'
        status.update(status='tracing', checkpoint_sha256=digest)
        save()
        execute(['bash', 'simulation/.runtime/run_roto2.sh', 'evaluate',
                 '--num-envs', '64', '--seed', '6101',
                 '--pressure-variant', cfg['pressure_variant'],
                 '--domain-profile', cfg['domain_profile'], '--hand-posture', cfg['hand_posture'],
                 '--reward-stage', cfg['reward_stage'], '--goal-direction', cfg['goal_direction'],
                 '--baseline-pose-weight', str(cfg['baseline_pose_weight']),
                 '--baseline-approach-weight', str(cfg.get('baseline_approach_weight', 0.)),
                 '--baseline-approach-pad-count', str(cfg.get('baseline_approach_pad_count', 4)),
                 '--reset-grasps', cfg['reset_grasps_file'], '--evaluate-checkpoint', checkpoint,
                 '--trace-policy-output', trace], 'trace')
        execute([PYTHON, '-m', 'script.audit_roto2_terminal_causes', '--trace', trace,
                 '--output', output / 'TERMINAL_CAUSES.json'], 'terminal_audit')
        execute([PYTHON, '-m', 'script.summarize_roto2_recontact', '--folder', output,
                 '--seed', '6101', '--trace', trace.with_suffix('.npz'),
                 '--evaluation', evaluation, '--final-training-result'], 'recontact')
        execute([PYTHON, '-m', 'script.plot_roto2_policy_diagnostic',
                 '--evaluation', evaluation, '--trace', trace,
                 '--output', output / 'diagnostic.png', '--final-training-result',
                 '--label', 'Pose0.30 final300M, independent seed6101'], 'plot')
        status.update(status='completed', visual_review_pending=True)
        save()
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
