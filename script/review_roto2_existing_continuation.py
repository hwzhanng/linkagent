"""Evaluate preserved V16 checkpoints; no training or configuration changes."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from review_roto2_reward_state_pair import inspect_episode_file, compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/simulation/rotation_v18/continuation_review'


def main():
    plan = json.loads((OUT / 'PLAN.json').read_text())
    path = OUT / 'STATUS.json'
    assert not path.exists(), 'Preserve previous evaluation'
    status = dict(status='evaluating', new_training_samples=0, goal_complete=False, evaluations=[])

    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(path)

    def verify():
        for file, digest in plan['frozen_files'].items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file

    try:
        verify()
        for arm in plan['arms']:
            for condition in plan['conditions']:
                label = arm['mode'] + '_' + condition['label']
                folder = OUT / label
                folder.mkdir(exist_ok=False)
                output = folder / 'evaluation.json'
                cmd = condition['command'].copy()
                cmd[cmd.index('--evaluate-checkpoint') + 1] = arm['checkpoint']
                cmd[cmd.index('--evaluation-output') + 1] = str(output)
                status.update(active_label=label, active_command=cmd)
                save()
                with (folder / 'run.log').open('w') as log:
                    proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                assert proc.returncode == 0, (label, proc.returncode)
                result, initial = inspect_episode_file(output, condition['horizon_seconds'])
                source = Path(plan['source_evaluations']) / ('source_' + condition['label']) / 'evaluation.json'
                baseline, original = inspect_episode_file(source, condition['horizon_seconds'])
                errors = {k: float(np.abs(initial[k] - original[k]).max()) for k in initial}
                assert max(errors.values()) < 1e-7, errors
                window = '24_48s' if condition['horizon_seconds'] == 48 else '10_24s'
                status['evaluations'].append(dict(label=label, checkpoint=arm['checkpoint'],
                    checkpoint_sha256=hashlib.sha256(Path(arm['checkpoint']).read_bytes()).hexdigest(),
                    result=result, source=baseline, comparison=compare(baseline, result, window),
                    matched_initial_errors=errors, window=window))
                verify()
                save()
        status.update(status='completed_pending_interpretation', active_label=None, active_command=None,
                      frozen_files_unchanged=True, evaluation_episodes=384)
        save()
        lines = ['# 原样续训检查点行为补测', '', '只评估V16已保存检查点，新增训练样本为0。两组均64次更新，不能作为V17长训或探索重置的等预算因果对照。', '',
                 '|检查点/条件|安全整圈 源→续训|失败 源→续训|后段净角 源→续训|', '|---|---:|---:|---:|']
        for row in status['evaluations']:
            a, b, w = row['source'], row['result'], row['window']
            lines.append(f"|{row['label']}|{a['safe_full']}→{b['safe_full']}|{a['physical_failures']}→{b['physical_failures']}|{a['windows'][w]['mean_raw_net_degrees']:.2f}→{b['windows'][w]['mean_raw_net_degrees']:.2f}°|")
        (OUT / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    except BaseException as error:
        status.update(status='failed_preserved', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
