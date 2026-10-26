"""Record median and best first episodes from an existing matched evaluation."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from validate_roto2_stage1_run import ROOT, verify_video_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder, output = args.folder.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'RECORDINGS.json'
    if status_path.exists():
        raise RuntimeError('Existing attempt preserved; do not silently overwrite')
    evaluation = json.loads((folder / f'evaluation_seed{args.seed}.json').read_text())
    pipeline = json.loads((folder / 'DIAGNOSTIC_PIPELINE.json').read_text())
    assert pipeline['status'] == 'completed'
    common = list(pipeline['commands'][0]['command'])
    assert common[2] == 'evaluate'
    common[2] = 'record'
    i = common.index('--evaluation-output')
    del common[i:i+2]
    checkpoint = Path(common[common.index('--evaluate-checkpoint') + 1])
    rows = sorted(evaluation['per_episode'], key=lambda r: r['after_10s_directed_turns'])
    examples = [('median', rows[len(rows)//2]),
                ('best', max(rows, key=lambda r: (r['success'], r['after_10s_directed_turns'])))]
    status = dict(status='recording', source_evaluation=str(folder), seed=args.seed,
                  selection='Median over all episodes by late rotation, and best safe late rotation.',
                  videos=[], stage2_accepted=False)

    def save():
        status_path.write_text(json.dumps(status, indent=2) + '\n')

    save()
    try:
        with np.load(folder / f'evaluation_seed{args.seed}.npz') as measured:
            for label, row in examples:
                video = output / f'{label}_seed{args.seed}_env{row["env"]}.mp4'
                assert not video.exists()
                cmd = common + ['--video-env-index', str(row['env']), '--record-video', str(video)]
                item = dict(label=label, episode=row, video=str(video), command=cmd)
                status['videos'].append(item)
                save()
                log_path = video.with_suffix('.log')
                with log_path.open('w') as log:
                    result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                item['returncode'] = result.returncode
                assert result.returncode in (0, 143, -15)
                if result.returncode:
                    assert 'SimulationContext cleared' in log_path.read_text(errors='replace')
                item['artifact_verification'] = verify_video_artifact(video, checkpoint, args.seed, row['env'])
                data = json.loads(video.with_suffix('.json').read_text())
                poses = np.array([r['fruit_pose_native'] for r in data['trace']])
                error = float(np.abs(poses - measured['poses'][:len(poses), row['env']]).max())
                assert error < 1e-6, error
                item['matched_evaluation_pose_max_error'] = error
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-vf',
                                'fps=1/4,scale=640:-1,tile=3x2', '-frames:v', '1',
                                str(video.with_suffix('.jpg'))], check=True)
                save()
        status['status'] = 'completed_pending_visual_review'
        save()
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    main()
