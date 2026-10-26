"""Finish existing V17 evidence collection; never launch or extend training."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/simulation/rotation_v17/ppo_pair'
SERVICE = 'roto2-v17-reward-state-pair.service'


def main():
    path = OUT / 'POSTPROCESS.json'
    assert not path.exists(), 'Preserve previous postprocessing evidence'
    state = dict(status='waiting_for_existing_pair', goal_complete=False,
                 training_launched=False, steps=[], started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(path)

    def run(script):
        command = [sys.executable, str(ROOT / 'script' / script)]
        if script.startswith('record_'):
            command.append('--execute')
        row = dict(command=command, status='running')
        state['steps'].append(row)
        save()
        with (OUT / (Path(script).stem + '.log')).open('x') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        row.update(returncode=proc.returncode, status='completed' if proc.returncode == 0 else 'failed')
        save()
        assert proc.returncode == 0, row

    try:
        save()
        deadline = time.monotonic() + 7200
        while True:
            pair = json.loads((OUT / 'STATUS.json').read_text())
            service = subprocess.run(['systemctl', '--user', 'is-active', SERVICE],
                                     capture_output=True, text=True).stdout.strip()
            state.update(pair_status=pair['status'], pair_service=service,
                         completed_evaluations=len(pair['evaluations']))
            save()
            assert pair['status'] != 'failed_preserved', pair.get('error')
            if pair['status'] == 'completed_pair_pending_scientific_review' and service == 'inactive':
                assert pair['core_hashes_unchanged'] and not pair['training_running']
                assert len(pair['evaluations']) == 9 and len(pair['training_runs']) == 2
                break
            assert service in ('active', 'activating', 'deactivating'), f'Unexpected stopped pair: {service}'
            assert time.monotonic() < deadline, 'Existing pair did not finish within two hours'
            time.sleep(15)
        state['status'] = 'reviewing_physical_metrics'
        save()
        run('review_roto2_reward_state_pair.py')
        review = json.loads((OUT / 'PHYSICAL_REVIEW.json').read_text())
        assert review['complete_pair_evidence']
        state['status'] = 'recording_representative_rollouts'
        save()
        run('record_roto2_reward_state_pair.py')
        videos = json.loads((OUT / 'videos/RECORDINGS.json').read_text())
        assert videos['status'] == 'completed_pending_visual_review'
        assert videos['video_count'] == 6 and all(v['verified'] for v in videos['videos'])
        state.update(status='completed_pending_visual_and_scientific_review',
                     scientific_success_claimed=False, video_count=6)
        save()
    except BaseException as error:
        state.update(status='failed_preserved', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
