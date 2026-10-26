"""Bounded matched critic-state PPO pair; no automatic continuation or selection."""
import json
import hashlib
import subprocess
import time
from pathlib import Path
import numpy as np
from summarize_roto2_sprint import summarize
from summarize_roto2_fresh_review import contact_windows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v17/ppo_pair'


def main():
    plan=json.loads((OUT/'PLAN.json').read_text())
    assert not (OUT/'STATUS.json').exists(), 'Preserve previous pair'
    status=dict(status='starting',training_running=False,goal_complete=False,
                evaluations=[],training_runs=[],active_child=None)
    def save():
        temp=OUT/'STATUS.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(OUT/'STATUS.json')
    def verify():
        for path,digest in plan['frozen_files'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256(Path(plan['source_checkpoint']).read_bytes()).hexdigest()==plan['source_sha256']
    def run(cmd,folder,training=False):
        folder.mkdir(exist_ok=False)
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status.update(active_child={'pid':proc.pid,'command':cmd,'output':str(folder)},training_running=training);save()
            try:
                while proc.poll() is None:
                    if training:
                        progress=folder/'observed/PROGRESS.json'
                        if progress.exists():
                            try:status['live_diagnostic']=json.loads(progress.read_text())
                            except json.JSONDecodeError:pass
                        status['current_child_elapsed_seconds']=time.monotonic()-started_child
                        save()
                    time.sleep(5)
                if proc.returncode:raise RuntimeError(f'Command failed with {proc.returncode}: {folder}')
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        status.update(active_child=None,training_running=False);save()
    def evaluate(checkpoint,prefix):
        nonlocal started_child
        for condition in plan['evaluations']:
            label=prefix+'_'+condition['label'];folder=OUT/label;cmd=condition['command'].copy()
            cmd[cmd.index('--evaluate-checkpoint')+1]=str(checkpoint)
            cmd[cmd.index('--evaluation-output')+1]=str(folder/'evaluation.json')
            status['status']='evaluating_'+label;save();started_child=time.monotonic();run(cmd,folder)
            report=summarize(folder/'evaluation.json',condition['horizon_seconds'])
            if condition['horizon_seconds']==48:report['contact_windows']=contact_windows(folder/'evaluation.json')
            if prefix!='source':
                with np.load(OUT/('source_'+condition['label'])/'evaluation.npz') as b,np.load(folder/'evaluation.npz') as a:
                    errors={k:float(np.abs(a[k]-b[k]).max()) for k in ['initial_pose','target_direction','target_speed']}
                assert max(errors.values())<1e-7,errors
                report['matched_initial_errors']=errors
            status['evaluations'].append(dict(label=label,status='completed',summary=report));save();verify()
    started_child=time.monotonic()
    try:
        verify();save();evaluate(Path(plan['source_checkpoint']),'source')
        for arm in plan['arms']:
            verify();mode=arm['mode'];status['status']='training_'+mode;status['live_diagnostic']=None;save()
            parent=ROOT/'logs/roto2_pressure_fruit'/('plain_seed'+str(plan['training_seed']))
            before=set(parent.iterdir()) if parent.exists() else set()
            started_child=time.monotonic();run(arm['command'],OUT/('train_'+mode),training=True)
            added=set(parent.iterdir())-before;assert len(added)==1;run_path=added.pop()
            done=json.loads((run_path/'training_result.json').read_text())
            assert done['training_completed'] and done['finite_parameters']
            assert done['ppo_updates']==plan['updates_per_arm'] and done['training_transitions']==plan['samples_per_arm']
            migration=json.loads((run_path/'critic_migration.json').read_text())
            assert migration['value_existing_adam_preserved'] and migration['critic_dimensions']==1
            status['training_runs'].append(dict(mode=mode,path=str(run_path),result=done,migration=migration));save()
            evaluate(Path(done['final_checkpoint']),mode)
        verify();status.update(status='completed_pair_pending_scientific_review',training_running=False,
                                total_training_transitions=sum(r['result']['training_transitions'] for r in status['training_runs']),
                                automatic_extension_disabled=True,core_hashes_unchanged=True)
        save()
    except BaseException as error:
        status.update(status='failed_preserved',training_running=False,error=repr(error));save();raise


if __name__=='__main__':main()
