"""One task-reward ablation, with a mandatory scientific review at100M.

Core/environment code is unchanged. The trainer is held during matched evaluation;
an explicit local review receipt from the researcher/agent controls continuation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from run_roto2_stage2 import ROOT, RUNTIME, sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',action='store_true');a=p.parse_args()
    out=a.output.resolve();source=ROOT/'artifacts/simulation/rotation_v9/sprint'
    old=json.loads((source/'run/STATUS.json').read_text())
    cmd=list(old['command'])
    changes={'--million-steps':'200.0','--baseline-pose-weight':'0.01',
             '--baseline-approach-weight':'0.0','--baseline-pad-quality-weight':'0.0',
             '--baseline-offpad-weight':'0.0','--baseline-contact-motion-floor':'1.0'}
    for flag,value in changes.items():cmd[cmd.index(flag)+1]=value
    if a.preview:print(json.dumps(dict(command=cmd,changes=changes),indent=2));return
    assert not (out/'STATUS.json').exists(),'Preserve existing attempts'
    out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((source/'FROZEN_CORE.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in frozen.items())
    write(out/'FROZEN_CORE.json',frozen)
    # The existing matched evaluator takes this immutable baseline manifest.
    (out/'baseline').mkdir()
    write(out/'baseline/STATUS.json',old)
    for row in old['evaluations']:
        if row['label'].startswith('source_'):
            os.symlink(source/'run'/row['label'],out/'baseline'/row['label'],target_is_directory=True)
    source_cp=Path(cmd[cmd.index('--resume-checkpoint')+1])
    previous=json.loads((source_cp.parent.parent/'training_result.json').read_text())
    offset=previous['cumulative_training_transitions']
    status=dict(status='starting',command=cmd,task_reward_changes=changes,goal_complete=False,
                source_checkpoint_sha256=sha(source_cp),evaluations=[],budget_samples=200000000)
    write(out/'STATUS.json',status)
    parent=ROOT/'logs/roto2_pressure_fruit/plain_seed9400';before=set(parent.iterdir())
    proc=None
    def save():write(out/'STATUS.json',status)
    def evaluate(snapshot,folder):
        ec=[str(RUNTIME),'script/evaluate_roto2_sprint_midpoint.py','--run',str(out/'baseline'),
            '--snapshot',str(snapshot),'--output',str(folder),'--training-pid',str(proc.pid),
            '--skip-general-diagnostic']
        with (out/(folder.name+'.log')).open('w') as f:
            subprocess.run(ec,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        with (out/'train.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status.update(status='training',training_pid=proc.pid);save()
            run=None;reviewed=False
            while proc.poll() is None:
                if run is None:
                    found=set(parent.iterdir())-before
                    if found:
                        assert len(found)==1;run=next(iter(found));status['run']=str(run);save()
                progress=run/'progress.json' if run else None
                if progress and progress.exists() and not reviewed:
                    current=json.loads(progress.read_text())
                    if current['training_transitions']>=100000000:
                        with (out/'snapshot100m.log').open('w') as f:
                            subprocess.run([str(RUNTIME),'script/snapshot_roto2_milestones.py',
                                '--run',str(run),'--output',str(out),'--pid',str(proc.pid),
                                '--milestones',str(offset+100000000)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                        os.kill(proc.pid,signal.SIGSTOP)
                        snap=out/f'milestone_{offset+100000000}'
                        status.update(status='evaluating100m_training_held',snapshot100m=str(snap));save()
                        evaluate(snap,out/'check100m_pairs')
                        status['status']='awaiting100m_scientific_review_training_held';save()
                        receipt=out/'REVIEW100M.json'
                        while not receipt.exists():
                            if proc.poll() is not None:raise RuntimeError('Held trainer exited before review')
                            time.sleep(5)
                        decision=json.loads(receipt.read_text());assert decision['action'] in ('continue','stop')
                        status['review100m']=decision
                        if decision['action']=='stop':
                            proc.terminate();os.kill(proc.pid,signal.SIGCONT);proc.wait(timeout=60)
                            status.update(status='stopped_after100m_scientific_review',training_returncode=proc.returncode);save();return
                        os.kill(proc.pid,signal.SIGCONT);reviewed=True;status['status']='training_after100m_review';save()
                time.sleep(5)
            status['training_returncode']=proc.returncode;save()
            if proc.returncode:raise RuntimeError(f'Training exit {proc.returncode}')
        done=json.loads((run/'training_result.json').read_text())
        assert done['training_completed'] and done['finite_parameters']
        assert abs(done['training_transitions']-200000000)<8192*16
        cp=Path(done['final_checkpoint']);snapshot=out/'final_snapshot';snapshot.mkdir()
        import shutil
        shutil.copy2(cp,snapshot/'agent.pt')
        write(snapshot/'PROVENANCE.json',dict(snapshot_sha256=sha(cp),stable_source_and_progress_verified=True,
            cumulative_training_transitions=done['cumulative_training_transitions'],requested_milestone=done['cumulative_training_transitions'],
            final_checkpoint=True,training_transitions=done['training_transitions']))
        status.update(status='evaluating200m',training_result=done);save()
        evaluate(snapshot,out/'check200m_pairs')
        assert all(sha(ROOT/k)==v for k,v in frozen.items())
        status.update(status='completed_pending_scientific_and_visual_review',core_hashes_unchanged=True);save()
    except BaseException as e:
        if proc is not None and proc.poll() is None:
            proc.terminate();os.kill(proc.pid,signal.SIGCONT)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        status.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');save();raise


if __name__=='__main__':main()
