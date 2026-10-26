"""Fresh plain PPO, held for review at100M and hard-ended at200M.

The review receipt is written by the supervising researcher/agent after reading
physical evaluations. This runner never changes PPO, observations or dynamics.
"""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
from run_roto2_stage2 import ROOT,RUNTIME,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args();out=a.output.resolve()
    plan=json.loads((out/'PLAN.json').read_text());cmd=plan['training_command']
    assert '--resume-checkpoint' not in cmd and '--resume-action-std' not in cmd
    assert cmd[cmd.index('--million-steps')+1]=='200'
    if a.preview:print(json.dumps(plan,indent=2));return
    assert not (out/'STATUS.json').exists(),'Existing work must be preserved'
    frozen=json.loads((out/'FROZEN_CORE.json').read_text())
    def verify():
        assert all(sha(ROOT/k)==v for k,v in frozen.items()),'Frozen core changed'
        assert sha(Path(plan['training_library']))==plan['training_library_sha256']
    verify();status=dict(status='starting',goal_complete=False,training_running=False,evaluations=[],command=cmd,budget_samples=200000000,fresh_initialization=True)
    def save():write(out/'STATUS.json',status)
    save();proc=None;held=False
    parent=ROOT/'logs/roto2_pressure_fruit'/('plain_seed'+str(plan['training_seed']));before=set(parent.iterdir()) if parent.exists() else set();run=None
    def evaluate(checkpoint,label,condition):
        folder=out/label;folder.mkdir(exist_ok=False);ec=list(condition['command']);ec[ec.index('--evaluate-checkpoint')+1]=str(checkpoint);ec[ec.index('--evaluation-output')+1]=str(folder/'evaluation.json')
        row=dict(label=label,condition=condition['label'],checkpoint_sha256=sha(checkpoint),status='evaluating',command=ec);status['evaluations'].append(row);save()
        with (folder/'run.log').open('w') as log:subprocess.run(ec,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=json.loads((folder/'evaluation.json').read_text());assert d['simulation_physx_rollout_completed'] and d['episodes']==64
        row.update(status='completed',safe_full=sum(r['success'] for r in d['per_episode']),failures=sum(r['failed'] for r in d['per_episode']),mean_after10_degrees=d['mean_after_10s_directed_turns']*360);save()
    def evaluate_checkpoint(checkpoint,prefix,include_source=False):
        verify()
        for condition in plan['evaluations']:
            if include_source:evaluate(Path(plan['reference_checkpoint']),'source_'+condition['label'],condition)
            evaluate(checkpoint,prefix+'_'+condition['label'],condition)
        # Actions/joint limits are needed for a scientific decision, not just rewards.
        condition=plan['evaluations'][0];ec=list(condition['command']);ec[ec.index('--evaluate-checkpoint')+1]=str(checkpoint);i=ec.index('--evaluation-output');del ec[i:i+2];ec+=['--trace-policy-output',str(out/(prefix+'_nominal48_trace.json'))]
        with (out/(prefix+'_trace.log')).open('w') as log:subprocess.run(ec,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify()
    try:
        with (out/'train.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);status.update(status='training',training_running=True,training_pid=proc.pid,training_pid_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19]);save();reviewed=False
            while proc.poll() is None:
                if run is None and parent.exists():
                    found=set(parent.iterdir())-before
                    if found:assert len(found)==1;run=next(iter(found));status['run']=str(run);save()
                progress=run/'progress.json' if run else None
                if progress and progress.exists() and not reviewed:
                    try:current=json.loads(progress.read_text())
                    except json.JSONDecodeError:time.sleep(1);continue
                    if current['training_transitions']>=100000000:
                        with (out/'snapshot100m.log').open('w') as snaplog:subprocess.run([str(RUNTIME),'script/snapshot_roto2_milestones.py','--run',str(run),'--output',str(out),'--pid',str(proc.pid),'--milestones','100000000'],cwd=ROOT,stdout=snaplog,stderr=subprocess.STDOUT,check=True)
                        os.kill(proc.pid,signal.SIGSTOP);held=True;status.update(status='evaluating100m_training_held',training_running=False,training_process_held=True);save()
                        evaluate_checkpoint(out/'milestone_100000000/agent.pt','at100m',True)
                        status['status']='awaiting100m_scientific_review_training_held';save();receipt=out/'REVIEW100M.json'
                        while not receipt.exists():
                            if proc.poll() is not None:raise RuntimeError('Held trainer ended before review')
                            time.sleep(5)
                        decision=json.loads(receipt.read_text());assert decision['action'] in ('continue','stop');status['review100m']=decision
                        if decision['action']=='stop':
                            proc.terminate();os.kill(proc.pid,signal.SIGCONT);held=False;proc.wait(timeout=60);status.update(status='stopped_after100m_scientific_review',training_running=False,training_process_held=False,training_returncode=proc.returncode);save();return
                        os.kill(proc.pid,signal.SIGCONT);held=False;reviewed=True;status.update(status='training_after100m_review',training_running=True,training_process_held=False);save()
                time.sleep(5)
            status.update(training_returncode=proc.returncode,training_running=False);save();assert proc.returncode==0
        done=json.loads((run/'training_result.json').read_text());assert done['training_completed'] and done['finite_parameters'] and done['resumed_from'] is None
        assert abs(done['training_transitions']-200000000)<8192*16
        status.update(status='evaluating200m',training_result=done);save();evaluate_checkpoint(Path(done['final_checkpoint']),'at200m')
        status.update(status='completed200m_pending_scientific_review',core_hashes_unchanged=True,automatic_extension_disabled=True);save()
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            if held:os.kill(proc.pid,signal.SIGCONT)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        status.update(status='failed_preserved',training_running=False,error=repr(error));save();raise

if __name__=='__main__':main()
