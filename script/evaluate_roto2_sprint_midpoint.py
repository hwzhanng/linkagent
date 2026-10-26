"""Matched source/midpoint evaluation of an immutable checkpoint; never controls training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from run_roto2_stage2 import ROOT, RUNTIME, write, sha
from summarize_roto2_sprint import update


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--skip-general-diagnostic',action='store_true',
                   help='Run paired evaluations directly when no general diagnostic is scheduled for this snapshot')
    p.add_argument('--preview',action='store_true');a=p.parse_args()
    root=a.run.resolve();snap=a.snapshot.resolve();out=a.output.resolve()
    original=json.loads((root/'STATUS.json').read_text())
    source_rows=[r for r in original['evaluations'] if r['label'].startswith('source_')]
    assert len(source_rows)==3 and all(r['status']=='completed' for r in source_rows)
    commands=[]
    for src in source_rows:
        cmd=list(src['command']);label=src['label'][7:]
        cmd[cmd.index('--evaluate-checkpoint')+1]=str(snap/'agent.pt')
        cmd[cmd.index('--evaluation-output')+1]=str(out/label/f'evaluation_seed{src["seed"]}.json')
        commands.append(dict(source_label=src['label'],label=label,command=cmd))
    if a.preview:print(json.dumps(commands,indent=2));return
    assert not (out/'STATUS.json').exists(),'Existing attempt preserved; no silent restart'
    out.mkdir(parents=True,exist_ok=True)
    status=dict(status='waiting_for_snapshot',training_mutated=False,evaluations=[],
                training_pid=a.training_pid,snapshot=str(snap),goal_complete=False)
    file=out/'STATUS.json';write(file,status)
    proc_path=Path(f'/proc/{a.training_pid}/stat')
    start_tick=proc_path.read_text().split(') ',1)[1].split()[19] if proc_path.exists() else None
    try:
        while not (snap/'PROVENANCE.json').exists():
            if not proc_path.exists() or proc_path.read_text().split(') ',1)[1].split()[19]!=start_tick:
                raise RuntimeError('Verified training process ended before the required snapshot; no restart attempted')
            time.sleep(30)
        prov=json.loads((snap/'PROVENANCE.json').read_text());cp=snap/'agent.pt'
        assert sha(cp)==prov['snapshot_sha256']
        assert prov['stable_source_and_progress_verified']
        assert prov['cumulative_training_transitions']>=prov['requested_milestone']
        write(out/'SNAPSHOT_PROVENANCE.json',prov)
        # Let the already-running general diagnostic finish before another GPU
        # evaluator is added. Training continues independently throughout.
        generic=snap/'DIAGNOSTIC_PIPELINE.json'
        status['status']='waiting_for_general_diagnostic';write(file,status)
        d={}
        while not a.skip_general_diagnostic:
            d=json.loads(generic.read_text()) if generic.exists() else {}
            if d.get('status') in ('completed','failed'):break
            helpers=json.loads((root/'STATUS.json').read_text()).get('helpers',[])
            ids=[h['pid'] for h in helpers if 'script/evaluate_roto2_snapshot.py' in h['command']]
            if ids and not any(Path(f'/proc/{i}').exists() for i in ids):break
            time.sleep(30)
        status['general_diagnostic_status']=('not_scheduled_for_this_snapshot' if a.skip_general_diagnostic
                                             else d.get('status','missing_or_process_ended'))
        frozen=json.loads((root.parent/'FROZEN_CORE.json').read_text())
        assert all(sha(ROOT/k)==v for k,v in frozen.items())
        for src in source_rows:
            source_folder=root/src['label'];folder=out/src['label'];folder.mkdir()
            for name in (f'evaluation_seed{src["seed"]}.json',f'evaluation_seed{src["seed"]}.npz'):
                shutil.copy2(source_folder/name,folder/name)
                assert sha(source_folder/name)==sha(folder/name)
            status['evaluations'].append(src)
        status['status']='evaluating';write(file,status);update(out)
        for src,plan in zip(source_rows,commands):
            folder=out/plan['label'];folder.mkdir()
            row={**src,'label':plan['label'],'command':plan['command'],
                 'checkpoint_sha256':sha(cp),'status':'evaluating'}
            for key in ('safe_full','failures','mean_after10_degrees'):row.pop(key,None)
            status['evaluations'].append(row);write(file,status)
            with (folder/'evaluation.log').open('w') as log:
                subprocess.run(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            evaluation=folder/f'evaluation_seed{src["seed"]}.json';r=json.loads(evaluation.read_text())
            assert r['simulation_physx_rollout_completed'] and r['episodes']==64
            with np.load(evaluation.with_suffix('.npz')) as measured, np.load(root/src['label']/f'evaluation_seed{src["seed"]}.npz') as baseline:
                errors={key:float(np.abs(measured[key]-baseline[key]).max())
                        for key in ('initial_pose','target_direction','target_speed')}
            assert max(errors.values())<1e-7,errors
            row.update(status='completed',safe_full=sum(ep['success'] for ep in r['per_episode']),
                       failures=sum(ep['failed'] for ep in r['per_episode']),
                       mean_after10_degrees=r['mean_after_10s_directed_turns']*360,
                       matched_initial_condition_max_errors=errors)
            write(folder/'DIAGNOSTIC_PIPELINE.json',dict(status='completed',scope='evaluation_only',
                  commands=[dict(command=row['command'])],source_pair=src['label']))
            write(file,status);update(out)
        # One48s action trace retains joint-limit and reward/contact evidence for
        # interpreting the paired outcome; it is not another training arm.
        last=status['evaluations'][-1];cmd=list(last['command']);i=cmd.index('--evaluation-output');del cmd[i:i+2]
        folder=out/last['label'];trace=folder/f'action_trace_seed{last["seed"]}.json'
        cmd+=['--trace-policy-output',str(trace)]
        status['status']='tracing_extended_episode';write(file,status)
        with (folder/'action_trace.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with np.load(trace.with_suffix('.npz')) as tr,np.load(folder/f'evaluation_seed{last["seed"]}.npz') as ev:
            assert np.max(np.abs(tr['pose']-ev['poses']))<1e-6
        assert all(sha(ROOT/k)==v for k,v in frozen.items())
        status.update(status='completed_pending_scientific_review',core_hashes_unchanged=True,
                      extended_trace_matches_evaluation=True,training_mutated=False)
        write(file,status);update(out)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}');write(file,status)
        raise


if __name__=='__main__':main()
