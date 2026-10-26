"""One fixed 600M task-curriculum continuation, with preserved checkpoints and evaluations."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_roto2_stage2 import ROOT, RUNTIME, command, sha, write, WEIGHTS, DEFAULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true')
    a=p.parse_args();out=a.output.resolve();lib=a.library.resolve();held=a.heldout.resolve()
    receipt=json.loads(a.validation.read_text())
    assert receipt['training_library_ready'] and receipt['training_library_sha256']==sha(lib)
    assert receipt['heldout_library_sha256']==sha(held)
    source=ROOT/'logs/roto2_pressure_fruit/plain_seed7401/2026-10-21_16-16-13'
    cfg=json.loads((source/'run_config.json').read_text())
    result=json.loads((source/'training_result.json').read_text())
    assert result['training_completed'] and result['finite_parameters']
    checkpoint=Path(result['final_checkpoint']); source_sha=sha(checkpoint)
    assert source_sha=='6108ba5deaa7f535d02bcfc48f6cd2931b9007de5f3d9fc9c1b4841ca314f216'
    cfg.update(reset_grasps_file=str(lib),domain_profile='gentle')
    spec=dict(num_envs=8192,training_seed=9400,additional_samples_per_arm=600_000_000,
              rollout_steps=16,common_resume_action_std=.5)
    cmd=command(cfg,spec,{},checkpoint)+['--episode-length-seconds','48']
    if a.preview: print(json.dumps(cmd,indent=2));return
    assert not (out/'STATUS.json').exists(),'Preserve existing attempt; explicit recovery required'
    out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'artifacts/simulation/rotation_v9/sprint/FROZEN_CORE.json').read_text())
    def verify_core():
        assert all(sha(ROOT/f)==digest for f,digest in frozen.items()),'Frozen core changed'
    verify_core()
    status=dict(status='evaluating_source',command=cmd,source_checkpoint_sha256=source_sha,
                training_started=False,goal_complete=False,evaluations=[],helpers=[])
    path=out/'STATUS.json';write(path,status)
    write(out/'PLAN.json',dict(training=spec,source_run=str(source),source_checkpoint_sha256=source_sha,
          training_library=str(lib),training_library_sha256=sha(lib),heldout_library=str(held),
          heldout_library_sha256=sha(held),physical_validation=str(a.validation.resolve()),
          training_horizon_seconds=48,original_evaluation_horizon_seconds=24,
          recipe_change='New reset distribution, gentle domain,48s task horizon; existing reward weights unchanged',
          no_core_changes=True,github_sync=False))
    handles=[];helpers=[]
    def spawn_helper(args,log):
        f=log.open('w');handles.append(f)
        proc=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);helpers.append(proc)
        status['helpers'].append(dict(pid=proc.pid,command=args));write(path,status)
        return proc
    def evaluate(run,cp,label,library,domain,seconds,seed):
        verify_core();folder=out/label;folder.mkdir(exist_ok=True)
        ec=['bash','simulation/.runtime/run_roto2.sh','evaluate','--num-envs','64','--seed',str(seed),
            '--pressure-variant','plain','--domain-profile',domain,'--hand-posture','palm_up',
            '--reward-stage','baseline','--goal-direction','positive','--reset-grasps',str(library),
            '--episode-length-seconds',str(seconds),'--evaluate-checkpoint',str(cp)]
        for key,default in zip(WEIGHTS,DEFAULTS):ec+=['--'+key.replace('_','-'),str(cfg.get(key,default))]
        file=folder/f'evaluation_seed{seed}.json';ec+=['--evaluation-output',str(file)]
        row=dict(label=label,seed=seed,domain=domain,horizon_seconds=seconds,library=str(library),
                 checkpoint_sha256=sha(cp),command=ec,status='evaluating')
        status['evaluations'].append(row);write(path,status)
        with (folder/'evaluation.log').open('w') as log:
            subprocess.run(ec,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=json.loads(file.read_text());assert d['simulation_physx_rollout_completed'] and d['episodes']==64
        row.update(status='completed',safe_full=sum(r['success'] for r in d['per_episode']),
                   failures=sum(r['failed'] for r in d['per_episode']),
                   mean_after10_degrees=d['mean_after_10s_directed_turns']*360)
        write(folder/'DIAGNOSTIC_PIPELINE.json',dict(status='completed',scope='evaluation_only',commands=[dict(command=ec)]))
        write(path,status)
        return folder
    try:
        old=ROOT/'simulation/roto2_pressure/reset_grasps_v8_validated256.json'
        evaluate(source,checkpoint,'source_old_nominal24_seed9501',old,'nominal',24,9501)
        evaluate(source,checkpoint,'source_heldout_gentle24_seed9501',held,'gentle',24,9501)
        evaluate(source,checkpoint,'source_heldout_gentle48_seed9601',held,'gentle',48,9601)
        status.update(status='training',training_started=True);write(path,status)
        parent=ROOT/'logs/roto2_pressure_fruit/plain_seed9400'
        before=set(parent.iterdir()) if parent.exists() else set()
        with (out/'train.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['training_pid']=proc.pid;write(path,status)
            while proc.poll() is None:
                dirs=set(parent.iterdir())-before if parent.exists() else set()
                if dirs and 'run' not in status:
                    assert len(dirs)==1;run=next(iter(dirs));status['run']=str(run);write(path,status)
                    milestone=result['cumulative_training_transitions']+300_000_000
                    spawn_helper([str(RUNTIME),'script/snapshot_roto2_milestones.py','--run',str(run),
                                  '--output',str(out),'--pid',str(proc.pid),'--milestones',str(milestone)],out/'snapshot.log')
                    spawn_helper([str(RUNTIME),'script/evaluate_roto2_snapshot.py','--run',str(run),
                                  '--snapshot',str(out/f'milestone_{milestone}'),'--training-pid',str(proc.pid),
                                  '--seed','9401'],out/'milestone_evaluation.log')
                time.sleep(10)
            status['training_returncode']=proc.returncode;write(path,status)
            if proc.returncode:raise RuntimeError(f'Training exit {proc.returncode}')
        run=Path(status['run']);done=json.loads((run/'training_result.json').read_text())
        assert done['training_completed'] and done['finite_parameters']
        assert abs(done['cumulative_training_transitions']-result['cumulative_training_transitions']-600_000_000)<8192*16
        cp=Path(done['final_checkpoint']);status.update(status='evaluating',training_result=done,final_checkpoint_sha256=sha(cp));write(path,status)
        old=ROOT/'simulation/roto2_pressure/reset_grasps_v8_validated256.json'
        for seed in (9501,9502):
            evaluate(run,cp,f'old_nominal24_seed{seed}',old,'nominal',24,seed)
            evaluate(run,cp,f'heldout_gentle24_seed{seed}',held,'gentle',24,seed)
        extended=evaluate(run,cp,'heldout_gentle48_seed9601',held,'gentle',48,9601)
        video_folder=out/'heldout_gentle24_seed9501'
        with (out/'videos.log').open('w') as log:
            subprocess.run([str(RUNTIME),'script/record_roto2_evaluation_examples.py','--folder',str(video_folder),
                            '--seed','9501','--output',str(out/'videos')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for h in helpers:
            if h.wait():raise RuntimeError('Milestone helper failed; final artifacts preserved')
        verify_core();status.update(status='completed_pending_scientific_and_visual_review',core_hashes_unchanged=True)
        write(path,status)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}');write(path,status)
        raise
    finally:
        for f in handles:f.close()


if __name__=='__main__':main()
