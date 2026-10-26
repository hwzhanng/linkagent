"""Prepare/record representative actual V17 rollouts after the pair has finished."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from review_roto2_reward_state_pair import inspect_episode_file
from validate_roto2_stage1_run import verify_video_artifact

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v17/ppo_pair'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT/'videos');args=parser.parse_args()
    status=json.loads((OUT/'STATUS.json').read_text());plan=json.loads((OUT/'PLAN.json').read_text())
    assert status['status']=='completed_pair_pending_scientific_review', 'Wait for the existing pair; do not rerun it'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    source=next(c for c in plan['evaluations'] if c['label']=='nominal48')
    checkpoints={'source':plan['source_checkpoint']}
    checkpoints.update({r['mode']:r['result']['final_checkpoint'] for r in status['training_runs']})
    jobs=[]
    for model in ['source','masked','validity']:
        path=OUT/(model+'_nominal48')/'evaluation.json'
        review,_=inspect_episode_file(path,48);evaluation=json.loads(path.read_text())
        late=np.asarray(review['windows']['24_48s']['raw_net_degrees_by_episode'])
        order=np.argsort(late,kind='stable');median=int(order[len(order)//2])
        safe=np.flatnonzero(review['safe_by_episode'])
        best=int(safe[np.argmax(late[safe])]) if len(safe) else int(np.argmax(late))
        for label,index in [('median_all_episodes',median),('best_safe' if len(safe) else 'best_available_no_safe_success',best)]:
            video=output/f'{model}_{label}_env{index}.mp4';cmd=source['command'].copy();cmd[2]='record'
            i=cmd.index('--evaluation-output');del cmd[i:i+2]
            cmd[cmd.index('--evaluate-checkpoint')+1]=checkpoints[model]
            cmd+=['--record-video',str(video),'--video-env-index',str(index),'--video-max-steps','960']
            jobs.append(dict(model=model,label=label,env=index,seed=source['seed'],checkpoint=checkpoints[model],
                             evaluation=str(path),late24_48_degrees=float(late[index]),episode=evaluation['per_episode'][index],
                             command=cmd,video=str(video)))
    video_plan=dict(selection='Median over all64 by24-48s net angle; best safe by same metric, or explicitly labeled best available failure if no safe success.',jobs=jobs)
    manifest=output/'VIDEO_PLAN.json'
    if manifest.exists():assert json.loads(manifest.read_text())==video_plan
    else:manifest.write_text(json.dumps(video_plan,indent=2)+'\n')
    if not args.execute:
        print(json.dumps({'prepared_videos':len(jobs),'plan':str(manifest),'recording_started':False}));return
    active=subprocess.run(['systemctl','--user','is-active','--quiet','roto2-v17-reward-state-pair.service'])
    assert active.returncode!=0, 'Training/evaluation service must exit before GPU recording'
    state_path=output/'RECORDINGS.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(status='recording',videos=[])
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for job in jobs:
            video=Path(job['video'])
            completed=next((r for r in state['videos'] if r['video']==str(video) and r.get('verified')),None)
            if completed:
                assert hashlib.sha256(video.read_bytes()).hexdigest()==completed['sha256'];continue
            assert not video.exists(), 'Preserve incomplete recordings; use a separate output directory'
            row={**job,'verified':False};state['videos'].append(row);save()
            with video.with_suffix('.log').open('w') as log:
                proc=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row['returncode']=proc.returncode
            assert proc.returncode==0
            row['artifact_checks']=verify_video_artifact(video,Path(job['checkpoint']),job['seed'],job['env'])
            meta=json.loads(video.with_suffix('.json').read_text());recorded=np.asarray([r['fruit_pose_native'] for r in meta['trace']])
            with np.load(Path(job['evaluation']).with_suffix('.npz')) as data:
                target=data['poses'][:len(recorded),job['env']]
            assert recorded.shape==target.shape
            error=float(np.abs(recorded-target).max());assert error<1e-6,error
            row['matched_pose_max_error']=error
            indices=np.linspace(0,len(recorded)-1,6,dtype=int)
            select='+'.join(f'eq(n,{int(i)})' for i in indices)
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-vf',
                            "select='"+select+"',scale=640:-1,tile=3x2",'-frames:v','1',str(video.with_suffix('.jpg'))],check=True)
            row.update(verified=True,sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                       storyboard=str(video.with_suffix('.jpg')),full_visual_review_completed=False)
            save()
        state.update(status='completed_pending_visual_review',video_count=len(state['videos']),task_success_claimed=False);save()
    except BaseException as error:
        state.update(status='failed_preserved',error=repr(error));save();raise


if __name__=='__main__':main()
