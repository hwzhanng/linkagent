"""Read-only physical review for the fresh long-horizon experiment.

Never resumes/stops training or writes a review decision. A 48s intermediate
screen cannot establish completion of the original24s validation or research.
"""
import argparse,json
from pathlib import Path
import numpy as np
from summarize_roto2_sprint import summarize
from scipy.spatial.transform import Rotation
from run_roto2_stage2 import write,sha


def contact_windows(path):
    report=json.loads(path.read_text())
    with np.load(path.with_suffix('.npz')) as z:
        active=z['active'].astype(bool);pads=z['pads'].astype(bool)
        q=np.concatenate([z['initial_pose'][None],z['poses']])[...,3:]
        m=Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3)
        diff=np.diff(np.arctan2(m[...,1,0],m[...,0,0]),axis=0)
        delta=np.arctan2(np.sin(diff),np.cos(diff))
    # Keep denominator over all episodes separate from occupancy while alive.
    valid=active & (np.abs(delta)<.5)
    for ei,row in enumerate(report['per_episode']):
        if row['failed']:valid[max(int(active[:,ei].sum())-1,0),ei]=False
    result={}
    for start,end in ((10,24),(24,48),(38,48)):
        sl=slice(round(start/.05),min(len(active),round(end/.05)));mask=valid[sl];count=int(mask.sum());present=pads[sl].any(-1)&mask
        result[f'{start}_{end}s']=dict(episodes=active.shape[1],time_steps=mask.shape[0],valid_active_steps=count,
            effective_any_pad_fraction_active_steps=float(present.sum()/count) if count else None,
            effective_any_pad_fraction_all_episode_time=float(present.mean()) if mask.size else None,
            valid_active_fraction=float(mask.mean()) if mask.size else None)
    return result


def compare_pair(before,after,horizon=48):
    with np.load(before.with_suffix('.npz')) as b,np.load(after.with_suffix('.npz')) as a:
        errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('initial_pose','target_direction','target_speed')}
    assert max(errors.values())<1e-7,errors
    old=summarize(before,horizon);new=summarize(after,horizon)
    return dict(initial_condition_max_errors=errors,safe_full_change=new['safe_full']-old['safe_full'],
        failures_change=new['physical_failures']-old['physical_failures'],
        safe_two_turn_change=new['safe_at_least_two_turns']-old['safe_at_least_two_turns'],
        late_window_net_angle_changes={k:new['windows'][k]['mean_net_degrees_all_episodes']-old['windows'][k]['mean_net_degrees_all_episodes'] for k in old['windows']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
    status=json.loads((root/'STATUS.json').read_text());plan=json.loads((root/'PLAN.json').read_text());evaluations={};pairs={};traces={}
    for row in status['evaluations']:
        if row['status']!='completed':continue
        path=root/row['label']/'evaluation.json';r=summarize(path,48)
        assert r['episodes']==64
        if row['label'].startswith('source_'):assert row['checkpoint_sha256']==plan['reference_checkpoint_sha256']
        evaluations[row['label']]={**r,'contact_windows':contact_windows(path)}
    for milestone in ('at100m','at200m'):
        for condition in plan['evaluations']:
            label=condition['label'];before='source_'+label;after=milestone+'_'+label
            if before in evaluations and after in evaluations:pairs[after]=compare_pair(root/before/'evaluation.json',root/after/'evaluation.json')
        trace=root/(milestone+'_nominal48_trace.json');label=milestone+'_nominal48'
        if trace.exists() and label in evaluations:
            with np.load(trace.with_suffix('.npz')) as tr,np.load(root/label/'evaluation.npz') as ev:
                assert tr['pose'].shape==ev['poses'].shape
                error=float(np.abs(tr['pose']-ev['poses']).max());assert error<1e-6
            traces[milestone]=dict(pose_trace_max_error=error,diagnosis=json.loads(trace.read_text()),json_sha256=sha(trace),npz_sha256=sha(trace.with_suffix('.npz')))
    readiness={m:all(m+'_'+c['label'] in pairs for c in plan['evaluations']) and m in traces for m in ('at100m','at200m')}
    result=dict(pipeline_status=status['status'],evaluations=evaluations,matched_pairs=pairs,action_traces=traces,
        review_evidence_ready=readiness,scientific_decision_written=False,training_mutated=False,goal_complete=False,
        final_validation_unproven=['Original24s evaluation on independent seeds','Actual video review','48s repeated rotation with effective fingertip contact','Tactile tradeoff curve','Equal-budget plain/scalar/jacobian >=3 training seeds with comparable rotation'],
        limitations=['All data are within one ancestral grasp family','Failure-inclusive net angles use all64 episodes; contact metrics have explicit separate denominators','Pad recontact is not proof of full-finger unloading','A fresh100M model need not beat a mature source to show learning; assess trajectory and failure mechanism before continuing','48s screen cannot substitute for original24s validation'])
    write(root/'PHYSICAL_REVIEW.json',result)
    lines=['# 新初始化长时域训练：物理复核','',f'运行状态：{status["status"]}。这里只汇总证据，不自动做继续训练决定。','',
        '|评估|安全整圈/64|失败/64|安全两圈/64|24–48秒净角°，全部回合|38–48秒净角°，全部回合|后10秒有效指腹占比，存活有效步|','|---|---:|---:|---:|---:|---:|---:|']
    for label,r in evaluations.items():
        pad=r['contact_windows']['38_48s']['effective_any_pad_fraction_active_steps'];padtext='无有效步' if pad is None else f'{100*pad:.2f}%'
        lines.append(f'|{label}|{r["safe_full"]}|{r["physical_failures"]}|{r["safe_at_least_two_turns"]}|{r["windows"]["24_48s"]["mean_net_degrees_all_episodes"]:.2f}|{r["windows"]["38_48s"]["mean_net_degrees_all_episodes"]:.2f}|{padtext}|')
    lines+=['','100M/200M复核材料就绪：'+json.dumps(readiness),'','有效指腹离开/返回不等于整根手指已卸载。表内48秒结果不能替代原24秒正式验收、视频和后续压力创新验证。','', '关节限位、探索标准差、配对初始条件误差、逐回合接触恢复与分母见 PHYSICAL_REVIEW.json。']
    (root/'PHYSICAL_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(evaluations=len(evaluations),matched_pairs=len(pairs),ready=readiness)))

if __name__=='__main__':main()
