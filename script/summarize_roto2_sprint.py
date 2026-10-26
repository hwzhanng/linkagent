"""Read-only sustained-motion metrics for the fixed V9 experiment.

Keeps original success definitions. Supplemental two-turn and pad-recontact
metrics never substitute for pressure benefit or a verified unloaded finger.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation


def recontact_bouts(pads, active, dt=.05):
    """Stable contact, >=0.5s off, then >=0.25s on, independently per pad."""
    count=np.zeros(pads.shape[1:],dtype=np.int64)
    on=np.zeros_like(count);off=np.zeros_like(count)
    seen=np.zeros_like(count,dtype=bool);armed=seen.copy()
    for t in range(round(1/dt),len(pads)):
        live=active[t,:,None]
        contact=pads[t]&live
        on=np.where(contact,on+1,0)
        off=np.where(live&~pads[t],off+1,0)
        armed|=seen&(off>=round(.5/dt))&live
        recovered=armed&(on>=round(.25/dt))&live
        count+=recovered;armed[recovered]=False
        seen|=(on>=round(.25/dt))&live
    return count


def summarize(path,horizon):
    report=json.loads(path.read_text())
    assert report['simulation_physx_rollout_completed']
    with np.load(path.with_suffix('.npz')) as z:
        poses=z['poses'];initial=z['initial_pose'];active=z['active'].astype(bool)
        pads=z['pads'].astype(bool);direction=z['target_direction']
    assert len(report['per_episode'])==poses.shape[1]
    q=np.concatenate([initial[None],poses])[...,3:]
    m=Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3)
    heading=np.arctan2(m[...,1,0],m[...,0,0]);diff=np.diff(heading,axis=0)
    delta=np.arctan2(np.sin(diff),np.cos(diff))*direction
    total=(delta*active).sum(0)/(2*np.pi)
    recorded=np.asarray([r['directed_pose_turns'] for r in report['per_episode']])
    assert np.max(np.abs(total-recorded))<1e-5
    failed=np.asarray([r['failed'] for r in report['per_episode']])
    safe=np.asarray([r['success'] for r in report['per_episode']])
    supported=np.asarray([r['supported_new_turns'] for r in report['per_episode']])
    # Remove the last failure frame from supplemental contact/speed metrics.
    diagnostic_active=active.copy()
    for ei in np.flatnonzero(failed):
        diagnostic_active[max(int(active[:,ei].sum())-1,0),ei]=False
    diagnostic_active &= np.abs(delta)<.5
    bouts=recontact_bouts(pads,diagnostic_active)
    windows={}
    for start,end in [(1,5),(10,20),(max(1,horizon-10),horizon)]+([(24,48)] if horizon>=48 else []):
        key=f'{start:g}_{end:g}s';sl=slice(round(start/.05),min(len(active),round(end/.05)))
        mask=diagnostic_active[sl];den=int(mask.sum())
        net=(delta[sl]*active[sl]).sum(0)
        windows[key]=dict(all_episode_denominator=len(failed),
            mean_net_degrees_all_episodes=float(np.degrees(net).mean()),
            episodes_with_observed_steps=int(active[sl].any(0).sum()),
            valid_active_step_denominator=den,
            directed_speed_rad_s_active_steps=float((delta[sl]*mask).sum()/(.05*den)) if den else None,
            effective_pad_count_active_steps=float((pads[sl].sum(-1)*mask).sum()/den) if den else None,
            mean_net_degrees_surviving_episodes=float(np.degrees(net[~failed]).mean()) if (~failed).any() else None)
    return dict(file=str(path),json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),
        horizon_seconds=horizon,episodes=len(failed),safe_full=int(safe.sum()),physical_failures=int(failed.sum()),
        safe_at_least_two_turns=int((safe&(total>=2)&(supported>=2)).sum()),
        mean_turns=float(total.mean()),mean_episode_peak_pressure_pa=report['mean_peak_pressure_proxy_pa'],
        windows=windows,effective_pad_recontact_bouts_by_episode=bouts.tolist(),
        episodes_with_any_pad_recontact=int((bouts.sum(-1)>0).sum()),
        episodes_with_repeated_same_pad_recontact=int((bouts.max(-1)>=2).sum()),
        mean_recontacts_per_episode=float(bouts.sum(-1).mean()),pose_metric_independently_matched=True,
        interpretation='Pad contact OFF does not prove whole-finger physical unloading. Single training seed, diagnostic episode counts, no pressure innovation claim.')


def update(root):
    status=json.loads((root/'STATUS.json').read_text()); rows=[]
    for ev in status.get('evaluations',[]):
        if ev['status']!='completed':continue
        rows.append(dict(label=ev['label'],**summarize(root/ev['label']/f'evaluation_seed{ev["seed"]}.json',ev['horizon_seconds'])))
    names={r['label']:r for r in rows};pairs=[]
    for label,r in names.items():
        if label.startswith('source_') and label[7:] in names:
            after=names[label[7:]]
            pairs.append(dict(source=label,final=label[7:],safe_full_difference=after['safe_full']-r['safe_full'],
                physical_failure_difference=after['physical_failures']-r['physical_failures'],
                safe_two_turn_difference=after['safe_at_least_two_turns']-r['safe_at_least_two_turns'],
                mean_turn_difference=after['mean_turns']-r['mean_turns']))
    result=dict(pipeline_status=status['status'],evaluations=rows,matched_condition_pairs=pairs,
        complete_goal=False,success_gate_unchanged=True,training_mutated=False)
    tmp=root/'SUSTAINED_MOTION.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'SUSTAINED_MOTION.json')
    lines=['# 持续转动与接触恢复诊断','',
           '|评估|整圈安全成功|至少两圈安全成功|物理失败|重新接触的回合|',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["label"]}|{r["safe_full"]}/{r["episodes"]}|{r["safe_at_least_two_turns"]}/{r["episodes"]}|{r["physical_failures"]}/{r["episodes"]}|{r["episodes_with_any_pad_recontact"]}/{r["episodes"]}|')
    lines+=['','补充指标不改变原成功标准。两圈要求同时满足原安全成功、实际净转角≥2圈与承托下新增转角≥2圈。',
            '重新接触定义：已有至少0.25秒有效指腹接触，连续离开有效接触至少0.5秒，再连续接触至少0.25秒。有效指腹接触消失不等同整根手指已卸载，不能单凭此指标宣布步态成功。',
            '后半程净转角以全部评估回合为分母，失败后的时段不计运动；有效步转速和接触数单独列出步数分母。单训练种子结果不能代替多种子压力研究。','',
            '详细窗口统计及每回合事件次数：SUSTAINED_MOTION.json。']
    (root/'SUSTAINED_MOTION.md').write_text('\n'.join(lines)+'\n')
    return status['status']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    last=None
    while True:
        signature=(a.run/'STATUS.json').read_bytes()
        if a.watch and signature==last:
            time.sleep(60);continue
        state=update(a.run)
        last=signature
        if not a.watch or state.startswith(('completed','failed')):break
        time.sleep(60)


if __name__=='__main__':main()
