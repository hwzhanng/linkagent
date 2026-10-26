"""Independent, failure-inclusive physical review of the fixed V17 pair.

Read-only with respect to training and original evaluation criteria.
Produces partial evidence while the pair is running, never a success receipt.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v17/ppo_pair'


def inspect_episode_file(path,horizon):
    r=json.loads(path.read_text());assert r['simulation_physx_rollout_completed']
    with np.load(path.with_suffix('.npz')) as z:
        poses=z['poses'];initial=z['initial_pose'];active=z['active'].astype(bool)
        pads=z['pads'].astype(bool);direction=z['target_direction'];speed=z['target_speed']
    n=active.shape[1];assert n==64 and len(r['per_episode'])==n
    q=np.concatenate([initial[None],poses])[...,3:]
    matrices=Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3)
    angle=np.arctan2(matrices[...,1,0],matrices[...,0,0]);change=np.diff(angle,axis=0)
    delta=np.arctan2(np.sin(change),np.cos(change))*direction
    actual=(delta*active).sum(0)/(2*np.pi)
    recorded=np.array([e['directed_pose_turns'] for e in r['per_episode']])
    assert np.max(np.abs(actual-recorded))<1e-5
    failed=np.array([e['failed'] for e in r['per_episode']]);success=np.array([e['success'] for e in r['per_episode']])
    valid=active & (np.abs(delta)<.5) & (matrices[1:,...,2,2]>.5)
    for env in np.flatnonzero(failed):valid[max(int(active[:,env].sum())-1,0),env]=False
    windows={}
    for start,end in [(10,horizon),((24 if horizon==48 else 10),horizon)]:
        name=f'{start}_{end}s'
        if name in windows:continue
        begin=round(start/.05);stop=round(end/.05);expected=stop-begin
        sl=slice(begin,min(stop,len(active)));mask=valid[sl]
        # All64 episodes remain in denominator; absent/failed future time gets zero contact.
        raw_degrees=np.degrees((delta[sl]*active[sl]).sum(0))
        stable_degrees=np.degrees((delta[sl]*mask).sum(0))
        anypad=(pads[sl].any(-1)&mask).sum(0)/expected
        threepad=((pads[sl].sum(-1)>=3)&mask).sum(0)/expected
        windows[name]=dict(expected_steps_per_episode=expected,episodes=n,
                           raw_net_degrees_by_episode=raw_degrees.tolist(),
                           valid_upright_net_degrees_by_episode=stable_degrees.tolist(),
                           any_pad_full_window_fraction_by_episode=anypad.tolist(),
                           three_pad_full_window_fraction_by_episode=threepad.tolist(),
                           mean_raw_net_degrees=float(raw_degrees.mean()),
                           mean_valid_upright_net_degrees=float(stable_degrees.mean()),
                           mean_any_pad_full_window_fraction=float(anypad.mean()),
                           mean_three_pad_full_window_fraction=float(threepad.mean()))
    recorded_late=np.array([e['after_10s_directed_turns'] for e in r['per_episode']])*360
    assert np.max(np.abs(np.array(windows[f'10_{horizon}s']['raw_net_degrees_by_episode'])-recorded_late))<.001
    return dict(episodes=n,safe_full=int(success.sum()),physical_failures=int(failed.sum()),
                safe_two_turns=int((success & (actual>=2) & (np.array([e['supported_new_turns'] for e in r['per_episode']])>=2)).sum()),
                safe_by_episode=success.tolist(),failed_by_episode=failed.tolist(),windows=windows,
                json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                npz_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),
                pose_reconstruction_max_error_turns=float(np.max(np.abs(actual-recorded))),
                mean_peak_pressure_proxy_pa=r['mean_peak_pressure_proxy_pa']),dict(initial_pose=initial,target_direction=direction,target_speed=speed)


def compare(before,after,window):
    result={}
    rng=np.random.default_rng(17880);draw=rng.integers(64,size=(5000,64))
    for key in ['raw_net_degrees_by_episode','valid_upright_net_degrees_by_episode',
                'any_pad_full_window_fraction_by_episode','three_pad_full_window_fraction_by_episode']:
        difference=np.array(after['windows'][window][key])-np.array(before['windows'][window][key])
        interval=np.quantile(difference[draw].mean(1),[.025,.975])
        result[key]=dict(mean_difference=float(difference.mean()),paired_episode_bootstrap95=interval.tolist())
    result.update(safe_full_change=after['safe_full']-before['safe_full'],
                  failure_change=after['physical_failures']-before['physical_failures'],
                  safe_two_turn_change=after['safe_two_turns']-before['safe_two_turns'])
    return result


def main():
    status=json.loads((OUT/'STATUS.json').read_text());plan=json.loads((OUT/'PLAN.json').read_text())
    results={};initial={};comparisons={}
    conditions={r['label']:r for r in plan['evaluations']}
    for record in status['evaluations']:
        assert record['status']=='completed'
        label=record['label'];condition=next(k for k in conditions if label.endswith('_'+k))
        result,starting=inspect_episode_file(OUT/label/'evaluation.json',conditions[condition]['horizon_seconds'])
        results[label]=result;initial[label]=starting
    for condition,cfg in conditions.items():
        horizon=cfg['horizon_seconds'];window=f'{24 if horizon==48 else 10}_{horizon}s'
        for old,new in [('source','masked'),('source','validity'),('masked','validity')]:
            a=old+'_'+condition;b=new+'_'+condition
            if a not in results or b not in results:continue
            errors={k:float(np.abs(initial[a][k]-initial[b][k]).max()) for k in initial[a]}
            assert max(errors.values())<1e-7,errors
            comparisons[b+'_vs_'+old]=dict(window=window,matched_initial_errors=errors,
                                           **compare(results[a],results[b],window))
    result=dict(pipeline_status=status['status'],evaluations=results,paired_comparisons=comparisons,
                complete_pair_evidence=len(results)==9 and len(comparisons)==9,
                success_criteria_changed=False,training_modified=False,goal_complete=False,
                confidence_boundary='Episode bootstrap within one training-seed pair; not training-seed generalization.',
                supplemental_contact_denominator='All64 episodes times the full planned window; missing/failed future steps count as zero; valid requires upright and no large pose step.',
                scientific_decision_written=False)
    (OUT/'PHYSICAL_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V17配对物理复核','',f'当前状态：{status["status"]}。完整配对材料：{result["complete_pair_evidence"]}。尚未作方法验收决定。','',
           '|模型/条件|安全整圈/64|失败/64|安全两圈/64|后段净角°，全64回合|后段≥1 / ≥3指腹接触，全计划时长|',
           '|---|---:|---:|---:|---:|---:|']
    for label,r in results.items():
        window='24_48s' if label.endswith('48') else '10_24s';w=r['windows'][window]
        lines.append(f'|{label}|{r["safe_full"]}|{r["physical_failures"]}|{r["safe_two_turns"]}|{w["mean_raw_net_degrees"]:.2f}|{w["mean_any_pad_full_window_fraction"]:.2%} / {w["mean_three_pad_full_window_fraction"]:.2%}|')
    lines+=['','后段为48秒评估的24–48秒或24秒评估的10–24秒。主净角保持原始实际位姿口径；额外剔除无效/倾斜步的净角独立保留，不能替换原成功判据。',
            '失败后的未观测时间在接触指标中计零，防止只看存活样本导致接触率虚高。触觉接触不等于整指卸载；需视频和动作轨迹验证。',
            '配对初始条件误差、逐回合数据和区间在PHYSICAL_REVIEW.json。区间只反映本组64个回合，不能替代多个训练种子。']
    (OUT/'PHYSICAL_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=status['status'],evaluations=len(results),comparisons=len(comparisons),complete_pair_evidence=result['complete_pair_evidence'])))


if __name__=='__main__':main()
