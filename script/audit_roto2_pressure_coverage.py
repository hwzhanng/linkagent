"""CPU-only decomposition of saved sparse-pressure blind spots. No new rollout or core edit."""
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from simulation.roto2_pressure.core import surface_sites

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v9/sprint/pressure_coverage'


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    rows=[];combined={}
    categories=['no_unmasked_sparse_site_above_3500','lost_only_to_panel_mask',
                'masked_signal_present_but_cost_below_gate','risk_active']
    for train_seed in (8301,8302,8303):
        for eval_seed in (9101,9102):
            file=ROOT/f'artifacts/simulation/rotation_v8/stage3/results/jacobian_seed{train_seed}/seed{eval_seed}/action_trace_seed{eval_seed}.npz'
            with np.load(file) as a:
                # Trace records raw Pa; the controller divides by5kPa and clips.
                p=np.clip(a['sparse_pressure_pre'][1:].astype(np.float64)/5000,0,4)
                mask=a['sparse_mask_pre'][1:].astype(bool)
                peak=a['peak_pressure_pa'][:-1]
                active=a['active'][1:]&a['active'][:-1]&~a['done'][:-1]
            risk=(np.maximum(p-.7,0)**2*mask).sum(-1)/np.maximum(mask.sum(-1),1)
            fullrisk=(np.maximum(p-.7,0)**2).mean(-1)
            fullmax=p.max(-1)*5000;maskedmax=(p*mask).max(-1)*5000
            classification=[fullmax<=3500,(fullmax>3500)&(maskedmax<=3500),
                            (maskedmax>3500)&(risk<=1e-4),risk>1e-4]
            assert np.all(np.stack(classification).sum(0)==1)
            for start,end in ((1,5),(10,15),(18,23),(1,24)):
                times=np.arange(1,len(p)+1)*.05
                window=active&(times[:,None]>=start)&(times[:,None]<end)
                for threshold in (3500,10000,18000):
                    chosen=window&(peak>threshold);count=int(chosen.sum())
                    nums=[int((chosen&c).sum()) for c in classification]
                    assert sum(nums)==count
                    row=dict(training_seed=train_seed,evaluation_seed=eval_seed,window=f'{start}_{end}s',
                             dense_peak_threshold_pa=threshold,dense_hot_step_count=count,
                             categories=dict(zip(categories,nums)),
                             risk_active_with_all_panels_count=int((chosen&(fullrisk>1e-4)).sum()),
                             risk_active_with_recorded_mask_count=int((chosen&(risk>1e-4)).sum()))
                    rows.append(row);key=(row['window'],threshold)
                    if key not in combined:combined[key]=dict(window=row['window'],dense_peak_threshold_pa=threshold,
                        dense_hot_step_count=0,categories={c:0 for c in categories},
                        risk_active_with_all_panels_count=0,risk_active_with_recorded_mask_count=0)
                    agg=combined[key]
                    for k in ('dense_hot_step_count','risk_active_with_all_panels_count','risk_active_with_recorded_mask_count'):agg[k]+=row[k]
                    for c in categories:agg['categories'][c]+=row['categories'][c]
    for r in combined.values():
        den=r['dense_hot_step_count'];r['category_fractions']={k:(v/den if den else None) for k,v in r['categories'].items()}
    # Geometric diagnostic only: uniformly distributed single-contact positions,
    # same32mm sphere and10mm Gaussian width, no added sensor/control information.
    dense,sparse=[x.numpy().astype(np.float64) for x in surface_sites('cpu')]
    k=np.arange(20000);z=1-2*(k+.5)/len(k);phi=k*np.pi*(3-np.sqrt(5))
    sites=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],-1)*.032
    def geometry(layout):
        distance=np.linalg.norm(sites[:,None]-layout[None]*.032,axis=-1).min(-1)
        attenuation=np.exp(-distance**2/(2*.01**2))
        return dict(nearest_site_distance_mm_percentiles=dict(zip(['p50','p95','max'],(np.percentile(distance,[50,95,100])*1000).tolist())),
            single_patch_peak_attenuation_percentiles=dict(zip(['min','p05','p50'],np.percentile(attenuation,[0,5,50]).tolist())),
            fraction_detecting_7000pa_true_single_patch_peak_at_3500pa_threshold=float((7000*attenuation>3500).mean()))
    result=dict(source='Six completed V8 Jacobian first-episode traces,384 episodes across3 training seeds',
        episodes=384,training_seeds=3,per_trace=rows,step_weighted_aggregate=list(combined.values()),
        timing='Unmasked sparse pressure_pre[t] paired with post-action dense peak[t-1]; first/terminal/reset transitions excluded',
        categories_are_mutually_exclusive_and_exhaustive=True,
        pressure_units='Raw trace Pa divided by5000 and clipped to[0,4], exactly as the controller input',
        full_panel_case='Offline signal-coverage counterfactual on fixed trajectories; not a controller intervention or new policy evaluation',
        geometry_only=dict(sample_contact_positions=20000,fruit_radius_m=.032,gaussian_width_m=.01,dense96=geometry(dense),sparse24=geometry(sparse)),
        core_source_sha256=hashlib.sha256((ROOT/'simulation/roto2_pressure/core.py').read_bytes()).hexdigest(),
        no_new_dense_actor_input=True,no_sensor_or_core_change=True,no_training_change=True)
    (OUT/'COVERAGE_DECOMPOSITION.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 压力风险漏检来源分解','',
      '固定历史轨迹做离线分解，不改变训练、传感器或压力算法。384回合来自3个训练种子，时间步不是独立训练重复。',
      '时间对齐：动作前稀疏读数[t]对应前一步动作后密集峰值[t−1]，排除初始及重置边界。',
      '', '|窗口|密集峰值阈值|热点步骤数|全部稀疏点均低于3.5kPa|仅面板掉线导致|成本聚合/门槛导致|风险已激活|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in combined.values():
        if r['dense_peak_threshold_pa'] not in (3500,10000):continue
        fractions=[r['category_fractions'][k] for k in categories]
        if r['dense_hot_step_count']:
            lines.append(f'|{r["window"]}|{r["dense_peak_threshold_pa"]}Pa|{r["dense_hot_step_count"]}|'+ '|'.join(f'{f:.1%}' for f in fractions)+'|')
    lines+=['','3.5kPa是当前风险函数的阈值，不是18kPa物理终止阈值。所有列均以对应密集热点步骤数为分母。',
      '“全部稀疏点低于阈值”使用掉线前24点读数，不能归咎于掉线；“仅掉线”表示至少一个完整面板读数超阈值，但所有有效读数均未超阈值。',
      '纯几何比较使用同一球面/高斯模型和均匀单接触位置，仅刻画布局覆盖，不是现实果实传感器标定，也不是控制效果。',
      '下一阶段应先验证可观测热点上的同策略降压，并明确稀疏布局的覆盖边界；这不能替代原有全表面密集峰值对照或缩小最终目标。关闭掉线不能恢复空间上未采到的热点。任何布局/尺度改动需单独报告并保留原条件对照；当前没有实施。']
    (OUT/'PRESSURE_COVERAGE.md').write_text('\n'.join(lines)+'\n')
    for r in combined.values():
        if r['dense_peak_threshold_pa']==3500:print(json.dumps(r),flush=True)
    print(json.dumps(result['geometry_only']),flush=True)


if __name__=='__main__':main()
