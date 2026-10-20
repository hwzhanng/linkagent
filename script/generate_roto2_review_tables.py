"""Generate readable indexes from package registries and adapt historical report links."""
from pathlib import Path
import json,re,os
root=Path(__file__).resolve().parents[1]; base=root/'research/roto2_review/2026-10-20'
runs=json.loads((base/'tables/runs.json').read_text()); ev=json.loads((base/'tables/evaluations.json').read_text())
fmt=lambda x: '未记录' if x is None else (f'{x:,}' if isinstance(x,int) else str(x))
lines=['# 全部运行与训练量','', '统计来自原始完成/进度JSON。训练完成只表示计算结束，不代表任务达标。累计继承量不可跨行相加。配置或事件单独存在的目录包括评估、探针、启动失败等，不能一律算作失败训练。','', '完整78目录：[CSV](tables/runs.csv) / [JSON](tables/runs.json)。原始TensorBoard：[标量CSV](tables/tensorboard_scalars.csv)，共2,310行；事件文件保存在各运行目录。','', '| 运行（配置） | 状态/版本/阶段 | 环境数 | 本次新增转移 | 权重累计转移 | PPO更新 | 秒 | 完成证据 |','|---|---|---:|---:|---:|---:|---:|---|']
for r in runs:
 if r['status']=='config_or_events_only':continue
 label=r['run'].replace('logs/roto2_pressure_fruit/','')
 cfg=r['config_path'] or r['run'];result=r['result_path'] or r['progress_path']
 lines.append(f"| [{label}](evidence/{cfg}) | {r['status']} / {r['reward_version']} / {r['reward_stage']} | {fmt(r['num_envs'])} | {fmt(r['additional_training_transitions'] if r['status']=='completed_training' else r['saved_progress_transitions'])} | {fmt(r['cumulative_lineage_transitions'])} | {fmt(r['ppo_updates'])} | {round(r['elapsed_seconds'],2) if r['elapsed_seconds'] else '未记录'} | [JSON](evidence/{result}) |")
lines += ['', '## 继承关系与原始配置','', '下表包含每次训练记录的来源；旧绝对路径是原机定位信息，恢复时使用仓库内对应相对路径。检查点在Release归档，参见[恢复说明](REPRODUCE.md)。','']
for r in runs:
 if r['status']=='config_or_events_only':continue
 lines += [f"- `{r['run']}`：源权重 `{r['resumed_from'] or '未记录/无恢复权重'}`；手姿 `{r['hand_posture']}`，域 `{r['domain_profile']}`，方向 `{r['goal_direction']}`；源码快照 `{r['source_snapshot_present']}`。"]
(base/'TRAINING_RUNS.md').write_text('\n'.join(lines)+'\n')
lines=['# 全部评估记录','', '**40份文档不是40个独立模型，也不是合计独立测试集。** 同检查点、同种子、不同课程或扰动条件可能重复。下表“成功”是各原JSON定义的成功；contact/regrasp等课程成功不可当作完整一圈成功。早期角速度指标和错误初始化保留作排错证据，不能混入有效旋转统计。','', '[完整CSV](tables/evaluations.csv) · [JSON与每项成功定义](tables/evaluations.json) · [2,048条逐回合CSV](tables/episodes.csv)','', '| 原始评估 | 回合 | 有效性 | 阶段/手姿/域 | 原定义成功率 | 失败率 | 净角度° | 严格承托率 |','|---|---:|---|---|---:|---:|---:|---:|']
pct=lambda x:'未记录' if x is None else f'{100*x:.3f}%'
for r in ev:
 lines.append(f"| [{r['source']}](evidence/{r['source']}) | {r['episodes']} | {r['validity']} | {r['stage']}/{r['posture']}/{r['profile']} | {pct(r['success_fraction'])} | {pct(r['failure_fraction'])} | {round(360*r['mean_directed_rotations'],4) if r['mean_directed_rotations'] is not None else '未记录'} | {pct(r['mean_secure_support_fraction'])} |")
lines += ['', '`pose_based_stage_specific`表示有姿态计数来源字段，仍需读该项成功定义和轨迹核验；它不是自动认证。`legacy_or_unverified_rotation_metric`不可作为已实现真实旋转的证据。`rejected_initialization`明确撤回。缺失值不补零。','', '当前最终权重的128回合汇总和最佳回报权重的64回合对照见[v7详细复核](reports/v7_review.md)。']
(base/'EVALUATIONS.md').write_text('\n'.join(lines)+'\n')
manifest={x['source_path']:x for x in json.loads((base/'manifest.json').read_text())}
reports=base/'reports';reports.mkdir(exist_ok=True)
for version in ['v6','v7']:
 for kind,src in [('design',f'artifacts/simulation/rotation_{version}/ANALYSIS_AND_DESIGN.md'),('review',f'artifacts/simulation/rotation_{version}/trial_seed{47 if version=="v6" else 48}/REVIEW.md')]:
  if version == 'v6' and kind == 'design': src=src.replace('ANALYSIS_AND_DESIGN','ANALYSIS_AND_CHANGES')
  text=(root/src).read_text()
  def replacement(m):
   path=m.group(1);item=manifest.get(path)
   if item and item['review_path']:dest=base/item['review_path']
   elif item: return '(../REPRODUCE.md)'
   else: dest=root/path
   return '('+os.path.relpath(dest,reports)+')'
  text=re.sub(r'\(/home/descfly/linkagent-ppo/([^\)]+)\)',replacement,text)
  text='> 阅读副本：仅将本机链接改为仓库链接；原始文档见[逐字证据副本](../evidence/'+src+')。模型链接统一指向恢复说明。\n\n'+text
  (reports/f'{version}_{kind}.md').write_text(text)
p=base/'METHOD_AND_PROTOCOL.md';p.write_text(p.read_text().replace('+r_motion =','r_motion ='))
print('Generated tables and four readable reports')
