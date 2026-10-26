"""Summarize actual PPO observations with the predeclared materiality screen."""
import argparse
import json
from pathlib import Path


def summarize(rows):
    total = sum(r['n'] for r in rows)
    error = sum(r['unclipped_squared_error_sum'] for r in rows)
    def fraction(key):
        return sum(r[key] for r in rows) / max(total, 1)
    return dict(minibatches=len(rows), prediction_evaluations=total,
                clipped_fraction=fraction('clipped_n'),
                wrong_direction_clipped_fraction=fraction('wrong_direction_clipped_n'),
                harmful_clipped_fraction=fraction('harmful_clipped_n'),
                clipped_zero_gradient_fraction=fraction('clipped_zero_gradient_n'),
                unclipped_nonzero_gradient_fraction=fraction('unclipped_nonzero_gradient_n'),
                unclipped_mse=error / max(total, 1),
                clipped_error_share=sum(r['clipped_squared_error_sum'] for r in rows)/max(error, 1e-20),
                wrong_direction_error_share=sum(r['wrong_direction_squared_error_sum'] for r in rows)/max(error, 1e-20))


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output
    plan=json.loads((out/'PLAN.json').read_text())
    rows=[json.loads(line) for line in (out/'observed/minibatches.jsonl').read_text().splitlines()]
    completion=json.loads((out/'observed/OBSERVER_COMPLETION.json').read_text())
    assert completion['completed'] and completion['updates']==plan['updates']
    assert len(rows)==plan['updates']*32
    assert all(r['clipped_nonzero_gradient_n']==0 for r in rows)
    assert len({(r['update'],r['epoch'],r['minibatch']) for r in rows})==len(rows)
    per_update=[dict(update=u,**summarize([r for r in rows if r['update']==u]))
                for u in range(1,plan['updates']+1)]
    gate=plan['predeclared_materiality_screen']
    hits=[r['update'] for r in per_update if r['update']>gate['ignore_startup_updates']
          and r['wrong_direction_clipped_fraction']>=gate['wrong_direction_clipped_fraction_at_least']
          and r['wrong_direction_error_share']>=gate['wrong_direction_squared_error_share_at_least']]
    result=dict(completed=True,source_policy_only_one_seed=True,
                interaction_transitions=plan['training_transitions'],
                repeated_predictions_are_not_independent_transitions=True,
                overall=summarize(rows),
                after_startup=summarize([r for r in rows if r['update']>gate['ignore_startup_updates']]),
                by_epoch=[dict(epoch=e,**summarize([r for r in rows if r['epoch']==e])) for e in range(8)],
                per_update=per_update,materiality_updates=hits,
                materiality_screen_passed=len(hits)>=gate['minimum_updates_meeting_both'],
                behavior_causality_proven=False,core_modified=False,formal_training_started=False)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    a=result['after_startup']
    report=f'''# V16 真实 PPO 价值梯度诊断

完成{plan['updates']}次更新、{plan['training_transitions']:,}条交互、{len(rows):,}个 minibatch。8192环境，训练环境8176，{plan['episode_seconds']}秒时限，成熟源检查点和原奖励，保留探索与优化器状态。诊断策略不进入候选模型选择。

## 去除前8次启动更新后的统计

- 预测被裁剪：{a['clipped_fraction']:.4%}
- 朝目标反方向移动且被裁剪：{a['wrong_direction_clipped_fraction']:.4%}
- 反方向裁剪样本占未裁剪平方误差：{a['wrong_direction_error_share']:.4%}
- 所有裁剪样本的实际 value loss 梯度均为零：已通过 hook 验证。
- 未裁剪且存在非零梯度：{a['unclipped_nonzero_gradient_fraction']:.4%}
- 达到事前双门槛的非启动更新：{len(hits)}/{plan['updates']-8}；要求至少8次。筛查结果：{result['materiality_screen_passed']}。

每次更新重复使用同一 rollout 8轮，不能把所有预测次数当成独立交互或独立统计样本。该筛查只判断本次成熟策略续训是否出现值得优先处理的现象；不能证明修改损失会改善转动，也不能外推到其他检查点。

## 对实现的校正

硬裁剪不是本仓库独有的确定性 bug：[Stable-Baselines3 官方实现](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html)在启用价值裁剪时采用相同的预测裁剪后 MSE，默认关闭该选项；[CleanRL 官方实现](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)采用裁剪与未裁剪平方误差的较大值。实现存在不同取舍，不能仅凭合成例子选择算法。

本轮未改变损失、反向传播、奖励、actor 或仿真核心。记录器通过参数、梯度、Adam状态逐位相等及随机数不变的独立检查。实际诊断源码哈希和每个 minibatch 记录在 observed/ 下。
'''
    (out/'REPORT.md').write_text(report)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_update','by_epoch']},indent=2))


if __name__=='__main__':main()
