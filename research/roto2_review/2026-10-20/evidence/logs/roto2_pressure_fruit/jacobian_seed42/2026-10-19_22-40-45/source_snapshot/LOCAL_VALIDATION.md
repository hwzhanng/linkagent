# 本机 PPO 验收记录（2026-10-19）

目录：`/home/descfly/linkagent-ppo`。分支：`research/roto2-object-pressure-track`。
源码基线：`cd97e7a75c83ddd20584816d8089fc08fac2dcad`；修正保留为本地改动，尚未推送。

## 环境

- RTX 5090 32 GB，驱动 595.84；Python 3.12，Torch 2.10.0+cu128。
- Isaac Sim 6.0.1.0；IsaacLab `ffff603eafc6b74264a5261cc0183d6a65390d78`。
- PPO 来自未改动的 `multimodal_rl@4ac55e854b4c732534d443715a0322d4220b461c`。
- 隔离解释器：`simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python`。
- NVIDIA EULA 已得到用户明确接受。没有操作真实机械手。

## 发现与修正

1. 转换后 USD 的非刚体子节点与刚体同名，PhysX 接触匹配重复。仅重命名非刚体包装节点；实际网格、关节和刚体路径保留。缓存加入转换器源码哈希。
2. 无接触点时 PhysX 返回 NaN。零接触力对应零压力贡献；有接触力却没有有限接触位置时仍报错退出。
3. 原预抓取姿态把水果放在四指上方、远离拇指。原始 16 环境各 200 步测试的稳定三指支撑占比为 0，平均约 4.8–4.9 步终止。现在用网格求解并经真实物理筛选的握姿初始化，保留关节/位置噪声和水果几何随机化。
4. **底层物理配置修正（已汇报）**：掌部包围指根 link0/link1 装配件；间接连接的 link1 没有被 PhysX 自动排除。只过滤掌部与这十个指根装配件的碰撞；保留掌部与远端、跨手指及手与水果的碰撞。空手对照中掌部最大内部接触力由约 952,383 N 降至约 1.24 N；数值来自 16 个候选姿态诊断，不是硬件测量。
5. 主动关节目标同时服从其被动 mimic 从动关节的可行范围。水果质量和惯量现在实际按照随机椭球几何及密度写入 PhysX，修复了原来只采样密度却仍使用固定质量的问题。
6. 每环境日志保留向量；终止后的堆叠帧重置；评估复位不改变训练环境的物理状态、观测或时间差分历史。四帧输入仍为 prop 280 + tactile 1728。
7. **奖励核心实现修正（已汇报）**：把原有 −8 失败惩罚覆盖到实际全部失败终止，包括超力和低位掉落；超时不扣该惩罚。幅度、旋转奖励、压力修正公式不变。
8. 异常退出返回非零状态；检查 PPO 的异常返回、有限参数及三个网络的实际参数变化。保存初始、当前、最终检查点及每次运行的源码快照。

## 已取得的证据

| 测试 | 样本口径 | 结果 |
|---|---|---|
| 压力算法 CPU 回归 | 6 项原有测试 | 6 项通过 |
| 实际 PhysX 冒烟 | 8 环境 × 20 控制步 | 观测尺寸、有限观测及奖励通过 |
| 修正握姿、质量后的保持测试 | 64 环境 × 500 步，seed42 | 78.156% 环境步具有三指支撑；有回合达到完整 479 步时限 |
| 同一配置的随机动作测试 | 64 环境 × 500 步 | 43.309% 支撑；2.109% 步骤接受 Jacobian 修正；全程有限 |
| 首次 PPO 接口测试 | 9,968 训练转移，5 次更新 | 三个网络都更新；检查点保存和加载通过 |
| 256 环境 PPO 验收（失败惩罚修正前） | 499,920 训练转移，65 次更新 | 三个网络参数有限且改变；140.7 秒完成 |
| 训练/评估隔离检查 | 256 环境，16 评估 + 240 训练 | 评估复位前后训练物理状态与四帧输入完全相同 |

最终奖励修正后的训练、恢复及独立评估结果将在本记录后续部分补充。原始日志与 JSON 在 `simulation/.runtime/setup/`。

上述结果证明环境有非零接触/压力信号、可更新的 PPO 和会实际触发的压力修正。它们不证明已学会连续转动，也不证明 Jacobian 方法优于 plain/scalar。减压优势仍须按 README 的共同随机种子、独立评估和匹配旋转表现标准比较；这里的压力是接触导出的代理量，不是实测果实损伤。

## 本机运行

```bash
cd /home/descfly/linkagent-ppo
bash simulation/.runtime/run_roto2.sh smoke
bash simulation/.runtime/run_roto2.sh train --million-steps 0.5 --validate-contracts
bash simulation/.runtime/run_roto2.sh evaluate \
  --evaluate-checkpoint /absolute/path/to/final_agent.pt \
  --evaluation-output /absolute/path/to/evaluation.json
```

恢复：在 train 命令后传 `--resume-checkpoint /absolute/path/to/final_agent.pt` 或 `last_agent.pt`；`--million-steps` 表示新增预算。恢复网络、优化器和归一化器，重建环境和 rollout；不是物理状态/RNG 逐位续跑。当前模型的独立评估应使用 `last_agent.pt` 或 `final_agent.pt`；上游在线评估使用冻结策略窗口，在线“最佳”分数不代替独立评估。

正式训练预算、服务状态和运行目录以 `simulation/.runtime/setup/status.json` 为准。进度不能当作完成标记；完成须同时存在 `training_result.json` 的 `training_completed=true` 以及最终检查点。
