# 强化学习可视化成果：固定标注与训练报告汇编

**文档用途**：为组会、阶段验收、项目汇报和 Isaac Sim 演示统一记录口径。

**整理日期**：2026-08-19  
**项目范围**：番茄采摘 PPO、左手固定腕部柔顺球抓取、掌托式五指包络抓取。  
**硬件边界**：本文所列训练和回放均为离线 Isaac Sim/PhysX 结果，除非另有明确说明，`hardware_command_allowed=false`，不代表实机验收或 sim2real 成功。

## 1. 先区分权重类型

同一个训练目录中可能同时存在多个文件，它们不一定代表多次独立训练：

| 文件类型 | 含义 | 可否继续 PPO 训练 | 推荐展示方式 |
|---|---|---:|---|
| `model_XXXX.pt` | Actor、Critic、归一化器、策略方差、优化器和迭代次数 | 可以 | 训练恢复、固定评估、完整审计 |
| `source_checkpoint.pt` | 完整检查点的归档副本 | 可以 | 发布包、校验和、复现说明 |
| `actor_deterministic_jit.pt` | 只用于确定性推理的 Actor | 否/不建议 | Isaac Sim 回放、部署接口演示 |
| `policy.pt` / `policy.onnx` | 导出的推理模型 | 否 | 独立推理、跨框架推理 |
| `eval_update_XXXX.pt` | 训练过程中的评估周期快照 | 通常可以 | 同一训练内的过程曲线和动作演化 |
| `best.pt` / `best_model_XXXX.pt` | 当前运行目录中的最佳副本 | 视内容而定 | 必须同时标注“本运行最佳”，不能自动称为全局最佳 |
| `bc.pt` | 行为克隆模型 | 不是 PPO 完整状态 | 教师轨迹、离线损失、Actor-only 诊断 |
| `teacher_dataset.pt` | 教师示范数据集 | 不适用 | 示范轨迹、动作分布、数据质量分析 |

仓库和本地工作区中存在大量周期性快照。`model_1350.pt` 与
`model_1350_best/source_checkpoint.pt`、`model_5150.pt` 与
`best_model_5150.pt` 可能是同一结果的不同归档，不能重复计为独立实验。

## 2. 所有视频、截图和报告必须使用的固定标注

每一段视频、每一张截图、每一张曲线图的标题或图注都应包含以下内容。不能确认的字段写“未记录”，不能用推测值补齐。

```text
实验族：<Minimal-v3 / Clean-v5 / Physics-v1 / Physics-v2 / PhysX-Sphere / Enveloping-v2>
权重：<相对仓库根目录的路径>
权重类型：<完整 PPO 检查点 / 推理导出 / BC / 教师数据>
训练迭代：<XXXX；若为 BC 或数据集则填写 N/A>
任务：<完整任务名>
手部与腕部：<左手/右手/双臂；固定腕部或包含手臂>
仿真环境：<Isaac Sim 版本、Isaac Lab 版本、PhysX 频率/策略频率>
物体模型：<番茄椭球 / 柔顺球 / 代理碰撞体；尺寸和质量>
观测/动作：<Actor 观测维度 / Critic 特权状态维度 / 动作维度>
并行环境：<num_envs>
随机化：<几何、质量、摩擦、恢复系数、触觉增益/延迟等>
评估协议：<deterministic 或 stochastic、种子、环境数、步数、完整回合数>
结果口径：<固定评估 / 训练滚动指标 / 几何诊断 / 脚本保持测试>
核心结果：<成功、保持、掉落、过载、超时等；未评估则明确写明>
硬件边界：<hardware_command_allowed=false；若不是，必须注明授权和安全状态>
证据：<报告、JSON、视频、曲线或截图路径>
```

### 推荐的简短视频角标

```text
LinkAgent | <实验族> | <权重> | iter=<XXXX>
<Actor obs>/<Critic state>/<Action> | <手部> | <物体>
Isaac Sim <版本> | seed=<seed> | deterministic | offline only
```

### 不允许使用的表述

- 只展示一次成功回放，却写“成功率 100%”。
- 将训练过程中的 rolling success 当作固定种子评估结果。
- 将 `teacher_dataset.pt` 或 `bc.pt` 称为 PPO 成功策略。
- 将虚拟接触、几何候选或脚本保持测试称为真实触觉抓取成功。
- 将不同观测/动作接口的权重放在同一张成功率图中直接比较。
- 将仿真力、刚度、摩擦先验写成“已完成实机标定”。

## 3. 可直接展示的训练报告

下面每一节可以直接作为汇报 PPT 的一页或一组页面。每一节都包含：训练身份、环境、当前结果、适合展示的证据和固定标注。

---

## 报告 A：Minimal-v3 早期 PPO 基线

### 训练身份

- 代表权重：
  `logs/rsl_rl/tomato_pick_minimal_v3/2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192/model_1450.pt`
- 任务：`LinkAgent-TomatoPick-Minimal-v3`
- 训练迭代：1450
- W&B：`pmt88pns`
- 权重类型：完整 PPO 检查点，可恢复训练

### 环境与接口

- Isaac Sim/Isaac Lab + PhysX
- 8192 个并行环境，策略滚动长度 32
- 约 8 秒 episode
- 66 维 Actor 观测、11 维动作
- 番茄采用相关椭球几何随机化
- 属于较早的臂手协同/番茄抓取基线，不是当前左手固定腕部专用策略

### 训练情况

| 指标 | 结果 |
|---|---:|
| 固定评估回合 | 23,842 |
| 总体成功率 | **62.57%** |
| 95% Wilson 区间 | 61.95%–63.18% |
| seed 17 | 61.18% |
| seed 42 | 62.88% |
| seed 2026 | 63.65% |

### 结论

该权重适合作为“问题起点”和基线，不应作为当前最终策略。它的价值是说明后续 Clean-v5、Physics-v1 和包络抓取方案相对于早期奖励与环境设计的改进幅度。

### 建议展示

1. 三种子成功率柱状图。
2. 基线策略单集 Isaac Sim 回放。
3. 与 Clean-v5 的并排成功/失败回放。
4. 66/11 接口示意图，说明后续实验并未沿用相同的任务假设。

### 固定标注

```text
Minimal-v3 | model_1450.pt | iter=1450
66D Actor obs / 11D action | 8192 envs | tomato ellipsoid randomization
fixed deterministic evaluation | seeds=17,42,2026 | 23,842 episodes | success=62.57%
offline Isaac Sim/PhysX only | not hand-only final policy | hardware_command_allowed=false
```

---

## 报告 B：Clean-v5 稳定基线与正式可视化候选

### 训练身份

- 代表权重：
  `logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch/model_1350.pt`
- 归档权重：`artifacts/simulation/tomato_pick_clean_v5/model_1350_best/source_checkpoint.pt`
- 推理导出：`actor_deterministic_jit.pt`、`policy.pt`、`policy.onnx`
- 任务：`LinkAgent-TomatoPick-Clean-v5`
- 训练迭代：1350
- W&B：`1vp9dp6x`

### 环境与接口

- Isaac Sim 6.0.1.0、Isaac Lab `v3.0.0-beta2.patch1`
- 8192 个并行环境，32 步 rollout
- 66 维 Actor 观测、11 维动作
- 番茄相关椭球几何随机化
- 固定腕部/手部细节不是该实验的主要研究对象，仍属于番茄采摘任务基线

### 训练与评估结果

| 指标 | 结果 |
|---|---:|
| 固定评估回合 | 26,077 |
| 成功 | 25,081 |
| 总体成功率 | **96.1805%** |
| 95% Wilson 区间 | 95.9410%–96.4065% |
| detach | 99.7929% |
| retained | 96.2036% |
| drop | 2.1015% |
| 最差种子 | 96.0623% |
| 种子差异 | 0.2148 个百分点 |

评估覆盖 1300–1600 多个候选检查点；`model_1350` 按三种子总体成功率选出，`model_1600` 为 95.6390%，没有被静默替换为最终迭代。

### 结论

Clean-v5 是当前最适合对外展示“训练成功率和复现流程”的权重，但不能把它理解为已经解决了左手固定腕部的柔性包络抓取。它更适合作为项目中的稳定番茄采摘基线。

### 建议展示

1. 单集和 8/16 环境并行 Isaac Sim 视频。
2. 三种子成功率、drop 和 retained 柱状图。
3. 1300–1600 检查点选择曲线。
4. `policy.onnx` 独立推理演示。
5. “基线 → 物理增强”架构演进图。

已有说明：[Clean-v5 复现报告](./PPO_CLEAN_V5_MODEL1350_REPRODUCTION.md)。

### 固定标注

```text
Clean-v5 | model_1350.pt | iter=1350 | selected checkpoint
66D Actor obs / 11D action | 8192 envs | correlated tomato ellipsoid
fixed deterministic evaluation | seeds=17,42,2026 | 26,077 episodes
success=96.1805% | drop=2.1015% | offline Isaac Sim/PhysX
hardware_command_allowed=false | simulation baseline, not real-hand validation
```

---

## 报告 C：Physics-v1 中期恢复点 model_1200

### 训练身份

- 权重：
  `logs/rsl_rl/tomato_pick_physics_v1/2026-08-10_03-22-34_tomato_pick_physics_v1_from_scratch_8192/model_1200.pt`
- 任务：`LinkAgent-TomatoPick-Physics-v1`
- 训练迭代：1200
- W&B：`hs0bad5e`
- 状态：历史恢复点，不是最终选定模型

### 环境与接口

- Isaac Sim 6.0.1.0、Isaac Lab `v3.0.0-beta2.patch1`
- 8192 个并行环境；120 Hz 物理；控制降采样 6
- 66 维 Actor 观测、83 维 Critic 状态、11 维动作
- 手部 mimic、材质、自碰撞过滤、果实物理属性和摘取/保持逻辑已启用
- 没有连接 SDK、CAN 或实机

### 训练情况

这是训练过程中的 rolling 指标，不是固定种子发布评估：

| TensorBoard step | success | retained | detach | drop | mean hold steps |
|---:|---:|---:|---:|---:|---:|
| 1200 | 40.7379% | 41.6353% | 88.0447% | 33.8255% | 8.2257 |
| 1208 日志末端 | 41.6606% | 未单独发布 | 未单独发布 | 未单独发布 | 未单独发布 |

### 结论

`model_1200` 的价值是展示从 Clean-v5 到显式 PhysX 之后的训练难度和恢复流程。不能引用 40.7379% 作为正式最终成功率；正式选择必须使用完整训练后的多种子固定评估。

### 建议展示

1. Physics-v1 环境结构、接触和保持逻辑示意图。
2. 1200 → 5150 的训练接续关系图。
3. rolling 指标与固定评估指标的区别说明。
4. 中期失败回放：接触、掉落和保持不足。

### 固定标注

```text
Physics-v1 recovery | model_1200.pt | iter=1200
66D Actor obs / 83D Critic state / 11D action | 8192 envs
explicit hand/material/self-collision/fruit-retention physics enabled
training rolling metrics only: success=40.7379% at step 1200
not fixed-seed evaluation | recovery point, not selected release
offline only | hardware_command_allowed=false
```

---

## 报告 D：Physics-v1 最终候选 model_5150

### 训练身份

- 选定权重：
  `logs/rsl_rl/tomato_pick_physics_v1/2026-08-12_09-39-56_tomato_pick_physics_v1_resume5000_to5400_8192/model_5150.pt`
- 便捷副本：`logs/qualification/tomato_physics_v1/convergence_report/best_model_5150.pt`
- 任务：`LinkAgent-TomatoPick-Physics-v1`
- 训练迭代：5150
- 完整续训链：0–5150，共 12 个 W&B 运行段

### 固定评估结果

| 指标 | 结果 |
|---|---:|
| 完成 episode | 20,102 |
| 成功 | 17,282 |
| 固定 deterministic 成功率 | **85.9715%** |
| 95% Wilson 区间 | 85.4846%–86.4448% |
| 最差种子成功率 | 85.6218% |
| drop | 6.7804% |
| 三种子差异 | 0.9301 个百分点 |
| 原定 90% 目标 | **未达到** |

### 结论

Physics-v1 已经证明显式物理、接触、摘取和保持链路可以在大规模并行环境中运行并完成统计评估，但仍没有达到 90% 目标。它是当前最适合展示“物理增强训练和严格评估”的成果，不应包装成已完成的真实番茄无损抓取。

### 建议展示

1. 已有 36.6 秒多环境 Isaac Sim Demo。
2. 5150 收敛曲线和 9 个固定检查点比较。
3. 20,102 个 episode 的成功、掉落和种子差异图。
4. 可达空间点云、初始无碰撞审计。
5. 成功抓持与断梗后掉落的近景对比。

已有材料：

- [Physics-v1 最终复现报告](./PHYSICS_V1_PERF_V7_FINAL_5150.md)
- [Physics-v1 展示页](./PHYSICS_V1_SHOWCASE.md)

### 固定标注

```text
Physics-v1 | model_5150.pt | iter=5150 | selected simulation checkpoint
66D Actor obs / 83D Critic state / 11D action | 8192 envs
120 Hz PhysX | tomato + compliant stem + hand material/self-collision physics
fixed deterministic evaluation | seeds=17,42,2026 | 20,102 episodes
success=85.9715% | worst seed=85.6218% | drop=6.7804% | target 90% not reached
offline Isaac Sim only | hardware_command_allowed=false
```

---

## 报告 E：Physics-v2 早期主线训练

### 训练身份

- 权重范围：`logs/rsl_rl/tomato_pick_physics_v2_learning/*`
- 任务：`LinkAgent-TomatoPick-Physics-v2`
- 主要用途：Physics-v1 的物理先验扩展和环境接口验证
- 状态：早期/阶段性训练，未形成可发布的最终策略

### 环境与接口

- 约 256 个并行环境
- 66 维 Actor 观测、11 维动作，继承 Physics-v1 的任务接口
- 加入相关的几何、质量、成熟度、湿润度、摩擦和接触先验
- 重点检查独立资产身份、真实 distal 几何、掌部接触过滤和接触传感器对齐

### 训练情况与结论

当前材料主要是学习早期日志、配置和诊断记录，没有通过完整的固定多种子发布门槛。因此不能给出“Physics-v2 最终成功率”，也不能把早期 checkpoint 标记为最终 PPO 模型。

### 建议展示

1. Physics-v1 → Physics-v2 的环境随机化对比图。
2. 番茄几何/质量/湿润度/接触参数的随机化矩阵。
3. 接触传感器与 distal 几何对齐示意图。
4. 早期训练曲线和已知失败类型。

### 固定标注

```text
Physics-v2 early learning | <具体 checkpoint> | early-stage diagnostic
66D Actor obs / 11D action | approximately 256 envs
correlated geometry/mass/ripeness/wetness/contact priors enabled
no final fixed-seed release evaluation | do not quote as final success
offline Isaac Sim/PhysX | hardware_command_allowed=false
```

---

## 报告 F：左手固定腕部 10D/110D PhysX 柔顺球训练

### 训练身份

- 代表目录：`gpu_overnight_v3_10d110d_final/`、`physx_sphere_final_v2/`
- 任务：左手固定腕部柔顺球包络抓取
- 手部资产：由供应商 `L20_8_l_s.urdf` 转换生成
- 权重类型：10D/110D 训练检查点，以及最终 PhysX 训练目录中的 `latest.pt`/周期权重

### 环境与接口

- Isaac Sim 4.5 / IsaacLab 2.3.2
- 120 Hz PhysX、30 Hz 策略更新
- 左手固定腕部；16 个主动关节，5 个 mimic 关节
- 35 mm 半径、75 g 动态柔顺球
- 五个逐指 PhysX 接触通道，包含 fingertip 和 proximal fallback reporter
- 110 维可部署 Actor 观测、148 维特权 Critic 状态、10 维混合动作
- 质量、摩擦、恢复系数、刚度先验、触觉增益/偏置/延迟/丢包随机化

### 训练情况与边界

接触探针、USD 转换、脚本化多指保持和短 PPO smoke 已通过；这证明环境、关节驱动和接触通道可以运行。现有资料不支持把脚本保持测试等同于 PPO 泛化成功率，因此当前报告不填写 PPO 最终成功率。

### 建议展示

1. 左手 URDF 转 USD 和 软球场景截图。
2. 五个接触通道的实时力曲线。
3. 脚本化保持测试，明确标注为“控制/几何 sanity check”。
4. 10D 动作分解：五个 synergy + 五个 digit residual。
5. 110D Actor 与 148D privileged Critic 的接口示意。

完整环境说明：[Final PhysX sphere-compliant grasp training](../../../ppo_tomato_grasp_stage01_work/simulation/ppo_tomato_grasp/PHYSX_FINAL_TRAINING.md)。

### 固定标注

```text
PhysX-Sphere | <checkpoint path> | left-hand-only fixed wrist
110D deployable Actor obs / 148D privileged Critic state / 10D hybrid action
Isaac Sim 4.5 / IsaacLab 2.3.2 | PhysX 120 Hz | policy 30 Hz
dynamic compliant sphere: radius=35 mm, mass=75 g | five contact channels
scripted hold/contact probe passed | PPO final success not established
simulation proxy, not tomato calibration | hardware_command_allowed=false
```

---

## 报告 G：旧版 `enveloping_grasp` 阶段式五指包络

### 训练身份

- 权重范围：`simulation/ppo_tomato_grasp/artifacts/enveloping_grasp/`
- 代表目录：`final_stage2/`、`final_phase/`、`final_core/`、`g0_core2/`、`g1b/`
- 训练方式：G0/G1/Stage-1～4 阶段化包络抓取、教师先验和安全约束对照

### 代表性交付结果

当前核心 nominal 交付基线：`stage4_band1_long/best.pt`，对应 35–40 mm 半轴核心大果径带。

| 指标 | 三种子结果 |
|---|---:|
| 严格成功率 | 34.01%、38.36%、45.14% |
| 三种子平均 | **39.17%** |
| 掉落范围 | 54.3%–64.6% |
| 过载范围 | 0.57%–1.36% |
| 掌托先行观测 | 100% |
| 四根直指远端参与 | 100% |
| 断梗后保持 | 57.8%–68.8% |

轻度材料、触觉和执行器随机化的三种子平均严格成功率为 23.82%，不能与 nominal 结果混合。宽域 38–50 mm 半轴压力测试约 0.9% 严格成功，说明当前交付范围必须限定为已验证核心尺寸带。

### 结论

该阶段证明了掌托先行、五指包络、拇指对置、远端参与和摘后保持可以被写入严格验证器，但掉落和断梗后保持仍是主要瓶颈。它是“复杂任务逻辑与失败结构”的展示，不是高成功率最终策略。

### 建议展示

1. G0 → G1 → Stage-4 的课程学习阶梯图。
2. `stage4_band1_long` 成功和掉落回放。
3. nominal 与轻度随机化的并排柱状图。
4. 掌托先行、远端参与、覆盖角和断梗后保持的指标雷达图。
5. `postbreak_refine`、`grace_refine` 失败对照，说明增加奖励并未自动带来泛化。

### 固定标注

```text
Enveloping-v1 | stage4_band1_long/best.pt | stage=4 | left hand, palm-up
five-finger palm-supported enveloping | core semi-axis=35–40 mm
strict three-seed nominal evaluation | mean success=39.17%
palm-first=100% | distal participation=100% | main failure=drop/post-break retention
nominal simulation baseline only | do not mix with 38–50 mm stress domain
hardware_command_allowed=false
```

---

## 报告 H：`enveloping_grasp_v2` 掌托式五指包络与离线 BC 诊断

### 训练身份

- 任务：`palm_supported_five_finger_enveloping`
- 权重范围：`simulation/ppo_tomato_grasp/artifacts/enveloping_grasp_v2/`
- 仓库研究产物：`research/palm_enveloping_grasp_v2/results/large_artifacts/`
- 代表权重：
  `t0ab_candidate297_priorppo_curr120_20260819/best.pt`
- 代表 BC：
  `t0u_offline_bc_candidate288_20260819/bc.pt`
- 大多数 v2 权重：128 维 Actor、177 维 Critic、14 维动作；少数 raw 分支使用 133/182 状态接口

### 环境与方法

- 左手、掌面向上、掌部先建立支撑，再进行五指包络和摘取
- 物体使用球形/椭球形物理代理，加入 fingertip proxy、掌部接触、拇指对置、远端关节参与和覆盖角约束
- 14D 动作保留旧前缀，并为食指、中指、无名指和小指增加独立远端残差
- 包含 PPO、teacher prior、在线 BC、离线 BC、DAgger 和动作反事实分支
- 所有研究产物均为离线仿真，`hardware_command_allowed=false`

### 当前训练/诊断情况

本轮最有价值的结果不是“已经成功抓取”，而是定位了从教师动作到可部署 Actor 的迁移问题：

- candidate288 的固定动作复核：五指实测接触约 98.44%，掌托支持 100%，掉落 0%，过载 1.56%，但四指同时保持约 30.77%，没有摘梗事件。
- t0u 离线 BC 完全匹配 Band-1：`success=0`，`measured_distal_all=0.4318`，`five_digit_rate=0.0038`。
- t0u 37.5 mm nominal：`success=0`，`measured_distal_all=0.7283`，`five_digit_rate=0.0075`，过载约 27.17%。
- t0r/t0s/t0y/t0z/t0aa/t0ac 等分支均未通过 T0 actor-only gate。

这些结果说明虚拟接触候选、教师活动或离线损失下降，不能直接替代 Actor-only 的五指同步包络成功。

### 结论

v2 是当前最适合做“研究问题和失败经验”展示的一族权重。应突出：接触观测校准、掌托先行、拇指对置、远端残差和教师到 Actor 的迁移难点；不能将 `best.pt` 的文件名当作整个项目的最佳策略。

### 建议展示

1. 掌托先行 → 根部预定位 → 远端卷曲 → 触觉微调 → 摘后保持流程图。
2. 五指接触占空比和同时接触率热力图。
3. candidate288 固定动作与 t0u Actor-only 的并排回放。
4. Teacher、BC、PPO Actor 的动作分布和轨迹差异。
5. 失败分类：拇指不对置、四指不同步、过载、超时、摘后掉落。

已有材料：[v2 工作日志](./enveloping_grasp_v2/WORK_LOG_20260819.md) 和
[第一版包络抓取报告](./enveloping_grasp_v2/FINAL_ENVELOPING_GRASP_REPORT.md)。

### 固定标注

```text
Enveloping-v2 | <checkpoint> | stage=<1/2> | palm_supported_five_finger_enveloping
128D Actor obs / 177D Critic state / 14D action (unless raw branch explicitly noted)
left hand | palm-up | five-finger envelope | physical fingertip proxy + contact diagnostics
actor-only strict gate | current selected diagnostics did not pass strict success
teacher/BC metrics are not PPO deployment success
offline PhysX only | hardware_command_allowed=false
```

---

## 4. 展示物料与训练族的推荐对应关系

| 汇报目的 | 首选权重/环境 | 首选物料 | 必须附带的说明 |
|---|---|---|---|
| 证明已有高成功率基线 | Clean-v5 `model_1350` | 视频、三种子柱状图、复现报告 | 不是左手固定腕部最终策略 |
| 证明物理建模变完整 | Physics-v1 `model_5150` | 36.6 秒 Demo、收敛报告、可达空间 | 85.9715%，90% 目标未达到 |
| 证明手部 PhysX 环境可运行 | `physx_sphere_final_v2` | 场景截图、接触通道、脚本保持 | 不是 PPO 最终成功率 |
| 展示复杂五指包络逻辑 | `stage4_band1_long/best.pt` | 流程图、接触热力图、失败回放 | nominal 39.17%，随机化下降 |
| 展示研究难点和改进方向 | v2 `t0ab` / `t0u` | Actor-only 对比、BC 损失、失败报告 | 未通过严格 T0，不可称最终策略 |
| 展示算法演进 | model_1450 → model_1350 → model_5150 | 统一指标表、训练曲线、时间线 | 不能忽略接口和环境变化 |

## 5. 组会汇报建议的最小素材包

如果只制作一套简洁但完整的汇报，建议准备以下六件材料：

1. **Clean-v5 成功视频**：展示稳定基线的直观效果。
2. **Physics-v1 多环境视频**：展示显式 PhysX、接触和保持。
3. **model_1450 / model_1350 / model_5150 成功率表**：明确环境和接口不同。
4. **PhysX-Sphere 接触通道截图**：展示左手专用柔顺物体环境。
5. **Enveloping-v1 掉落与保持失败回放**：展示真实问题，而不是只展示成功样例。
6. **Enveloping-v2 Teacher → BC → PPO Actor 对比图**：说明当前最重要的研究瓶颈。

推荐叙事顺序：

```text
早期 62.57% 基线
    → Clean-v5 96.18% 稳定基线
    → Physics-v1 85.97% 物理增强但仍未达 90%
    → 左手 PhysX 柔顺球环境验证
    → 掌托式五指包络的接触与迁移问题
```

## 6. 当前展示结论

- **最适合做“已取得成果”展示**：Clean-v5 `model_1350`、Physics-v1 `model_5150`。
- **最适合做“环境和控制能力”展示**：`physx_sphere_final_v2`。
- **最适合做“研究问题、失败经验和下一步计划”展示**：`enveloping_grasp` 与 `enveloping_grasp_v2`。
- **不应作为最终成果单独展示**：early Physics-v2、`eval_update_*.pt`、教师数据、BC 权重、单次候选动作和脚本化保持结果。

所有正式图片和视频都应能回溯到：权重、环境配置、评估 JSON、训练曲线和运行版本。无法回溯的截图只能作为过程记录，不能作为定量成果。
