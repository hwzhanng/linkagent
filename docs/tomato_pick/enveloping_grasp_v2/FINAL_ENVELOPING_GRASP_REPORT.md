# 第一版掌托式五指包络抓握成果报告

日期：2026-08-16  
任务：左手、掌面向上、大果径柔顺球形番茄、五指包络与摘取  
状态：第一版完整 PhysX 训练链路、接触点适配、分档训练、核心大果径三种子独立评估、无界面随机回放和 v0.8 断梗后反应窗口消融已完成；最终 80% 严格成功门槛尚未达到。当前交付权重为可复现实验基线，所有新精炼候选已单独封存且未升级。

## 1. 结论先行

本轮已经不再是旧的拇指主导捏取策略：策略、动作接口和严格验证器都明确要求掌托先建立、五指同步接触、空间覆盖、拇指对置、四根直指远端关节参与，以及摘取后的保持。旧 Band-4 捏取结果保持封存，没有被覆盖。

当前交付候选 `stage4_band1_long/best.pt` 固定在已验证的 35--40 mm 半轴（约 70--80 mm 直径）核心大果径带，并完成 3 个独立 seed 的 64 环境阶段二评估；它仍没有达到原计划的 80% 严格成功率目标：

| 指标 | 结果 |
|---|---:|
| 独立 seed 20260820 严格成功 | 34.01% |
| 独立 seed 20260821 严格成功 | 38.36% |
| 独立 seed 20260822 严格成功 | 45.14% |
| 三种子平均严格成功 | **39.17%** |
| 掉落范围 | 54.3%--64.6% |
| 过载范围 | 0.57%--1.36% |
| 断裂后保持范围 | 57.8%--68.8% |
| 最大覆盖代理 | 约 228°--231° |
| 掌托先行观测 | 100% |
| 四根直指远端参与观测 | 100% |
| 实测 PhysX 接触点峰值比例 | 0.3125% |

这说明在已验证核心尺寸和 nominal 物理下，掌托先行、五指包络和慢速摘取链路已经可重复出现，但断梗后的长期保持仍是主要瓶颈。评估已经启用 PhysX 接触点追踪并记录实测比例；由于接触流仍稀疏，缺失帧按逐指显式 fingertip fallback 处理，不能把该结果解释为完整真实接触点级别的 sim2real 保证。更大的 38--50 mm 半轴随机域仍属于物理可达性压力测试，不与核心成功率混报。

同一基线在 38--50 mm 半轴、材料/传感器/执行器全随机的压力测试中严格成功约 0.9%、掉落 87.5%、过载 11.6%，因此当前交付范围明确限定为 35--40 mm 半轴核心带；这不是放宽成功判据，而是把已知不可达/未标定域独立标记，留给后续几何与动力学扩展。

在核心 35--40 mm 半轴带上追加的轻度材料、触觉和执行器随机化三种子验收结果如下。该组比 nominal 更接近后续鲁棒性验证，因此单独报告，不能与 nominal 均值混合：

| 指标 | seed 20260820 | seed 20260821 | seed 20260822 | 三种子平均 |
|---|---:|---:|---:|---:|
| 严格包络成功 | 24.44% | 22.22% | 24.81% | **23.82%** |
| 掉落 | 74.07% | 76.19% | 74.42% | **74.89%** |
| 过载 | 1.48% | 1.59% | 0.78% | **1.28%** |
| 摘后保持 | 71.88% | 75.00% | 87.50% | **78.13%** |
| 触发摘取 | 90.63% | 85.94% | 93.75% | **90.10%** |

因此本轮最终结论是“核心 nominal 可复现、轻度随机化仍不鲁棒、严格 80% 门槛未通过”。这不是训练链路失败：掌托先行和四根直指远端参与在三种子中均保持 100% 观测；当前主要瓶颈仍是摘取前后掉落和扰动下的接触/执行器响应。交付权重仅用于离线复现实验和后续改进，不得直接用于实机。

精炼候选 `stage4_band1_postbreak_refine/best.pt` 的内部评估最高约 40.6%，但三种子独立 nominal 平均约 29.1%，低于当前交付基线约 39.2%，因此明确封存为失败对照，不覆盖基线。GPU PhysX 的同 seed 重复也存在非确定性，正式结论以独立 JSON 中的完整回合数和多 seed 均值为准，不引用单次内部峰值。该结果验证了“加大断梗后密集奖励”并不自动带来泛化提升，后续应优先做接触/动力学建模和阶段课程，而不是继续放大同一奖励。

### v0.8 断梗后反应窗口实验结论

为给断梗瞬间的触觉闭环留出有限响应时间，新增了默认关闭的 `post_break_grace_steps`。该开关只延迟有限策略步的高度掉落终止，不改变掌托在先、五指包络、压力、断梗和保持等严格判据。对照结果如下：

| 配置 | nominal 三种子严格成功 | 轻度随机三种子严格成功 | 轻度随机过载 | 结论 |
|---|---:|---:|---:|---|
| 基线，grace=0 | 39.17%（历史独立集） | 23.82% | 1.28% | 当前交付 |
| 基线，grace=1 | — | 21.35% | 0.84% | 成功下降，拒绝 |
| 基线，grace=2 | — | 27.42% | 2.77% | 越过 2% 安全门槛，拒绝 |
| 基线，grace=4 | 47.88%（本轮消融） | 23.16% | 3.37% | 随机域退化，拒绝 |
| grace=4 续训候选 | 39.62% | 未作为候选验收 | seed21=2.78% | 不超过基线，封存 |

因此最终交付仍是 `stage4_band1_long/best.pt`，复现实验默认 `post_break_grace_steps=0`。`stage4_band1_postbreak_grace_refine/best.pt` 只用于后续研究对照；不能因为某个 seed 的 nominal 成功率上升，就把有限窗口误报成稳定泛化改进。

## 2. 产物与检查点

全部产物位于：

`simulation/ppo_tomato_grasp/artifacts/enveloping_grasp/`

- 最终阶段 2 候选：`final_stage2/best.pt`
- 阶段编码改进候选：`final_phase/best.pt`
- 最终随机化适应候选：`final_core/best.pt`
- G1 最佳包络候选：`g1b/best.pt`
- G0 固定核心候选：`g0_core2/best.pt`
- 最终独立评估：`eval_final_stage2_128.json`
- 阶段编码独立评估：`eval_final_phase_128.json`
- G0 独立评估：`eval_g0_core2_256.json`
- G1 独立评估：`eval_g1a4_best_256.json`
- 训练曲线：各检查点目录下的 `metrics.jsonl`
- 教师启动数据：`enveloping_teacher_g0.pt`、`enveloping_teacher_g0_branch0.pt`
- 几何与无围挡审计：`p1_geometry_sweep_report.json`、`p2_norim_80mm.json`、`p2_norim_100mm.json`
- 检查点校验：`FINAL_CHECKSUMS.sha256`
- 当前大果径等径入口候选：`stage4_large_iso_band1/best.pt`
- 当前核心交付基线：`stage4_band1_long/best.pt`（35--40 mm 半轴）
- 断梗后保持精炼对照：`stage4_band1_postbreak_refine/best.pt`（独立验收未超过基线，封存）
- 断梗后反应窗口精炼对照：`stage4_band1_postbreak_grace_refine/best.pt`（500 更新；独立验收未超过基线，封存）
- 核心基线三种子独立验收：`eval_baseline_nominal_seed20.json`、`eval_baseline_nominal_seed21.json`、`eval_baseline_nominal_seed22.json`
- 精炼对照三种子独立验收：`eval_postbreak_refine_nominal_seed20.json`、`eval_postbreak_refine_nominal_seed21.json`、`eval_postbreak_refine_nominal_seed22.json`
- 核心基线宽域压力测试：`eval_baseline_stress_38_50_seed20.json`（严格成功 0.89%、掉落 87.5%、过载 11.6%，明确不作为交付域）
- 核心基线轻度随机化三种子验收：`eval_baseline_random35_40_seed20.json`、`eval_baseline_random35_40_seed21.json`、`eval_baseline_random35_40_seed22.json`（三种子均值严格成功 23.82%，明确不与 nominal 混报）
- 大果径随机化桥接候选：`stage4_large_bridge_b2_p25/best.pt`
- 大果径教师数据（仅弱先验对照）：`large_iso_teacher.pt`
- 三种子独立评估：`eval_large_iso_band1_seed20260820.json`、`eval_large_iso_band1_seed20260821.json`、`eval_large_iso_band1_seed20260822.json`
- 后台随机轨迹：`replay_large_iso_seed20260840.pt` 及同名 `.json`；小规模烟测为 `replay_large_iso_smoke.pt`
- 断梗后反应窗口消融：`eval_grace_refine_nominal_seed20.json`、`eval_grace_refine_nominal_seed21.json`、`eval_grace_refine_nominal_seed22.json`
- grace=1/2/4 随机化消融：`eval_baseline_grace1_random35_40_seed*.json`、`eval_baseline_grace2_random35_40_seed*.json`、`eval_baseline_grace_random35_40_seed*.json`
- 保持反射/种子库诊断：`retention_seed_bank_v1.pt`、`retention_seed_bank_nominal.pt`（未启用为交付策略）
- 最终接触缓存/110D/14D 烟测：`p2_smoke_final.json`（8 环境、8 步、退出码 0）

旧的捏取结果仍在 `archives/pinch_small_fruit_band4_20260815/`，本轮没有写入旧目录。

## 3. 本轮实现的关键改动

1. 14D 动作接口：保留旧 10D 前缀，为食指、中指、无名指、小指加入独立远端残差，避免直指只能依赖根部。
2. 阶段门控：掌部支撑 → 根部预定位 → 近端/远端卷曲 → 逐指触觉微调 → 保持 → 茎秆加载/断裂 → 摘后保持。
3. 严格验证器：掌面向上、掌托在先、五指同步、角覆盖、拇指对置、半球分布、载荷份额、远端参与、压力、保持和断裂后保持共同决定成功。
4. 连续拓扑奖励：对覆盖、对置、半球展开和远端关节的连续进度提供奖励，不再只在达到硬阈值后才有信号。
5. 修正半球统计：旧在线统计以初始化为 1 的最大值做通过判定，会恒定显示为 0；现在按有效几何窗口记录最小半球分数。
6. 随机化课程：G0 固定核心 → 大果径等径 35--40 mm 半轴入口 → 37--45 mm 椭球/中扰动 → 38--50 mm 完整域；stage-4 band 优先于旧 G0/G1/final 别名，避免隐式域跳变。
7. 安全对照：训练和独立验收均显式使用 `--enveloping-safe-force 12`；该先验下的最终独立成功仍不足，不能据此直接进入实机。
8. 物理诊断开关：新增 `joint_projection_enabled` 与 `joint_projection_after_break`，并完成阶段二对照；关闭断梗后投影没有改善，交付配置保留安全投影。

## 4. 复现命令

使用 Isaac Sim 环境：

```bash
export PYTHONPATH=/home/miobarnacle/linkbot/ppo_tomato_grasp_stage01_work:/tmp/isaaclab-v2.3.2-Tj7F/source/isaaclab
export ISAAC_PYTHON=/home/miobarnacle/anaconda3/envs/isaacsim/bin/python
```

独立最终评估：

```bash
$ISAAC_PYTHON simulation/ppo_tomato_grasp/evaluate_enveloping_checkpoint.py \
  --headless --stage 2 \
  --checkpoint simulation/ppo_tomato_grasp/artifacts/stage4_band1_long/best.pt \
  --output simulation/ppo_tomato_grasp/artifacts/enveloping_grasp/eval_large_iso_recheck.json \
  --num-envs 64 --steps 220 --coverage-deg 200 --hold-steps 10 \
  --axis-low .035 --axis-high .040 --safe-force 12 \
  --nominal-physics --pick-speed .010 --pick-displacement .012
```

严格判定的最终结果应以独立评估器 JSON 为准，而不是训练过程中的单次 64 环境 eval。

无界面随机回放：

```bash
$ISAAC_PYTHON simulation/ppo_tomato_grasp/replay_enveloping_checkpoint.py \
  --headless --checkpoint simulation/ppo_tomato_grasp/artifacts/stage4_band1_long/best.pt \
  --output simulation/ppo_tomato_grasp/artifacts/enveloping_grasp/replay_band1_seed20260840.pt \
  --num-envs 32 --episodes 16 --steps 220 --seed 20260840 \
  --axis-low .035 --axis-high .040 --safe-force 12
```

## 5. 当前失败结构与下一轮优先级

按优先级排列：

1. 掉落：核心带 nominal 独立评估仍约 60% 掉落，需继续做断裂前后动力学/接触点诊断，不能只增加奖励权重。
2. 断裂后保持：当前精炼对照的三种子均值低于基线，下一轮应使用更连续的接触状态/摩擦与关节响应课程，而非直接复用本轮 8/3/5 密集奖励。
3. 过载：采用每指独立的安全带和压力变化率惩罚；小指、无名指目前峰值更高，应单独调节。
4. 真实接触点：升级或适配接触传感器字段，替换 fingertip proxy，并加入无围挡/穿模反例回归。
5. 训练稳定性：保持 G1 → final 的渐进门槛，禁止把覆盖、保持和执行器扰动同时一步拉满。
6. 回放交付：轨迹导出已完成；若需要视频，应基于 `.pt` 轨迹离线渲染，不能把 GUI 中一次成功录像当作随机策略表现。最终交付默认回放核心基线，精炼权重只作为对照。

本轮工作清单已完成 P1--P9 的可交付项；后续若继续，优先从“真实接触/摩擦状态估计、断梗后动力学课程、扰动下逐指闭环”中选择一个明确假设重新开实验，不再无目标延长当前奖励精炼。

## 6. 安全边界

本轮全部运行在离线 PhysX 仿真，`hardware_command_allowed=false`。报告中的力、刚度、摩擦、压力阈值均为仿真先验，不能直接用于实机番茄或灵巧手安全控制；实机前仍需完成关节、接触点、传感器和软体材料标定。
