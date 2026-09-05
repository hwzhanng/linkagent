# 小果三指捏取：model150 向 52–80 mm 分段扩域执行计划

日期：2026-09-05（Asia/Shanghai）  
状态：**计划已冻结；分桶采样与独立评估能力尚未实现；未启动扩域训练**  
范围：Isaac Lab 3.0 / PhysX，固定左腕、无果梗、只扩展果实尺寸；禁止连接实机

## 1. 决策与边界

本计划默认以已经通过 B0 验收的权重为唯一父候选：

```text
logs/rsl_rl/
small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/
model_150.pt
SHA-256: e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d
```

文件当前大小为 `5,779,299` bytes，SHA-256 已于本计划建立时复核。其发布含义仍然只是
B0 64–70 mm 等效直径域；三种子严格成功率为 `99.80% / 99.41% / 99.61%`，聚合
`99.61%`。本计划不覆盖、不改名该文件。

本轮目标是训练**一个**覆盖 52–80 mm 的统一 14D Actor。扩域期间保持以下内容不变：

- 固定左腕，不加入机械臂搜索或接近策略；
- 不启用果梗、采摘轨迹、Sim2Real 或真实硬件；
- 拇指、食指、中指是严格必需指，无名指/小指继续隔离；
- `TIM_strict_success`、80+40 步连续保持、扰动、压力带和掉落定义不变；
- B0 已验收的 guided-residual、逐指力投影、保持锚定和 120 Hz 目标插值合同不变；
- 第一轮不同时扩大位置、姿态、轴比、摩擦、触觉或执行器随机化。

只允许“尺寸及由相同密度产生的自然质量变化”进入本轮。其余域随机化属于后续 P6，
不得与本轮尺寸课程混训后再把收益归因于果径扩展。

## 2. 唯一尺寸口径

正式分桶统一使用等效直径：

```text
D_eq = 2 * (a * b * c)^(1/3)
```

其中 `a/b/c` 为椭球半轴。区间采用左闭右开 `[low, high)`；最终 `78–80 mm` 桶包含
80 mm 端点。任何报告必须同时记录 `a/b/c` 与 `D_eq`，禁止用平均半轴替代等效直径。

当前 `native_train.py` 和 `native_evaluate.py` 的 `--fruit-axis-min-mm`、
`--fruit-axis-max-mm` 参数表示**半轴**。例如 52–54 mm 直径对应 26–27 mm 半轴。
现有实现还会独立均匀采样三个半轴，因此等效直径在宽区间内会向中部集中。它只能用于
单一窄桶的诊断，不能用于正式的多桶均衡训练或验收。

当前几何又是在 prestartup 时为每个环境生成一次，同一环境的后续 reset 会复用同一
几何。因此：

- 正式样本量以“独立几何的首回合”计数；
- 同一环境的重复回合不得伪装成新的尺寸样本；
- 重复回合如果用于动力学统计，必须按 `scene_seed + env_id` 聚类计算区间；
- 训练阶段必须轮换多个 prestartup 几何 cohort，不能在同一批 256 个果实上训练到底。

## 3. 扩域前置工程门 E0

在任何扩域 PPO 训练前，先完成以下只影响采样、日志和评估的工程工作。

### 3.1 分桶采样器

新增按桶配额采样能力，建议接口为：

```text
--diameter-bins-mm 62:64,64:66,66:68,68:70
--diameter-bin-weights 0.40,0.20,0.20,0.20
--geometry-cohort-id <stable-id>
```

建议流程：

1. 先按给定权重选择 2 mm 尺寸桶；
2. 在桶内均匀采样 `D_eq`；
3. 从冻结的 B0 形状比例分布采样 `a:b:c`；
4. 对三轴统一归一化，使几何平均半轴严格等于 `D_eq / 2`；
5. 密度继续使用 B0 范围，质量按椭球体积和密度计算；
6. 保存 `scenario_id/bin/D_eq/a/b/c/density/mass/scene_seed/env_id`。

本轮不得直接把形状比例扩展到 P6 的 0.85–1.15。首先保持 B0 形状分布，避免把
“尺寸扩展”和“形状扩展”混成一个不可解释实验。

### 3.2 独立评估器

评估器至少输出 JSON/JSONL 中的以下字段：

- 每个 2 mm 桶的首回合分母、成功、掉落、过载、自碰撞和安全投影；
- acquisition success、4 s retention、6 s retention、扰动恢复；
- `P(4 s retained | acquired)`；
- 掌部接触和掌部辅助成功；
- 三指分别的接触率、压力带率、峰值力和过载；
- 保持段动作变化、速度、加速度、jerk、方向反转和高频能量；
- checkpoint/config/source/asset/scenario-list SHA-256；
- Isaac Lab、Isaac Sim、PhysX、PyTorch、CUDA 和 GPU 标识。

### 3.3 E0 工程验收

- 固定 seed 重跑时，`scenario_id`、分桶和几何参数逐项一致；
- 每桶实际首回合数量与请求配额完全一致；
- 采样得到的 `D_eq` 不越桶，端点归属无重复；
- 相同 `D_eq` 下改变轴比时，体积和质量计算正确；
- 关闭真实接触或破坏传感器路径时，严格成功必须为 0；
- 64–70 mm 新采样器下，model150 相对冻结 B0 基线的成功率绝对差不超过 2 pp；
- B0 掉落、过载和平滑性不得恶化；
- 单元测试、compile 检查和 2 环境 PhysX smoke 全部通过。

E0 未通过时，禁止启动 E1。

## 4. 2 mm 单侧前沿课程

默认交替扩展小果端和大果端。E0 零样本扫描可以交换同一对阶段的先后顺序，但不得跳过
困难桶，也不得一次同时引入两个新桶。

| 阶段 | 新前沿桶（直径） | 累计验收域 | 单桶诊断半轴 | 里程碑 |
|---|---:|---:|---:|---|
| E0 | 52–80 mm 零样本扫描 | 64–70 mm | 26–40 mm | 可达性地图 |
| E1 | 62–64 mm | 62–70 mm | 31–32 mm | 微门 |
| E2 | 70–72 mm | 62–72 mm | 35–36 mm | M1 |
| E3 | 60–62 mm | 60–72 mm | 30–31 mm | 微门 |
| E4 | 72–74 mm | 60–74 mm | 36–37 mm | M2 |
| E5 | 58–60 mm | 58–74 mm | 29–30 mm | 微门 |
| E6 | 74–76 mm | 58–76 mm | 37–38 mm | M3 |
| E7 | 56–58 mm | 56–76 mm | 28–29 mm | 微门 |
| E8 | 76–78 mm | 56–78 mm | 38–39 mm | M4 |
| E9 | 54–56 mm | 54–78 mm | 27–28 mm | 微门 |
| E10 | 78–80 mm | 54–80 mm | 39–40 mm | M5 |
| E11 | 52–54 mm | 52–80 mm | 26–27 mm | M6/最终域 |

已有 B0 同样拆为 `64–66 / 66–68 / 68–70 mm` 三个 2 mm 回归桶。因此最终正式域
总计 14 个互斥桶。

## 5. 每个阶段的训练分布

每个微阶段按环境数而不是期望概率强制配额：

| 样本来源 | 配额 | 用途 |
|---|---:|---|
| 当前新前沿桶 | 40% | 学习新增尺寸 |
| 与新桶相邻的已通过桶 | 20% | 保持边界连续性 |
| B0 64–70 mm 三桶 | 25% | 锚定 model150 能力 |
| 其余已通过桶 | 15% | 防止早期扩展域遗忘 |

阶段较早、分类重叠时，将空余配额回填 B0，但旧通过域总配额始终不得低于 60%。不得按
episode 结束速度重新加权，否则容易快速终止的失败桶会占据过多 PPO transition。

### 5.1 默认优化方式

- 从上一个正式晋级 checkpoint 使用 full-state `--resume`；
- 原因是动作、观测、奖励、Verifier 和控制合同均未改变，只有采样域增加 2 mm；
- 学习率起点 `5e-5`，上限 `7.5e-5`；
- entropy coefficient 起点 `5e-4`，不得高于 B0 的 `1e-3`；
- `desired_kl=0.01`，PPO clip 建议在实现显式选项后固定为 `0.10`；
- 每 25 update 保存 checkpoint 并短评；单阶段最多 150 update。

预先定义的唯一备用恢复路线：若第一个 25-update 窗口出现 Critic loss 非有限、连续两次
更新 KL 超过 0.015，或新域成功率下降且 B0 同时下降超过 3 pp，则回到上一个晋级模型，
改用 Actor MLP + Actor normalization 初始化，重置 Critic、Adam 和探索方差，
`init_std=0.02–0.03`。两种恢复路线必须放在不同目录，禁止拼接证据。

### 5.2 几何 cohort 预算

256 环境时，每个 cohort 只有约 102 个新前沿几何。每阶段至少使用 5 个独立训练 cohort：

```text
cohort 1: smoke 5 update + train 20–25 update
cohort 2–5: 各 train 25–30 update，跨 cohort 从完整 checkpoint 继续
```

这样新桶至少约 510 个独立训练几何。若使用 512 环境，则至少 3 个 cohort。不得以同一
cohort 的 reset 次数代替独立几何数量。

## 6. 单阶段执行循环

每个 E1–E11 严格按以下顺序执行：

1. **准静态可达性门**：不加载 PPO，用可解释闭合扫描当前新桶；
2. **零样本 Actor 基线**：上一个晋级模型在新桶上确定性评估；
3. **GPU smoke**：64 环境 × 5 update；
4. **正式适配**：按 25 update 为一个窗口训练，最多 150 update；
5. **窗口短评**：开发 seed 评估新桶、相邻桶和 B0；
6. **双 checkpoint 稳定门**：连续两个 checkpoint 都通过短筛才允许送审；
7. **微阶段晋级**：冻结候选、配置、场景清单和 SHA-256；
8. **成对里程碑**：E2/E4/E6/E8/E10/E11 后复测全部累计域；
9. **失败回滚**：只回到上一个已经晋级的 checkpoint，失败 checkpoint 只读保留。

## 7. 科学阶段门槛

### 7.1 准静态物理可达性门

每个新前沿桶至少 256 个独立几何：

- 拇指/食指/中指真实同时接触率 ≥90%；
- 拇指必须接触，食指和中指不能由无名指替代；
- 过载率 ≤1%；
- 掌部接触率 <5%，掌部辅助成功为 0；
- 自碰撞率、安全投影率均为 0；
- 首接触横向弹射速度 P95 ≤0.08 m/s。

若三指接触率不足，本阶段判为几何/工作区阻塞，不得用延长 PPO、增大奖励或放大碰撞
offset 处理。

### 7.2 训练健康门

每 25 update 检查一次：

- 无 NaN/Inf、观测/动作维度漂移、传感器失效或终止分母错误；
- 连续两次更新 KL >0.015：停止当前分支并执行预登记备用路线；
- 任一短评出现自碰撞、安全投影或掌部辅助成功：立即停止；
- 任一桶过载率 >5%：立即停止，不等待训练“自行恢复”；
- B0 相对上一个晋级模型下降 >3 pp：暂停并回滚；
- 新桶成功率连续两个 25-update 窗口提升均 <2 pp：停止当前 2 mm 配置。

### 7.3 微阶段短筛门

开发集使用 2 个固定、训练外 seed；每个 `seed × 被测桶` 至少 128 个独立首回合。连续
两个相邻 checkpoint 均须满足：

- 新桶 `TIM_strict_success` ≥80%；
- 新桶 acquisition success ≥92%；
- 新桶 `P(4 s retained | acquired)` ≥88%；
- 新桶掉落率 ≤5%，过载率 ≤2%；
- B0 严格成功率 ≥95%，且相对上一晋级模型下降 ≤3 pp；
- 掌部辅助成功、自碰撞、安全投影均为 0；
- 第 7.5 节的保持平滑硬门全部通过。

开发 seed 可用于阶段调参，禁止进入最终盲评集合。

### 7.4 成对里程碑晋级门

E2/E4/E6/E8/E10/E11 后，用 3 个训练外里程碑 seed 对累计域全量评估。每个
`seed × 2 mm 桶` 至少 256 个独立首回合：

- 累计域严格成功率点估计 ≥87%；
- 每个 2 mm 桶成功率 ≥82%；
- 最差 `seed × 桶` ≥77%；
- B0 成功率 ≥94.6%，即相对冻结的 99.61% 不下降超过 5 pp；
- 每桶 acquisition success ≥95%；
- 每桶 `P(4 s retained | acquired)` ≥90%；
- 任一桶掉落率 ≤4%，过载率 ≤1.5%；
- 掌部辅助成功、自碰撞和安全投影均为 0；
- 保持平滑与扰动恢复硬门全部通过。

### 7.5 保持平滑与扰动硬门

扩域阶段继续使用 B0 的严格门，不提前消耗 B3 放宽余量：

| 指标 | 上限/下限 |
|---|---:|
| 保持动作单步变化 P95 | ≤0.04 |
| 保持关节速度 P95 | ≤0.20 rad/s |
| 保持关节加速度 P95 | ≤3.0 rad/s² |
| 保持关节 jerk P95 | ≤80 rad/s³ |
| 保持方向反转 | ≤2.0 次/(s·joint) |
| 3 Hz 以上关节目标能量 | ≤15% |
| 扰动后恢复三指接触 | ≤0.75 s |
| 扰动后恢复软压力带 | ≤1.0 s |
| 扰动恢复成功率 | ≥85% |

## 8. 统计判定方法

阶段报告必须同时给点估计和不确定性：

- 成功率使用 95% Wilson 区间；
- 掉落、过载等稀有失败使用单侧 95% Clopper–Pearson 上界；
- B0 回归使用相同 `scenario_id` 的配对 bootstrap；
- 多回合复用同一几何时，使用以 `scene_seed + env_id` 为 cluster 的 bootstrap；
- 禁止把每个 physics step、hold step 或同一几何 reset 后的回合当作独立样本；
- 训练回报、单 seed 峰值或最新 update 不参与 best checkpoint 决策。

里程碑除点估计硬门外，还应满足：

- 累计域成功率 Wilson 95% 下界 ≥85%；
- 每个 2 mm 桶成功率 Wilson 95% 下界 ≥78%；
- B0 配对成功率差的 95% 下界 ≥-5 pp；
- 掉落和过载的单侧 95% 上界分别不超过阶段门。

如果点估计过门但置信界未过，结论是“证据不足”，应增加独立几何样本，不得继续训练来
改变统计结果。

## 9. 失败分流与 1 mm 自适应拆分

| 观测 | 根因优先级 | 动作 |
|---|---|---|
| acquisition <90% | 工作区/预张开/远端可达性 | 停止 PPO，回到准静态扫描 |
| acquisition 高、retention 低 | 压力闭环/保持控制/信用分配 | 只改一个控制或优化变量做 A/B |
| 小尺寸掉指 | 闭合行程不足、指尖相互干涉 | 检查远端关节与指尖几何 |
| 大尺寸过载 | 预张开不足、瞬态冲量、拇指/食指压力 | 检查逐指力投影，不放宽硬压力门 |
| B0 下降 3–5 pp | 灾难性遗忘开始 | 旧域回放提高到 70%，降低学习率 |
| B0 下降 >5 pp | 晋级失败 | 回滚上一个里程碑模型 |
| 2 mm 桶两轮不进步 | 桶内难度跨越过大 | 拆成两个 1 mm 桶 |
| 1 mm 桶仍不可达 | 物理/机构边界 | 缩减发布域或改变任务结构，不伪造接触 |

启用 1 mm 拆分前必须保留原 2 mm 失败证据。只拆分失败桶，例如 58–60 mm 拆为
58–59 和 59–60 mm；已经通过的尺寸桶不得重定义。

## 10. model150 控制合同命令片段

在正式分桶接口实现前，以下公共参数用于单桶诊断；训练和评估必须复用同一组参数：

```text
--guided-residual
--geometry-prior
--geometry-prior-scale 0.15
--force-servo
--force-servo-seek-gain 0.04
--force-servo-target-gain 0
--force-servo-retract-gain 0.35
--force-servo-alpha 0.30
--force-servo-max-offset 0.25
--force-servo-warning-fraction 0.78
--direct-force-projection
--direct-force-warning-fraction 0.50
--direct-force-high-gain-by-digit 0.80,0.40,1.00
--direct-force-low-gain 0.06
--direct-force-alpha 0.10
--direct-force-max-open 0.50
--direct-force-root-fraction 0.45
--force-gate-start-fraction 0.60
--precontact-residual-fraction 0
--residual-scale 0.04
--distal-residual-scale 0.06
--residual-require-all-contact
--residual-trust-region
--residual-trust-warning-fraction 0.55
--residual-trust-in-band-scale 0.05
--hold-speed 0.08
--hold-acceleration 0.70
--natural-frequency 3.0
--offset-x-mm 20
--offset-y-mm 30
--offset-z-mm 24
--fruit-contact-offset 0.001
--fruit-rest-offset 0.0001
--robot-max-depenetration-velocity 0.3
--fruit-max-depenetration-velocity 0.3
--hold-joint-target-anchor-fraction 0.40
--hold-hand-stiffness-by-joint 2,12,12,12,12,2,12,12,12,12,2,2,4,4,12,12
--hold-hand-damping-by-joint 3,0.6,0.6,3,3,3,0.6,0.6,3,3,3,0.6,0.6,0.6,3,3
--manual-decimation-control
--position-target-interpolation 6
```

正式训练还保留 `--joint-smoothness-weight 0.01`。不得在扩域过程中隐式启用
`--geometry-prior-size-adaptive`、`--allow-nonrequired-digits`、
`--freeze-required-distal` 或任何真实硬件接口。

## 11. 最终 52–80 mm 独立验收

正式测试 seed 在 E1 前生成并封存，之后不得用于训练、早停、调参或 checkpoint 选择。
当前 B0 使用过的 `20260902 / 20261411 / 20261530` 只能作为回归 seed，不能再称为完全
隐藏的最终 seed。

最终建议样本量：

```text
14 个 2 mm 桶 × 5 个全新 seed × 每 seed/桶 96 个独立首回合
= 6,720 个独立几何回合
```

最终发布硬门：

- 52–80 mm 总体 `TIM_strict_success` ≥85%；
- 最差 seed ≥80%；
- 最差 2 mm 桶 ≥78%；
- B0 64–70 mm ≥92%，目标保持 ≥94.6%；
- 总体掉落率 ≤4%，过载率 ≤1.5%；
- 4 s 持稳成功率 ≥85%，6 s 长保持成功率 ≥78%；
- `P(4 s retained | acquired)` ≥90%；
- 扰动恢复成功率 ≥85%；
- 掌部辅助成功、自碰撞和安全投影均为 0；
- 保持动作单步变化 P95 ≤0.05、jerk P95 ≤96 rad/s³、方向反转 ≤2.4；
- 总体成功率 Wilson 95% 下界 ≥83%，每桶下界 ≥75%；
- 至少保存每个尺寸桶一条成功轨迹，以及每类主要失败轨迹；
- 发布包包含 checkpoint、完整控制合同、源码、USD、场景清单和全部 SHA-256。

虽然最终合同允许比阶段门稍宽的平滑上限，选模时仍优先选择满足 B0 严格平滑门的候选。

## 12. 当前下一步

当前可以安全开始的是 E0 工程工作，而不是 E1 长训：

1. 实现等效直径分桶采样与 cohort 标识；
2. 为独立评估器加入逐桶首回合 JSON 证据；
3. 增加采样、分母、终止原因和统计区间测试；
4. 用 model150 完成 64–70 mm 新旧采样器等价性门；
5. 对 52–80 mm 做逐 2 mm 零样本/准静态可达性地图；
6. E0 全部过门后，才允许从 E1 开始 PPO 扩域。

