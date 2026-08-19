# 掌托式五指包络抓握 PPO v2 改进执行计划

文档版本：1.8.87  
建立日期：2026-08-17  
状态：T0 DYNAMIC STABILITY / ACTION TRUST-REGION ITERATION / LIVING DOCUMENT  
任务范围：左手固定掌面向上，番茄位于掌心小范围随机位置；只训练手部掌托、五指包络、果梗摘断和摘后保持，不训练机械臂寻找果实。  
历史基线：`simulation/ppo_tomato_grasp/artifacts/stage4_band1_long/best.pt`，只读保留。  
历史计划：`PALM_SUPPORTED_ENVELOPING_GRASP_WORKPLAN.md`，作为 v1 审计记录，不在 v2 中覆盖。

> 本文件是下一轮工作的唯一执行计划。每完成一个任务、否决一个假设或改变一个门槛，都必须同步更新状态、证据、产物路径和更新日志。旧捏取成果、v1 包络权重及其评估文件均不得覆盖。

## 1. 结论先行

本次成果不理想的主要原因不是单纯的 PPO 训练时长不足，而是训练任务定义、观测、随机化配置和评估证据之间存在多处不一致。当前最可信的失败链路是：

`果径/形状信息饱和或配置错误 → 先短暂形成五指接触 → 阶段二奖励允许退化到三指支撑 → 远端关节只需产生位移即可获得参与奖励 → 断梗后逐指接触丢失、果实偏心 → 掉落`

随机化会进一步放大该问题，因为策略没有可靠的果径编码、完整的 14D 动作历史和分指节触觉信息。评估器中的批量标量广播和自动重置后采样又把部分诊断指标错误地报告为 100% 或 0%，使训练选择受到误导。

因此 v2 的顺序必须是：

`先修证据链与配置 → 再复评旧权重 → 再改任务/观测/奖励 → 分阶段训练 → 最后扩展随机化`

在 P0/P1 审计门通过前，不开始新的长时间 PPO 训练。

## 2. 已确认问题与证据等级

### 2.1 高置信度、必须先修复

| 编号 | 问题 | 本地证据 | 直接影响 |
|---|---|---|---|
| R1 | Band-1 球形开关与设计相反 | `physx_tomato_train.py:482-486` 注释要求 Band-1 等径、Band-2 起椭球，但实现为 `stage4_band > 1` 时等径 | 训练域与计划文档不一致；课程顺序被倒置 |
| R2 | “轻度随机评估”实际使用接近完整随机域 | `evaluate_enveloping_checkpoint.py:147-163` 固定使用 ±4 mm、摩擦 0.25–0.90、3 步延迟、较大死区/回差 | 23.82% 结果不能被准确解释为 Band-1 匹配评估；训练域和评估域混报 |
| R3 | 大果径观测饱和 | `physx_tomato_env.py:1335` 使用 `(axis-0.025)/0.012` 并截断；半轴约 37 mm 以上均变成 1 | 策略无法区分核心 37–40 mm 及更大果实，尺寸泛化缺少输入依据 |
| R4 | 阶段二仍奖励三指支撑 | `physx_tomato_env.py:1468` 定义拇指+任意两指为 `secure_now`；`1492-1502` 的摘后保持也允许该拓扑；`1638-1652` 持续支付对应奖励 | PPO 可通过捏取/三指局部最优获得高回报，与目标五指包络冲突 |
| R5 | 远端参与指标只证明“动过”，不证明接触或承力 | `physx_enveloping_env.py:174-177` 以累计关节位移锁存；`255-272` 直接奖励远端位移 | 指尖在自由空间卷曲也可通过门槛，不能证明远端自由度改善抓握 |
| R6 | 批量诊断指标被广播成全环境结果 | `evaluate_enveloping_checkpoint.py:261-266` 把一个 batch 标量写给所有环境 | 报告中的掌托先行 100%、远端参与 100% 不是逐回合统计，不能作为验收证据 |
| R7 | 终止步指标在自动 reset 后读取 | `evaluate_enveloping_checkpoint.py:245-258` 在 `env.step()` 后读取接触/阶段状态 | `five_digit_current_on_done_rate=0` 等指标可能只是重置后状态，失败分解不可信 |
| R8 | 指标分母不一致 | success/drop 以完成回合计数；pick/break/hold 使用“每个并行环境曾发生过”布尔量 | 不同指标不能直接组成同一漏斗，无法定位从包络到摘后的真实损失 |
| R9 | 14D 策略只观察前 10D 动作历史 | `physx_tomato_env.py:1390-1394` | 在死区、回差和限速存在时，远端四关节控制具有额外部分可观测性 |
| R10 | 触觉特征包含重复值和大量复制阶段位 | `physx_tomato_env.py:1345-1381`；4 个阶段量被复制到五指的 20 个槽位 | 有效信息密度低，挤占分指节接触和完整动作历史 |
| R11 | 所谓刚度随机化没有落实到实际接触材料 | 初始材料刚度固定在 `physx_tomato_env.py:316-321`；reset 只随机 `object_stiffness` 并写摩擦/恢复系数，见 `1811`、`1829-1836` | Critic/损伤代理看到随机刚度，但 PhysX 接触动力学仍可能保持固定，形成伪随机化 |
| R12 | 检查点没有保存完整环境配置 | checkpoint 只包含网络、优化器、update 和 metrics | 无法从权重独立还原训练分布，后续对照容易再次发生参数漂移 |

### 2.2 结果证据支持，但仍需专项实验确认

| 编号 | 假设 | 当前证据 | 验证方式 |
|---|---|---|---|
| H1 | 断梗后的主要失败是逐指接触丢失和偏心，而非过载 | nominal 掉落约 59.96%、过载约 0.87%；随机域掉落约 74.89%、过载约 1.28% | 记录断梗前后 ±20 步的逐指接触、果心偏移、速度和关节目标 |
| H2 | 当前摘取触发过早 | `prepick_hold_steps=3`，即约 150 ms；当前目标要求稳定掌托包络 | 对 3/6/10/20 步触发做控制变量实验 |
| H3 | 断梗力释放产生的瞬态需要策略主动响应 | 4 步 grace 在 nominal 提升、在随机域增加过载，说明单纯延迟终止不是解法 | 测量断梗瞬间 Δv、角速度、接触丢失顺序；训练显式摘后恢复阶段 |
| H4 | 当前触觉/接触几何不足以指导包络 | 实测 PhysX 接触点峰值比例只有约 0.31%–0.63%，其余使用 fingertip fallback | 分链节记录 link3/link4 接触，修复接触点缓存并报告 measured-only 指标 |
| H5 | PPO 更新本身没有稳定收敛 | 精炼时 critic loss 长期约 1e4–2e4，std 约 0.06 几乎不变，内部成功率 18.75%–37.5% 波动 | 奖励归一化、价值归一化、独立验证集和学习率/探索小矩阵对照 |
| H6 | 人工掌托面和 20 mm 边缘影响了动力学 | 原 base collision 被关闭，使用平面与边缘代理 | 原掌碰撞、平面无边缘、当前边缘三种物理对照；记录静态滚落率 |

## 3. 当前结果应该怎样重新解释

现有权重不是无效结果，它证明了手掌向上、五指短暂接触、摘断和部分保持可以在 PhysX 中出现。但以下数字必须保守解释：

- 核心 nominal 三种子严格成功均值约 39.17%，掉落约 59.96%，过载约 0.87%。
- 当前所谓随机化三种子均值约 23.82%，掉落约 74.89%，但其配置更接近完整压力域，不能称为 Band-1 训练匹配结果。
- `palm_first_observed_rate=100%` 和 `distal_all_observed_rate=100%` 受批量广播统计影响，需要重新计算。
- 断梗后窗口和精炼没有形成稳定收益：窗口会在随机域增加过载，精炼权重独立均值约 39.62%，未超过基线。
- 当前失败最集中在摘后掉落；继续增加同类奖励或延长训练，可能只会强化三指局部最优。

## 4. v2 不可偏离的设计原则

1. 成功必须是“掌托先行 + 五指持续包络 + 摘断 + 摘后稳定保持”，三指支撑只能作为中间状态，不能作为阶段二正向终点。
2. 远端关节贡献必须由接触位置、承力持续时间和反事实消融证明，不能用“关节曾移动 0.03 rad”代替。
3. 训练、验证和压力测试使用命名配置文件，不再依赖散落的 CLI 默认值。
4. 每个 checkpoint 必须保存完整解析后的配置、源码哈希、父 checkpoint、随机种子和观测/动作契约版本。
5. 先在 35–40 mm 等径核心带解决任务，再逐项引入椭球、位置、材料、传感器和执行器扰动。
6. 任何提升必须同时满足成功率、掉落、过载、五指持续性和远端贡献门槛；不得只看单一 seed 或内部最高值。
7. v1 基线和旧小果径捏取策略全部只读；v2 使用独立代码和产物目录。

## 5. 分阶段执行流程

### P0：冻结历史成果并建立 v2 命名空间

目标：保证新实验可回退、不会再次混淆策略。

工作清单：

- [x] P0-01 校验并记录 v1 基线、失败精炼候选、报告和源码 SHA256。
- [x] P0-02 创建 `simulation/ppo_tomato_grasp_v2/`，只复制必要模块，不修改 v1 代码。
- [x] P0-03 创建 `simulation/ppo_tomato_grasp/artifacts/enveloping_grasp_v2/`，训练与评估分目录。
- [x] P0-04 定义 v2 独立 contract（先以 `contract.py` 固化，JSON 导出在 P1-07 完成）。
- [x] P0-05 定义配置档：`v1_nominal_reliable`、`v2_band1_match`、`v2_band2_shape`、`v2_band3_sensor_actuator`、`v2_band4_stress`。

退出门槛：v1 校验全部通过；v2 不会写入任何旧权重目录；所有配置能解析并导出为 JSON。

### P1：修复证据链和配置，不训练

目标：先让每一个数字都代表真实的逐回合事件。

工作清单：

- [x] P1-01 修正等径/椭球开关：v2 Band-1 profile 强制 `isotropic_axes=True`；Band-2 起才允许独立三轴采样。
- [x] P1-02 评估器改为读取命名配置，不再硬编码“全随机默认值”。
- [x] P1-03 在环境 reset 前写入 terminal snapshot（阶段/终止码、五指接触、掌托、逐指力、远端参与、断梗和掉落事件）。link3/link4 的逐链节快照仍列为 P3 接触适配任务。
- [x] P1-04 所有 v2 evaluator 比率使用逐完成回合快照计数；不再广播 batch 标量。
- [x] P1-05 v2 snapshot 保留 measured geometry 布尔字段；fallback 与 measured-only 的分项报告将在接触链节适配后补齐。
- [x] P1-06 修复果径归一化，使 30–50 mm 半轴全范围均不饱和；增加边界单元测试。
- [x] P1-07 checkpoint 保存 resolved config、观测契约版本、源码哈希、父权重和训练命令。
- [x] P1-08 验证并标注 compliant stiffness：当前 API 路径只可靠改变 damage/tactile prior，不能宣称真实材料刚度已随机化。
- [x] P1-09 为 2/32/256 环境运行 reset、接触、自动重置和终止事件一致性测试；64 环境保留为后续压力测试。

退出门槛：

- terminal snapshot 与回放逐帧人工抽查一致；
- `success + drop + overload + damage + timeout = completed episodes`；
- 相同回放离线重算与在线报告误差为 0；
- 等径 Band-1 三轴差小于数值容差，Band-2 才出现长宽比变化；
- 35、37.5、40、45、50 mm 在观测中映射到不同值且不饱和。

### P2：重新评估 v1，定位主要退化来源

目标：不训练，先得到可信的基线和单因素敏感度。

评估矩阵：

| 配置 | 变化内容 | 用途 |
|---|---|---|
| E0 | 37.5 mm 等径、中心、无传感器/执行器扰动 | 最小可解释基线 |
| E1 | 35–40 mm 等径 | 纯尺寸泛化 |
| E2 | E1 + 小范围位置随机 | 位置敏感度 |
| E3 | E2 + 椭球长宽比 0.9–1.1 | 形状敏感度 |
| E4 | E3 + 摩擦/质量 | 物体动力学敏感度 |
| E5 | E4 + 传感器延迟/偏置/掉帧 | 触觉鲁棒性 |
| E6 | E5 + 执行器响应/死区/回差 | 控制鲁棒性 |
| E7 | 当前 stress_full | 只作压力测试 |

工作清单：

- [x] P2-01 每个配置运行 6 个未见种子（20260820–20260825），每个配置累计约 1540–1552 个完成回合（每 seed ≥256）。
- [x] P2-02 输出逐回合终止码、成功/掉落/过载、掌托、五指终止接触、远端参与和实测几何漏斗；完整汇总见 `baseline_recheck/p2_aggregate.json`。
- [x] P2-03 输出每根手指的首次接触、断梗前/后接触占空比、link3/link4 载荷和远端实测占空比；字段写入 terminal snapshot/evaluator。
- [x] P2-04 完成 full14/freeze_distal/roots_only 反事实：markov T0 seed20260820 分别为 18.75%/28.68%/12.31% 严格成功；远端实测覆盖接近 0，证明旧远端动作存在干扰且 joint2 指标不能代替真实接触。
- [ ] P2-05 做断梗事件窗分析：断梗前后各 20 步的果实速度、角速度、中心偏移、五指力和关节目标。
- [ ] P2-06 对 3/6/10/20 步预摘保持做无训练物理对照。

退出门槛：至少能解释总失败中的 80%；每个主要失败被归入接触丢失、偏心滑移、过载、提前摘取、几何不可达、传感器/执行器扰动或物理异常之一。

### P3：构建 v2 观测、动作和触觉闭环

目标：让策略真正拥有使用所有手指自由度所需的信息。

工作清单：

- [x] P3-01 采用新的 128D actor/177D critic 观测契约，保留完整 previous_action[14]。
- [x] P3-02 阶段量只编码一次，避免复制到五根手指的 20 个槽位。
- [x] P3-03 分开 link3/link4 力和接触占空比；Actor 使用可映射触觉特征，Critic 保留 PhysX 特权量。
- [x] P3-04 对每指加入 force、force-rate、contact-age、lost-contact-age、proximal/distal load ratio 和传感器有效位。
- [x] P3-05 保留 14D 动作，根部即时通道和远端残差使用不同低通速率。
- [x] P3-06 远端动作奖励仅在至少四指接触且对应接触质量成立时计入，并新增近果面/接触后动作门控，禁止自由空间位移正奖励。
- [x] P3-07 为动作滤波、远端雅可比、迟滞/状态恢复建立离线单测；完整 PhysX 回差曲线仍在 T1 验收。

退出门槛：

- 各远端残差只影响对应 joint2；
- 动作历史完整可观测；
- 触觉失效位能区分无接触和传感器掉线；
- 远端自由空间运动不再获得正奖励；
- 反事实关闭远端动作会明显降低远端接触占空比，而不是只改变关节位移指标。
- [x] scripted link3/link4 probe 已完成并把 measured/virtual 分开；当前 measured link4 覆盖为 0%，因此以下“远端接触”只能按代理通道解释。
- [ ] 修复 URDF 碰撞体与 ContactSensor 配置，使四根直指 measured link4 在固定名义探针中均取得非零覆盖；未通过前不得把 T0 成功率写成真实触觉成功。

### P4：重写阶段目标和奖励

目标：消除三指捏取局部最优，将五指持续包络设为唯一正向摘取路径。

推荐使用显式阶段机，PPO 只控制手指动作：

1. `SETTLE`：果实在掌心落稳，手掌支撑持续成立。
2. `PRESHAPE`：拇指对置、四指根部快速预定位，不允许摘取。
3. `ENVELOP`：近端与远端沿果面闭合，形成五指空间覆盖。
4. `STABILIZE`：五指和掌托连续保持，降低果实相对速度和峰值压力。
5. `HARVEST`：外部模块施加固定摘取轨迹，策略仅保持/恢复包络。
6. `POSTBREAK`：断梗后继续至少 1–2 秒五指低压力保持，并接受小扰动。

工作清单：

- [x] P4-01 删除阶段二 `secure_now=拇指+两指` 的持续正奖励；它只作为诊断中间事件。
- [x] P4-02 严格成功触发要求五指、掌托、覆盖、对置、力安全和连续保持。
- [x] P4-03 逐指接触质量和严格几何门共同决定成功，单指失联不会被其余四指补偿。
- [x] P4-04 摘后奖励包含接触连续性、低速度、低压力和远端接触恢复项。
- [x] P4-05 已加入远端/过载/接触质量诊断；根部饱和与掌缘卡住反事实留在 P2-04。
- [x] P4-06 成功/失败在 reset 前写入独立 terminal snapshot。
- [x] P4-07 reward、return、value 和非有限参数保护已加入 JSONL 训练日志。

退出门槛：脚本化成功轨迹的回报必须显著高于三指捏取、瞬态五指、远端空动和掌缘卡住反例；所有反例都有单元测试。

### P5：状态一致的分阶段训练

目标：先分别学会包络与保持，再合并完整摘取，避免一次同时学习所有难点。

训练分支：

- Branch A：从 v1 actor 做行为蒸馏/初始化，但重置 Critic、优化器和探索参数。
- Branch B：v2 从零训练。
- 两分支使用完全相同的训练/验证种子和预算；不得直接加载 v1 Adam 状态继续低学习率漂移。

课程：

| 阶段 | 场景 | 训练目标 | 进入下一阶段门槛 |
|---|---|---|---|
| T0 | 37.5 mm 等径、中心、无摘取 | 掌托后五指包络 | 成功 ≥90%，过载 <1% |
| T1 | 35–40 mm 等径、小位置随机 | 1 秒稳定包络 | 成功 ≥88%，掉落 <8% |
| T2 | 从完整物理 snapshot 起步 | 断梗后 2 秒动态保持 | 成功 ≥85%，过载 <2% |
| T3 | nominal 完整摘取 | 包络→摘断→保持 | 成功 ≥85%，掉落 <12% |
| T4 | 小椭球与摩擦/质量随机 | 形状和材料鲁棒 | 成功 ≥80%，过载 <2% |
| T5 | 逐步加入触觉和执行器扰动 | 闭环鲁棒 | 训练匹配域成功 ≥80% |

状态一致 retention snapshot 必须同时保存并恢复：

- 16D 关节位置和速度；
- 14D 动作、16D 目标、方向、死区/回差内部状态；
- 番茄位置、姿态、线速度、角速度和三轴尺寸；
- 触觉历史、延迟、掉线状态和逐指接触链节；
- 掌托、摘取进度、茎秆状态、断梗事件和损伤状态。

恢复后先执行短暂物理 settle，并验证接触/速度连续；仅保存关节位置的旧 seed bank 不再使用。

PPO 稳定性措施：

- reward/return 归一化或 PopArt；
- value clipping、梯度裁剪和 explained variance 监控；
- 独立 validation env，不用训练环境挑选 best；
- 以成功率 Wilson 下界、掉落和过载共同选择 checkpoint；
- 8 GB GPU 初始采用 32–64 env、64 horizon，确认内存后再扩展；
- 本地 JSONL 为权威记录，W&B 只做镜像展示。

### P6：逐项扩展随机化

目标：确认每一种随机因素造成的损失，并在损失可控时才进入下一档。

顺序固定为：

`尺寸 → 位置 → 椭球率 → 质量/摩擦 → 茎秆 → 触觉噪声/延迟 → 执行器响应/死区/回差 → 组合域`

工作清单：

- [ ] P6-01 每加入一类随机化，保持其余参数不变并运行 matched validation。
- [ ] P6-02 单项引入后成功率下降超过 5 个百分点，停止扩域并针对该因素训练。
- [ ] P6-03 过载达到 2% 立即拒绝当前课程，不通过放宽安全阈值补救。
- [ ] P6-04 stress_full 始终单独报告，不与交付域均值混合。
- [ ] P6-05 对不同果径输出策略动作、载荷份额和各指远端贡献曲线。

### P7：最终独立验收与可视化交付

交付域暂定为 35–40 mm 半轴的大果径核心域；38–50 mm 保持压力测试身份，除非 P6 明确通过。

最终验收至少包含 5 个从未参与训练/选模的 seed，每个 seed 至少 256 个完成回合，总计不少于 1280 回合。

| 指标 | 交付门槛 |
|---|---:|
| 严格成功率 95% Wilson 下界 | ≥80% |
| 掉落率 | <15% |
| 过载率 95% Wilson 上界 | <2% |
| 掌托先行率 | ≥98% |
| 摘取前五指持续包络率 | ≥95% |
| 摘后 2 秒五指保持率 | ≥90% |
| proximal-only / pinch topology | ≤3% |
| 四根直指远端接触占空比 | 每指 ≥60% |
| 指尖/远端反事实贡献 | 关闭后成功率下降 ≥10 个百分点，且非由过载造成 |
| 碰撞伪成功、掌缘卡住伪成功 | 0 |

交付物：

- 最优 checkpoint、最后 checkpoint 和完整配置快照；
- 训练/验证曲线、独立评估 JSON、失败分类和置信区间；
- 随机环境实时权重回放，而非成功录像循环；
- 成功、临界、掉落、过载、缺指和近端主导的代表视频；
- 逐帧轨迹，可离线重算所有严格指标；
- 源码、权重、配置和报告 SHA256；
- `hardware_command_allowed=false`，在实机标定完成前禁止直接部署。

## 6. 选模、停止和回退规则

1. 任一指标统计不一致，停止训练，回到 P1。
2. 训练内部成功上升但独立验证下降超过 5 个百分点，不升级。
3. 过载率达到 2%，不论成功率多高都不升级。
4. 若 distal 指标上升但 distal-contact 反事实无影响，视为奖励投机，回到 P3/P4。
5. 若某随机因素造成超过 10 个百分点下降，拆成更小课程，不直接扩大训练时长。
6. 不允许通过缩短保持窗、增加 grace、提高压力上限或降低五指门槛包装成功率。
7. v2 未达到门槛时，继续保留 v1 作为可复现基线；不得覆盖历史结果。

## 7. 预计产物目录

```text
ppo_tomato_grasp_stage01_work/
├── PALM_ENVELOPING_GRASP_V2_IMPROVEMENT_EXECUTION_PLAN.md
├── simulation/ppo_tomato_grasp_v2/
│   ├── configs/
│   ├── env/
│   ├── metrics/
│   ├── train/
│   ├── evaluate/
│   └── tests/
└── simulation/ppo_tomato_grasp/artifacts/enveloping_grasp_v2/
    ├── audit/
    ├── baseline_recheck/
    ├── t0_static_enclosure/
    ├── t1_size_position/
    ├── t2_retention/
    ├── t3_full_harvest/
    ├── t4_material_shape/
    ├── t5_sensor_actuator/
    ├── final_evaluation/
    └── release/
```

## 8. 第一批执行任务

下一次开始实施时，严格按下列顺序：

- [x] A-01 冻结 v1 校验清单并建立 v2 目录。
- [x] A-02 编写配置解析与快照测试。
- [x] A-03 修正 Band-1 等径逻辑和果径观测归一化。
- [x] A-04 实现 terminal snapshot 与逐回合指标累加器。
- [x] A-05 v2 保留 measured geometry 位，并已加入逐链节 link3/link4 分解字段。
- [x] A-06 用 2/32/256 env 完成自动 reset、指标一致性、action/obs contract GPU 烟测。
- [x] A-07 对 v1 权重完成 E0–E7 首个未见 seed 重评估，输出新的根因排序。
- [x] A-08 P1/P2 的 matched validation 完成后开始 P3/P4 任务重构；T0/T1 训练验收仍未通过。
- [x] A-09 完成 512 候选的静态多通道 reach primitive 搜索；确认四指在运动学上存在可达组合，但不把候选姿态当作成功示范。
- [x] A-10 完成候选 57 的 4 mm physical proxy 分阶段路径反事实；确认接触动力学会产生高掉落/过载，T0-B 不通过。
- [x] A-11 完成第一版 raw link4 力反馈的逐指低力接触—保持—退让诊断；逐指观察提高但仍掉落 37.50%、过载 9.38%，不进入 PPO。
- [x] A-12a 固定果心/掌托 contact-window：修正 link4-only debug 碰撞过滤，并完成候选57、8 mm proxy、middle 残差尺度0.7、0.10 rad/s 的 256-env×3 seed 复核；四指实测接触率 100%、掌托支持 100%、果心漂移 0、过载 0%，各指最大力均低于对应 ceiling。
- [x] A-12b 动态 retention-only 单指承力窗：禁用摘取/断梗，解除果实运动学固定，按 index→middle→ring→little 分阶段测试低速进入、接触保持和高力退让；nominal 256-env×3 seed 已完成，candidate85 + per-digit force-band（lower=0.10 N，upper=4.0/3.0/3.0/2.5 N）逐指保持 99.1%/78.2%/95.1%/98.9%、四指同时73.2%、掉落0%、过载0%。该结论只覆盖37.5 mm等径 nominal 动态保持，不代表摘梗或随机化通过。
- [x] A-12b-0 动态 retention screening：完成 256-env 动态多指与 retention-only seed hold；index/ring 实测承力余量不足，四指同时接触为0%，未满足每指保持门，正式 256×3 通过矩阵暂不启动。
- [x] A-12c 动态 seed signed-Jacobian/尖端距离审计：固定掌托与已接触手指，对 index/middle/ring/little 分别记录 root/distal/global 通道的有符号 tip 距离、果心偏移和载荷反事实；结果确认动态果心重排与多通道耦合是 index/middle 余量不足的主因，详见 v1.8.36–v1.8.38。
- [x] A-12d 接触映射修复候选：在仿真命名空间固定 6 mm physical proxy、contactOffset=0.3 mm、restOffset=0.1 mm、mid-cup 与 per-digit reach-assist 后，candidate85 通过固定果心与动态 seed 双闸门；后续 256-env×3 seed force-band hold 逐指99.11/78.16/95.08/98.91%、四指同时73.16%、掉落/过载0%。该几何/动作基元仅作为 simulation-only 接触映射证据，不写回未标定 URDF/硬件。
- [~] A-13 已在 A-12b/A-12d 名义闸门通过后恢复 T0-B PPO；首轮安全信赖域续训已完成，但独立 actor-only 评估未通过，继续留在名义域改进，不进入 T1/扩域。
- [~] A-14 已完成 residual base controller 的名义安全基线复核：默认改为“掌部/根部/拇指先行 → 四指逐窗插入 → 低力带保持”，240-step approach + 60-step/digit 在 16-env、37.5 mm nominal 下无掉落/过载、掌托支持100%、四指实测观察75%；仍未达到五指同时包络门，不能直接进入长训。后续在该安全节奏上仅调可观测 residual 和小指 reach，不放宽终止阈值。
- [~] A-15 residual PPO smoke 的全通道 actor 在 0.15 residual scale 下破坏安全基线（drop=25%、measured distal-all=50%）；新增 `residual-mask`（默认 thumb+distal）和 approach 后 phase gate，禁止 PPO 在粗接近阶段扰动掌部/根部。旧 actor 在 phase-gated replay 下恢复到 drop=0、overload=0、palm=1、measured distal-all=0.75，但尚无接触提升；下一轮只训练“训练时即带 mask+phase gate”的 residual actor。

## 9. 当前风险与暂存疑事项

- 当前没有真实番茄和实机标定数据，因此接触刚度、果皮损伤、摩擦和果梗强度仍是仿真先验；v2 可以提高内部一致性，不能直接宣称真实无损。
- 灵巧手 URDF 与实机存在已知误差；任何策略都必须在后续实机部署前重新标定关节零位、方向、行程和指端位置。
- Isaac Sim 4.5 接触点回传稀疏，若无法提高 measured-only 覆盖率，必须明确使用链节接触代理并降低 sim2real 结论等级。
- 人工掌面和平面边缘是否符合真实掌部几何尚未确认；P2 必须通过无边缘/原碰撞对照确定影响。
- v1 权重可作为行为先验，但其奖励局部最优可能被蒸馏到 v2，因此必须保留从零训练对照。

## 10. 更新日志

- 2026-08-17 / v1.8.0：完成一次 v2-only 动作空间修复与证据闭环。原 14D 末端残差只作用 link4，导致大果径时指尖到不了球面；新增接近果面、尚无末端负载时的对应近端耦合，加入仿真专用 virtual distal tactile pad（不启用物理碰撞，不代表实机传感器）以及可配置的粗动作平滑。512 候选 nominal 搜索首次观察到“四根 link4 + 五指 + 掌托 + 在手内”分支；Band-1 随机域中仍未找到无过载、可持续保持的静态全五指分支，说明必须训练动态触觉调节。
- 2026-08-17 / v1.8.1：`t0_band1_curriculum120_ppo`（120 update）独立评估 279 回合：成功 0%、掉落 98.21%、过载 1.79%；代理接触率不能替代终端 link4 载荷，checkpoint 不升级。`t0_band1_assist_ppo180` 使用候选先验，30 update 后回报约 -4，因先验诱发过载/掉落而中止，失败权重保留。
- 2026-08-17 / v1.8.2：发现全通道平滑（alpha=0.25）会削弱 nominal link4 接触，标记为失败配置并将正式训练恢复为 direct coarse targets。nominal direct PPO（160 update）中段回报达到正值，但 best checkpoint 独立评估 256 回合仍为成功 0%、掉落 59.00%、过载 3.83%、超时 37.16%、distal_contact_all 31.80%；尚未达到 T0，禁止进入 T1/P6。
- 2026-08-17 / v1.8.3：当前仍保持 `hardware_command_allowed=false`；所有 virtual contact、reach assist 和 candidate search 仅服务于仿真可达性/观测性审计，旧 v1 权重和历史训练目录不覆盖。下一步优先修订 retention reward/teacher 的阶段时序，并在 nominal 通过严格五指保持后再做 Band-1 多 seed。
- 2026-08-17 / v1.8.4：完成 scripted link3/link4 接触闸门。新增 measured 与 virtual link4 分离字段并在 snapshot/evaluator 中保留；15 环境固定名义探针显示四根直指 measured link4 覆盖均为 0%、最大力均为 0 N，virtual proxy 覆盖均为 100%（0.388–0.485 N），原始接触点有效标记全为 false。该证据解释了“训练曲线/代理 distal_contact 上升但真实 distal 传感器不变”的表面矛盾。PPO 扩域和 T1–T5 暂停，下一阶段必须先完成碰撞体/传感器审计；在此之前任何成功率只能标注为 proximal+virtual-contact 仿真指标。
- 2026-08-17 / v1.8.5：训练器的 `best.pt` 现在只从 teacher 完全退出后的 rollout 选择；4 环境/2 update smoke 通过且 checkpoint 更新号符合预期。该证据链修复完成后，继续训练仍受 measured link4 接触闸门约束，不得用 virtual proxy 覆盖率替代传感器验证。
- 2026-08-17 / v1.8.6：对 `t0_nominal_distalshaping_ppo180/best.pt` 进行 128 环境 actor-only 验证，完成 133 回合：代理口径成功 0%、掉落27.82%、过载2.26%、掌托支持69.92%；virtual distal-contact-all 28.57%，measured distal 覆盖仅食/中指各0.75%、环/小指0%，measured geometry 0%。T0 不通过，不进入 T1–T5；当前执行状态保持在 contact-observability gate。
- 2026-08-17 / v1.8.7：measured link4 字段接入后再次运行 2-env retention snapshot smoke，79 个字段最大恢复误差 0，状态一致性通过；该修复不会破坏恢复链路，但不改变 measured contact 闸门结论。
- 2026-08-17 / v1.8.8：完成新一轮失败复盘并建立 `PALM_ENVELOPING_GRASP_V2_FAILURE_REPLAN_20260817.md`。三次 palm-first audited 运行均为严格成功 0%；α=0.25 平滑分支将掉落从 95–99% 降至 19.12%，但仍为五指终止 0%、实测远端接触仅 8.82%、超时 68.38%、过载 12.50%。checkpoint 逐步轨迹确认粗/根部/远端动作在接触后长期饱和，把果实从掌心推出后自由落体；平滑只能减小冲量，不能替代真实远端承力。后续执行状态转为 T0 动态稳定与动作信赖域迭代：先做教师退出续训、动作窗口/果心稳定性日志和反事实对照，再恢复五指远端课程，禁止直接扩大随机化。
- 2026-08-17 / v1.8.9：教师退出续训完成，独立评估 131 个完成回合：成功 0%、掉落 23.66%、过载 8.40%、超时 67.94%、掌托支持 73.28%、五指终止接触 0%、四指实测远端同时接触 12.21%。该结果不通过 T0-A；teacher 退出后未自动形成稳定五指实测承力。新的失败复盘与下一轮 P1 动作信赖域/果心稳定性修复详见 `PALM_ENVELOPING_GRASP_V2_FAILURE_REPLAN_20260817.md`。
- 2026-08-17 / v1.8.10：P1 trust-region 分支完成：独立评估成功 0%、掉落 21.71%、过载 5.43%、掌托 76.74%、五指终止 0.78%、实测四指 distal 同时接触 9.30%。同权重反事实 `freeze_distal` 掉落 13.28%/掌托 83.59%，`roots_only` 五指终止 1.52%但拇指力积分占比 49.6%；说明远端自由动作是稳定性瓶颈，而冻结远端本身不能解决接触学习。下一步改为逐指 link3 承力后释放 link4 residual 的课程，不扩大随机化。
- 2026-08-17 / v1.8.11：P2 逐指 link3 闸门分支完成：独立评估成功 0%、掉落 40.00%、过载 7.69%、掌托 57.69%、五指终止 2.31%，但实测 distal-all 提升至 23.08%（各指实测观察约 11.5%/16.9%/20.8%/44.6%）。逐指释放提高了真实远端接触，却暴露出释放后的粗/根部继续推离掌托；该权重不升级，下一轮限制掌托稳定后的粗动作上限并延长渐进窗口。
- 2026-08-17 / v1.8.12：P2b 粗动作上限 0.45/渐进 120 步反而掉落 50.37%、掌托 48.89%，P2c 第 60 步开放 distal 掉落 25.58%、过载 10.08%、掌托 73.64%，均未通过 T0-A。8 mm fingertip proxy 诊断使 measured distal-all 从 23.08% 提升到 27.13%、掉落从 40.00%降到32.56%，但 link4 平均载荷仍接近零；下一步转向 proxy 几何、接触 offset/rest offset、质量和持续承力矩阵，禁止继续盲加动作限制。
- 2026-08-17 / v1.8.13：P3 几何/质量 screening 矩阵完成（18 格：proxy 半径 6/7/8 mm、contactOffset 2/1 mm、低/中/高质量密度；固定 P2 checkpoint、动作课程和 strict measured gate）。矩阵汇总 `artifacts/enveloping_grasp_v2/p3_matrix_screen/p3_matrix_summary.json`：18 格中 9 格达到 screening 的 ≥32 个完成回合，T0-A 通过格为 0；最佳稳定性格仍掉落约 79%、掌托约 16%，而 8 mm/default 格 measured distal 观察率可到约 83%，持续 link4 力积分和掌托仍不足。由此排除“只调 proxy 半径/offset/质量即可恢复抓握”的假设；P3 后续转为同一几何下的阶段保持/接触后退让训练与 P4 反事实，几何矩阵仅保留为碰撞可见性校准证据，不能用作成功率选模。
- 2026-08-17 / v1.8.14：新增 `palm_first_motion_hold` 外移抑制层并完成反事实。常规阈值配置的 48 个完成回合为掉落 91.67%、过载 8.33%、掌托 6.25%、实测 distal-all 35.42%；极端冻结粗动作配置 26 个回合全部掉落。结论：硬冻结或被动 hold 不能替代 `PALM_SETTLE → 继续闭合远端` 的可学习阶段，后续训练必须让 distal/远端微调在保持掌托的同时继续推进，并把 coarse 横向冲量作为约束而非全局关闭。
- 2026-08-17 / v1.8.15：P4 半径 8 mm 分支续训至 160 update 的独立评估（128 回合）为掉落 25.78%、过载 0%、掌托 74.22%、超时 74.22%、measured distal-all 19.53%，较 80-update 检查点退化。该结果验证了“继续更新不一定改进”，160-update 权重冻结为失败样本；当前候选回退至 80-update 权重，先完成 ≥256 回合多 seed 复核，再决定下一次课程调整。
- 2026-08-17 / v1.8.16：在 80-update 权重上将 `palm_first_coarse_max_scale` 设为 0.4、渐进窗口设为 160 步，完成 256 回合独立评估：掉落 9.77%、过载 1.95%、掌托 89.45%，T0-A 首次通过；但 measured distal-all 8.20%、五指终止 0%，T0-B 未通过。该配置冻结为 T0-A 安全基线，后续只在其安全信赖域内优化 distal/逐指承力，不扩大随机化。
- 2026-08-17 / v1.8.17：T0-A 安全基线内的 measured-only 续训（关闭 virtual distal shaping）独立评估 128 回合：掉落 5.47%、过载 0%、掌托 92.19%，但 measured distal-all 仅 5.47%，各指观察约 5.47%/6.25%/5.47%/52.34%。安全门保持，T0-B 未改善；该结果证明远端载荷反馈稀疏/几何可达性仍是瓶颈，下一步增加单指接触保持课程和 link4 代理可达性探针，不放宽安全阈值。
- 2026-08-17 / v1.8.18：在 T0-A 基线 `max=0.4/ramp=160` 上完成 probe 强度审计。0.14 probe（128 回合）掉落 7.81%、过载 0%、掌托 92.19%、measured distal-all 10.16%；0.20 为掉落 7.81%、过载 0.78%、掌托 91.41%、measured distal-all 9.38%；0.30 为掉落 17.97%、掌托 82.03%，长 ramp/hold 仍掉落 21.48–25.00%。远端探针存在明显安全拐点，未通过 T0-B。
- 2026-08-17 / v1.8.19：新增并验证仿真专用 staggered probe 和逐指 probe action vector。逐指调度最高得到 measured distal-all 18.32%、各指约 20.61/21.37/22.14/66.41%，但掉落 19.85%、过载 2.29%；staggered 版本掉落 23.85%。该功能只用于动作可达性诊断，不进入实机控制，也不用于选模。
- 2026-08-17 / v1.8.20：完成三类受控续训。联合弱教师 + probe 的 256 回合评估掉落 17.37–22.52%、measured distal-all 13.13–16.79%；冻结粗动作仅训练 distal 的分支掉落 15.27%、measured distal-all 12.60%；加入连续 raw link4-force shaping 后掉落 17.42%、过载 2.27%、measured distal-all 14.02%。这些结果均不通过 T0-A/T0-B，且说明“加奖励/冻结粗动作”不能单独解决 link4 到达。当前计划回退为逐指单通道动作方向/Jacobian 审计与单指承力课程，保持 T0-A 基线只读封存，不扩大 probe/assist 或随机化。
- 2026-08-17 / v1.8.21：新增只读 `action_jacobian_probe.py`，对 T0-A checkpoint 的四指 distal ±0.18 和 matching proximal ±0.18 做同名义状态有限差分；index/middle/ring 均未因小扰动获得稳定 measured link4，little 的既有接触占主导。增大到 distal ±0.50 + proximal ±0.60 时，只有 index 正向短暂出现约 0.21 N measured link4，但该分支掉落；其余手指仍未达标。下一步从“每指四个动作通道的方向组合/动作基元”开始单指课程和 Jacobian 记录，暂不进行全局 PPO 长训或扩域随机化。
- 2026-08-17 / v1.8.22：将 Jacobian 审计扩展到 global flexion/curl/balance/spread/thumb opposition 五个 synergy 通道 ±0.25。单通道扰动仍不能稳定形成 index/middle/ring measured link4，部分 curl/balance/spread 方向只改变 little 接触时序；结合 residual/近端的大扰动也会先触发掉落。计划更新为“每指多通道 reach primitive → 单指 measured 承力窗 → 逐指合并”，暂不进行全局 PPO 长训或随机化扩域。

- 2026-08-17 / v1.8.23：完成 link4/URDF 碰撞几何审计。原始直指 link4 碰撞 mesh 约 1×2×1 mm，reset 时四个直指 link4 到果心约 84.9–97.2 mm；v2 proxy 以 link4 原点为中心、半径 8 mm。该证据支持“virtual distal 信号比 raw link4 更早出现”的解释，但仍不能把 proxy 原点当作实机指腹标定。
- 2026-08-17 / v1.8.24：完成 physical proxy 接触路径反事实。8 mm 物理 proxy 能在随机动作中触发四指 measured link4，但掌部峰值约 7–11 N 且掉落/过载显著；慢动作仍不能稳定保持。4 mm proxy + contactOffset 0.3 mm + restOffset 0.1 mm 能降低冲量，但当前 256 候选输出中仍没有全四指实测接触、在手内且不过载的安全候选。该分支只作为几何诊断，不进入训练默认，也不改变 T0-A 基线。
- 2026-08-17 / v1.8.25：为 `link4_action_search.py` 增加仿真专用 proxy offset CLI，并把下一阶段明确为“每指多通道 reach primitive / signed Jacobian → 单指 measured 承力窗 → 逐指合并”。在该证据完成前暂停全局 PPO 长训、probe/assist 放大和随机化扩域；T0-A 安全 checkpoint 继续只读封存，旧 v1 目录不改。
- 2026-08-17 / v1.8.26：新增 v2-only 可选 `per_digit_root_residual_span_fraction` 和 `multichannel_reach_primitive.py`。隔离探针关闭手指碰撞、virtual distal 和 stem 动力学，只测动作对指尖距离的运动学影响；默认映射保持不变。root span 0.40/0.75 的单通道扰动仍只有亚毫米级目标改善，global flexion 的共同位移约 2 mm，说明必须求多通道组合基元；该参数暂不进入任何 PPO checkpoint，继续停在 T0-B 可达性审计。
- 2026-08-17 / v1.8.27：在 T0-A checkpoint 上对 root span 0.40 做独立反事实（130 完成回合）得到掉落 13.85%、掌托 85.38%、过载 0%、measured distal-all 0%；相较 T0-A 基线掉落变差且无真实 link4 改善。结论：root span 只能作为后续组合 primitive 的候选自由度，暂不训练、不选模。
- 2026-08-17 / v1.8.28：完成 512 候选的静态多通道 reach primitive 搜索；在关闭碰撞动力学、冻结果实的诊断条件下，四指同时进入 39 mm 壳层的比例为 36.72%，候选 57 的四指最小距离约 32.6/27.6/29.3/32.5 mm 且果心无漂移。随后以相同候选执行 4 mm 物理 proxy 分阶段接触，64 回合出现掉落 65.63%、过载 34.38%、掌托支持 21.88%，虽 distal-all 87.50% 但实测四指同时接触仅 20.31%，严格成功 0%。这说明当前瓶颈不是“没有一组姿态能到达”，而是接触时序、低冲量进入和承力保持；不将候选 57 写入教师/默认训练，T0-B 继续保持未通过。下一步转为逐指低力接触闭环基元，并要求接触后保持窗和退出退让反事实同时通过。
- 2026-08-17 / v1.8.29：完成第一版逐指低力接触—保持—退让闭环诊断（32 回合）。4 mm physical proxy、offset 0.3/0.1 mm、virtual distal 关闭时，逐指 raw link4 观察率 50.0/59.4/62.5/46.9%，四指同时观察 9.38%，但掉落 37.50%、过载 9.38%、严格成功 0%；最大指尖力 18.0 N、平均掌部峰值 12.52 N。结论：力反馈已经能提高部分真实远端观察，却未解决接触进入冲量；下一阶段必须转为 retention-only、先逐指建立低速低力承力窗，禁止在当前路径上直接继续 PPO。
- 2026-08-17 / v1.8.30：A-12 单指固定果心/掌托 smoke 完成。新增 `retention_single_digit_probe.py` 的物理 proxy 半径、动作速率和接触占空比记录，并将 debug 碰撞过滤收紧为“掌托 + 目标 digit 的 link4/proxy”，避免目标指的 link3 先遮蔽 link4。8 mm proxy 下，index（候选57，32-env/260-step）峰值约 1.20 N、果心漂移约 0；middle 使用独立物理搜索候选 206，低速 0.10 rad/s 时接触占空比 27.4%、峰值 8.97 N（低于 9.75 N ceiling 但未达到 60% 保持门）；ring 占空比 26.0%、峰值 4.71 N；little 占空比 75.1%、峰值 5.05 N；middle 以 8 mm/常规速率会出现约 13.6 N 峰值并触发过载，证明“单一 proxy/速率/阈值”不可直接泛化。该阶段仍未通过 A-12：尚无四指均达到掉落≤10%、过载≤2%、接触保持≥60% 的 256×3 seed 证据，禁止合并 PPO；下一步先做按 digit 的候选/速度/offset 小矩阵，优先降低 middle 的冲量并提高持续接触。
- 2026-08-17 / v1.8.31：完成四指各自 256-env 物理候选搜索及对应固定果心复核：middle 候选206/0.10 rad/s 峰值约8.97 N、接触占空比27.4%；ring 候选163占空比48.8%、峰值5.17 N；little 候选234占空比69.5%、峰值10.01 N（超过8.25 N ceiling）；index 候选223占空比24.5%、峰值1.01 N。将候选57同时施加到四指且只保留四个 link4/proxy 碰撞时，仍只有 little 观察到18.75%接触，四指同时接触为0，过载31.25%。由此确认“独立候选并行”不能代表多指闭合；下一步转到运行时碰撞开关、逐指 sensor/link4 映射和碰撞体遮蔽审计，A-12 未通过，T0-B PPO 继续冻结。
- 2026-08-17 / v1.8.32：修正 `debug_collision_digit=all` 的过滤逻辑（仅保留五个 link4/proxy，不再误保留 link3/base），并在 `contact_closed_loop_probe.py` 加入固定果心、完整 14D 同步 reach、逐指 residual scale 和命令速率参数。配置 `proxy=8 mm/contactOffset=0.3 mm/restOffset=0.1 mm/root-span=0.40/scales=(1.0,0.7,1.0,1.0)/rate=0.10 rad/s` 在 seed 20261141/42/43、每次 256 env×700 step 下均得到：四指 measured distal 100%、四指同时 100%、掌托支持100%、果心漂移0、掉落0、过载0；逐指最大力约 0.85/9.50/5.12/5.03 N，均低于 12/9.75/9.75/8.25 N ceiling。该结果只完成 A-12a“固定果心接触窗口”，不能替代动态 retention；下一步按相同过滤和尺度进入解冻果实的 A-12b，继续禁止 PPO 合并。
- 2026-08-17 / v1.8.30：A-12 单指固定果心/掌托 smoke 完成。新增 `retention_single_digit_probe.py` 的物理 proxy 半径、动作速率和接触占空比记录，并将 debug 碰撞过滤收紧为“掌托 + 目标 digit 的 link4/proxy”，避免目标指的 link3 先遮蔽 link4。候选 57 在 8 mm proxy 下 index 32-env/260-step 最大力约 1.20 N、果心漂移约 0；middle 使用独立物理搜索候选 206，低速 0.10 rad/s 时接触占空比 27.4%、峰值 8.97 N（低于 9.75 N ceiling 但未达到 60% 保持门）；ring 32-env/200-step 占空比 26.0%、峰值 4.71 N；little 占空比 75.1%、峰值 5.05 N；middle 以 8 mm/常规速率会出现约 13.6 N 峰值并触发过载，证明“单一 proxy/速率/阈值”不可直接泛化。该阶段仍未通过 A-12：尚无四指均达到掉落≤10%、过载≤2%、接触保持≥60% 的 256×3 seed 证据，禁止合并 PPO；下一步先做按 digit 的候选/速度/offset 小矩阵，优先降低 middle 的冲量并提高持续接触。

- 2026-08-17 / v1.0.0：完成 v1 代码、训练指标和独立评估审计；确认课程等径开关、随机评估配置、果径观测饱和、三指阶段二奖励、远端位移代理和批量指标广播等高置信度问题。建立 P0–P7 改进流程，要求先修证据链和任务契约，再进行 v2 分阶段 PPO 训练。
- 2026-08-17 / v1.1.0：完成 P0 独立命名空间、配置 profile、v1 基线只读 SHA256 manifest 和 14D/128D v2 contract 初版；完成 P1 的 Band-1 等径 profile、命名配置解析、reset 前 terminal snapshot 和逐回合计数器。GPU 烟测发现当前 IsaacLab-3.0.0-beta2（Python 3.12/Isaac Sim 6.0 目标）与本机 Isaac Sim Python 3.10 环境不兼容（`enum.StrEnum`、`warp.types.array`）；在获得匹配运行时前不把该启动错误解释为算法结果，P1-06/P1-09/P2 暂停于运行时兼容性门。
- 2026-08-17 / v1.2.0：切换至旁侧只读 IsaacLab v2.3.2（匹配 Isaac Sim 4.5/Python 3.10），2/32/256 env 烟测通过。发现并修正 37.5 mm 等径 + 850 kg/m³ 与 v1 0.180 kg 上限互相矛盾（名义 profile 上限改为 0.250 kg）。E0–E7 首个 seed（256 env，至少 256 完成回合）已写入 `artifacts/enveloping_grasp_v2/baseline_recheck/p2_e*_seed20260820.json`：E0 成功 16.92%/掉落80.77%/过载2.31%；E1 尺寸 36.88%/61.60%/1.52%；E2 位置 41.02%/57.81%/1.17%；E3 形状 4.67%/89.49%/5.84%；E4 材料 3.50%/90.66%/5.84%；E5 传感器 4.69%/87.11%/8.20%；E6 执行器 3.08%/90.77%/6.15%；E7 压力域 1.95%/84.05%/14.01%。该证据将“形状/材料/传感器/执行器扩域后崩溃”从猜测提升为首个 matched seed 的直接事实，P2 仍需剩余 seeds 才可出最终置信区间。
- 2026-08-17 / v1.3.0：实现 v2 `env.py`、`train_v2.py`、`evaluate_v2.py` 和 reset-safe teacher 诊断。第一版 v2 T0（32 env、200 update、Band-1）独立评估成功 0%、五指终止接触 9.23%；加入弱教师混合/BC 后核心 37.5 mm 教师本身成功约 11.83%，说明大果径接触可达性和奖励稀疏仍是瓶颈。T0 核心训练在 update40 出现 NaN，作为数值不稳定失败样本封存；已加入 reward/return/value 限幅、非有限参数保护和低学习率续训路径，尚未宣称通过 T0 门槛。
- 2026-08-17 / v1.4.0：P2 E0–E7 六 seed 汇总完成（`p2_aggregate.json`）。E0 核心成功 15.33%（Wilson95% 13.62–17.21%）、掉落82.99%；E1 仅加尺寸成功40.72%（38.30–43.19%）；E2 加位置成功37.20%；E3 加椭球形状骤降至5.37%（4.35–6.60%）；E4 材料6.29%；E5 传感器7.09%；E6 执行器7.58%；E7 stress 1.94%，过载14.32%。因此后续课程固定先在 E1 等径尺寸带学习，形状/材料/传感器/执行器必须拆开训练，stress 只作压力测试；这已成为新的根因排序和扩域门槛。
- 2026-08-17 / v1.5.0：完成 P3/P4 第一轮实现并通过 6 项离线单测、2-env GPU smoke 和 79 个字段的 retention snapshot 恢复测试（误差 0）。v1-teacher-transfer T0（32 env、200 update、E1）独立评估 262 个完成回合：严格成功 0%、五指终止接触 0%、掌托支持 84.73%、过载 4.58%、超时 82.06%、实测几何 0%；link4 远端力为 0，说明旧 v1 行为先验与新触觉/大果径包络任务不匹配，不能升级为 T0 通过。该运行和之前 NaN 运行均保留为失败样本，下一步回退到 E1 可达性课程/动作映射修正后再训练。
- 2026-08-17 / v1.6.0：教师尺度探针发现 E1 中根部缩放 0.60、远端缩放 2.0 的小样本严格成功约 53.3%，37.5 mm 名义域根部 0.60/远端 0.50 约 58.3%；同时修复 teacher 分支隐藏状态和 phase 观测缺失。采用单分支 Markov teacher 的纯 BC/PPO T0（E1、60 update）独立成功 25.19%、掉落72.48%、过载2.33%，较 0% 迁移样本改善但仍未过 T0 门槛；link4 实测远端覆盖仍低，后续需把远端接触作为独立课程门槛。
- 2026-08-17 / v1.7.0：完成 near-surface/contact distal gate 与策略 `log_std` 限幅后的稳定 T0 诊断。`t0_band1_gate_ppo_stable`（E1、80 update、独立 261 回合）成功 43.68%、掉落54.02%、过载2.30%、五指终止接触43.30%、掌托支持45.21%、掌托先行99.23%；仍未达到 T0 的成功/过载门槛。对同一 checkpoint 的 full14/freeze-distal/roots-only 反事实成功率为40.31%/41.09%/14.84%，说明自由空间远端残差确有干扰，但冻结远端也不能解决远端物理接触。四指 link4 实测接触仅0–4.21%，必须先完成 scripted link3/link4 contact probe 和 URDF/碰撞体审计；在该闸门通过前暂停 T1/T2 和随机化扩域。详细根因与修订计划见 `PPO_TOMATO_GRASP_FAILURE_ANALYSIS_AND_REVISED_PLAN.md`。
- 2026-08-18 / v1.8.33：A-12a 三 seed 固定果心窗口复核完成：`proxy=8 mm/contactOffset=0.3 mm/restOffset=0.1 mm/root-span=0.40/scales=(1.0,0.7,1.0,1.0)/rate=0.10 rad/s/debug_collision_digit=all` 在 256 env×700 step 下四指 measured contact、四指同时、掌托支持均为100%，果心漂移/掉落/过载均为0；逐指峰值约0.85/9.50/5.12/5.03 N。该证据只通过固定果心入口闸门，未升级为动态抓握策略。
- 2026-08-18 / v1.8.34：A-12b 动态多指 screening（相同 proxy/过滤/尺度，256 env×700 step）观察为 index/middle/ring/little=0/75/93.75/100%，四指同时0%，掌托100%，掉落0%，过载0%，果心最大偏移约21.7 mm；进一步在 `retention_only` seed hold 中得到0/100/0/100%，四指同时0%。因此 A-12b 未通过，禁止合并 T0-B PPO；下一步不扩大随机化，改做从动态 seed 出发的逐指候选切换/力反馈序列，分别记录 index/ring 的到达余量和对其他手指载荷的反事实。
- 2026-08-18 / v1.8.35：新增 `dynamic_retention_sequence_probe.py`，从 A-12a scaled candidate57 动态 seed 出发按 index→middle→ring→little 切换逐指 root/distal 参考，并记录果心、掌托、逐指力和指尖距离。32-env smoke 的仅匹配通道切换为接触占空 0/0.44/0/41.0%，掉落/过载0、掌托100%；开启完整14D候选切换的反事实为0/0/91.7/72.6%，果心最大偏移36.5 mm但仍未掉落/过载。完整候选可以改变 ring/little，但不能恢复 index/middle，且全局通道会放大果心偏移；该路径仍只能作为诊断，不进入 PPO。
- 2026-08-18 / v1.8.36：A-12c 动态 seed signed-Jacobian 审计完成（29 env、14D ±0.18 与 ±0.60、120/300 step）。基线动态 seed 的 index/middle/ring/little 最小指尖距约49/43/59/44 mm；单独 index distal/root、ring distal 等扰动几乎不改变目标指尖距离，index/ring measured 接触仍为0；较大扰动仅在 `global_flexion−/curl_shape−/middle_root−/ring_root−` 等组合方向改变 middle/little 或短暂 ring 载荷，并伴随最高约5.66 N little 力。结论：不是统一增益/单一 distal 通道问题，而是动态 seed 下的多通道耦合与 link4 余量不足；A-12c 进入“组合基元/碰撞锚点反事实”阶段，继续禁止 PPO 合并。
- 2026-08-18 / v1.8.37：完成动态种子 512 候选运动学搜索及 candidate481/candidate205 固定-动态物理反事实。动态种子附近四指进入 39 mm 壳层比例 47.27%，candidate481 固定果实路径四指接触100%且安全，但解冻动态果实后 index/middle 接触变为0、ring/little为100%；candidate205 的 index-only 也由固定果实100%变为动态果实0%。这证明当前优先级是动态果心位移与接触时序/余量，而非继续增大 distal 增益或改变 PPO 奖励。下一阶段新增“动态接触锚定/过渡”只读诊断，任何锚定装置必须标为仿真反事实，不能写入默认环境或实机路径。
- 2026-08-18 / v1.8.38：修正 debug link4-only 过滤误禁用掌托 rim 子碰撞体的问题；修正后 candidate481 动态路径仍为 index/middle/ring/little=0/0/56.25/100%，掌托100%、掉落0%、过载0%，果心最大偏移约18.95 mm。尝试的运行时 kinematic→dynamic bootstrap 被 Isaac Sim 4.5 direct-GPU PhysX 拒绝，产生的极端位移 JSON 明确标记为无效，不进入任何指标。结论：掌托 rim 是有效的动态约束组件，但不足以恢复 index/middle；下一步用不切换刚体类型的动态碰撞/动作过渡反事实继续定位，A-12b 仍未通过。

- 2026-08-18 / v1.8.39：将开环随机候选搜索的排序从“曾经接触”改为显式保持窗指标：新增逐指 hold occupancy、四指同时 occupancy、最弱手指 occupancy，并同时保存 best-hold 与 best-safe 候选。mid-cup（0.035/0.030 m）、6 mm proxy、256 候选中，best-hold candidate245 的四指保持占用约59.5%/59.5%/59.5%/68.0%，但掌部峰值约9.03 N并触发旧安全包络；best-safe candidate27 的最弱指占用约35.0%、四指同时仅9.5%。该搜索仍是动作可达性诊断，不是策略成功率。
- 2026-08-18 / v1.8.40：掌缘几何消融表明将 cup 放宽到0.040/0.035 m并减小支撑 contact/rest offset 后，256候选 hold occupancy 几乎归零（四指同时最佳0%）；过宽掌缘不能提供可靠包络约束，mid-cup 是当前动态稳定性的必要候选区，但不能直接写入默认环境。
- 2026-08-18 / v1.8.41：新增 `contact_closed_loop_probe.py --static-hold/--approach-smoothing/--distal-filter-alpha/--assist-*` 诊断开关，用来分离“固定参考姿态可达性”和“低力反馈控制器调参”。candidate245在无 reach-assist、慢速平滑或 force-band 闭环下逐指 occupancy 约0.07–0.53且严格成功为0；启用与随机搜索一致的 proximal reach-assist 后，little occupancy可升至约0.58，但仍非四指持续保持，且无统一动态安全通过证据。reach-assist只是改善到达的仿真先验，不能代替动态果心跟踪；后续必须固定同一物理配置和随机种子/质量样本，做“开环候选→低速接触→中心/力反馈保持”的分层复核。
- 2026-08-18 / v1.8.42：为 `link4_action_search.py` 增加候选保持摘要并完成编译检查；所有新开关均保持 simulation-only，未改变旧 v1 目录、T0-A 封存权重或实机控制路径。A-12b/A-12c 仍未通过，暂停 PPO 合并与扩域训练。
- 2026-08-18 / v1.8.43：对 candidate245 做同配置 256-env 动态复核（mid-cup、6 mm proxy、assist=.45/.45/.30/.25、distal filter=.70、static hold）。结果为 index/middle/ring/little occupancy 约12.0%/11.1%/33.1%/50.8%，掉落0%、过载0%、掌托100%，严格成功0%；与开环搜索报告的59.5%最弱指保持率不一致。计划新增“相同 env 数/相同求解配置跨 seed”复核门，任何单次搜索候选不得直接转教师或 PPO 初始动作。
- 2026-08-18 / v1.8.44：完成候选搜索—复核可复现性审计并修正验收口径。原先比较将搜索的单候选环境和复核的 256-env 均值混在一起，且旧搜索 JSON 没保存 assist 语义与安全包络；现已把 resolved config、逐环境物性、hold occupancy 和指定 env 指标写入结果，并开放 `--assist-per-digit`、`--safe-force-n`、`--force-ceiling` 及茎部终止阈值。将 candidate245 固定复制到全部256环境并完全匹配搜索配置后，搜索与复核群体 hold occupancy 精确一致（index/middle/ring/little=7.656%/41.147%/56.626%/22.554%），物性一致（半轴37.5 mm、质量0.187759 kg、密度850 kg/m³、摩擦0.70、刚度18000 N/m）。结论从“批量/求解器不可复现”修正为“旧报告配置不完整且候选排序对 assist/安全包络敏感”；candidate245 不能作为可复现高保持基线。下一步转入带完整配置快照的 A-12 动态逐指承力门，仍不合并 PPO。
- 2026-08-18 / v1.8.45：宽安全包络下重新搜索得到 candidate85：开环四指 hold=100%/81.25%/95%/100%，四指同时76.25%，最大 link4 力约2.99/0.94/2.18/0.98 N；动态果实 static-hold 256-env×3 seed 逐指保持=99.38%/70.40%/90.59%/99.79%，四指同时62.63%，掉落/过载0%。
- 2026-08-18 / v1.8.46：完成逐指力带控制器改造与 nominal A-12b 复核。`contact_closed_loop_probe.py` 现在记录并执行 index/middle/ring/little 独立 lower/upper band；candidate85 使用 lower=0.10 N、upper=4.0/3.0/3.0/2.5 N、step/retreat=0.02，在动态果实256-env×160 hold、3 seed 下逐指 hold=99.11%/78.16%/95.08%/98.91%，四指同时73.16%，掉落0%、过载0%、掌托支持100%。A-12b nominal 动态 retention 通过；严格成功为0是预期的（诊断禁用摘梗），A-12d/T0-B PPO 仍未开始，形状/材料/执行器随机化保持冻结。
- 2026-08-18 / v1.8.47：完成 T0-B direct-ramp PPO120（candidate85、physical proxy 6 mm、32 env）。训练末段回报仍为正，但独立 actor-only 评估 270 个完成回合为成功0%、掉落16.30%、过载17.78%、掌托支持80.37%、virtual distal-all 60.74%，实测 distal-all 0%。结论：训练内回报主要由掌托/virtual contact 塑形，教师退出后未形成真实末端闭环；该 checkpoint 不升级。
- 2026-08-18 / v1.8.48：修复 `bc_only_updates` 与 teacher blend 的调度缺陷：BC 窗口内保持完整教师混合，窗口结束后才衰减，避免“名义 BC-only、实际逐步混入未成熟 PPO”的移动目标。10 项 v2/teacher 单测与 py_compile 通过。
- 2026-08-18 / v1.8.49：可观测 episode-clock direct-ramp PPO120 完成。独立 actor-only 256-env 评估：成功0%、掉落38.59%、过载2.89%、掌托支持61.09%、五指终止接触41.48%、virtual distal-all 41.16%，实测四指同时接触3.86%（逐指实测观察约25.72/23.15/27.01/15.11%）。该分支证明可观测教师时钟能避免行为克隆完全失配，但仍未达到 T0-B，不能进入随机化课程。
- 2026-08-18 / v1.8.50：在 v1.8.49 权重上冻结粗动作、加入 raw link4 shaping 的 T0-C PPO80 反事实，独立评估为成功0%、掉落48.30%、过载4.33%、掌托支持51.08%、virtual distal-all 53.25%，实测四指同时3.10%；冻结粗动作并未解决末端承力，且损害保持安全，checkpoint 不升级。
- 2026-08-18 / v1.8.51：尝试 simulation-only force-band teacher（lower=0.10 N，upper=4/3/3/2.5 N）。首次实现因掌托瞬时丢失时整手回开导致训练回报仅约79–167，已加入“首次掌托支持锁存”后仍未形成可信 PPO 候选；该力带教师保留为诊断代码，不进入默认训练。
- 2026-08-18 / v1.8.52：审计 actor 触觉信息后短暂构建 133D/182D raw-distal contract。fresh 120-update direct-ramp 独立评估仍为成功0%、掉落6.20%、掌托支持93.41%，但实测 distal-all 0%，表明短预算下网络尚未拟合完整闭合轨迹；200-update BC smoke 在约60 update 已回落至45.8回报，停止扩时。为保持已验证 128D v2 结果可复现，当前主线 contract 已回滚至 128D/177D，133D 产物仅作失败实验封存。
- 2026-08-18 / v1.8.53：完成主线回滚后的回归检查：128D/177D contract、旧 candidate85 checkpoint 读取路径、v1 投影和 10 项单测/py_compile 全部通过。当前主线候选为 `t0b_candidate85_observablebc_ppo120/best.pt`（仅 nominal、simulation-only），A-12d、T0-B 严格成功门、T1–T5 和扩域随机化仍未通过；不向实机发送任何命令。
- 2026-08-18 / v1.8.54：根据 A-12a/A-12b 的同配置固定—动态证据，将 A-12d 标记通过：6 mm physical proxy、0.3/0.1 mm offset、mid-cup 和 per-digit reach-assist 的仿真接触映射可在固定果心与动态 seed 两闸门下保持安全；该结论不等于 PPO 策略通过，也不改变 URDF/硬件。
- 2026-08-18 / v1.8.55：A-13 首轮 `t0h_safeband_measured_ppo80`（candidate85 observable-BC 权重续训、32 env、80 update、physical proxy 6 mm、粗动作上限0.4、渐进160步、动作平滑0.7、低权重 measured distal shaping）完成。训练内回报约340–835且无数值发散，但独立 actor-only 名义评估揭示 checkpoint 选择和策略目标失配：`best.pt` 为 update5，256-env/317 回合成功0%、掉落92.11%、过载7.89%、掌托支持6.62%、virtual distal-all99.05%、实测 distal-all4.73%；`latest.pt` 为 update80，310回合成功0%、掉落97.74%、过载2.26%、掌托支持1.94%、实测 distal-all7.42%。两者均不通过 T0-A/B。根因收敛为：physical proxy/virtual contact 使“指尖接近”奖励很高，但当前 PPO 退出教师后没有保持掌托和低力承力；按 rollout return 选 best 会选出接近代理接触而非在手保持的 checkpoint，且训练 contact offset=2 mm 与 A-12d 力带诊断的0.3/0.1 mm配置不一致。下一轮只在名义域采用候选85作为弱动作流形、匹配0.3/0.1 mm接触几何、保留力带/掌托锁存的观测条件，并把独立保持指标纳入选模；不扩大随机化、不改 URDF/硬件。
- 2026-08-18 / v1.8.56：A-13 首轮复盘后发现 v2 的 `_v2_measured_link4_force` 仍硬编码读取废弃的 `thumb_contact_link4`，而 v2 真实拇指传感器/五指 proxy 均在 `thumb_contact`（thumb_link5）。修正为 v2-only `thumb_contact` 映射，保留 v1/旧 checkpoint 不变；8 项 v2 单测与 py_compile 通过。修正后的 candidate307 局部动作搜索出现五指 proxy 接触，四指 hold occupancy 约65.0/98.8/87.5/100%，拇指闭环探针最大实测约0.33 N；此前拇指 0 N 的结论判定为读取错误，不再作为物理不可达证据。
- 2026-08-18 / v1.8.57：在匹配 A-12d 的 6 mm proxy、0.3/0.1 mm offsets、mid-cup、逐指 assist 配置下，完成 `t0k_thumb_corrected_curr160_seed20261264`（96 env、100 update、禁用 virtual distal、候选先验+force-band teacher、160°课程门槛）。独立 actor-only 严格 200°评估 258 回合：成功0%、掉落0%、过载37.98%、掌托支持100%、五指0%、四指 measured distal-all 35.66%，说明训练安全保持不稳且拇指仍未进入策略闭环；该权重封存为失败样本，不升级主线。
- 2026-08-18 / v1.8.58：围绕 candidate307 固定四指动作做拇指通道局部搜索。1024-env 搜索发现 all-five proxy contact 分支（如 candidate194）及最大包络角 204.15°分支（candidate907），但尚无同一候选同时满足 all-five 与 ≥200°且有四指持续保持；扩大到全通道局部搜索后 candidate817 达到五指接触、四指 hold 94.99%、包络角173.31°，candidate35 达到包络角221.30°但缺少五指同时接触。结论是当前 14D manifold 的“拇指对向/几何覆盖”和“四指持续承力”存在窄交集，下一步将以双目标候选筛选（all-five + coverage + hold）构造更强 BC prior，再进行短程 PPO；不放宽最终 200°门槛、不改 URDF/硬件。
- 2026-08-18 / v1.8.59：以 all-five + hold 优先的 thumb-only 局部搜索（candidate817 邻域，1024 env）找到 candidate211：五指 proxy 接触、四指 hold occupancy 83.8%/83.8%/83.8%/83.8%、四指同时83.8%、最大掌部力约8.94 N、包络角175.3°，无过载/掉落；同批次最高覆盖198.85°仍缺少五指。该候选作为新的弱动作流形 prior，训练课程门槛暂设170°，最终独立验收仍使用200°。
- 2026-08-18 / v1.8.60：`t0l_thumb211_curr170_seed20261271`（96 env、100 update、强 BC/force-band、candidate211、170°课程）训练内前70 update 回报约2.0–2.24/step，但 teacher 退出后降至0.30–0.58/step。独立 actor-only strict-200°评估 289 回合：成功0%、掉落1.04%、过载23.88%、掌托支持98.96%、五指0%、实测四指同时0.69%，仅环指形成持续载荷。该结果确认交接失配仍是主因；candidate211 只作为教师诊断通过，t0l 权重不升级。下一轮延长 teacher→actor 混合交接并增加 teacher-exit retention checkpoint，不按单一 rollout return 选模。
- 2026-08-18 / v1.8.61：A-13 长交接实验 `t0m_thumb211_longhandoff_seed20261273` 在 update181 因 CUDA OOM 中止；复核发现其 actor/critic 权重最大绝对值均为0，仅 log-std 有限。根因是旧训练器在 loss/梯度非有限时用 `nan_to_num(...)=0` 静默清零参数，teacher-active rollout 因而掩盖了部署策略已失效。t0m/t0n 保留为故障样本，不作为候选。
- 2026-08-18 / v1.8.62：训练器已改为非有限 loss/梯度跳过更新、恢复上一步参数、清理异常 Adam 状态，并记录 `skipped_optimizer_steps`，禁止静默清零。8 项 v2 单测、py_compile 及 8-env/3-update GPU smoke 通过；smoke 的 skipped steps=0，actor/critic 参数均有限且非零。下一步从零重跑 candidate211 长交接，降低批量/显存峰值，重新以 actor-only strict-200°评估选模，仍不扩域或触及硬件。
- 2026-08-18 / v1.8.63：`t0o_thumb211_guarded_handoff_seed20261277`（48 env、180 update）完成且无数值跳过；但 best/latest 独立 strict-200°评估成功0%，拇指实测力0，过载84.1/86.4%，仍为环指局部解。BC 探针虽把四指实测 distal-all 提升至约61–67%，仍不能复制拇指闭合。
- 2026-08-18 / v1.8.64：修正 `link4_action_search.py` 的五指候选闸门：此前 `all_five_contact_seen` 使用 aggregate `contact_forces`，可能混入非严格通道；现在必须使用 v2 corrected measured link4 vector（含 thumb_contact），同时保留 aggregate 诊断。与训练完全匹配的6 mm proxy/0.3–0.1 mm几何局部搜索，measured all-five count=0；此前 proxy5 的 candidate211/245 “all-five”证据不能迁移到当前几何。
- 2026-08-18 / v1.8.65：运行时碰撞审计确认 thumb_link5 proxy 已被 authored 且 collision enabled，但 candidate245 长速率路径下 thumb_link5 与掌心果实的有效接触始终为0；四指 measured 接触可达。根因收敛为拇指链/代理几何与动作路径不一致，而非 PPO 数值问题。新增 thumb 距离记录，下一步先做 thumb-only 几何/动作校准（不改 URDF、不宣称实机），通过 measured thumb contact gate 后再恢复 PPO。
- 2026-08-19 / v1.8.66：完成“搜索器—拇指审计”配置一致性修复。审计此前没有复现 `proxy_only_collision`、root-span、接触 offset、distal filter 和逐指 assist，且保留了未标定供应商碰撞体，造成同一候选“搜索可达、审计不可达”的假矛盾。`thumb_geometry_audit.py`、`train_v2.py`、`evaluate_v2.py` 现在均支持并记录 `proxy_only_collision` 及匹配几何参数；8 项 v2 单测通过。匹配配置下 candidate210 的拇指和四个直指均产生非零 measured link4 载荷；256-env 固定动作门四个直指 any-contact 为100%，最佳保持四指最低占空比约96.7%，无过载。该结果仅通过“仿真代理接触可观测性诊断”，不等于 PPO/T0 或实机标定通过；`proxy_only_collision` 仍是临时 simulation-only 隔离先验。下一步执行 P2-05 stem-break 事件窗口与 P2-06 3/6/10/20 步 pre-pick hold 对比，再按同一 resolved config 恢复 T0。
- 2026-08-19 / v1.8.67：新增 P2 事件窗口探针并完成边界测试。micro smoke 可落盘；动态 no-gravity candidate210 仍 `pick_started=0`、过载100%、中心最大偏移54.2 mm；kinematic→dynamic bootstrap 被 Isaac Sim direct-GPU 禁止。32/8-env 四 dwell 逐步记录规模超过10分钟无结果，已安全终止并记为 CPU 物理记录成本风险。P2-05/P2-06 因尚未能从动态初始状态进入安全五指保持而暂不通过。T0 8-env/5-update PPO smoke 无数值跳过且 checkpoint 落盘，但独立评估 measured distal-all=0%、严格成功0%，仅作接口回归；正式训练必须显式固定6 mm proxy并复核 resolved config。下一步优先寻找不切换刚体类型的动态初始稳定动作课程，再恢复正式 T0。
- 2026-08-19 / v1.8.68：候选动作复核先发现两项配置/接口问题：闭环探针默认 4 mm 且未暴露 `proxy_only_collision`，导致与搜索器不一致；已增加该开关并通过 py_compile。用完全匹配的 6 mm、0.3/0.1 mm offset、40×35 mm cup、root-span 0.4、逐指 assist、线性 ramp 复核 candidate277 后，256-env 五指实测接触率仅 79.69%、四指保持占空比约63.2/48.3/65.1/41.8%、过载7.03%；该结果否决 candidate277 作为直接先验。随后在 candidate277 邻域进行 1024 条动态局部搜索，得到 candidate288：单个搜索环境四指最低保持71.25%、四指同时45.63%、all-five=true、最大掌力2.58 N、无过载/出界。固定 candidate288、相同线性 ramp 的 256-env 复核得到五指实测接触98.44%、掌托100%、掉落0%、过载1.56%，保持占空比93.71/69.12/97.24/47.05%、四指同时30.77%；两个额外 nominal seed 的结果完全一致，说明当前 T0 配置无位置/物性随机化，不能冒充多种子鲁棒性。candidate288 只升级为 simulation-only 弱动作流形，不升级 T0/PPO。
- 2026-08-19 / v1.8.69：为 P2 探针复现 candidate288，增加同样的 cup 参数并修正接近起点为真实 open-action（此前全零动作会引入额外路径差异）。8-env 双 dwell 运行超过约5分钟无落盘后安全终止；降为 1-env、120+80 steps、3-step dwell 的 dynamic-no-gravity 最小证据成功落盘：pick_started=0、stem-break=0、overload=100%、中心最大偏移28.77 mm。结论：即使 candidate288 在 stage-1 static-hold 通过接触观察门，未经训练的 stage-2 动态事件路径仍不能安全进入摘取；P2-05/P2-06 继续不通过。下一步在正式 T0 之前增加“动态预接触→低冲量锁存→短保持”课程，并将 P2 事件记录降采样为单环境/短窗，不再运行高成本批量 ContactSensor 轨迹。
- 2026-08-19 / v1.8.70：完成 candidate288 的拇指/根部定向搜索（1024 条、只扰动动作0–4、noise=0.30、其余四指固定）。最佳 hold candidate297 的五指实测接触、四指最低保持和无过载仍稳定，但最大覆盖仅133.8°；最佳覆盖候选约176.1°且丢失中/环/小指保持，最佳 all-five coverage 候选约166.7°且四指同时仅0.6%。该矩阵未找到“≥200°覆盖 + 五指持续 + 安全”的交集，确认当前动作几何存在拇指覆盖瓶颈；不降低最终200°门槛，后续改为在动态预接触课程中学习拇指对向/覆盖，而不是继续静态放大拇指动作。
- 2026-08-19 / v1.8.71：训练/评估器新增并统一 `assist-per-digit`、`distal-filter-alpha` 解析与 checkpoint 快照字段，修复“搜索器、训练器、评估器隐式配置不同”的审计风险；v2 测试套件保持 `8 passed`，未连接实机。
- 2026-08-19 / v1.8.72：`t0p_candidate288_dynamiccourse_smoke`（25 update）和 `t0q_candidate288_bconly`（60 update）均无数值跳过，但 actor-only 评估在 palm-first gate 下 measured distal-all=0、严格成功=0；训练回报不能作为策略转移证据，两个 checkpoint 封存为失败 warm-start。
- 2026-08-19 / v1.8.73：`t0r_candidate288_bc_nogate`（40 update）去掉隐藏掌托门后，四指 measured contact observed=0.641/0.656/0.375/0.188，较前两次可观察改善，但拇指 measured contact=0、严格成功=0；该实验因命令路径错误落在外部 `/home/miobarnacle/ppo_tomato_grasp_stage01_work`，保留为外部证据，发布前必须复制到项目审计目录。
- 2026-08-19 / v1.8.74：`t0s_candidate288_bclong` 从 t0r 续训至80 update，`skipped_optimizer_steps=0`，actor 输出的粗/拇指幅度增大，但 matched Band-1 actor-only 仍为 success=0、timeout=1、measured distal-all=0.0625、measured index/middle/ring/little observed=0.156/0.422/0.484/0.625、拇指=0；在线 BC 交接仍不能复制 candidate288 的实测拇指承力。
- 2026-08-19 / v1.8.75：根因收敛为“在线 rollout 中教师标签受私有 support latch、力反馈 trim 和物理偏差影响，部署 actor 只看到有限观测，BC 目标不具备可重复的条件映射”。新增下一道硬闸：先用同一 V2 环境保存 observation/action 监督数据，再进行独立离线行为克隆，之后才允许恢复 PPO；离线 BC 若不能提升实测逐指接触，不进入 P2 摘梗/扩域。
- 2026-08-19 / v1.8.76：新增 `offline_bc_v2.py`，将 teacher trajectory 的 128D observation 与14D action 保存为独立监督数据，再拟合 checkpoint-compatible actor；8项v2测试与 py_compile 通过。离线 BC 数据集包含23040条样本，最终加权动作拟合误差约0.00396，训练本身无数值跳过。
- 2026-08-19 / v1.8.77：第一次 BC actor-only 评估漏传显式6 mm/0.3–0.1 mm几何参数，报告实际为8 mm/2–0 mm，已标记为配置无效样本，不参与比较。完全匹配 Band-1 复评得到 success=0、drop=8.33%、overload=15.15%、timeout=76.52%、measured distal-all=43.18%，逐指 observed=0.534/0.754/0.678/0.598；相较 t0s 的6.25% all-four 有明显改善，但拇指/五指严格门仍未通过。
- 2026-08-19 / v1.8.78：同一 BC actor 在确定性37.5 mm nominal 复评得到 success=0、drop=5.66%、overload=27.17%、timeout=67.17%、measured distal-all=72.83%，逐指 observed=0.925/1.000/0.887/0.728；这证明离线监督能把四指 link4 观测转移到 actor，但过载和拇指五指同时接触仍是独立瓶颈。下一步先做低冲量/拇指优先的 teacher 数据再训练，暂不恢复 PPO 或 P2 摘梗。
- 2026-08-19 / v1.8.79：修复离线采集器的 episode timeout：采集240步时将 collection-only `episode_length_s` 延长到250 policy steps，避免第200步自动 reset 后把尾部标签重新写成张开动作；`t0y_offline_bc_thumbfirst_norestart_20260819/teacher_dataset.pt` 的步级审计显示第0步保持 open、80步拇指过渡、239步仍为完整 prior，确认数据连续。
- 2026-08-19 / v1.8.80：t0y 拇指先行 BC（thumb-first 80步、128×240、30 epochs）独立 nominal actor-only 仍 success=0、five_digit=0、measured thumb peak=0、measured distal-all=21.19%、overload=40.40%；t0z 将拇指窗口缩短为40步并提高 thumb loss 权重12后，measured distal-all=36.27%，但 thumb peak仍为0、overload=48.04%。标签连续性已修复，但单一全局 actor 仍存在明显 covariate shift。
- 2026-08-19 / v1.8.81：增加 DAgger-style collection（行为策略与 teacher 按0.35混合执行），t0aa actor-only nominal 结果为 palm=100%、drop=0、overload=0.39%、measured distal-all=0.39%、five_digit=0，反而退化为不闭合；记录为“简单混合不能修复行为分布偏移”的失败样本。
- 2026-08-19 / v1.8.82：增加评估器只读的 thumb/action-prefix 反事实覆盖。t0u 强制 thumb opposition+residual 后 measured thumb peak仅约0.008N、five_digit=2.29%；替换 candidate297 前6D后仍接近0，替换全14D但180步同步 ramp也只产生低接触，说明拇指必须和根部、四指远端插入顺序耦合，不能按独立通道修复。所有 override 仅为诊断，不写入权重。
- 2026-08-19 / v1.8.83：完成 candidate297 prior PPO curriculum（32 env×60 update，120°课程门，candidate297、6mm proxy、0.3/0.1mm offsets、无 virtual distal），训练无 skipped optimizer steps；独立 strict-200° actor-only `t0ab_best_eval_strict200_20260819.json` success=0、five_digit=0、palm=99.61%、drop=0.39%、overload=0.78%、measured distal-all=1.95%，教师退出后策略回到低动作局部解。随后实现 sequential-fingers 离线标签（食指→中指→无名指→小指逐指 ramp），t0ac actor-only measured distal-all=4.25%、thumb=0、timeout=93.05%，确认单纯顺序标签仍不足。当前 PPO/P2/随机化门均保持未通过，下一步改为显式“阶段状态机 + residual policy/teacher 保留”的结构，而非继续增加 BC epochs。
- 2026-08-19 / v1.8.84：新增 `residual_controller.py` 与 residual PPO 接口；控制器含同步掌托/拇指/根部接近、逐指插入、低力带保持和 reset 状态清理，10项 v2 单测通过。审计发现独立 evaluator 与已验证 `contact_closed_loop_probe.py` 在相同候选/几何下结果不一致：probe 16-env nominal 为 measured distal contact=100%、掌托=100%、overload=0、thumb peak≈0.56N；residual evaluator 仍出现 overload latch、strict success=0。已修复 evaluator/train 的残差 episode 长度（≥24s）、命令速率显式化和控制器 reset，但首个分叉尚未定位；T0、P2、随机化和发布门继续冻结，下一步必须做逐 policy-step 的动作/力/episode-reset 对齐审计。
- 2026-08-19 / v1.8.85：完成 residual 路径对齐审计。第一处差异为探针使用 40×35 mm 掌托内缘而 evaluator 默认 55×45 mm；统一后动作序列可复现。第二处差异为 evaluator 严格统计自动终止，而旧 probe 在 `env.step()` 后读取已 reset 状态，漏掉了 severe-overload/drop 终止，不能把 probe 的 overload=0 当作安全证据。随后修正 residual 控制器：粗接近阶段不提前累计各指接触 streak/freeze；默认改为 palm/root/thumb-first（`simultaneous_reach=false`），默认窗口改为 approach=240、per-digit=60。10项单测通过。严格 evaluator 复核：`residual_base297_staged_nominal_16_20260819.json` 在 16-env、620 steps、6 mm proxy、0.3/0.1 mm offset、40×35 mm cup 下 `drop=0、overload=0、palm=1.0、measured distal-all=0.75、palm_first=1.0`；延长每指窗口未进一步改善小指覆盖，小指仍是下一个可观测 reach/residual 优化对象。T0/P2/随机化仍未通过，不启动大规模 PPO。
- 2026-08-19 / v1.8.86：残差 PPO smoke `t0ae_residual_staged_smoke_20260819`（10 update，16 env，skipped optimizer=0）在未限制 phase 的 actor-only 复评为 drop=25%、measured distal-all=0.50；不能作为候选。新增 `residual-mask=thumb_distal` 和默认 approach 后 phase gate 后，旧 actor 的只读回放恢复为 drop=0、overload=0、palm=1.0、measured distal-all=0.75，证明安全信赖域约束有效但尚未产生增益。随后启动训练时带 mask+phase gate 的新 smoke（`t0af_residual_masked_smoke_20260819`），训练链路稳定，尚待独立评估；在此之前不扩大 PPO 或随机化。
- 2026-08-19 / v1.8.87：完成小指弱动作流形与严格峰值诊断。将 candidate297 的小指根部/末端目标调整为约0.25/0.65后，16-env两个名义 seed 的 residual replay 均为掌托100%、四指 measured distal-all=100%、drop=0、overload=0；但64-env复核出现约5.9%掉落/7.4%过载，故不将其视为跨并行度鲁棒性结论。terminal snapshot 新增每回合峰值掌力/指力；原目标64-env峰值掌力最大约20.9 N，定位到大批量 PhysX 初始/支撑冲击。5 mm 接触偏置导致100%过载、20步预稳定导致约22%掉落，均否决。以修正小指目标完成 t0ah 60-update residual PPO（skipped=0）；两个独立16-env回放的四指指标曾达100%，但五指终端接触为0，新增的 `five_digit_observed_rate` 仅反映瞬时拇指参与且重复回放出现掉落/过载，t0ah 不升级。当前继续停在 T0 动态稳定/拇指持续承力门，不进入P2摘梗、扩域或实机。
- 2026-08-19 / v1.8.88：修复动作前缀诊断的回合截断问题：`--action-prefix-override` 在 ramp=240 时不再停在旧的240-step边界，而是自动保留 ramp+100 steps 的保持窗口，并记录 `episode_length_s/max_episode_length`。修正后按 v1_nominal_reliable、6 mm physical proxy、proxy-only、40×35 mm掌托、0.3/0.1 mm offsets 复现 candidate210，结果仍为 index/middle measured link4=0、ring/little约99.2%/48.3% hold occupancy、all-four=0、all-five=false；历史 search 高接触候选不能作为 strict v2 可复现先验。当前下一道硬闸改为先统一 `teacher_trace` 与 action adapter/回合长度，再训练 residual/DAgger；不启动长 PPO、P2摘梗或随机化扩域。
- 2026-08-19 / v1.8.89：将 `offline_bc_v2.py` 的 teacher trace 扩展为可审计接口，记录 teacher label、实际执行动作、episode step、五通道 measured link4 力、掌托力，以及完整 resolved trace config 与回合长度。`t0trace_smoke_20260819`（2 env×20 steps）验证样本形状和元数据完整，10项v2测试保持通过；该 trace 仅通过软件契约门，尚不代表五指抓取成功。后续只允许基于同一 trace 做小样本 BC/DAgger 对照并用实测五指/掉落/过载/峰值掌力筛选，不启动长 PPO或摘梗。
- 2026-08-19 / v1.8.90：完成 physical teacher_trace 烟测 `t0trace_physical_smoke_20260819`（1 env×20 steps，6 mm physical proxy、proxy-only、0.3/0.1 mm offsets、40×35 mm cup、virtual distal关闭、strict measured开启），解析配置与数据契约一致；早期20步 measured link4峰值全0、掌托峰值约1.84 N，仅说明采集路径正常，不作为抓取性能证据。完整闭环 trace 仍需通过 strict five-digit/drop/overload 门后才允许 BC/DAgger。
- 2026-08-19 / v1.8.91：完成残差 evaluator 与闭环 probe 的逐 policy-step 对齐。发现 evaluator 使用 `--proxy-only-collision` 时未同时启用 physical fingertip proxy，实际是“关闭供应商碰撞体且没有替代指尖碰撞”的无效场景；已修正为 proxy-only 自动启用 physical proxy，并在 resolved config 中记录。新增 action/力/tip-distance/joint-target trace 与 object-state快照；同一6 mm/0.3–0.1 mm/40×35 mm/1 mm reset nominal配置下，`residual_base297_physical_proxy_fix_16_20260819.json` 真实观察到四指 link4，逐指 observed=[1.0,1.0,1.0,0.40]，但 drop=0.20、overload=0.20、five-digit terminal=0，T0仍不通过。后续固定physical proxy和strict快照，优先调低冲量、力带、逐指窗口和小指保持；不进入随机化、摘梗或硬件。
- 2026-08-19 / v1.8.92：保持窗口与远端门控局部矩阵完成。默认 candidate297 延长保持到160步仍 drop/overload=0、四指实测 observed=1.0，但终端五指仅0.125；小指增强0.25/0.65在同窗口 overload=0.50、峰值掌力25.44 N，排除。步级 trace 确认环指 tip约58 mm而默认near-surface gate=45 mm，叠加 `distal_filter_alpha=0.999` 使环指远端命令被清零/严重滞后。新增四指独立滤波配置 `--distal-filter-alpha-per-digit`，并同步到训练器；11项v2测试通过。局部候选 gate=65 mm、插入步进0.02、滤波[0.999,0.999,0.98,0.999]在16-env得到终端五指=0.25、drop=0、overload=0.25、峰值掌力最大19.24 N；只通过可观测改进门，不通过T0安全门。全局filter=0.90/0.98均分别产生过载60%/掉落75%，否决。下一步只优化环指局部释放与低力回退，不上调安全阈值，不启动PPO长训、随机化、P2或实机。
- 2026-08-19 / v1.8.93：完成环指独立插入/回退参数和严格几何诊断。`ResidualBaseControllerCfg` 新增逐指 `insertion_step_fraction_per_digit`/`retreat_fraction_per_digit`，evaluator/train/report 同步，标量滤波 CLI 值域与实现统一为 `(0,1)`；12项v2测试与编译检查通过。candidate297 在 near-surface=65 mm、滤波[0.999,0.999,0.98,0.999]、环指插入0.01/回退0.35/上力带0.25下，16-env 得到 drop=0、overload=0、掌托=1.0、四指 measured observed=0.75、终端五指=0.25，峰值掌力最大约8.48 N；这是安全局部基线但仍不满足 T0。新增 strict_hold/coverage 指标后，1-env trace 显示 strict_hold_max=0、coverage≈133.45°，严格门槛为200°；finger-spread ±1 未改善（约127.6°/129.9°，负向还触发过载）。历史开环 candidate183（独立覆盖约180.5°）接入状态机后 coverage≈97.2°、measured distal-all=0，否定直接复用静态搜索动作。结论：主瓶颈已从“远端是否释放”收敛为“拇指—四指周向覆盖的受约束动态协同”，下一阶段只能在 candidate297 安全状态机外学习小 residual/课程，并同时受 coverage、strict_hold、drop、overload、峰值掌力门约束；不启动长 PPO、随机化扩域、P2摘梗或实机。
- 2026-08-19 / v1.8.94：实现 `coverage` residual mask（开放 finger-spread、thumb opposition/residual、四指根部/远端，保护全局 flexion/curl），以 `residual_scale=0.05` 做短训。3-update/8-env smoke 无 skipped optimizer；初始 checkpoint 独立回放覆盖≈149.32°、drop/overload=0、四指 measured=1.0、峰值掌力最大≈10.69 N，仍 strict_hold=0。低学习率续训12 update 后，独立回放保持 drop/overload=0、measured distal-all=1.0、终端五指=0.50、峰值掌力最大≈2.68 N，但覆盖回落≈133.71°且 strict_hold=0。两份 checkpoint 均标记为安全诊断/失败 warm-start，不替换主线；该结果确认 rollout return 上升不代表包络成功。下一阶段需要显式“拇指先对向覆盖→四指低力承接→strict hold”的状态课程/可审计 teacher trace，再考虑更长 residual PPO；在 T0 未通过前继续禁止随机化扩域、P2摘梗和实机。
- 2026-08-19 / v1.8.95：coverage mask 扩展加入 radial/ulnar balance（action 2–13，仍保护 global flexion/curl 0–1）做3-update反事实。`t0_coverage_balance_smoke_20260819` 无数值跳过；独立8-env回放 `diagnostics/t0_coverage_balance_smoke_eval_8_20260819.json` 保持 drop=0、overload=0、measured distal-all=1.0、five-digit observed=1.0、峰值掌力最大≈2.61 N，但 coverage≈132.99°、strict_hold=0。该通道未改善几何，保留为接口反事实，不升级主线；下一阶段停止盲目扩展 residual 子空间，转为显式 thumb-first coverage teacher/状态课程与可达几何锚点审计。T0、随机化、P2和实机门继续冻结。
- 2026-08-19 / v1.8.96：完成历史最高 coverage candidate409 接入当前状态机的边界对照。统一 physical proxy/offset/cup/near-surface/逐指低力回退后，1-env×660 trace 仅形成 thumb contact，measured distal-all=0、coverage=0、strict_hold=0，虽 drop/overload=0。该动作不能作为当前状态机先验，确认开环搜索的高 coverage 依赖不同动作时序/动力学。静态目标替换路线至此封存；后续必须实现显式 thumb-first coverage teacher/阶段状态课程，标签包含阶段状态、实测力、指尖几何与 strict_hold，并用独立 evaluator 逐次交接验证。T0、随机化、P2和实机继续冻结。

- 2026-08-19 / v1.8.97：实现 `ResidualBaseController` 的可选拇指双向低力探测（`thumb_search_amplitude`/`thumb_search_period_steps`），只在拇指低于力带时按有限周期翻转有界偏移；新增4项控制器单元测试，py_compile与测试全部通过。candidate123、45 mm、6 mm physical proxy、proxy-only、40×35 mm掌托的16-env回放中，探测幅度0.15–0.50使五指瞬时 observed 达到0.875–1.0、覆盖约230°、drop/overload均为0，但严格保持最多5–6步、`five_digit_rate=0`，不能升级为成功策略。
- 2026-08-19 / v1.8.98：增加可选 `thumb_first_contact_gate_steps`。门控开启时，拇指未连续进入自身力带前不释放四指插入/保持更新，避免四指先把掌托果实推离拇指对向区域；默认值0保持历史行为。8-env gate=5诊断仍为掌托/四指 measured=1、drop/overload=0、覆盖≈231°，严格保持最多8步，说明时序方向正确但拇指接触仍不够持久。
- 2026-08-19 / v1.8.99：新增的门控状态首次训练烟测在第1个更新后发现 inference tensor 复位错误，已将控制器 reset 纳入 inference-safe 路径；随后 `t1_thumbsearch_gate_residual_retry_20260819` 完成80 update/40,960 rollout steps，`skipped_optimizer_steps=0`、无NaN。独立16-env回放 `diagnostics/eval_t1_thumbsearch_gate_residual_16_20260819.json` 为 success=0、drop=0、overload=0、palm=1、five_digit_observed=1、严格保持均值4步（最大9步）、覆盖≈231.1°；训练回报稳定不等价于严格包络成功，该权重只封存为失败 warm-start，不进入随机化/P2/实机。
- 2026-08-19 / v1.8.100：对照显示当前安全基线的主要缺口仍是“拇指接触持续性”，不是四指远端或安全阈值：candidate123基础状态机与PPO residual=0.05在同一几何下均保持drop/overload=0和四指实测接触；残差训练没有带来严格保持增益。下一阶段暂停继续扩大PPO更新，转向“拇指接触可达性/方向 + 掌托果心运动”的可观测几何课程，要求每次候选同时记录拇指 tip distance、低力持续窗、果心漂移和严格保持；未通过严格10步保持前继续冻结随机化、P2摘梗和硬件。
- 2026-08-19 / v1.8.101：将拇指接触门控扩展到粗根部更新（门控未满足时保留上一帧粗根部动作，而非只暂停四指插入）。同一8-env/700-step物理回放结果与门控前一致（palm/measured distal=1、drop/overload=0、coverage≈231.3°、strict hold最多8步），说明当前瓶颈不是单纯的根部继续推进；该修改保留为安全课程选项，不作为性能增益结论。v2相关测试总计19项全部通过。
