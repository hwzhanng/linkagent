# 小果抓取 B 阶段唯一工作基线：无 PPO 的真实动力学抓取与撤支撑保持

基线编号：`small_fruit_phase_b_v1.1`。编制日期：2026-09-22；修订日期：2026-09-28。状态：**B′ 执行中；仅授权推进至 T4 正式门完成后暂停**。原 v1.0 执行及 `BLOCKED_B` 归档保留原身份，不能追溯改判。

唯一规范文件：`/home/miobarnacle/linkbot/linkagent/.omo/plans/small-fruit-phase-b-baseline-20260922.md`。

本文件供其他对话独立执行，不要求读取原对话。后续用户指令和项目硬件安全规则优先；B 阶段的范围、顺序、门槛和交付以本文件为准。旧路线图、旧报告、代码中的历史注释只作为参考，不能自动恢复旧的训练任务、放宽本文件门槛或扩大阶段范围。同目录之外的草稿、代理报告和执行日志均不是第二份基线。

## TL;DR (For humans)
**交付目标：** 得到一条可重复的、由真实执行器连续执行的三指抓取教师轨迹：拇指预对向，食中指协同闭合，取得合格接触，撤去果梗支撑，再完成保持及扰动验证。交付轨迹、完整失败记录、可复现实验包及明确的阶段结论。

**核心选择：** 先用独立关节目标控制验证几何和动力学是否可行，将“机器人做不到”和“14D 动作流形表达不了”分开。教师可以使用仿真真实状态和接触反馈；这不代表 PPO 已学习成功，也不代表当前策略能在原初始分布中成功。

**范围：** 固定基座、仅左手、67 mm 球形果实为主验收场景；64/70 mm 和小位移仅作邻域诊断。本阶段没有训练、模仿学习、动作维度扩张、自然断梗学习或真机运行。

**工作量：Large。风险：High。** 当前没有经 A 合同验收的物理正例，可能需要在固定预算内给出有证据的阻塞结论，不能保证通过增加搜索量就一定成功。

**量化目标：** 主验收 24/24 回合通过 A 的最终保持门及 B 的物理真实性门；完成 56 回合邻域诊断、16 回合物理反例和 8 回合批次一致性复核。其余门槛见下文，不允许用平均回报替代。

执行对话收到明确“执行本基线”的指令后开始。通过 B 后立即停止，等待用户审阅；若预算耗尽仍无正例，则交付 `BLOCKED_B` 并停止，不得宣称 B 验收通过。

---

> 执行摘要：无 PPO、真实连续关节控制、A 合同不降低、正反例验证、完整归档；PASS_B 或有证据的 BLOCKED_B 后停止，不自动进入下一阶段。

## Scope
### Must have

### 1. 起点与已知事实

项目根目录 `ROOT=/home/miobarnacle/linkbot/linkagent`，它是唯一 Git 项目；父目录 `/home/miobarnacle/linkbot` 不是项目仓库。本文未特别说明的相对路径均相对于 ROOT。

| 项目 | 已核实的起点 |
|---|---|
| A 报告 | `docs/tomato_pick/small_fruit_pinch_v1/PHASE_A_REVIEW_20260922.md` |
| A 证据目录 | `research/small_fruit_pinch_v1/evidence/20260922_phase_a/` |
| A 运行清单 | 上述目录下 `eval_model150_frozen/run_manifest.json` |
| A 运行身份 | `e28e1a6d3dc0663b4942976e4e314e4b7c9ecd96867ccc9d4befb44620a186c4` |
| A 冻结归档 | `frozen_run.tar.gz`，SHA-256 `be9c767404a772fc40b367bb2b25972bce317c4f8c49df82306b7f81ef280ac4` |
| A 验证 | 86 项相关测试通过；两环境物理事件接线通过；旧 model_150 两回合接触获取/最终保持均为 0 |
| A 限制 | 物理事件 smoke 显式注入过阶段标志，不是自主抓取正例；独立审查因额度中断，未给出最终通过结论 |
| 仓库记录 | 编制时分支 `improve/tomato-ppo`，HEAD `65b3e17653d295119dd70d9e6dd7cb38b606db34`；工作树存在大量未提交改动，HEAD 不等于实验完整源码 |
| 已知全仓问题 | A 全仓回归 394 passed、58 failed、89 subtests passed，失败清单在 `pytest_full.xml/log`；B 不负责修复无关采摘流程和历史文件迁移 |

历史 model_150 的 99% 左右成功率属于旧验收合同。旧的 6273 个“静态有效候选”、v7/v8 教师、CEM 零损失、运动学预演均不能直接升级为 B 正例。它们可用作搜索提案，必须从干净 reset 连续重放重新验收。

### 2. 目标分解

1. 统一关节、刚体、碰撞壳、末端标记和果心的坐标/索引，确认目标真正写入预期关节。
2. 验证三指末端接触是真实碰撞，排除掌托、近端受力、自碰撞、外部支撑和运行时状态重写。
3. 建立与 Actor 解耦的连续关节教师入口，保留真实驱动、果梗解除、扰动和 A Verifier。
4. 在固定预算内找到并冻结一个教师和合法预抓取起点，完成独立评估及反例。
5. 交付是否需要后续修改 14D 流形的定量证据；B 不直接修改策略维度。

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 不启动任何 PPO 更新、actor-init 训练、行为克隆或新 `.pt` 策略导出；教师保存在 JSON/轨迹数据中，不包装成“成功策略权重”。
- 不修改 A 的软/硬力阈值、几何阈值、保持时长、扰动时序或张量维度来换取通过；发现 A 实现缺陷可加反例修复，但必须另存运行身份并重新验证，不覆盖 A 归档。涉及成功语义变化时停止并向用户报告。
- 不在正式轨迹开始后写关节位置/速度状态，不锁果心，不逐步重置果实位姿或速度，不关闭重力，不施加承重补偿、隐形吸附、果实运动钳制或反馈重定位。初始化、原果梗物理模型和 A 的指定扰动写入必须分别留痕，不能扩展为任意状态修正。
- 不关闭正式回合的 done/reset 语义以拼接成功窗口；不把前一回合状态、无效传感器或人工阶段标志当成正例证据。
- 不启用正式场景的几何接触代理或 `base_link↔Tomato` 碰撞过滤；不换成无名指/小指承载来规避必需三指门。
- 不通过增大摩擦、减轻果实、加大执行器上限、降低指垫区域门、调高果梗支撑或改变积分步长制造成功；允许修复有证据的映射/实现错误，不允许任意参数优化绕过这些约束。
- 不扩展到 52–80 mm、椭球泛化、臂手接近、自然断梗控制、运输放置、真实 CAN/网关/机械臂。无实机、无凭据、无外部服务消费。
- 不改 upstream Isaac Lab、不改只读树莓派快照或父目录旧 SDK 副本；不覆盖用户已有改动，不自动创建提交，不推送、不部署。

### 3. 冻结物理及成功合同

主验收沿用 A 的 `small_fruit_pinch_v1.2.0` / `support_retention_v1`，外加 B 的真实性审计门 `phase_b_physical_audit_v1`。B 的附加门只能拒绝不可靠结果，不能放松 A。

| 条件 | B 的绑定值/规则 |
|---|---|
| 场景 | 1 个固定基座左手 articulation，21 个手部关节，16 个独立目标通道及原 mimic 关系；无臂、无右手 |
| 接口 | Actor 114、Critic 175、动作 14 不变；教师绕过 Actor 输出生成，直接提供 canonical 16 关节目标，不冒称 14D 策略动作 |
| 时钟 | PhysX `dt=1/120 s`，decimation 6，控制 20 Hz，正式单回合最多 320 个控制步 |
| 重力 | `[0,0,-9.81] m/s²`；果实必须为动态刚体 |
| 材料 | A 果实 static/dynamic friction `0.70/0.58`，restitution `0.01`；其余材料、质量/密度、惯量和接触参数以 A 清单及所选几何解析值冻结，不可为通过而调整 |
| 执行器 | 沿用 A 的方向 PD 与 mimic；独立关节 stiffness/damping `20/1`，followers `1/0.5`，effort limit `0.2 N·m`、velocity limit `1.2 rad/s`；仍受各关节实际配置更严格限制 |
| 力软带 | 拇指 `[0.20,1.80] N`、食指 `[0.15,1.50] N`、中指 `[0.12,1.30] N`，合格保持窗内逐帧满足 |
| 力硬上限 | 拇/食/中/无名/小指 `[4.0,3.5,3.5,3.0,3.0] N`，任一原始力越限使回合失败 |
| 掌面/禁触 | 掌面受力 `<0.08 N`；原禁触力通道均 `<0.02 N`；B 正例整段轨迹均不得依靠这些部位承重 |
| 其他承重 | 除原果梗在释放前的支撑外，果实只允许必需三指承载；果实与地面、额外刚体或其他手部区域的接触必须监测，不能以其支撑完成保持；任何未声明的接触支撑使正例失败 |
| 指垫区域 | **当前 A 实际代码 `fingertip_axial_progress_min=0.55`**，不是旧文档的 0.68；定义、局部坐标和碰撞壳需在任务 3 实测确认，不得进一步调低。不能靠平均接触点掩盖同一刚体上的近端接触 |
| 几何 | 对向角 ≤35°，夹持线段偏心/等效半径 ≤0.45，食中支撑跨度/等效半径 ≥0.40，拇食 OBB 间隙 ≥2 mm |
| 稳定 | 果心相对本回合固定参考位置漂移 ≤12 mm，相对手速度 ≤0.035 m/s；不得在回合中移动参考位置 |
| 时序 | 连续合格接触至少 80 步；解除果梗后连续合格保持至少 120 步；第 40 个脱支撑保持步后施扰，实际施扰确认后至少 40 步恢复 |
| 扰动 | `Δv=0.122625 m/s`，`J=mΔv`，世界水平 `+X/-X/+Y/-Y`；实际写入且下一物理步确认。失稳重开保持窗时必须重新施扰，旧确认不得复用 |
| 成功名称 | 主门为 `support_retained`；默认 `audit_release`，不要求自然断梗；`harvest_success` 不作为 B 通过依据 |

`self_collision_proxy_terminates=False` 是 A 当前配置事实，**不是已经证明无自碰撞**。B 必须补足正式轨迹的真实自碰撞/穿透证据，详见任务 3；不能只读取该代理的 False 值就判安全。

### 4. 初始化与教师的授权边界

允许创建独立的 `phase_b_local_feasibility` 场景，在 reset 时一次性设置果心与无接触预抓取姿态。该场景用于验证局部可行性，**不同于 A 默认掌前 55–75 mm 初始距离分布**；必须同时保存差异表，不能将局部可行性的结果推广为原分布成功率。

初始条件必须满足：真实碰撞表面最小初始间隙 ≥2 mm、五指原始果实接触力均 `<0.04 N`、无非相邻手部碰撞/穿透、无关节越限。预抓取可让拇指已部分对向，但初始姿态不得已经夹住果实。果实可由原果梗支撑，支撑参数不提高。有效回合起点、initial q、果心、姿态、质量、几何及所有初始化状态写入必须记录。

开始计时后只允许目标/努力控制、原果梗计算和 A 扰动。A 接触获取后由环境主动撤支撑，教师不得手动设置 `success`、`support_hold_age` 或 `disturbance_applied`。即使无果实状态写入，也要核对撤支撑后 `_last_stem_force_w/_last_stem_torque_w` 为零、外部支撑力为零。

教师为确定性有限状态控制器，可读取仿真 q/qdot、三指原始/有效力和接触几何；该特权状态是“物理可行性教师”的权限，不能被描述成可部署策略。主要阶段为预对向、拇指轻触、食中闭合、三指稳定、撤支撑保持。用 `_postprocess_joint_targets` 等窄接口写 canonical 目标，经过原 `_apply_action` 和执行器，不照搬旧探针的锁果心或关闭 done 覆盖。

目标速度上限：接近阶段 `min(0.5 rad/s, 实际关节限速)`；获取稳定接触后 `min(0.15 rad/s, 实际限速)`；目标加速度上限分别 `2.0/0.7 rad/s²`。按物理子步插值，不能只在记录中显示平滑而实际一次跳变。力软带超限时停止继续闭合并有界回撤；硬失败立即终止，不重置违规计数继续拼接。

阶段建议预算：预对向 ≤40 步、拇指轻触 ≤20 步、食中闭合 ≤40 步、连续合格接触 ≥80 步，争取在第 180 步前获得接触；剩余约 140 步用于撤支撑和最终保持。第 180 步仍未获取接触时将候选标为 `acquisition_late`，允许记录到回合结束但不能加长正式回合。可以有更长的诊断回放，结果必须单列并不能通过 B。

## Verification strategy
所有门由执行者运行脚本、读取原始输出和逐回合证据进行判定；用户最后审阅交付，不承担补跑测试的义务。视频用于审阅动作和碰撞，不能代替数字门。

测试策略：坐标/索引、事件计数、状态写入约束及教师状态机采用失败反例先行的 pytest 回归；物理能力通过实际 PhysX 回放验证。不要为了只证明“文件存在”堆叠单元测试，也不能用 mocks 代替物理正例。

### 5. 评估矩阵与分母

先冻结 `teacher.json`、场景集合、源码与哈希，再进入正式评估。仅报告该冻结候选的完整结果；一旦调整控制器或选点，旧验证全部作废，换 run_id 重跑对应完整集合。验证种子不得用于搜索和调参。

| 集合 | 固定安排 | 数量与门槛 |
|---|---|---|
| 搜索 | 67 mm 球；seed `20260922/20260923/20260924`；合法果心/无接触 q0/控制关键点联合搜索 | 只选候选，不计验收成功率 |
| 主验收 `nominal` | seed `20261001/20261002/20261003` × 4 扰动方向 × 2 初始化版本 | **24/24** 完成最终保持且 B 真实性门通过 |
| 初始化版本 | r0 为冻结名义场景；r1 用对应 seed 的独立 RNG 对果心各轴加 `U[-0.5,0.5] mm`、独立初始关节加 `U[-0.005,0.005] rad` | 采样结果预写场景集合；不合法初始化算失败，不丢弃重采到通过 |
| 邻域 `neighborhood` | seed `20261004`；直径 `64/70 mm` × 果心偏移 `{0,±2 mm X,±2 mm Y,±2 mm Z}` × 4 扰动方向 | 56 回合全部报告；不设置成功率硬门，不能据此宣称尺寸泛化完成 |
| 物理反例 `negative` | 缺拇指、保持果梗支撑、不执行扰动、关闭果实与必需指碰撞，各 ×4 方向；seed `20261005` | 16 回合，原始 Verifier 最终成功必须全为 0，且观察到预期反例条件 |
| 批次一致性 `batchcheck` | 选主验收 seed20261001/r0 的4个方向；分别逐一执行与4环境并行执行 | 8 回合全通过；使用相同场景ID/方向，不由临时 env_id 改变扰动方向 |

主验收包含零失败要求是本次制定的**局部重复性工程门槛**，不是对仿真统计显著性、真实果实安全或泛化能力的声明。质量/密度按冻结配置处理，不人为减重；所有实际质量写入各场景记录。邻域只改变尺寸/初始偏移，教师、摩擦、控制增益和其他配置保持不变。

上表为第一次正式候选的种子。若在总搜索预算内启用第二次正式候选，必须新建run_id，并将上表所有验证seed统一加10，即主验收 `20261011/20261012/20261013`、邻域 `20261014`、反例 `20261015`，其余集合构造不变；第一候选的全部失败保留。第二组种子预先登记但不得在第二候选冻结前运行，不再允许第三次正式候选。

正式执行每个场景只计第一个完整回合。某环境先结束后不得反复 reset 占据其他场景的配额；使用暂停/独立批次，或记录并排除明确标记的额外回合。超时、初始化失败、物理硬失败、NaN 均进入预注册分母。仅初始化前基础设施崩溃可以按原 scene_id 原参数重试一次，两次日志都保留；失败不能从分母中消失。

batchcheck 比较物理事件时间差 ≤2 个控制步、同阶段原始力 P95 差 ≤0.10 N、果心终点差 ≤2 mm。超过时先调查共享状态/环境间干扰，不能将并行结果作为可信证据；不要求 CUDA 浮点逐位相同。

物理反例必须真实触发条件并同时读取原始 Verifier，不能只在汇总层将 `diagnostic_only` 强行变为失败来证明反例有效。负例“保持果梗”需在专用诊断配置阻止自然断裂且确认支撑仍在；“不执行扰动”保留 due 请求但不调用 writer、不伪造 ack；负例不属于正例物理配置。单位测试继续覆盖软带失效、陈旧扰动、非法形状、非有限观测及跨环境 reset，所有相关 A 反例必须继续通过。

### 6. 证据格式

统一输出根：`research/small_fruit_pinch_v1/evidence/phase_b/<run_id>/`，run_id 采用 UTC 时间加候选编号，禁止覆盖已有目录。每个运行同时保存本文 SHA-256；执行进度写单独日志，不直接勾改这份规范。

必须交付：

```text
entry_state.json                 # Git状态、A身份校验、运行环境、当前源文件差异
baseline_diff.json               # B与A解析配置/场景/源码的逐项差异及理由
scenarios/search.json
scenarios/frozen_validation.json # 全部104个正式scene_id、种子、方向和初始化值
01_preflight/                    # 依赖、A反例回归、物理运行入口
02_frames_joints/                # 命名映射、坐标回归、真实单轴跟踪
03_contacts/                    # 接触区域/传感器负例、自碰撞与穿透审计
04_teacher/                     # 控制状态机与完整入口检查
05_search/                      # 每批候选及失败记录，包含未选中样本
teacher.json                    # 不含Actor权重的冻结教师
06_nominal/ 07_neighborhood/ 08_negative/ 09_batchcheck/
run_manifest.json               # 聚合索引；每个物理运行另有自己的A式清单
episodes.jsonl                  # 一行一个预注册场景的最终记录
traces/                         # 每回合20Hz完整数据及事件附近120Hz原始数据
videos/                         # 主验收成功与典型失败，可标注叠加图
tests/                          # 命令、退出码、pytest XML、编译输出
action_manifold_fit.json         # 教师关节轨迹到既有14D动作的可表达性分析
acceptance.json                 # 各门布尔结果及PASS_B/BLOCKED_B/INVALID_EVIDENCE
PHASE_B_REPORT.md
frozen_run.tar.gz + SHA256SUMS
```

逐回合至少记录：scene_id/重复编号、三层版本号和本文哈希、run_manifest哈希、seed、果实三半轴/质量/实际惯量、初始q/qdot及果心姿态、模式/反例类型、接触获取/撤支撑/扰动/终止物理步号、扰动方向与实际冲量、所有A审计标志、B附加门、终止原因、最大原始力、禁触/掌面/无名小指峰值、最小自碰撞/穿透裕量、各保持计数。

时序至少记录：canonical目标和实测q/qdot/effort、五指原始力及区域过滤后的力、果实过滤的平均接触位置和有效掩码、可取得的原始接触点及数量、接触区域进度、六路近端禁触力、掌面力、果心/速度、果梗力矩与状态、外部施力、对向角/偏心/跨度/OBB间隙、教师阶段、全部状态写入事件。缺失数值用 null+原因，不能用0冒充“无力/无误差”。

每个物理子步维护峰值/硬失败监控；保留事件前后至少0.25秒的120Hz数据，防止20Hz采样漏掉接触冲量或短暂禁触。A Verifier仍按20Hz执行，B补充门以子步异常否决本回合。传感器容量溢出、空过滤矩阵或在明确有力时接触点缺失均判证据无效。

## Execution strategy
### Parallel execution waves
按依赖执行，不为并行而重复启动仿真。推荐主执行者负责运行入口/教师，坐标与接触审计可由两个独立代理分别负责；全局最多4个子代理，只有一个代理拥有同一源文件写权限。最多一个 `lazycodex-qa-executor` 或同类 economy QA，均为叶子角色；GPU同时只跑一个PhysX进程。

代理额度不足时由主执行者接手，不把“代理没有返回”视为通过；逐项保存证据即可继续授权范围内的工作。不要再为此次计划启动新用户对话，执行只在用户指定的 B 对话进行。

本机基准为 RTX 4060 Laptop 8 GB；静态批次默认128，连续动力学搜索批次最多16，正式评估默认4。OOM只能减少并行环境数，必须保留相同场景清单、候选数和随机种子，不能暗改物理参数。

### 7. 固定搜索预算和失败分流

- 第一层：3个搜索seed，各最多32批×128个静态提案，总计≤12288。允许在独立提案环境瞬时设置姿态做FK/几何筛选，但这些结果全部标 `proposal_only`，不能贡献最终成功。
- 初始候选以真实左手基座FK定义；果心中心参考取 A reset 后实际果心转换到 `base_link` 坐标的值，搜索范围为其各轴±40 mm；允许从旧候选生成提案，但中心仍须在这个冻结范围内。各独立关节搜索域为 A canonical safe limits 与实际 USD joint limits 的交集；不得使用手写 body 序号或 marker 中心冒充碰撞壳表面。
- 第二层：3个搜索seed，各最多10代×16个连续轨迹候选，总计≤480个完整动力学候选；每个从合法无接触reset开始、最多320控制步。使用确定性 CEM，elite fraction=0.125，位置/阶段关键点及有界闭合增量为优化变量，不优化摩擦、质量、力阈值或执行器上限。
- 评分按约束优先：无证据/有硬违规的候选全部淘汰；随后按最长三指有效接触窗、最小几何违规量、脱支撑保持长度排序；原始回报仅作诊断。任何一次静态“零损失”都不能跳过第二层。
- 本基线只允许最多2个不同冻结教师进入完整正式评估；第一候选失败后可用剩余搜索预算改进并重新冻结，第二候选必须用第5节预登记的第二组验证seed。用过的验证数据只能作为已见失败样本，不再作为改进后候选的独立验证；第二次仍不通过即停止。报告分别列出两个候选的完整分母，不能合并挑选成功回合。
- 当12,288个提案或480个连续候选用尽、或必要真实性条件无法满足时，输出 `BLOCKED_B`。这表示执行到达明确阻塞、**不是 B 通过**，不自动加预算、降门槛或转入训练。

失败归因只能由证据支持：`mapping_error`、`unreachable_in_declared_search_domain`、`tracking_failure`、`forbidden_first_contact`、`palm_support`、`self_collision`、`force_out_of_band`、`hard_overload`、`support_loss`、`disturbance_recovery_failure`、`sensor_invalid`、`acquisition_late`、`timeout`。固定预算搜索失败只证明“在本域/预算内未找到”，不等于全局无可行解。

### 8. 运行命令合同

以下前两条是当前已有命令；`phase_b.cli` 及 `test_phase_b_*.py` 是本计划要求实现的**新入口**，编制时尚不存在，不得直接声称已执行通过。任务1创建统一CLI；每一子命令启动前校验输出目录、参数、scene_id及禁止项，不能静默回退为代理/运动学模式。

```bash
cd /home/miobarnacle/linkbot/linkagent
export LINKAGENT_ISAACLAB_ROOT=/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1
export PYTHONPATH=/home/miobarnacle/linkbot/linkagent
export PYTHONDONTWRITEBYTECODE=1
export B_PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
export B_RUN="research/small_fruit_pinch_v1/evidence/phase_b/<本次唯一run_id>"

"$B_PY" -m pytest -q research/small_fruit_pinch_v1/source/tests
"$B_PY" -B -m research.small_fruit_pinch_v1.source.native_physx_smoke --num-envs 2 --device cuda:0 --verification-events --evidence-json "$B_RUN/01_preflight/physx_events.json"

"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli preflight --run-root "$B_RUN"
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli audit --suite frames-joints --run-root "$B_RUN" --device cuda:0 --num-envs 2
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli audit --suite contacts --run-root "$B_RUN" --device cuda:0 --num-envs 2
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli search --budget-profile B1 --run-root "$B_RUN" --device cuda:0 --num-envs 16
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli freeze-teacher --run-root "$B_RUN"
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli replay --suite nominal --teacher "$B_RUN/teacher.json" --scenarios "$B_RUN/scenarios/frozen_validation.json" --run-root "$B_RUN" --device cuda:0 --num-envs 4
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli replay --suite neighborhood --teacher "$B_RUN/teacher.json" --scenarios "$B_RUN/scenarios/frozen_validation.json" --run-root "$B_RUN" --device cuda:0 --num-envs 4
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli replay --suite negative --teacher "$B_RUN/teacher.json" --scenarios "$B_RUN/scenarios/frozen_validation.json" --run-root "$B_RUN" --device cuda:0 --num-envs 4
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli replay --suite batchcheck --teacher "$B_RUN/teacher.json" --scenarios "$B_RUN/scenarios/frozen_validation.json" --run-root "$B_RUN" --device cuda:0 --num-envs 4
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli fit-manifold --run-root "$B_RUN"
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli summarize --run-root "$B_RUN"
"$B_PY" -m research.small_fruit_pinch_v1.source.phase_b.cli verify-bundle --run-root "$B_RUN"
```

将 `<本次唯一run_id>` 替换为真实标识后使用；不要复制占位符作为目录名。`batchcheck` 子命令内部依次运行四次1环境和一次4环境。每个已完成子命令重复执行时应明确拒绝覆盖或验证原产物后只读返回；重跑需要新attempt目录。

`summarize` 必须从轨迹/episode与清单重算门，不接受调用者传入 `success=true`；缺件返回 `INVALID_EVIDENCE`。预期物理负例可正常完成命令但结果必须是失败；基础设施异常返回非零。Kit 的退出码可能掩盖异常，因此验收同时检查 traceback、实际回合数、结果JSON及证据哈希。

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 入口与冻结 | 无 | 2/3/4 | 无 |
| 2 坐标与关节 | 1 | 5 | 3/4的CPU实现；GPU审计串行 |
| 3 接触与碰撞 | 1 | 5 | 2/4的CPU实现；GPU审计串行 |
| 4 教师入口 | 1 | 5 | 2/3，保持文件所有权不重叠 |
| 5 候选搜索 | 2/3/4 | 6 | 无 |
| 6 主验收 | 5 | 7/8 | 无 |
| 7 邻域、负例、批次 | 6 | 9 | 8的纯CPU分析 |
| 8 14D表达分析 | 6 | 9 | 7 |
| 9 归档与结论 | 7/8 | F1–F4 | 无 |

## Todos
每一项包含实现与验证。默认不提交 Git；实际进度记入本次证据目录，保持本文规范不变。以下以 `SRC=research/small_fruit_pinch_v1/source` 表示路径前缀。

- [ ] 1. 保存接手状态，建立 B 独立入口和场景/结果合同
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/{__init__,contracts,cli}.py` 和 `SRC/tests/test_phase_b_contract.py`；不改A归档。实现第8节CLI、不可覆盖输出、scene_id和结果schema，记录执行环境、本文哈希、所有工作树文件差异及B/A配置差异。
  - 依赖：无；解锁2/3/4。
  - 参考：`SRC/run_manifest.py`、`native_run_manifest.py`、`contract.py`、A报告与实际清单。先逐文件校验A归档，不要求当前工作树等于Git HEAD；若当前已被其他任务修改，记录差异并隔离运行，禁止自动reset恢复。
  - 验收：A相关测试零新增失败；运行preflight和现有2环境smoke；scene_id重复、输出覆盖、非法数值、正式模式代理/过滤/状态注入选项均在启动PhysX前拒绝。产出 `entry_state.json`、`baseline_diff.json`、`01_preflight/`。
  - QA：运行 `"$B_PY" -m pytest -q research/small_fruit_pinch_v1/source/tests/test_phase_b_contract.py`，然后第8节preflight/smoke。正例有效配置可解析；反例向已存在证据目录写入必须失败且原文件哈希不变。
  - Commit: N。

- [ ] 2. 审计统一坐标和 canonical→实际执行器通路
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/geometry.py`、`SRC/tests/test_phase_b_coordinates.py`；在CLI的frames-joints审计复用真实articulation状态。必要时最小修复现有名称映射，但不重复实现第二套动作流形。
  - 依赖：1；解锁5。
  - 参考：`SRC/native_env.py::configure_native_module_for_left/_pre_physics_step/_apply_action`、`pinch_action_manifold.py`、`vendor/hand_manifold.py`、`single_target_env_probe.py`。旧probe只参考，不默认继承其直接状态写入。
  - 工作：导出16个canonical名称与21实际关节/mimic映射；分别列出末端marker、受力壳和碰撞体索引，禁止列表局部索引冒充articulation id。统一 `world↔env_origin↔base_link↔grasp_basis` 转换，明确四元数排序与单位。用实际body transform和碰撞壳计算距离，记录转换后的同一果心。
  - 验收：CPU往返位置误差≤1e-6 m；同场景CPU变换与实际USD/PhysX位姿差≤0.1 mm、姿态差≤0.1°；环境origin平移不改变相对几何。对每个独立关节执行空载±0.10rad目标（距边界不足则取可用较小幅度并记录），每轴每方向100控制步，前60步有界移动、后40步保持，末20步跟踪误差P95≤0.05rad、最大≤0.10rad，无符号反转或错误轴；mimic符合冻结关系。其余独立轴保持原目标，只有既定mimic可跟随；该审计的果实在reset时置于无接触位置，不属于正例场景。不可观测幅度不算通过。
  - QA：运行 `test_phase_b_coordinates.py` 和第8节 `audit --suite frames-joints`。故意置换两个关节名字、混用mm/m、加入env_origin两次的反例必须报错；证据 `02_frames_joints/` 含每轴实测轨迹，不能只比较目标数组。
  - Commit: N。

- [ ] 3. 验证末端接触、传感器完整性及正式轨迹碰撞约束
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/contact_audit.py` 和相关测试；必要时对 `native_env.py` 补充只读诊断，或对项目内B专用USD派生物启用真实自碰撞。所有资产变化另存新路径/hash，原A资产不覆盖。
  - 依赖：1；解锁5。
  - 参考：`SRC/native_env.py::_physical_finger_force/_forbidden_proximal_force/_self_collision_proxy`、`verification_geometry.py`、`contact_binding_probe.py`、`assets/left_hand_urdf/`（相对于 `research/small_fruit_pinch_v1/`）。
  - 工作：逐指用已知无接触、远端壳接触、近端接触测试位置和区域门；明确平均接触点的含义。多接触点混合时若不能排除近端接触，必须标无效；不能仅因平均位置在指垫内就认证末端接触。记录原始接触数/传感器容量，无法读取容量状态时用可控密集接触场景验证无截断，并记录方法与限制。
  - 碰撞门：对非相邻指链、指链与掌面启用并检查实际接触；固定/mimic邻接排除名单必须按具名关节关系导出，不能排除拇食、拇中或指掌以获取通过。A代理保留但不作为完整证据。实际碰撞或穿透检测不能取得时B阻塞。几何粗筛可用OBB，最终不得把保守包围盒相交直接当实际穿透、也不得把未检测当无穿透。
  - 验收：有力（≥0.08N）的必需指样本中有限接触点比例≥99.5%；用于成功窗口的样本有效率100%；原始力非有限/空矩阵/截断零容忍。关闭对应物理碰撞后有效接触和最终成功均为0。远端正例可通过区域门，近端及混合近端反例不可通过。正式正例无非允许自接触，分离距离若数值低于-0.1mm判穿透失败；该容差不能用于放宽拇食2mm门。
  - 若掌部碰撞壳与参考网格明显失真：只可依据本项目可追溯网格/URDF生成修正派生资产并保存几何对照，不凭视觉猜尺寸、不关闭碰撞；无可追溯修正依据则 `BLOCKED_B`。几何修正后重新跑2/3和物理回归。
  - QA：相关pytest及第8节 `audit --suite contacts`，证据 `03_contacts/` 包含正反接触图、具名碰撞对、原始力和点集。不同环境同时接触不得串线。
  - Commit: N。

- [ ] 4. 实现连续关节教师与运行时真实性审计
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/{runtime,teacher}.py`、`SRC/tests/test_phase_b_teacher.py`。只使用具名16关节目标和第4节有限状态控制；原Actor不运行，不生成PPO经验缓冲。
  - 依赖：1；正式运行需要2/3通过；解锁5。
  - 参考：`SRC/native_env.py::_postprocess_joint_targets/_apply_action/_prepare_verification_support/_apply_verification_impulse/_get_dones/_reset_idx`、`verifier.py`、`simulation/tomato_pick/isaac/env.py::_apply_attachment_wrench`。`direct_joint_trajectory_probe.py` 当前锁果心并关闭done，禁止直接作为正式父类。
  - 工作：对关节/果实状态写入和额外wrench增加来源标记与运行期审计，白名单只含reset、原果梗模型及A扰动；教师只能写目标/允许努力。撤支撑后果梗力矩为零。记录原始终止快照后再reset，各回合状态独立。
  - 验收：目标速度/加速度不超上限，实际位置方向正确；软带越界停止闭合并有界回撤；硬失败终止；正式轨迹无白名单外的状态写入/支撑力；两环境交错终止时快照和保持计数不串回合。环指/小指保持开位，在正例中力均<0.05N，不辅助承载。
  - QA：`test_phase_b_teacher.py` 验证正常阶段推进、缺证据不推进、超力回撤、提前施扰拒绝、违规writer被捕获；再以2环境实际运行检查输出。人为注入阶段标志仅可作为测试反例，必须标注并导致正式证据不合格。证据 `04_teacher/`。
  - Commit: N。

- [ ] 5. 有预算地搜索并冻结动态候选
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/search.py`，读取第7节预算和只含搜索seed的场景；输出完整候选及失败账本。可复用几何/CEM数值代码，不复用锁果心、运动学teacher和关闭终止的运行模式。
  - 依赖：2/3/4全部通过；解锁6。
  - 参考：`physical_contact_cem.py`、`dynamic_contact_cem.py`、`opposition_surface_cem.py`；这些脚本只是提案算法参考。候选坐标统一由任务2计算。
  - 工作：按静态提案→连续执行两层筛选；q0/果心只在reset写入，后续闭合动作必须真实执行。教师参数/状态机、q0/中心、材料和动力学均保存。冻结前至少在3个搜索seed各出现一次完整 A 保持成功且无 B 违规；不满足则不进入正式集。
  - 验收：冻结 `teacher.json` 可在新进程加载；包含具名目标/关键点/阶段最大步数/反馈增益/限幅及来源hash；`checkpoint_loaded=false`。`freeze-teacher` 同时写出全部104个验证scene_id，读取文件就能还原输入，不依赖进程随机状态。预算没有超出、没有只保留top-k成功记录。
  - QA：第8节search/freeze-teacher；错误关节名、越限teacher和缺少阶段参数均拒绝加载。无可行候选时输出带最佳失败轨迹的 `BLOCKED_B`，不得伪造空教师继续跑正式评估。
  - Commit: N。

- [ ] 6. 执行24回合主验收
  - Recommended task executor category: unspecified-high
  - 范围：冻结后运行 nominal；不在此阶段改教师、门槛、初始化样本或碰撞过滤。每个run使用 `freeze_native_run` 等A清单能力，扩展记录teacher和scenario文件哈希。
  - 依赖：5；通过后解锁7/8。
  - 验收：24个唯一scene_id与预注册集合一致；24/24均在320步内真实达到最终保持，计入合格接触/保持窗口的每帧满足A门，整个回合满足子步B硬门；接触尚未建立的接近阶段不要求满足三指软带。自然断梗字段与主动释放不混淆。保持窗中原果梗力矩为0、无其他承重源。保留全部24回合20Hz数据，至少保存4方向各一段完整视频。
  - QA：第8节 `replay --suite nominal`；从落盘trace重新运行独立Verifier计算成功并与环境输出逐回合比较。缺一回合、状态写入违规、窗口接缝或人工标志均不通过。失败按第7节预算分流，不能删除失败场景。
  - 证据：`06_nominal/`、`episodes.jsonl`及`traces/videos`。
  - Commit: N。

- [ ] 7. 完成邻域、物理反例和批次一致性复核
  - Recommended task executor category: unspecified-high
  - 范围：完全复用冻结教师，依次运行 neighborhood、negative、batchcheck，不并发启动多个GPU仿真进程。
  - 依赖：6；解锁9；可与8的CPU分析并行。
  - 验收：56邻域回合全部完成并逐尺寸/方向/偏移报告；失败不影响名义场景存在性结论，但任何邻域硬异常必须记录，不能扩域宣传。16物理负例均为0成功且对应破坏条件确实发生；8批次回合满足第5节一致性门。
  - QA：第8节对应三个replay命令；重算分母及反例原因；故意重复scene_id或只传一个成功视频给汇总器时必须拒绝验收。证据 `07_neighborhood/08_negative/09_batchcheck/`。
  - Commit: N。

- [ ] 8. 分析教师轨迹能否由现有14D动作表达
  - Recommended task executor category: unspecified-high
  - 范围：新增 `SRC/phase_b/manifold_fit.py`，纯离线分析，调用当前 `pinch_action_to_joint_targets`，不训练Actor、不改变动作/观测维度。
  - 依赖：6；解锁9。
  - 工作：从成功轨迹提取关键阶段及每5个控制步的canonical目标，在 `a∈[-1,1]^14` 内拟合流形；每样本8个确定性初值、最多200次优化迭代，报告逐关节目标残差。该优化是数值反解，不是PPO训练；本阶段不把拟合动作送回环境作为额外正式策略试验。
  - 验收：输出最大/P95关节残差、误差集中关节、相邻阶段符号/耦合冲突及拟合是否收敛。P95≤0.05rad且max≤0.10rad记“局部静态可表达”；其余记“当前拟合未证明可表达”，不能仅凭优化失败宣称全局不可能。即使可表达也不能宣称经过动作滤波/延迟后可动态复现。
  - QA：第8节fit-manifold；用实际从14D采样生成的目标作正例，应可回代近似重构；超关节限位目标反例应拒绝。证据 `action_manifold_fit.json`，明确下一阶段是否优先检查拇指解耦、滤波或接触反馈，不能在B直接实施。
  - Commit: N。

- [ ] 9. 生成自足报告、冻结包及机器可读结论
  - Recommended task executor category: unspecified-high
  - 依赖：7/8；若之前阻塞，改为冻结已完成任务及失败证据，不伪造未执行项。
  - 工作：实现summarize/verify-bundle，从磁盘产物重算门；冻结本次源文件、实际资产及USD场景、teacher、scene集合、原始trace/视频/日志、测试结果、本文副本和版本。清单中的路径/hash逐项存在、归档中对应字节一致；大文件可按明确索引分包但不能只留失效外盘链接。
  - 报告必须回答：最早阻塞点是什么；是否有真实正例；每个集合的实际分母/成功数；旧失败属于几何、驱动、观测还是表达能力问题；B相对A改了什么；剩余限制是什么。保留所有失败，不用“基本完成”替代结果。
  - 验收：只可输出 `PASS_B`、`BLOCKED_B`、`INVALID_EVIDENCE`；A相关及B新增测试零新增失败，compile通过。全仓运行一次并与A失败清单逐项比较；新增失败若由B引入必须修复，无关漂移需列明证据，不能把58当固定豁免额度。
  - QA：第8节summarize/verify-bundle；运行 `"$B_PY" -m pytest -q tests research/small_fruit_pinch_v1/source/tests` 与 `"$B_PY" -m compileall -q research/small_fruit_pinch_v1/source`。篡改teacher/hash、丢失一个负例或漏掉一个scene_id的复制包必须返回 `INVALID_EVIDENCE`；正常冻结包可通过验证。输出 `PHASE_B_REPORT.md/acceptance.json/SHA256SUMS`。
  - Commit: N。

## Final verification wave
以下四项是验收视角，不要求启动四个代理。主执行者完成F1/F3/F4；有可用额度时由一个只读审查代理负责F2，若代理不可用则主执行者自审并明确标记“未独立审查”，不得编造审批。用户最终审阅仍保留。

- [ ] F1. 逐门合规复核
  - 用落盘数据检查24/24主门、16/16负例拒绝、8回合批次门、56回合邻域完整性、A时序和B状态写入约束；逐项链接证据。少一项不能PASS。
- [ ] F2. 代码与证据生成链复核
  - 检查教师入口没有绕过物理、状态机边沿/重置正确、汇总器不能从自报成功通过、真实故障未被异常吞掉；核对修改范围和新测试。将具体发现及解决证据写入报告，不以“审查已派出”替代结论。
- [ ] F3. 实际回放与归档复核
  - 审阅已经运行的四方向成功视频及至少两个失败视频/快照，核对其scene_id与trace；验证归档中文件字节与清单一致，说明包不包含Python/CUDA/Isaac完整安装环境。若真实实验从未成功，明确BLOCKED，禁止用渲染动画顶替。
- [ ] F4. 范围与停止复核
  - 确认没有PPO更新、没有策略维度变更、没有硬件写入、没有改A证据或未授权提交；报告PASS_B或明确阻塞后立即停止，等待用户审阅。不得自动进入C或额外扩域。

## Commit strategy

本任务不授权提交。保存工作树差异、实际源码快照与hash供审阅，不运行 `git reset --hard`、`checkout --`、清理未跟踪文件或覆盖既有实验。确需隔离时以当前真实工作树为源复制必要项目文件并记录来源，不能只取HEAD而漏掉A未提交代码。用户另行要求提交后再执行提交流程。

## Success criteria

### 9. 最终验收表

| 门 | PASS_B的必要条件 | 未满足时 |
|---|---|---|
| G0 身份 | A及当前源码差异可追溯，B场景/teacher/本文/运行清单全部冻结 | INVALID_EVIDENCE |
| G1 映射驱动 | 坐标和关节测试通过，真实空载跟踪满足任务2，非仅目标数组正确 | BLOCKED_B |
| G2 真实性 | 真碰撞、有效传感器、无掌托近端/无名小指承重、完整碰撞审计、无运行期作弊写入 | BLOCKED_B或证据缺失时INVALID_EVIDENCE |
| G3 正例 | 预注册24/24在320步内通过A保持及B附加门 | BLOCKED_B |
| G4 反例 | 16/16物理反例真实触发且最终成功为0，相关A/B反例回归通过 | BLOCKED_B |
| G5 批次 | 8回合重复成功，事件/力/位移差在第5节容差内 | BLOCKED_B |
| G6 邻域和表达分析 | 56回合完整报告，14D拟合分析完成且表述不超证据 | 缺件INVALID_EVIDENCE；低邻域成功率本身不否决局部PASS |
| G7 交付 | 报告、trace、可审阅视频、可校验归档、测试差异及所有限制齐全 | INVALID_EVIDENCE |

只有G0–G7全部满足才能说“B阶段验收通过”。`BLOCKED_B` 时必须交付最优失败轨迹、失败分型、已花预算、排除过的解释和下一次需要用户决定的具体变更；停止后由用户决定调整范围/预算/资产或后续计划。不得将有证据的失败交付表述为物理目标已经完成。

### 10. 主要代码与参考的读法

- A口径以 `SRC/contract.py`、`verifier.py`、`verification_geometry.py` 及A实际清单为准；A报告供解释。发现冲突要记录并解决，不能挑宽松者。
- `SRC/native_env.py` 与 `simulation/tomato_pick/isaac/{env,env_cfg,geometry_sampling}.py` 是真实运行通路。
- `SRC/native_run_manifest.py/run_manifest.py` 可复用，但当前只扫描既有source顶层及部分目录：新增 `phase_b/**/*.py`、teacher、scenario、派生资产必须显式纳入，不允许漏哈希。运行清单需记录解析后的实际控制参数，而非仅CLI。
- `docs/tomato_pick/small_fruit_pinch_v1/THUMB_OPPOSITION_PINCH_EXECUTION_PLAN_20260907.md`、`research/small_fruit_pinch_v1/evidence/20260910_candidate_audit/README.md` 提供历史反例；其中短训、长回放、放宽指垫门、锁果心或运动学示教授权不自动继承。
- 保留 `model_150.pt` 为只读历史反例即可；B正式教师无需读取它，也无需重新下载历史实验。

### 11. 给新对话的启动指令

将下段直接发给执行对话：

> 请执行 `/home/miobarnacle/linkbot/linkagent/.omo/plans/small-fruit-phase-b-baseline-20260922.md`，将其作为小果抓取 B 阶段的唯一工作基线。先核验A冻结证据和当前工作树，再按依赖实现并验收；只做无PPO的真实动力学教师与证据闭环。不得降低A成功门，不得锁果心、运动学回放冒充物理成功，不得训练或启动真机。通过全部B门后立即停止等我审阅；若固定预算耗尽或真实性门阻塞，提交明确BLOCKED_B及完整失败证据后停止，不擅自扩展任务。

### 12. 修订规则

本基线的通过门、物理常数、预算和阶段范围被冻结。执行者可选具体内部实现、修复实现bug及按显存降并行度；不能自行降低门槛、换验收样本、提高物理助力或无限追加搜索。需要此类变化时记录原因、影响及拟修订条款，等待用户决定。经用户同意后在**本文件**增加版本号和修订记录，再冻结新执行身份，避免产生两个互相冲突的工作基线。

### v1.1 修订记录（2026-09-28，用户选择 B′ 并同意修订 1/2/3 与先诊断后正式验收的顺序）

本节是同一份唯一基线的现行修订；与上文旧条款冲突时以本节为准。v1.0 的冻结副本、旧 run、结果及失败分母保持不变。新证据必须建立新 run_id，记录本文件的新 SHA-256 与资产/源码哈希；不得把旧 run 的 G1/T5 结果追溯标记为 v1.1 正式通过。修订来源和证据见同目录 `small-fruit-phase-b-baseline-20260922-revision-proposal-1.md`、`20260927T093558Z_g1_armature_fix1` 归档及 `THREE_FINGER_FEASIBILITY_DIAGNOSTIC_20260928.md`；这些材料不是第二份基线。

1. **执行器稳定性（修订 1）**：`SRC/native_env.py` 两组手部执行器 `armature` 从 `0.0001` 调至 `0.01 kg·m²`。stiffness/damping、effort/velocity 上限、dt、材料及其他冻结门不变。该变更影响动力学，故新 run 必须重新运行 T2/G1，不能仅引用旧审计。
2. **反转噪声判据（修订 2）**：`SRC/phase_b/geometry.py` 的 `sign_reversal` 数值容差为 `max(1e-6 rad, 1e-3·|本轴指令幅度|)`；仍须逐轴逐方向检查实际运动，真实反转不通过。新 T2 结果记录旧零容差与新容差的逐回合反事实判定。
3. **B′ 搜索（修订 3）**：果心搜索域中心改为 `base_link` 系闭合腔心 `(0.0584, -0.0219, 0.177) m`，各轴仍为 `±40 mm`；具体运行时精确中心从 A cfg 的 `closed_grasp_fruit_center_m` 推导并记录。动态候选须用另存路径和 SHA-256 的 `left_hand_l20_physx_selfcollide.usda` 派生资产检查闭合全路径：110 个具名非相邻手部对在任一中间状态发生实际接触均为硬违规；接触偏移导致的异常力值不得用作 G2/抓取通过证据。CEM 可加入食指/中指 `joint0` spread 错峰偏置，每轴幅度限 `±0.26 rad` 且最终目标严格钳入原 safe∩USD 限位。静态 OBB 命中只是提案，必须以真实实体/PhysX 复核；冻结门仍为三个搜索 seed 各至少一次完整 A 保持且无 B 违规。预算仍为每 seed 最多 32×128 静态、10×16 连续候选，不自动追加。
4. **顺序修订**：允许 T2/G1 通过后，对 T3/T4 尚未通过所需的局部位形先做 `proposal_only` 的连续食指接触与闭合路径自碰撞诊断，以及候选生成；这些记录不得计作 T3/T4/T5 的正式通过，也不得冻结 `teacher.json` 或运行验证集。先修正当前 15 维候选序列化回归，并确认真实食指垫区接触、禁触/掌面零承重、具名路径自碰撞与分离通道可靠；再用新候选重跑 T3/G2 和 T4 的完整机制及物理门。只有 T2/T3/T4 全部正式通过，才解锁任务 5 的正式预算搜索与教师冻结。这替代上方依赖矩阵中“任何 T5 代码/诊断都不得先行”的含义，正式 T5 对 T2/T3/T4 的依赖不变。
5. **本轮停止边界**：用户此前要求 T4 完成后暂停，本次 B′ 同意未取消该边界。T3 或 T4 的必要真实性条件无法满足时，保存失败轨迹、预算与原因并报告 `BLOCKED_B`，不通过放宽门槛继续。T4 正式通过后立即暂停并提交证据；T5 正式搜索及 T6–T9 需后续指令方可执行。A 默认掌前分布与 B 局部可行场景的差异继续逐项保存，不得将局部结果宣传为原分布成功率。

原 v1.0 编制说明（历史记录）：2026-09-22 编制时，本计划只交付目标，尚未执行 B 仿真、搜索、训练或代码实现。此句不描述 2026-09-28 的执行状态；现况以本节 v1.1 修订记录及各 run 的落盘结果为准。
