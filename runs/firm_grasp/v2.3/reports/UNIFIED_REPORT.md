# F3/F4 仿真PPO抓牢续作统一报告

日期：2026-10-19。执行人：外部单Agent（本次续作）。所有结果均在独立新根内产生；旧版 v1.6、F2、v2.2 及其证据保持只读。

## 0. 结论与停止原因

**已完成并封存（小样本、低成功率边界）**：

1. F3 三种模式均有真实 3 秒三指 G2 成功窗：quiet 4/4、force(axis0+) 3/4、torque(axis0+) 4/4（开发 4 回合口径）。
2. 新 F4 预登记 20 项中，PPO 候选 **15/20**，同条件未更新初始化 **10/20**；quiet 6/8 vs 4/8，force 4/6 vs 3/6，torque 5/6 vs 3/6。正式矩阵中力矩成功 5/6。
3. 独立重载（同清单、声明 seed 111）重复出完全相同的结果：候选 15/20，初始化 10/20；20 个 case 的逐个结果均一致。该复核是重复性确认，不是分布外泛化。
4. 候选全部 15 个正式成功窗均通过独立数值 G2 重算和原始完整网格 PAD 重分类；开发 11 个成功窗同样通过。
5. 候选在三种任务身份下均可 strict reload：quiet 4/4；force 身份 3/4；torque 身份 4/4。force/torque 的 strict 产物是把同一 actor_state_dict 逐字节复制、仅重新签署目标身份后落盘的核验件（未训练、未改 actor）；7/7 成功窗数值+网格 PASS。旧版权重/资产哈希复验未变。

**明确限制**：

- 只做了 10 次真实 PPO 更新（10,240 transitions），不是高成功率或高可靠性策略。
- 正式矩阵仍有 5 个失败：force axis0+、force axis2+、torque axis1+，以及 quiet 的 2 个 case。
- 单固定 67 mm 球、固定果位、固定初态；不证明果位/果形泛化、提举、机械臂运动或真机能力。
- 一个成功窗的最大相对角速度 0.493 rad/s，距 0.5 上限很近；成功是真实的，但余量薄。
- 独立重载只换 seed，未做独立分布；不能把“15/20 两次一致”写成统计显著性。

停止原因：已达到用户定义的理想边界（F3 三模式成功可重载、F4 候选优于本线初始化且不低于 v2.2、正式力矩出现成功、独立重载方向复现），按授权封存，未继续调参或开启 F5/提举/真机。

## 1. 输入、隔离与身份

|项目|实际值|
|---|---|
|新根|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-continuation-20261019`|
|主源码|`NEW/source`，分支 `codex/firm-grasp-f3-f4-continuation-20261019`|
|继承基线|`source-f3-v2_2` commit `f60b912ce85f1361cd02bc11da15a10e302b1cbe`|
|继承 critical source SHA256|`7460ecdec4fce9a0e258fbb667d8aa61f18dda155d32d4c080dd7b48f7398167`（新副本逐文件复算一致）|
|C01 训练时源码|commit `04444652423ab9f440ed049e0633ad7ad7a7ff48`；冻结副本 `NEW/source-v2.3.0-c01`|
|当前主源码|commit `5faf919712d9a7f13d03dcabde6cbdcc0c312e30`（增加 v2.3 actor seed 兼容）|
|新版本号|`firm_grasp_v2.3.0`|
|F4 预登记清单|`NEW/inputs/F4_CASES_CONTINUATION.json`，SHA256 `a6a3546f1fefb24bbe1b5466aeaae8f78af7c9c151e110c459ca0dd6f9e7c6f8`|
|GPU 锁|`/tmp/linkagent-phase-b-cuda0.lock`；真实 GPU/PhysX 进程始终串行，最多 1 个|
|CAN/真机|未开启；未修改上游 IsaacLab 或依赖环境；未清理 canonical 脏工作区|

输入哈希在 `RECEIPT.json` 与 `reports/PROTECTED_INPUTS_AFTER.json`。旧 v2.2 更新/初始化、F2、v1.6、批准 PAD、source geometry、原完整网格分类器均 `MATCHED_READ_ONLY`。

并发说明：本次没有派发子 Agent。当前工具只能选择 Agent 类型，不能显式指定并核实“同一模型标识+同一思考强度”，按用户授权由主 Agent 自行执行全部诊断、训练、核验与报告，并在此记录该限制；未切换到 economy 异构路由，也未创建额外用户对话。

隔离说明：新根内没有写入任何隐瞒支撑、果实状态瞬移、质量/摩擦/PD/驱动力上限修改；支撑在策略步 120 起严格为零；所有成功都要求真实 120 Hz 接触与批准 PAD。ramp40 只是准备期支撑退出课程的改变，已单独标注为条件收益。

## 2. 关键物理发现：撤支撑课程从 20 步延长到 40 步

继承条件为支撑 6 秒、最后 20 个策略步（1 秒）退出，`support-release-power=4`。本续作测试了计划中未执行的 ramp40：从策略步 80 到 120 渐退，**第 120 步起仍严格为零**，正式 G2 的“支撑为零”不变。

|同 actor（v2.2 update_000001）|quiet|说明|
|---|---:|---|
|ramp20（继承）|1/4|D01，严格重载，identity `2620b17d...`，与旧训练身份一致|
|ramp40（新准备条件）|3/4|D02；3 个窗数值 PASS、完整网格 PASS|
|ramp40 + 20 case 旧顺序矩阵|13/20|R0；quiet 6/8、force 4/6、torque 3/6|

这是**准备条件收益，不是 PPO 学习收益**。支撑实际为零、PAD、质量/摩擦/PD/驱动力上限、七类硬故障、360 子步与几何/旋转门槛均未改变。

同时确认：之前“首次 reset 与再次 reset 初态不同”的主要差异是 trace 采样相位差（相差一个物理子步，果速 0.034 m/s 对 0）；不是 reset 写入了不同初态。未据此修改 reset 代码。

## 3. 真实 PPO 训练

### 3.1 配置

从 v2.2 更新 actor + 匹配 normalizer 做 actor-only seed；fresh critic/optimizer；新任务身份 `firm_grasp_v2.3.0`。

|项目|值|
|---|---|
|更新数|10（另有 2 次 smoke，共 12 次真实更新）|
|每次采样|4 env × 256 策略步 = 1,024 transitions|
|总交互|10,240 transitions（smoke 另 2,048）|
|学习率|2e-5，fixed schedule|
|探索 std|5e-4，冻结 log_std|
|约束|冻结 actor normalizer；每 update 比较当前 actor 与冻结 prior 在同一 rollout 观测上的平均绝对 mean 偏移，超过 0.002 时向 prior 线性收缩|
|实际偏移|0.000200–0.000962，10 次均未触发收缩|
|参数变化|output layer 权重范数 0.000452 → 0.011509；hidden 层相对变化约 0.9%–1.8%；normalizer count 保持 1024；log_std 未变|

这不是“reward 上升即学会”：下面所有成功都来自 120 Hz 真实接触、零支撑、批准 PAD 与可重载权重的同条件复核。

### 3.2 训练期的物理迹象（不作为验收）

训练回合中出现了 env0 的 quiet 成功、env2 的成功和多次接近成功的窗；这些只用于判断方向，不进入正式分母。

## 4. 同条件开发复核（4 回合/模式）

条件：ramp40、g2 3 秒、同一 seed、同一 reset/pose；force/torque 为 axis0 正方向。

|actor|quiet|force|torque|合计|
|---|---:|---:|---:|---:|
|本线初始化（v2.2 update_000001）|3/4|2/4|2/4|7/12|
|PPO 候选（C01 final）|4/4|3/4|4/4|11/12|
|候选成功窗核验|4/4 PASS|3/3 PASS|4/4 PASS|11/11 数值+网格 PASS|

开发结果显示候选把 env0 的 quiet 失败转为成功，并提高了 force/torque 的保持率。

## 5. 新 F4 正式 20 项（预登记交错清单）

清单：8 quiet + 6 force（每轴 ±）+ 6 torque（每轴 ±），case 顺序 `continuation_interleaved_v1`。所有 actor 使用同一清单、同一 ramp40 条件；矩阵以 actor-only seed 评估（与旧 F4 协议一致），strict reload 单独验证。

|actor|quiet|force|torque|总计|FAILURE|UNKNOWN|
|---|---:|---:|---:|---:|---:|---:|
|本线初始化（v2.2 update）|4/8|3/6|3/6|**10/20**|10|0|
|PPO 候选（C01）|6/8|4/6|5/6|**15/20**|5|0|
|独立重载 初始化（seed111）|4/8|3/6|3/6|10/20|10|0|
|独立重载 候选（seed111）|6/8|4/6|5/6|15/20|5|0|
|旧 v1.6 适配（历史参考）|0/8|0/6|0/6|**0/20**|20|0|
|开掌负对照|0/4|—|—|0/4|4|0|

旧 v2.2 参考：继承条件下（ramp20、旧顺序）最终 20 项为 3/20；ramp40、旧顺序为 13/20；ramp40、新交错顺序为 10/20。候选在新清单上 15/20，高于本线初始化和 v2.2 在三种口径下的成绩，未发现总体退步。

说明：本线的“PPO 前初始化”就是 v2.2 `update_000001` actor（不是旧 v2.2 的常量先验 `initial.pt`）；因此本线初始化与“当前 v2.2 更新 actor”是同一权重，区别只在评估条件。旧 v2.2 常量先验没有在本轮重新训练，其历史 3/20 属于继承条件，不能与新表混作同一分母。R0（ramp40、旧顺序）与 R2（ramp40、新顺序）之间 13→10 的差异来自 case 顺序/源码版本，不是学习退步。

最佳与最后 checkpoint：同一个 `C01_train10/final.pt`（SHA256 `ffbf0917...a163`）。本轮只训练 10 次更新，没有在最终清单上挑选多个 checkpoint；开发阶段选定后即冻结。更新前对照为 `C01_train10/initial.pt`（其 actor 来自 v2.2 `update_000001`，SHA256 `af961f8e...60d9`）。

### 5.1 候选逐 case

|case|env|模式|轴/符号|结果|峰值 G2 子步|核验|
|---:|---:|---|---|---|---:|---|
|0|0|quiet|—|SUCCESS|365|PASS|
|1|1|force|0/+|FAILURE|138|—|
|2|2|torque|0/+|SUCCESS|365|PASS|
|3|3|quiet|—|SUCCESS|360|PASS|
|4|0|force|1/−|SUCCESS|364|PASS|
|5|1|torque|1/−|SUCCESS|360|PASS|
|6|2|quiet|—|SUCCESS|361|PASS|
|7|3|force|2/+|FAILURE|164|—|
|8|0|torque|2/+|SUCCESS|364|PASS|
|9|1|quiet|—|SUCCESS|361|PASS|
|10|2|force|0/−|SUCCESS|362|PASS|
|11|3|torque|0/−|SUCCESS|361|PASS|
|12|0|quiet|—|FAILURE|287|—|
|13|1|force|1/+|SUCCESS|363|PASS|
|14|2|torque|1/+|FAILURE|10|—|
|15|3|quiet|—|FAILURE|235|—|
|16|0|force|2/−|SUCCESS|360|PASS|
|17|1|torque|2/−|SUCCESS|362|PASS|
|18|2|quiet|—|SUCCESS|360|PASS|
|19|3|quiet|—|SUCCESS|360|PASS|

15 个 SUCCESS 的成功窗起止、最小 PAD 力/份额、速度/漂移/旋转/角速度/对向几何见 `reports/R1_verification/numeric/report.json` 与 `reports/R1_verification/WINDOW_AUDIT.json`。

### 5.2 成功窗余量（15 窗汇总）

|指标|最小值/最大值|门槛|
|---|---:|---:|
|食指最小 PAD 力|0.192 N|≥0.15 N|
|中指最小 PAD 力|0.184 N|≥0.15 N|
|食指最小承载份额|7.34%|≥5%|
|中指最小承载份额|6.83%|≥5%|
|最大相对速度|0.0193 m/s|≤0.05 m/s|
|最大锚点漂移|1.10e-4 m|≤0.01 m|
|最大相对旋转|0.0031 rad|≤0.174 rad (10°)|
|最大相对角速度|0.493 rad/s|≤0.5 rad/s（余量薄）|
|起窗子步|0–46|≤60|
|支撑为零|全部 true|必须|
|七类硬故障|全部 false|必须|

## 6. F3 三模式与可重载

- quiet：候选 4/4；其中 4 个窗数值+网格 PASS。
- force(axis0+)：3/4；3 个窗 PASS。
- torque(axis0+)：4/4；4 个窗 PASS。
- 严格重载：从 C01 训练身份源码 `source-v2.3.0-c01` 以 `--policy checkpoint` 重载 C01 final，quiet 4/4；该源码副本保留在 `NEW/source-v2.3.0-c01`。
- force/torque 的 strict reload：把 C01 final 的 `actor_state_dict` 逐字节复制，仅用目标 force/torque 环境身份重新落盘为 `C01_identity_artifacts/{force,torque}_strict.pt`（无训练、无 actor 修改）；以 `--policy checkpoint` strict 重载后 force 3/4、torque 4/4，7/7 成功窗数值+网格 PASS。这补足了旧协议中“actor-only seed 后在该身份保存并严格重载”的步骤。
- force/torque 的开发对照同时保留了 actor-only seed 版本（`C01d_eval_*`），结果与 strict 版本一致；两者都不改变 actor 字节。
- 独立重载：同清单 seed111 的 20 个 case 结果与 seed110 逐 case 相同；说明固定初态下的重复性，不证明独立分布。

## 7. 历史参考、负对照与录像

- 旧 v1.6 经原 161 维/ring-little 动作语义适配，在 ramp40 + 新清单下运行：0/20 SUCCESS、20 FAILURE、0 UNKNOWN（failures 为 drop/timeout，峰值 G2 均为 0）。这复现了旧 actor 在新条件下的不可用性，不是新候选的退步。
- 开掌负对照：0/4 SUCCESS、4 FAILURE、0 UNKNOWN（全部 drop），说明成功不是“开掌/无接触”伪通过。
- 两视角录像：前视 `F4v_front_c`、侧视 `F4v_side_c`，均由 C01 训练身份源码严格重载并记录 4/4 quiet 成功；两视角各 93 帧、10 fps、9.3 秒，选中 env2（成功窗 sim_step 758–1117）。两视角 trace 均 4/4 数值 PASS，env2 成功窗原始完整网格 PASS。录像只作解释，不替代数值/网格核验。录像回执保留了旧 `thumb_pad_grasp_v1` task_id 字段（record_video 继承 schema），实际任务身份以 run_metadata 的 `firm_grasp_v2.3.0` 为准；原回执未改写。

## 8. 复现命令

```bash
NEW='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-continuation-20261019'
OLD='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018'
EP='/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python'

# 1) 候选训练（C01）
FIRM_SOURCE_NAME=source "$EP" "$NEW/execute.py" C01_train10 --mode train --policy fresh \
  --actor-seed-checkpoint "$OLD/evidence/F3_train_margin_bounded02/update_000001.pt" \
  --iterations 10 --goal g2 --hold-seconds 3 --support-release-power 4 \
  --support-ramp-policy-steps 40 --exploration-std .0005 --learning-rate 2e-5 \
  --freeze-actor-normalizer --freeze-actor-log-std --actor-mean-shift-max .002

# 2) 严格重载（必须使用 C01 训练同身份源码）
FIRM_SOURCE_NAME=source-v2.3.0-c01 "$EP" "$NEW/execute.py" C01b_eval_quiet \
  --mode evaluate --policy checkpoint --checkpoint "$NEW/evidence/C01_train10/final.pt" \
  --episodes 4 --goal g2 --hold-seconds 3 --support-release-power 4 \
  --support-ramp-policy-steps 40 --learning-rate 2e-5

# 3) 新 F4 20 项（候选；初始化把 checkpoint 换成 v2.2 update）
FIRM_SOURCE_NAME=source FIRM_ENTRY_MODULE=matrix "$EP" "$NEW/execute.py" R1_matrix_C01_interleaved \
  --matrix --support-release-power 4 --support-ramp-policy-steps 40 --learning-rate 2e-5 \
  --force-ratio .1 --torque-ratio .01 \
  --actor-seed-checkpoint "$NEW/evidence/C01_train10/final.pt"

# 4) 数值与原始完整网格核验
CUDA_VISIBLE_DEVICES='' "$EP" "$NEW/verify_f4_all.py" \
  "$NEW/evidence/R1_matrix_C01_interleaved" --output-dir "$NEW/reports/R1_verification"
```

## 9. 资源账

见 `reports/RESOURCE_LEDGER_CONTINUATION.json`（最终版）。本次新根共 27 个 GPU/PhysX 启动回执：20 个 exit 0，7 个快速失败（argparse/身份/重试命名，合计约 67 秒，无物理结果被冒充）；成功运行墙钟合计约 2.57 小时（含 Kit 启动/关闭），真实交互 59,636 transitions，PPO 更新 12 次（2 smoke + 10 正式），GPU 进程始终最多 1 个。CPU 核验未单独计量；失败与中断全部保留。

## 9.1 交付物索引

- 小体积可移植包：`NEW/delivery/`（权重、报告、两视角视频、输入清单、核心源码快照、`DELIVERY_MANIFEST.json`）。
- 完整源码树：`NEW/source`（主源码）与 `NEW/source-v2.3.0-c01`（C01 训练同身份冻结副本）。
- 原始轨迹/失败产物：`NEW/evidence/<run>/`，索引见 `NEW/RAW_EVIDENCE_INDEX.json`。
- 权重：`NEW/delivery/weights/C01_final.pt`、`C01_initial.pt`、`v22_update.pt`、`v22_initial.pt`、`v16_update1.pt`；C01 final SHA256 `ffbf0917...a163`。

## 10. 未完成项与下一步

- 未做：扰动机/轴随机化的训练课程；F2 actor 起点；reset 小范围随机化；更多 update；提举/机械臂/果位果形泛化/真机。
- 最有价值下一步：在训练中随机化 disturbance mode/axis/sign（或分段 force/torque 训练），针对本清单仍失败的 force axis0+/axis2+、torque axis1+ 做有界课程；正式门槛不变，仍需重新预登记清单与完整分母。
- 不得把本报告的 15/20 写成高可靠性或统计显著；它是固定初态、小样本、同条件重复性结果。
