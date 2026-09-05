# 小果三指捏取 PPO 执行日志（2026-09-05）

本日志记录 2026-09-05 对既定小果捏取基线的结构审计、可回退探针和正式训练启动情况。历史约 89% 权重、大果掌托包络任务及其目录均为只读，本日志不修改或替代它们。

## 1. 本次执行口径

- 任务：左手三指小果捏取；拇指、食指、中指为严格必需指，无名指/小指隔离在预抓取开位。
- 物理：Isaac Lab 3.0 / PhysX，20 Hz 策略、120 Hz 物理，64–70 mm 等效果径标称域。
- 严格判据：三指真实接触、分指压力带、连续持稳与扰动恢复；独立 Verifier 的 `strict_success` 才用于结论。
- 历史资产：`logs/rsl_rl/small_fruit_pinch_v1_structural_x20_contact001_impulsafe_smoothreward2_256x120/model_50.pt` 只作为本轮 Actor warm-start 源，不覆盖该文件。
- 本轮所有真实硬件接口保持关闭；没有 CAN、机械臂或灵巧手实机命令。

## 2. 结构候选与消融证据

### 2.1 选择性保持执行器（当前主线）

当前候选组合如下，保持段将目标以当前实测关节位置按 0.8 比例锚定，并对不同关节使用不同的 PhysX 刚度/阻尼：

```text
--hold-joint-target-anchor-fraction 0.8
--hold-hand-stiffness-by-joint 2,12,12,12,12,2,12,12,12,12,2,2,4,4,12,12
--hold-hand-damping-by-joint 3,0.6,0.6,0.6,0.6,3,0.6,0.6,0.6,0.6,3,3,3,3,0.6,0.6
```

其余主线结构保持：几何弱先验、逐指力投影、三指齐备后才开放 residual、残差 trust-region、保持段目标速度/加速度限制。该组合在 3 个独立 seed、每 seed 64 个请求回合（批量执行实际完成约 64 个）上得到：

| seed | 严格成功 | 掉落 | 过载 | 自碰撞 | 平均峰值力 (N) |
|---:|---:|---:|---:|---:|---:|
| 20260902 | 1.000 | 0 | 0 | 0 | 1.380 |
| 20261411 | 1.000 | 0 | 0 | 0 | 1.377 |
| 20261530 | 1.000 | 0 | 0 | 0 | 1.387 |

seed 20261530 的保持段统计为：动作增量 P95 `0.000897`、关节速度 P95 `0.303` rad/s、加速度 P95 `4.120` rad/s²、jerk P95 `145.39` rad/s³；这比未锚定结构明显稳定，但尚未满足计划中更严格的平滑门，因此不能把该探针直接称为最终发布权重。

### 2.2 远端关节冻结消融

使用完全相同的主线候选，仅加入 `--freeze-required-distal`，冻结拇指/食指/中指远端 canonical 轴：

| seed | 实际完成回合 | 严格成功 | 掉落 | 过载 | 自碰撞 |
|---:|---:|---:|---:|---:|---:|
| 20260902 | 256 | 0.000 | 0.000 | 0.0469 | 0.000 |
| 20261411 | 259 | 0.000 | 0.000 | 0.0849 | 0.000 |
| 20261530 | 259 | 0.000 | 0.000 | 0.0849 | 0.000 |

未冻结时三种子均为 1.000 严格成功；冻结后全部为 0。这是远端自由度对持稳/恢复有实际贡献的反事实证据，不是以奖励强迫指尖运动。冻结开关仅用于评估，禁止用于训练和发布。

### 2.3 已拒绝的速度阻尼前馈

新增的 `--hold-velocity-damping-effort` 探针在四档增益下均导致严格成功率 0，并出现约 0.50–0.74 的过载率；原因是该实现覆盖了位置驱动努力通道，而不是在其上叠加阻尼。该选项保留为关闭状态的诊断代码，不进入训练命令。

之前的当前姿态阻抗、接触后全局刚度降档、目标斜率/速度硬帽和按关节 anchor 向量等探针也因严格成功下降或过载上升被拒绝，保留其临时输出但不作为主线。

## 3. 工程验证

- `compileall`：`native_env.py`、`native_train.py`、`native_evaluate.py`、`guided_residual_env.py` 通过。
- 小果专项单测：`research/small_fruit_pinch_v1/source/tests/test_temporal_and_verifier.py` 为 **4 passed**（带 Isaac/PhysX 配置弃用警告）。
- 全仓 pytest 当前有 57 个与本轮无关的既有失败，主要集中在旧 harvest-cycle 注册、设备/导出策略固定资源和历史快照一致性；未将这些失败归因于小果结构改动，也没有修改其只读区域。
- 5 更新 × 64 环境 GPU 烟雾训练通过，无 NaN；训练侧接触/带内覆盖约 0.87/0.87，保持动作增量约 0.0015，生成 `logs/rsl_rl/small_fruit_pinch_v1_selective_hold_smoke_20260905/model_final.pt`。

## 4. 正式结构重训（已完成）

训练入口：`research/small_fruit_pinch_v1/source/native_train.py`。

```text
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_smoke_20260905/
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_256x200_20260905/
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/
```

执行过程：

1. 先完成 5 更新 × 64 环境 GPU 烟雾训练；无 NaN、无维度/接触崩溃，接触覆盖约 0.85–0.87，确认 14D Actor 和结构环境可运行。
2. 正式 256 环境、总计 200 更新、seed `20260907`、GPU `cuda:0`；第一次进程运行至约 61 更新时被外部中断，已从完整 `model_50.pt` 恢复，补跑 150 更新至 `model_199.pt`，最终同时生成 `model_final.pt`。未拼接损坏 checkpoint，也未覆盖历史约 89% 或大果任务权重。
3. 训练只加载结构候选的 14D Actor MLP 与观测归一化，Critic/Adam/探索方差重置；`init_std=0.05`、学习率 `1.5e-4`、熵系数 `0.001`、保持段关节平滑奖励 `0.01`。第 168–199 更新期间回报约 475–493，接触率约 0.74–0.81、带内约 0.74–0.81、动作增量约 0.0021–0.0027、无 NaN 或安全故障。

新正式候选目录中各 checkpoint 的 SHA-256：

```text
model_50.pt   ffc9bbb5b2f38d8fcd4ec32a2e57b7e82677d448303b453347a5b449adc40e1f
model_100.pt  4d9e66a2b07bcccb4fde4ab54c460988399c8f98aff9e6c99f16e66b77cd3a37
model_150.pt  e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d
model_199.pt  e338ced69d976c709fd0493899c082c4d51b1754f7f2d07d8781f657c3d087d8
model_final.pt bf133eb2292f58e315b46ba92f31a8bb97af10b85060f348442b3ab13aa65966
```

## 5. 正式候选验收（已完成）

### 5.1 三种子完整盲评

候选 `model_150.pt` 在 64–70 mm 等效直径域使用 128 并行环境、每种子请求 512 个完成回合，严格判据为独立 Verifier 的三指真实接触、压力带、连续持稳和扰动恢复：

| seed | 回合 | 严格成功率 | 掉落率 | 过载率 | 自碰撞率 | 保持速度 P95 | 保持加速度 P95 | 保持 jerk P95 | 保持反转率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260902 | 512 | 0.9980 | 0 | 0 | 0 | 0.179846 | 1.823307 | 65.669678 | 0.944087 |
| 20261530 | 512 | 0.9941 | 0 | 0.0039 | 0 | 0.178496 | 1.842477 | 66.511581 | 0.982927 |
| 20261411 | 512 | 0.9961 | 0 | 0.0039 | 0 | 0.176391 | 1.838386 | 66.473778 | 0.979754 |

平均严格成功率 `0.9961`，最差 seed `0.9941`，平均过载 `0.0026`，最大过载 `0.0039`；保持段最大速度/加速度/jerk/反转率为 `0.179846 / 1.842477 / 66.511581 / 0.982927`，均通过 B0 门槛。原始输出保存在：

```text
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20260902.txt
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20261530.txt
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20261411.txt
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20260902_per_joint.txt
```

逐关节保持段 P95（canonical 顺序：拇指 0–4、食指 0–3、中指 0–3、无名指 0–3、小指 0–3）也已写入 `model150_seed20260902_per_joint.txt`，证明远端轴在实际执行轨迹中有独立运动，而不是只看 Actor 输出。

### 5.2 远端关节消融

同一合同加入 `--freeze-required-distal` 后，三种子实际完成 530/521/532 回合，严格成功率均为 `0`，过载率为 `0.0660/0.0729/0.0771`，保持 jerk P95 为 `80.3315/80.3041/80.1602`。该反事实结果说明冻结指尖会破坏稳定持稳，远端自由度确实被策略利用。

## 6. 当前交付状态与后续边界

- **已交付**：B0（64–70 mm）三指小果捏取的结构化 14D 仿真候选，`model_150.pt` 通过三种子严格成功、过载、掉落、自碰撞和保持平滑门；最终 SHA-256 为 `e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d`。
- **保留不动**：历史约 89% 权重（`f2703db7...cb702f`）和大果掌托包络训练目录；本轮仅以新结构候选为独立产物，没有覆盖或替代它们。
- **仍未宣称完成**：52–80 mm B3 扩展、果梗/采摘动力学、真实触觉标定、Sim2Real 和任何实机控制；这些工作必须另行按计划阶段执行。
- 本次运行已结束，后台没有继续训练或 Isaac Sim 可视化进程。
