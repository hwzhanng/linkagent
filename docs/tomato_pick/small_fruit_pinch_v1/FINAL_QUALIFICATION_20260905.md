# 小果三指捏取 PPO：B0 最终验收报告

日期：2026-09-05（Asia/Shanghai）  
范围：Isaac Lab 3.0 / PhysX，左手三指仿真捏取；不连接实机  
状态：**B0 64–70 mm 等效直径域合格；B3 扩展与采摘动力学未执行**

## 1. 发布候选

```text
/home/miobarnacle/linkbot/linkagent/logs/rsl_rl/
small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/model_150.pt
```

SHA-256：

```text
e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d
```

该权重为结构重训目录中的 `model_150.pt`，不覆盖历史约 89% 权重或大果掌托包络成果。历史只读权重仍为：

```text
/home/miobarnacle/linkbot/ppo_tomato_grasp_stage01_work/archives/
pinch_small_fruit_band4_20260815/artifacts/stage4_main_band4_train_256x120/best.pt
SHA-256: f2703db72d18fe9c0e55271625748be18c8a9be0a08af4739c291a03bbcb702f
```

## 2. 训练证据

- 5 更新 × 64 环境 GPU 烟雾阶段通过，无 NaN、无维度或接触崩溃。
- 正式训练使用 256 环境、总计 200 更新、seed `20260907`、`cuda:0`。第一次进程中断后从完整 `model_50.pt` 恢复，补跑 150 更新至 `model_199.pt`，并生成 `model_final.pt`。
- 只继承 14D Actor MLP 和 Actor 观测归一化；Critic、Adam 状态与探索方差重置。训练过程没有启用真实硬件接口。
- 训练末期（168–199 更新）回报约 475–493，接触/带内覆盖约 0.74–0.81/0.74–0.81，动作增量约 0.0021–0.0027，无非有限值或安全故障。

## 3. 评估合同

评估必须同时使用训练期的结构环境和如下控制合同，不能只加载权重后省略结构层：

```text
fruit-axis: 32–35 mm (等效直径 64–70 mm)
guided-residual + geometry-prior(scale=0.15)
force-servo: seek=0.04, target=0, retract=0.35, alpha=0.30,
             max_offset=0.25, warning_fraction=0.78
direct-force-projection: warning=0.50,
  high_gain_by_digit=(0.80,0.40,1.00), low_gain=0.06,
  alpha=0.10, max_open=0.50, root_fraction=0.45
force_gate_start=0.60, precontact_residual_fraction=0
residual_scale=0.04, distal_residual_scale=0.06
residual_require_all_contact + residual_trust_region(0.55,0.05)
hold_speed=0.08, hold_acceleration=0.70, natural_frequency=3.0
working_offset=(20,30,24) mm
fruit_contact_offset=0.001 m, fruit_rest_offset=0.0001 m
robot/fruit max_depenetration_velocity=0.3 m/s
hold_anchor_fraction=0.40
hold_stiffness_by_joint=(2,12,12,12,12,2,12,12,12,12,2,2,4,4,12,12)
hold_damping_by_joint=(3,0.6,0.6,3,3,3,0.6,0.6,3,3,3,0.6,0.6,0.6,3,3)
manual_decimation_control, position_target_interpolation=6
```

严格成功必须由独立 Verifier 计数：拇指/食指/中指真实接触、三指压力带、连续持稳和扰动恢复；不使用掌面支撑、不使用隐形捕获、不以训练回报代替成功率。

## 4. 三种子盲评结果

每个 seed 请求 512 个完成回合，实际完成数均为 512。

| seed | 严格成功率 | 掉落率 | 过载率 | 自碰撞率 | 保持动作 Δ P95 | 保持速度 P95 | 保持加速度 P95 | 保持 jerk P95 | 保持反转率 | 保持高频能量 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260902 | 0.9980 | 0 | 0 | 0 | 0.000902 | 0.179846 | 1.823307 | 65.669678 | 0.944087 | 0.043407 |
| 20261530 | 0.9941 | 0 | 0.0039 | 0 | 0.000898 | 0.178496 | 1.842477 | 66.511581 | 0.982927 | 0.044093 |
| 20261411 | 0.9961 | 0 | 0.0039 | 0 | 0.000899 | 0.176391 | 1.838386 | 66.473778 | 0.979754 | 0.045854 |

聚合：平均严格成功率 **99.61%**，最差 seed **99.41%**，平均过载 **0.26%**，最大过载 **0.39%**；最大保持速度/加速度/jerk/反转率/高频能量为 `0.179846 / 1.842477 / 66.511581 / 0.982927 / 0.045854`。相对于 B0 门槛（严格成功≥92%、最差 seed≥88%、掉落≤2%、过载≤1%、速度≤0.20、加速度≤3、jerk≤80、反转≤2、HF≤0.15）全部通过。

原始证据：

```text
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20260902.txt
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20261530.txt
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/evaluation/model150_seed20261411.txt
```

## 5. 远端自由度消融

加入 `--freeze-required-distal` 冻结拇指/食指/中指远端 canonical 关节后：

| seed | 实际完成回合 | 严格成功率 | 过载率 | 保持速度 P95 | 保持 jerk P95 |
|---:|---:|---:|---:|---:|---:|
| 20260902 | 530 | 0 | 0.0660 | 0.205812 | 80.331512 |
| 20261530 | 521 | 0 | 0.0729 | 0.201860 | 80.304131 |
| 20261411 | 532 | 0 | 0.0771 | 0.199499 | 80.160187 |

冻结后三种子成功率均为 0，过载上升到 6.60–7.71%，说明远端指尖自由度确实参与了低力持稳与恢复。该开关仅为反事实诊断，不是发布配置。

## 6. 逐关节保持段证据

`model150_seed20260902_per_joint.txt` 记录了 16 个 canonical 手部关节的保持段速度、加速度、jerk 和目标速度 P95。远端轴（各指末端关节）具有非零实测目标/速度，不是只观察 Actor 原始输出或把动作锁在一套终姿。

## 7. 后续边界

本报告只批准 B0 64–70 mm 仿真捏取候选。52–80 mm B3 尺寸扩展、果梗断裂、真实触觉矩阵标定、Sim2Real 和实机控制均未完成；后台训练及 Isaac Sim 可视化进程已停止。
