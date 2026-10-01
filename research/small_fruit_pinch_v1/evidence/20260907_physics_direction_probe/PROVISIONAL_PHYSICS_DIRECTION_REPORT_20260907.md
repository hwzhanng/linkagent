# 64–70 mm 左手三指捏取首版物理方向报告

日期：2026-09-07（Asia/Shanghai）  
状态：**provisional / 物理动作检查版，不是最终采摘权重**

> 后续审计已发现本版 guide-only 贡献过强，并产生不覆盖本权重的 residual16/24 第二候选。
> 当前人工查看优先使用 `RESIDUAL16_24_CANDIDATE_REPORT_20260907.md` 中的权重与命令；
> 本文及原权重继续作为只读回归锚点。

## 1. 本轮目标

本轮只验证“仅固定基座左手、64–70 mm 果实、拇指先横向进入食指和中指另一侧，再形成三指接触”的动作方向。
成功率不是本轮硬门；硬门仍包括单左手拓扑、真实接触传感器、无 NaN/自碰撞/过载，以及不能靠固定姿态标签冒充成功。
不连接实机，不加载双臂或右手。

## 2. 训练产物

训练目录：

```text
logs/rsl_rl/small_fruit_pinch_v1/left_hand_physical_direction_probe_20260907_candidate_yshift_64_70/
```

首选检查点：

```text
model_final.pt
SHA-256: 105075581b7e7a94fa3dd4f81cb76105dbab63aaa99ad977b21a2024bd71409d
```

训练设置：

- 50 PPO iterations，128 并行环境，GPU `cuda:0`；
- 14D guided residual，Actor 输出层零残差初始化，残差幅度 `0.08`，远端 `0.12`，接触前残差比例 `0.05`；
- 果径语义为直径 64–70 mm，事件参数使用椭球半轴 32–35 mm；
- 训练场景只引用 `research/small_fruit_pinch_v1/assets/left_hand_urdf/left_hand_l20_physx.usd`；21 个左手关节（16 主动 + 5 mimic），无臂关节、右手链节或臂 actuator；
- 本轮启用 `palm_fruit_collision_filter`，仅过滤导入 USD 的 `base_link↔Tomato` 粗糙掌部 AABB。该开关是模型几何诊断条件，**不得直接作为最终发布或实机依据**；指节碰撞、三指接触、传感器和压力保护仍由 PhysX 计算。

完整训练命令（从 `linkagent/` 运行）：

```bash
PYTHONPATH=/home/miobarnacle/linkbot/linkagent \
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
research/small_fruit_pinch_v1/source/native_train.py \
  --num-envs 128 --iterations 50 --seed 20260907 \
  --log-dir logs/rsl_rl/small_fruit_pinch_v1/left_hand_physical_direction_probe_20260907_candidate_yshift_64_70 \
  --guided-residual \
  --guide-action-values 0.5563,0.3046,-0.4787,0.1579,0.3133,0.6795,0.1310,-0.1900,0.2070,0.5084,-0.2590,-0.6060,-0.1727,0.0229 \
  --zero-residual-init --residual-scale 0.08 --distal-residual-scale 0.12 \
  --precontact-residual-fraction 0.05 --init-std 0.06 --explicit-hand-pd \
  --center-local-mm -52.42 -16.09 -177.48 --palm-fruit-collision-filter \
  --logger tensorboard --run-name left_physical_direction_probe_yshift
```

## 3. 独立评估结果

4 个独立回合的短评估（仍绑定上述诊断配置）：

```text
strict_success_rate=0.0000
drop_rate=0.0000 overload_rate=0.0000 self_collision_rate=0.0000
mean_palm_force_n=0.000000
mean_peak_force_n=0.521548 max_peak_force_n=0.601330
action_delta_p95=0.010000 joint_velocity_p95=0.012445
joint_acceleration_p95=0.383810 joint_jerk_p95=29.752085
direction_reversal_rate=0.184748 high_frequency_target_energy=0.167236
hold_action_delta_p95=0.000000 hold_joint_velocity_p95=0.000000
```

`strict_success_rate=0` 在本轮是预期结果：本轮未把“断梗并摘后保持”作为训练目标，不能把未摘取包装成成功。
它同时说明短评估没有发生掉落、过载、自碰撞或掌部承力。

## 4. 策略实际动作与接触点证据

单环境策略轨迹审计确认：

- Actor 输出并非固定终姿回放；第 0 步已有非零残差，后续动作持续变化；
- 仅拇指、食指、中指开放残差，环指和小指保持安全开位；
- 轨迹第 227 步首次形成三指共同有效接触，接触力峰值约 `[0.474, 0.512, 0.591] N`（拇指、食指、中指），无名指/小指为 0；
- 首次真实接触顺序为 `拇指(step 14) → 食指(step 205) → 中指(step 211)`；三指共同达到
  `0.08 N` 阈值的累计步数为 17。无名指和小指在全轨迹中没有被策略打开参与。
- 以真实 `ContactSensor.contact_pos_w` 计算，而非 frame tip：

```text
best_step=227
contact_xyz(m) = [
  (0.08923, -0.03405, 0.17341),  # thumb
  (0.05450, -0.01748, 0.21373),  # index
  (0.05678, -0.00383, 0.21577),  # middle
]
fruit_xyz(m) = (0.07560, -0.01156, 0.19122)
theta_opp = 43.47 deg
d_chord = 11.59 mm
```

本轮采用“本地继续门”而不是严格目标：对向角低于 60°、弦偏差低于约 `0.40 R_eq`（64–70 mm 球的等效半径约 32–35 mm），因此足以供人工检查动作方向；尚未声称达到计划中的 `30°/0.40R` 严格门。

接触顺序和轨迹原始输出见 `policy_trace_250step.txt` 的
`first_contact_step=[[14,205,211,-1,-1]] contact_order=[[0,1,2]]`。

8 环境并行短复核中，拇指首接触率为 8/8，三指共同接触率为 4/8；无名指和小指在 8/8
回合均未接触，且没有掉落、过载或自碰撞终止。完整数组见
`multi_env_direction_check.md`。这比单环境轨迹更能支持动作方向判断，但仍不作为成功率
估计。

作为负对照，关闭掌部诊断过滤后使用同一果心和同一权重，reset clearance 将果实推到远离手指的
位置，250 步内无任何有效接触（见 `no_filter_negative_control.md`）。这说明当前过滤开关仍是
导入掌部粗糙 AABB 的几何补丁，不能把该条件误报为最终物理真实性已完成。

## 5. 结论和限制

**可以交付用于检查动作方向的首版 provisional 权重**：它确实在单左手、64–70 mm 场景中先摆拇指、再出现食指/中指接触，并保持低压力；不会因为严格采摘成功率为 0 而被误判为训练崩溃。

但它还不是最终“抓紧并摘下”权重，原因是：

1. 诊断分支过滤了掌部粗糙 AABB，仍需用精确掌面碰撞体或无过滤模型复核；
2. 当前策略的最佳实测对向角约 43.5°，方向正确但仍有明显余量；
3. 本轮没有以断梗、摘后 3–6 s 保持和扰动恢复为训练目标，严格成功率 0 是诚实结果；
4. 本轮只覆盖 64–70 mm，未开始扩域或高成功率优化。

## 6. 复现与下一步

策略轨迹审计：

```bash
PYTHONPATH=/home/miobarnacle/linkbot/linkagent \
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
research/small_fruit_pinch_v1/source/policy_trace_probe.py \
  --checkpoint logs/rsl_rl/small_fruit_pinch_v1/left_hand_physical_direction_probe_20260907_candidate_yshift_64_70/model_final.pt \
  --num-envs 1 --steps 250 --explicit-hand-pd \
  --guide-action-values 0.5563,0.3046,-0.4787,0.1579,0.3133,0.6795,0.1310,-0.1900,0.2070,0.5084,-0.2590,-0.6060,-0.1727,0.0229 \
  --residual-scale 0.08 --distal-residual-scale 0.12 --precontact-residual-fraction 0.05 \
  --center-local-mm -52.42 -16.09 -177.48 --palm-fruit-collision-filter
```

下一步必须先完成无过滤/精确掌面碰撞复核，并把 `theta_opp` 压到严格门，再开始任何成功率优化；不得用增加迭代、放宽压力或固定终姿掩盖几何问题。
