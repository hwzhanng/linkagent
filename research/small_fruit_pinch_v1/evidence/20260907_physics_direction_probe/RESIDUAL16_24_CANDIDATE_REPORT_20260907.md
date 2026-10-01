# residual16/24 物理方向候选报告（2026-09-07）

状态：**可供人工查看动作方向的第二候选；不替换旧锚点，不代表最终摘取成功**

## 1. 产生原因

首版锚点在 zero-policy 消融中仍可依靠固定 guide 完成 5/8 三指短接触，说明接触前
`residual_scale=0.08 × precontact_fraction=0.05` 使 Actor 物理权限过小。本轮只做一次
克制调整：根部残差 `0.08→0.16`、远端残差 `0.12→0.24`、接触前比例 `0.05→0.15`。
单左手拓扑、64–70 mm、guide、果位、安全硬门、显式 PD 和诊断掌部过滤均保持不变。

## 2. 权重身份

```text
logs/rsl_rl/small_fruit_pinch_v1/
left_hand_physical_direction_probe_20260908_residual16_24_64_70/model_final.pt
SHA-256: 7bd678fb2588a47241bbf025a1c1bd686b37bcd0bc62b6568efb57160f0d836c
```

对应左手 USD：

```text
research/small_fruit_pinch_v1/assets/left_hand_urdf/left_hand_l20_physx.usd
SHA-256: b5ab656b3b58b2e19b349755158c9f11f84f9b48c2211c474fac6fd66b7ddb57
```

## 3. 训练设置与训练内趋势

- 30 PPO iterations，128 并行环境，seed `20260908`；
- Actor 输出层零残差初始化，探索标准差 `0.06`；
- 果径 64–70 mm（PhysX 椭球半轴 32–35 mm）；
- 训练日志：`contact_ratio 0.251→0.423`，`in_band_ratio 0.084→0.275`；
- `mean_reward -5.52→47.12`，中途最佳 70.47；
- `palm_force_n=0`；严格摘取成功仍为 0，符合本轮尚未训练断梗/摘后保持的范围。

完整训练命令：

```bash
PYTHONPATH=/home/miobarnacle/linkbot/linkagent \
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
research/small_fruit_pinch_v1/source/native_train.py \
  --num-envs 128 --iterations 30 --seed 20260908 \
  --log-dir logs/rsl_rl/small_fruit_pinch_v1/left_hand_physical_direction_probe_20260908_residual16_24_64_70 \
  --guided-residual \
  --guide-action-values 0.5563,0.3046,-0.4787,0.1579,0.3133,0.6795,0.1310,-0.1900,0.2070,0.5084,-0.2590,-0.6060,-0.1727,0.0229 \
  --zero-residual-init --residual-scale 0.16 --distal-residual-scale 0.24 \
  --precontact-residual-fraction 0.15 --init-std 0.06 --explicit-hand-pd \
  --center-local-mm -52.42 -16.09 -177.48 --palm-fruit-collision-filter \
  --logger tensorboard --run-name left_physical_direction_probe_residual16_24
```

## 4. 训练外 full/zero-policy 对照

固定 seed `20260909`、8 环境、每环境 250 策略步；两组除 Actor 输出是否置零外完全一致。

| 指标 | 完整策略 | zero-policy | 判读 |
|---|---:|---:|---|
| 拇指首接触 | 8/8 | 8/8 | 正确顺序保留 |
| 三指补齐环境 | 5/8 | 5/8 | 覆盖率尚未拉开 |
| 三指共同接触累计步数 | 74 | 41 | 完整策略提高 80.5% |
| 有效几何环境 | 5/8 | 5/8 | 样本仍少 |
| 最佳对向角（有效环境） | 43.79/53.54/55.83/45.87/44.48° | 49.96/57.51/61.59/56.90/46.89° | 完整策略整体更接近对向 |
| 弦偏差（有效环境） | 11.71/13.13/13.96/12.15/11.30 mm | 12.91/14.52/16.07/14.80/11.77 mm | 完整策略整体更小 |
| 环指/小指接触 | 0/8 | 0/8 | 未偷用非目标手指 |
| 终止 | 0 | 0 | 短窗无掉落/过载/自碰撞终止 |

完整策略 Actor 输出：

```text
mean_abs=0.009316 std=0.012511 delta_p95=0.018588 max_abs=0.073175
```

这表明策略贡献已经可以测得，但仍是“接触持续时间和几何改善”，而不是高覆盖率或最终
抓牢证明。旧锚点在同一 seed/放宽后运行时为 6/8 环境、累计 48 步：旧锚点覆盖略高，
新候选接触更持久；因此两者均保留，不做单指标强行替换。

## 5. 16 回合独立安全/平滑评估

seed `20260910`：

```text
strict_success_rate=0.0000
drop_rate=0.0000 overload_rate=0.0000 self_collision_rate=0.0000
mean_palm_force_n=0.000000
mean_peak_force_n=0.522699 max_peak_force_n=1.107885
action_delta_p95=0.010000
joint_velocity_p95=0.011977
joint_acceleration_p95=0.366397
joint_jerk_p95=25.443043
direction_reversal_rate=0.175658
high_frequency_target_energy=0.159722
```

本轮没有引入过载、自碰撞或掌部承力；jerk 比首版 4 回合短评的 29.75 更低。保持段
`hold_high_frequency_target_energy=0.630936` 的样本量和持续时间很小，暂不作通过判断，
必须在后续长保持课程中重新统计。

## 6. 晋级决定

- 旧锚点继续作为“覆盖较广、guide 主导”的回归基线；
- 新候选作为“PPO 贡献可测、共同接触更持久”的人工动作检查候选；
- 当前只满足 64–70 mm 的 provisional 方向门，尚未满足 `theta_opp≤35°`、高覆盖率、
  断梗后 3–6 s 保持、扰动恢复和精确掌面碰撞硬门；
- 在人工确认物理动作之前，不启动长时间成功率训练、不扩果径、不进入实机。

## 7. 人工可视化验收命令

该命令直接运行训练结果权重，在 64–70 mm 随机尺寸环境中逐回合推理，不循环播放预选录像；
GUI 场景只含独立左手、果实和地面：

```bash
cd /home/miobarnacle/linkbot/linkagent
PYTHONPATH=/home/miobarnacle/linkbot/linkagent \
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
research/small_fruit_pinch_v1/source/native_evaluate.py \
  --checkpoint logs/rsl_rl/small_fruit_pinch_v1/left_hand_physical_direction_probe_20260908_residual16_24_64_70/model_final.pt \
  --num-envs 1 --episodes 4 --seed 20260911 --guided-residual \
  --guide-action-values 0.5563,0.3046,-0.4787,0.1579,0.3133,0.6795,0.1310,-0.1900,0.2070,0.5084,-0.2590,-0.6060,-0.1727,0.0229 \
  --residual-scale 0.16 --distal-residual-scale 0.24 \
  --precontact-residual-fraction 0.15 --explicit-hand-pd \
  --center-local-mm -52.42 -16.09 -177.48 --palm-fruit-collision-filter \
  --hand-only-view --visualization-delay-ms 25 --viz kit \
  --experience /home/miobarnacle/linkbot/linkagent/research/small_fruit_pinch_v1/isaaclab.python.gui.nodbg.kit
```

人工检查重点：拇指是否先横摆到食指/中指对侧、食指和中指是否随后闭合、三指共同接触后
是否继续小幅调整、环指/小指是否保持开位，以及是否存在明显抽搐、穿透或果实异常弹射。
