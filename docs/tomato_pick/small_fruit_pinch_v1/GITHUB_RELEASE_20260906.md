# Small-fruit Pinch-v1 GitHub 发布说明

日期：2026-09-06（Asia/Shanghai）  
发布范围：左手、三指（拇指/食指/中指）小果捏取仿真；Isaac Lab 3.0 / PhysX  
硬件状态：`hardware_enabled=false`，不会连接 CAN、厂商网关或实机

## 交付内容

- 训练/评估源码：[`research/small_fruit_pinch_v1/`](../../../research/small_fruit_pinch_v1/)
- 训练计划、过程日志与验收报告：本目录下的 `SMALL_FRUIT_PINCH_PPO_OPTIMIZATION_PLAN_20260831.md`、
  `TRAINING_LOG_20260905.md`、`FINAL_QUALIFICATION_20260905.md` 和资格清单 JSON。
- 仅左手 Isaac Sim GUI 兼容 experience：
  [`isaaclab.python.gui.nodbg.kit`](../../../research/small_fruit_pinch_v1/isaaclab.python.gui.nodbg.kit)
- 选定权重（Git LFS）：
  [`model_150.pt`](../../../logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/model_150.pt)
- 逐种子严格评测原始文本、逐关节保持段证据和远端关节冻结消融证据，位于同一训练目录的
  `evaluation/` 子目录。

## 选定权重与验收摘要

```text
相对路径：
logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/model_150.pt
SHA-256：e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d
```

固定 64–70 mm 等效果径域、三种独立评测种子下，平均严格成功率为 99.61%，最差种子为
99.41%；掉落率为 0，最大过载率为 0.39%，自碰撞率为 0。保持段的最大 P95 速度、加速度、
jerk 和高频目标能量分别为 0.179846、1.842477、66.511581 和 0.045854。

这些数字来自独立 Verifier 的最终成功计数，不是训练面板回报，也不代表番茄真实物性、
断梗动作或 sim2real 已完成。

## 仅左手可视化

GUI 仅隐藏显示图元，不改变 PhysX 场景、关节动作合同或触觉/接触计算：完整 USD 仍作为
仿真载体，视图中只保留左手和果实，右臂、右手以及左臂外部结构被隐藏。启动时使用：

```bash
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  "$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python" \
  research/small_fruit_pinch_v1/source/native_evaluate.py \
  --viz kit \
  --experience research/small_fruit_pinch_v1/isaaclab.python.gui.nodbg.kit \
  --device cuda:0 \
  --checkpoint logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/model_150.pt \
  --num-envs 1 --episodes 1000000 --seed 20260902 \
  --fruit-axis-min-mm 32 --fruit-axis-max-mm 35 \
  --guided-residual --geometry-prior --geometry-prior-scale .15 \
  --force-servo --force-servo-seek-gain .04 --force-servo-target-gain 0 \
  --force-servo-retract-gain .35 --force-servo-alpha .30 \
  --force-servo-max-offset .25 --force-servo-warning-fraction .78 \
  --direct-force-projection --direct-force-warning-fraction .50 \
  --direct-force-high-gain-by-digit .80,.40,1.00 \
  --direct-force-low-gain .06 --direct-force-alpha .10 \
  --direct-force-max-open .50 --direct-force-root-fraction .45 \
  --force-gate-start-fraction .60 --precontact-residual-fraction 0 \
  --residual-scale .04 --distal-residual-scale .06 \
  --residual-require-all-contact --residual-trust-region \
  --residual-trust-warning-fraction .55 --residual-trust-in-band-scale .05 \
  --hold-speed .08 --hold-acceleration .7 --natural-frequency 3.0 \
  --offset-x-mm 20 --offset-y-mm 30 --offset-z-mm 24 \
  --fruit-contact-offset .001 --fruit-rest-offset .0001 \
  --robot-max-depenetration-velocity .3 --fruit-max-depenetration-velocity .3 \
  --hold-joint-target-anchor-fraction .40 \
  --hold-hand-stiffness-by-joint 2,12,12,12,12,2,12,12,12,12,2,2,4,4,12,12 \
  --hold-hand-damping-by-joint 3,.6,.6,3,3,3,.6,.6,3,3,3,.6,.6,.6,3,3 \
  --manual-decimation-control --position-target-interpolation 6 \
  --hand-only-view --visualization-delay-ms 50
```

该命令只启动仿真回放。不要把 `--hand-only-view` 误解为实机授权；实机控制仍需单独的
现场安全确认、配置和环境变量放行。

## 未纳入本次发布的内容

- Isaac Lab 上游源码、Omniverse 缓存、TensorBoard/W&B 运行时目录和 Python 缓存；
- 大果掌托包络 v3 的研究分支；它保留在工作区，避免与本次小果捏取合同混淆；
- 真实触觉标定、番茄非胡克物性、断梗动力学、sim2real 和任何实机控制命令。

