# 固定顶置相机采摘闭环的 Isaac 验收

## 验收对象

仿真使用一台固定安装在双臂中间顶部、斜向前下看的 RGB-D 相机：

- world 安装位置：`(0.02, -0.05, 2.38) m`；
- world 注视点：`(0.43, -0.08, 1.76) m`；
- calibration ID：`isaac_top_rgbd_v1`；
- device ID：`isaac-top-rgbd-01`。

`active_view_reposition` 不移动相机。它只把机械臂移动到审核过的有限去遮挡位，再从同一
固定外参抓取严格更新的 RGB-D 帧。验收必须同时证明相机位姿零漂移、图像因机械臂运动发生
变化、目标 track 保持不变。

## 权威边界

- Ark 临时替代 YOLO，只输出二维 box、点和语义分数；
- 果实三维中心、尺寸和方向来自对齐深度、内参与固定相机外参；
- `IsaacTrainingDistributionGraspPlanner` 使用 Clean-v5 reset 的真实末端参考位姿和 grasp basis；
- model1350 的 66D actor、11D action、checkpoint、动作缩放、残差平移/旋转限制和 15 N
  仿真终止阈值保持不变；
- 采后撤退、运输、投放和复位通过独立 Cartesian 原子控制口执行，不扩大 PPO residual envelope；
- 完成判定来自 PhysX 果实刚体、五指接触、手部关节、容器碰撞体和独立 verifier，不接受
  脚本文本或 mock 状态作为替代。

## 运行命令

使用项目记录的 Isaac Lab 环境：

```bash
export LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1
export ISAAC_PYTHON="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
```

每次运行使用新的输出目录，避免把旧 Ark 帧或旧报告解释成当前代码证据。

固定相机、真实机械臂 USD、RGB-D 和小幅 MoveL 探针：

```bash
"$ISAAC_PYTHON" script/isaac_harvest_cycle_probe.py \
  --headless \
  --output artifacts/simulation/harvest_cycle_acceptance/probe-current.json
```

Ark box 与对齐深度三维几何：

```bash
"$ISAAC_PYTHON" script/isaac_harvest_perception_acceptance.py \
  --headless \
  --output-root artifacts/simulation/harvest_cycle_acceptance/perception-current
```

单场景完整闭环：自检、Ark 识别、去遮挡、重观察、点云净空、训练分布准备位、冻结 PPO、
持果撤退、运输、投放、独立放置验证和下一目标复位：

```bash
"$ISAAC_PYTHON" script/isaac_harvest_pre_pick_acceptance.py \
  --headless \
  --full-cycle \
  --output-root artifacts/simulation/harvest_cycle_acceptance/full-cycle-current
```

不分配 RGB-D renderer、只隔离验证正式 PPO 与采后原子技能：

```bash
"$ISAAC_PYTHON" script/isaac_tomato_pick_runner_acceptance.py \
  --headless \
  --full-cycle \
  --output artifacts/simulation/harvest_cycle_acceptance/ppo-post-current/acceptance.json
```

物理掉果失败注入：冻结 PPO 成功持果后真实打开 G20，要求 PhysX 报告掉落、
`payload_retreat` 在发送机械臂命令前中止，并由掉果观察端口完成定位和处置分类：

```bash
"$ISAAC_PYTHON" script/isaac_tomato_pick_runner_acceptance.py \
  --headless \
  --inject-drop-before-retreat \
  --output artifacts/simulation/harvest_cycle_acceptance/drop-recovery-current/acceptance.json
```

该失败注入只有在末端零运动误差不超过 4 mm、失败码为 `payload_dropped`、定位成功、
`motion_authorized=false` 且 `frozen_hanging_pick_ppo_reuse_allowed=false` 时才通过。

## 通过条件

完整闭环报告只有同时满足下列条件才允许 `passed=true`：

1. 两次 Ark/RGB-D 观测使用同一 track，第二帧严格更新，固定相机 pose 漂移不超过 `1e-6`；
2. 三维目标通过尺寸、置信度、可见度、workspace、点云净空和训练分布准备位门禁；
3. 冻结 model1350 的独立 pick verifier 确认 detached、retained、未掉落；
4. 回撤与运输各段均保持至少两指 `>=0.60 N` 的仿真接触；
5. 释放后果实刚体位于容器碰撞体范围，手部为空；
6. 机械臂返回 reset reference，且本次结果没有复用旧报告。

仓库内已有的 `perception-top`、`pre-pick-top6` 等目录是开发过程证据，不自动代表当前 HEAD
通过。修改相机、几何、规划、PPO adapter 或采后技能后，必须按上述命令写入新的输出目录。
