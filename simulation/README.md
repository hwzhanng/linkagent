# C2 仿真与 PPO 训练环境

本目录与 Agent 主运行环境隔离。导入 `module`、`task` 或默认 CLI 时，不会导入这里，
也不会加载 Isaac Lab、Gymnasium、Torch、RSL-RL 或设备 SDK。

当前建立了三种运行角色：

1. `agent_core`：现有 Agent/Task/Mock，继续保持 Python 3.11+ 标准库边界；
2. `contract_smoke`：运行训练接口、结果/终止和诊断标签的无物理门禁；
3. `isaac_ppo`：唯一允许加载 Isaac Lab、GPU 和 RSL-RL 的仿真/训练进程；主运行时已
   独立安装，DirectRLEnv 已完成 GPU 冒烟与正式 PPO 训练。

`contract_smoke` 不是仿真物理证据，也不能生成可部署 checkpoint。它的价值是在番茄、
果梗和指尖碰撞资产确定前，把环境 API、阶段状态、一次性断梗事件、成功不变量和训练
配置固定下来。

## 当前可执行命令

```bash
python3.12 -m script.sim_env preflight --role contract_smoke
python3.12 -m script.sim_env smoke --episodes 100 --seed 0
python3.12 -m script.sim_env preflight --role isaac_ppo
python3.12 -m script.train_tomato_ppo
```

前两条当前应通过。第三条未提供主路径时会列出路径阻塞；使用下方已安装路径后，运行时、
任务资产和 adapter 检查通过；PPO 启动门禁以 6D/16 关节物理 smoke、奖励审计和优化器
短探针为准。

使用已安装的主运行时：

```bash
export LINKAGENT_ISAACLAB_ROOT="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
python3.12 -m script.sim_env preflight --role isaac_ppo
unset LINKAGENT_ISAACLAB_ROOT
```

实际训练必须由 `${LINKAGENT_ISAACLAB_ROOT}/isaaclab.sh train --rl_library rsl_rl`
启动，不能用系统 Python、Agent Python 或 SDK gateway Python 直接启动。预检不会创建/修改 Conda 环境，不执行
`pip install`，不下载资产，不访问 CAN，也不读取模型 API 凭据。

在它自己的解释器中重复运行时版本和 GPU 探针：

```bash
${LINKAGENT_ISAACLAB_ROOT}/isaaclab.sh -p \
  -m script.isaac_runtime_probe --with-gpu
```

运行已实现的任务环境物理冒烟（不启动 PPO）：

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/descfly/linkagent \
  ${LINKAGENT_ISAACLAB_ROOT}/env_isaaclab/bin/python \
  /home/descfly/linkagent/script/isaac_tomato_pick_env_smoke.py \
  --num-envs 4 --steps 16 --policy zero --viz none
```

当前该命令能创建/重置/推进环境，并验证 actor `66D`、critic `83D`、action `11D`、
张量有限性及尺寸—密度—质量相关采样。它不是四阶段成功证明。

`python3.12 -m script.train_tomato_ppo` 默认只打印解析后的训练计划。即使以后所有门禁
打开，仍必须显式加 `--execute` 才会创建训练进程。执行器不使用 shell，并从子进程
环境移除常见模型 API key；它不安装包、不自动下载资产、不连接机器人。

## 资产边界

- 机器人资产沿用 C2-0 选定的 `linkbot_bimanual_sdk_training_mass1kg`；
- 番茄椭球和向量化 Kelvin–Voigt 果梗已实现并通过短物理冒烟；
- G20 精确指尖 collider/contact reporter overlay 与自碰撞 pose sweep 仍是训练前阻塞项；
- `simulation/config/assets_v1.yaml` 已通过物理、奖励和优化器启动门禁；
- 旧 Fruit-Grasp 代码、reward、27D action、checkpoint 和训练产物不进入本目录。

## 后续物理环境落点

`simulation/tomato_pick/isaac/` 已包含 `DirectRLEnv`、批量 66D actor observation、
83D privileged critic、11D 连续动作、可断果梗、结构化 episode metrics 和
RSL-RL 注册。6D 末端 IK、16 个独立 G20 关节目标、接触 reporter、奖励审计和两次
PPO 优化更新均已通过。完整路线见
[`C2_SIM2REAL_TRAINING_PREP.md`](../docs/tomato_pick/C2_SIM2REAL_TRAINING_PREP.md)。

正式模型 1450 的环境、评估和续训命令见
[`PPO_MODEL1450_REPRODUCTION.md`](../docs/tomato_pick/PPO_MODEL1450_REPRODUCTION.md)。
