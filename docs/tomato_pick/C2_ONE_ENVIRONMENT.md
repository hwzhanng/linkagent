# C2-1 环境骨架与 C2-2 物理实现进展

状态：C2-1 完成；C2-2 DirectRLEnv 主体完成，规则策略验收进行中  
日期：2026-08-04

## 1. 当前有两级环境证据

第一级是无物理契约环境：

```text
114D observation → 规则策略 → 7D action
```

它连续 100 个 seed 完成 `CONTACT → GRASP → DETACH → HOLD`，验证 observation/action
布局、四阶段、一次性断梗事件、成功不变量和失败分类。它始终标记
`physics_valid=false`，不能生成 checkpoint。

第二级是已经实现的 Isaac Lab `LinkAgent-TomatoPick-Direct-v0`：

- Sim 6 / RTX 5090 / CUDA PhysX；
- 正式双臂/双 G20 USD，首版激活右臂；
- 120 Hz physics、20 Hz policy；
- 114D actor、33D privileged critic、7D action；
- 相关采样三轴番茄、可一次性断裂的向量化果梗；
- batch reset、reward、termination、metrics 和 RSL-RL 注册。

4 环境、16 步 CUDA reset/step 冒烟通过，PPO 没有启动。规则策略仍会因临时关节反力
触觉代理出现假过力终止，因此物理四阶段与 100-seed 验收尚未通过。

## 2. 三套环境边界

|角色|用途|状态|
|---|---|---|
|`agent_core`|Agent、Task、Ark adapter、离线 mock|不导入 Isaac/Torch|
|`contract_smoke`|114D/7D 无物理回归|已通过 100 seed|
|`isaac_ppo`|Isaac 物理环境与以后训练|环境可启动；PPO gate 关闭|

机器配置：

- `simulation/environments/runtime_manifest.yaml`：角色、权限和任务注册回调；
- `simulation/environments/isaaclab_runtime_v1.yaml`：本机版本/GPU/robot smoke；
- `simulation/config/assets_v1.yaml`：资产和 physics/PPO 双门禁；
- `simulation/config/sim2real_v1.yaml`：部署一致观测、随机化和课程；
- `simulation/config/ppo_v1.yaml`：唯一 PPO 初始配置，状态仍为 blocked。

预检/启动器不会自动安装依赖、下载资产、读取模型凭据或访问 CAN。真正训练只能通过
Isaac Lab launcher，并且即使门禁以后打开也必须显式 `--execute`。

## 3. 冻结的 PPO 边界

- Actor 114D，最后 3 维为 RGB-D 椭球拟合得到的番茄 `x/y/z` 半轴归一化值；
- Actor 只能使用真机可获得的测量；精确质量、断裂阈值和约束状态只允许给 critic；
- action 7D：抓取坐标系末端残差 4 + closure/opposition/spread 3；
- 预抓取由规划器和 MoveJP/MoveL 完成，PPO 只处理接触附近局部动作；
- 仿真用 differential IK，真机用 bounded FollowP，共享 action scale 和安全 supervisor；
- checkpoint 必须携带接口、归一化、mapper、资产和 sim2real manifest；
- 旧 Fruit-Grasp 的 27D action、reward、checkpoint 和成功定义不得复用。

## 4. 当前验证命令

```bash
python3.12 -m script.sim_env preflight --role contract_smoke
python3.12 -m script.sim_env smoke --episodes 100 --seed 0

export LINKAGENT_ISAACLAB_ROOT="/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1"
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/descfly/linkagent \
  ${LINKAGENT_ISAACLAB_ROOT}/env_isaaclab/bin/python \
  /home/descfly/linkagent/script/isaac_tomato_pick_env_smoke.py \
  --num-envs 4 --steps 16 --policy zero --viz none
python3.12 -m script.train_tomato_ppo
unset LINKAGENT_ISAACLAB_ROOT
```

前三项分别验证契约、无物理规则闭环和物理 reset/step；最后一项必须返回 blocked，且
不能创建训练进程。

## 5. 下一步只有一条主线

1. 为正式 G20 生成指尖 collider/contact reporter overlay，不修改源 USD；
2. 做 open/mid/closed pose sweep 和自碰撞 pair 过滤；
3. 替换关节反力代理，跑通单环境物理规则策略；
4. 逐级验证 1/16/128/512 环境和 100-seed 物理规则策略；
5. 用本项目拉/弯/扭测量收敛果梗参数；
6. 通过后进入 C2-3，才启动 PPO。

完整 sim2real 与 to-real 门禁见
[C2_SIM2REAL_TRAINING_PREP.md](./C2_SIM2REAL_TRAINING_PREP.md)。
