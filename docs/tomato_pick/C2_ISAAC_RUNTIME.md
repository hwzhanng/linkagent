# C2 Isaac Sim / Isaac Lab 运行环境

状态：已安装；正式机器人 USD 与番茄 DirectRLEnv CUDA smoke 均已完成  
日期：2026-08-04

## 1. 结论

C2 的主运行时已经固定为：

|项目|版本或位置|
|---|---|
|Isaac Sim|`6.0.1.0`|
|Isaac Lab|源码版本 `6.1.14`，tag `v3.0.0-beta2.patch1`|
|Python|`3.12.3`|
|Torch|`2.10.0+cu128`|
|RSL-RL|`5.0.1`|
|GPU|RTX 5090，CUDA capability `12.0`|
|源码根目录|`/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1`|
|隔离环境|`/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab`|

选择 Sim 6 不是追新。正式
`linkbot_bimanual_sdk_training_mass1kg.usda` 由 Sim 6 工具链生成，包含 Sim 6 的
Newton mimic 属性；Sim 5.1 虽能运行普通 Isaac Lab smoke，但加载该资产时 articulation
拓扑失败。Sim 6.0.1 已实际解析出固定底座、71 个刚体、56 个独立关节和 10 个已
author 的 mimic 关系。正式 USD 没有被修改。

机器可读记录见
[`simulation/environments/isaaclab_runtime_v1.yaml`](../../simulation/environments/isaaclab_runtime_v1.yaml)。

## 2. 环境隔离

- Agent 默认 Python 环境不导入 Isaac、Torch 或 RSL-RL；
- Isaac/PPO 运行时是独立 uv venv，不修改 `base`、`trex` 或其他 Conda 环境；
- 安装和 smoke 没有导入 vendor SDK、连接 CAN、读取机器人或使用模型 API key；
- `OMNI_KIT_ACCEPT_EULA=YES` 只在用户明确授权后的安装和 smoke 进程中设置，没有写入
  shell profile 或全局环境；
- Sim 5.1 / Isaac Lab 2.3.2 环境保留作框架参考，禁止用于 C2 资产、环境或训练。

两次因缺少 Git LFS 而中止的源码检出已移到两个可恢复目录，没有被删除，也不会被
launcher 使用：

```text
/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1.incomplete-20260804
/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1.incomplete-lfs-20260804
```

## 3. 已完成验证

1. Torch 在 RTX 5090 上分配 CUDA tensor 并完成计算；
2. Isaac `AppLauncher` 无界面启动；
3. `SimulationContext` 在 `cuda:0` reset，并以 120 Hz 运行 PhysX；
4. 正式双臂/G20 USD 运行 8 个物理步，无初始化错误；
5. 实际读取到 71 bodies、56 independent joints、fixed base；
6. 双臂 `L1..L7` / `R1..R7` 和左右共 10 个指尖 link 全部存在；
7. 资产 payload 静态包含 10 个 mimic relation。

可重复的综合 smoke：

```bash
cd /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1
source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
  /home/descfly/linkagent/script/isaac_runtime_smoke.py \
  --viz none --steps 16
deactivate
```

成功时输出以 `ISAAC_RUNTIME_SMOKE_REPORT=` 开头的 JSON，并以返回码 0 退出。

## 4. 依赖健康说明

官方安装器返回 0，真实 App/PhysX/robot smoke 通过，但 `pip check` 不是 clean。原因是该
beta tag 与 Sim 6.0.1 的发布元数据存在互相不可同时满足的精确约束：

- Isaac Lab 安装器明确选择 Torch 2.10 cu128，Sim core 元数据请求 Torch 2.11；
- `isaaclab-rl` 请求 `packaging<24`，Sim core 元数据请求 `packaging==26`；
- Isaac Lab 请求 Pillow 12.2，MoviePy 2.2.1 请求 Pillow `<12`；
- Sim kernel 与 Isaac Lab 对 NumPy、coverage 的精确版本不同。

现在不强行逐项“修绿” `pip check`，因为那会破坏官方安装器为 Isaac Lab 选择的训练
版本，且真实运行已经通过。升级规则是：只在独立新环境中尝试，重新运行 GPU、PhysX、
正式 USD 和后续 C2 环境回归后，才切换主运行时。Isaac Lab 官方 changelog 也明确记录
了 Sim 6.0.1 依赖和运行时冲突修复：
[Isaac Lab changelog](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/refs/changelog.html)。

## 5. 当前门禁

“运行时和环境可启动”不等于“PPO 可开始”。当前已打开
`isaac_runtime_preflight_passed`、`task_assets_ready_for_physics` 和
`direct_rl_env_cuda_smoke_passed`。下列门禁继续关闭：

- 指尖 collider/contact reporter overlay 与自碰撞 pose sweep 尚未完成；
- 关节反力临时触觉代理仍会造成假过力终止；
- 物理规则策略名义闭环与 100-seed smoke 尚未通过；
- PPO profile 状态仍是 `blocked_until_physics_rule_policy_acceptance`。

环境实现、实际冒烟命令和 sim2real 门禁见
[C2_SIM2REAL_TRAINING_PREP.md](./C2_SIM2REAL_TRAINING_PREP.md)。
