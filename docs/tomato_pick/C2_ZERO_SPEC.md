# C2-0 番茄仿真开工规范

状态：完成  
日期：2026-08-04  
机器可读规范：
[`tomato_pick_v1.yaml`](../../botSDK/urdf/02_project_derived/simulation_baselines/tomato_pick_v1.yaml)

## 1. C2-0 解决了什么

C2-0 不实现 Isaac 环境，也不训练 PPO。它把 C1 Skill、正式机器人模型、G20 触觉和
未来实机 adapter 之间容易漂移的事实收敛成一个可自动检查的入口。

冻结结果：

- 正式安装方向为右手 `rpy=(π,0,+π/2)`、左手 `rpy=(π,0,-π/2)`；
- actor 使用 114D observation、7D action、20 Hz 和四个固定 phase；新增的 3 维是
  可由 RGB-D 目标几何提供的番茄 `x/y/z` 半轴尺寸归一化值；
- 预抓取由 planner + MoveJP/MoveL 完成，PPO 只输出局部末端残差和三个手部协同量；
- 当前实机已验证的触觉是五指 `12×6`、`0..255` 的无量纲矩阵权重；
- 名义 `20 N` 量程不是 raw→N 标定曲线；逐指标定前真实 gateway 必须令
  `force_valid=false` 且 N 值为空；
- C2 可使用明确标记的仿真合力代理，但不得把它写成实机标定；
- 右 G20 直连/触觉和双臂双手同时拓扑仍为待验收；
- 旧 `agricultural_grasp_no_equipment_proxy` 只保留物理先验，不再是训练入口。

## 2. C2 环境边界

C2 的任务是单个带梗番茄的：

```text
CONTACT -> GRASP -> DETACH -> HOLD
```

果梗必须使用一次性断裂约束。成功同时要求：果梗约束已断、果实仍稳定持有、没有
滑落/掉落且 Verifier 有效。critic 可以读取精确接触、果梗约束和随机化参数；actor
不能读取这些仿真私有真值。

首版允许使用宽工程先验建立环境，但这些范围只用于 domain randomization，不是
真机参数声明。正式 PPO 训练必须等规则策略完成名义四阶段闭环，并通过 100 seed
reset/step 稳定性检查。

首版物理频率固定 120 Hz、策略 decimation 为 6，即 20 Hz；phase 上限暂定
`30/30/40/20` 步，动作尺度与 C1 mock 起点一致。reward 采用按 phase 开启的进度项，
加一次性断梗和最终任务奖励；滑移、掉落、过力、越界和动作变化率为惩罚。精确权重
属于版本化环境配置，可以调参，但不得改变“未断梗或已掉落绝不成功”的终止语义。

## 3. 三个运行环境

- Agent/Task：Python 3.11+，默认不导入 vendor SDK、Isaac 或 torch；
- SDK gateway：独立进程/环境，以 Python 3.12 为当前参考，C2 期间只做离线或只读；
- Isaac/PPO：使用 Isaac 自带 Python，不连接真实 CAN，也不导入模型供应商 adapter。

跨进程只交换冻结 DTO/Port 数据，不能传 SDK client、Isaac 对象或 torch tensor 到
Agent/Task。

## 4. C2-0 定义的下一验收门

C2-1 已完成无物理的 114D/7D 契约环境、运行环境隔离、规则策略 100-seed smoke 和
受控 PPO 启动计划；详见
[`C2_ONE_ENVIRONMENT.md`](./C2_ONE_ENVIRONMENT.md)。以下 1～3、5～7 项仍必须在
Isaac 物理环境中重新验收，不能用 contract smoke 代替。

下一里程碑不是“PPO 开始训练”，而是“规则策略环境验收通过”：

1. 正式 `mass1kg` 资产能被新环境加载；
2. 指尖低面数碰撞体和自碰撞过滤通过接触 smoke；
3. 番茄/果梗可以 reset，断裂事件只发生一次；
4. C1 observation/action wrapper 逐值匹配 114D/7D manifest；
5. 规则策略完成接触、抓取、拉脱和保持；
6. 未断梗、滑移或掉落永远不能成功；
7. 100 个随机种子无 NaN、爆炸或无界动作。

静态准入检查入口：

```bash
python3.12 botSDK/lbot_arm_api-master/tools/isaac/check_agricultural_grasp_entry.py
```

该检查不启动 Isaac、不连接 ROS/CAN，也不需要硬件。
