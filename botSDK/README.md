# botSDK 核心目录

该目录是当前农业机器人下游设备能力的唯一硬件资料区，已经移除旧 PPO、树莓派
机械臂链路、重复模型和历史演示。

## 保留结构

| 目录 | 用途 |
|---|---|
| `lbot_arm_api-master/tools/local_arm_can/` | 双臂 7 关节本地 USB-CAN 控制、FK/IK 和高级运动 |
| `linkerhand-ros2-sdk-main/` | LinkerHand 官方 ROS 2 SDK，只作为上游依赖 |
| `linkerhand-control-workspace/` | G20 直接控制、压力读取、标定、配置与实验记录 |
| `urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/` | 唯一正式双臂双手 URDF |
| `urdf/02_project_derived/isaac_assets/` | 两套正式 Isaac Sim 双臂训练资产 |
| `urdf/02_project_derived/simulation_baselines/tomato_pick_v1.yaml` | 当前 C2 番茄/果梗仿真唯一机器规范 |
| `urdf/01_vendor_original/` | 模型来源和厂商原始资产 |
| `INTEGRATION_MANIFEST.json` | 唯一接入清单：正式路径、设备形状、能力和待真机验证项 |

## 接入原则

- Agent/Task 层不直接调用底层 CAN 或 ROS SDK；
- 机械臂通过冻结的 Arm/EndEffector Tool 接口调用 `local_arm_can`；
- PPO 通过冻结的 Pick Skill 接口接收观测并输出归一化动作；
- G20 通过灵巧手适配层调用官方 SDK，不再使用树莓派转发；
- 生成目录、虚拟环境、checkpoint 和训练日志不得提交到 `botSDK`。
- RealAgent adapter 放在 `module/robotics/adapters/`，不复制或改写 vendor SDK 到应用层；
- 真机部署参数放在仓库根 `config/`，CAN 拓扑变更时只改部署配置与验收记录。
- 名义 `20 N` 量程不是触觉 raw→N 换算；逐指标定前 gateway 的 N 通道必须无效。

设备控制和模型入口分别见：

- [`lbot_arm_api-master/README.md`](lbot_arm_api-master/README.md)
- [`linkerhand-control-workspace/README.md`](linkerhand-control-workspace/README.md)
- [`urdf/02_project_derived/README.md`](urdf/02_project_derived/README.md)
