# LinkBot 本机机械臂控制

本目录只保留当前仍在使用的双臂控制实现。机械臂为左右各 7 关节，电脑通过
USB-CAN 直接连接设备；运行时不依赖树莓派、网络控制器或旧版 LBot 网络 SDK。

## 唯一控制入口

`tools/local_arm_can/` 是机械臂控制的唯一入口：

```text
Agent / Skill / ROS 2 客户端
        ↓
tools/local_arm_can/ros_node.py
        ↓
advanced_controller.py
        ├── kinematics.py：FK / IK
        └── trajectory.py：MoveJ / MoveL 轨迹
        ↓
service.py + protocol.py + socketcan.py
        ↓
SocketCAN → USB-CAN → 7 关节机械臂
```

当前已经实现并完成真机验证：

- 双臂状态读取与软限位；
- FK、带种子和关节限位的 IK；
- MoveJ、MoveJP、MoveL、FollowP；
- 反馈看门狗、抢占、软件急停和控制权门禁。

MoveC 不可用。当前控制器不包含环境碰撞检测或绕障规划，因此 IK 有解不代表整条
路径无碰撞。

## 保留内容

- `tools/local_arm_can/`：控制源码、ROS 2 接口源码和回归测试；
- `tools/isaac/`：正式双臂资产的生成、检查与仿真准入工具；
- `docs/本机直连机械臂高级控制器.md`：真机验证记录。

ROS 2 的 `build/`、`install/` 和 `log/` 是可重建产物，不纳入工作区。首次使用先执行：

```bash
cd /home/descfly/linkagent/botSDK/lbot_arm_api-master
tools/local_arm_can/build_ros_interfaces.sh
python3 -m unittest discover -s tools/local_arm_can/tests -v
```

详细启动、安全门禁和接口说明见
[`tools/local_arm_can/README.md`](tools/local_arm_can/README.md)。

## 已移除的历史链路

以下内容已经弃用并从工作区清除：

- Fruit-Grasp PPO baseline、训练日志与 checkpoint；
- 树莓派机械臂快照、网络 SDK、控制器固件和旧 ROS 工作区；
- ROSBridge 示教轨迹、WD40/水瓶演示和动作编排器；
- 旧网页控制台以及通过树莓派转发的右手控制脚本。

新的 PPO/Skill 实现不得依赖上述历史链路，应通过冻结的 Skill 接口调用当前机械臂
控制与 G20 灵巧手能力。
