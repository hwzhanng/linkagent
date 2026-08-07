# 本机直连机械臂 SDK、高层运动库与本地服务

## 1. 当前结论

这是当前机械臂的本机控制主链路：电脑通过 USB-CAN 直接连接机械臂，在电脑上运行
CAN HAL、状态服务、FK/IK 和高层运动控制。运行时不需要树莓派转发，也不访问
`192.168.10.21:8000`。

```text
Python / VLA / ROS 2 客户端
        ↓
ros_node.py（本地 ROS 2 服务）
        ↓
advanced_controller.py（高层运动）
        ├── kinematics.py（FK / IK）
        └── trajectory.py（MoveJ / MoveL 轨迹）
        ↓
service.py（CAN HAL、安全门禁、反馈看门狗）
        ↓
protocol.py + socketcan.py
        ↓
SocketCAN canX → USB-CAN → 机械臂
```

右臂已经在电脑 `can1`、1 Mbit/s、CAN ID `0x33～0x39` 上完成真机验收。USB
重新插拔后 `canX` 序号可能变化，每次启动都必须按 CAN ID 和实物线缆重新确认。

本目录只负责机械臂，不包含或管理末端灵巧手。

## 2. SDK 和高层运动库位置

| 文件/目录 | 层级 | 用途 |
|---|---|---|
| `protocol.py` | CAN 协议 | 经典电机与 Motorevo 帧编解码 |
| `socketcan.py` | CAN 传输 | SocketCAN 发现、配置、收发和 `can0` 保护 |
| `service.py` | 本地 CAN HAL | 双臂通用运行时、HTTP/SSE、遥测、使能、软限位和反馈看门狗 |
| `kinematics.py` | **高层运动库** | 双臂精确 FK、带软限位和种子的 IK、基座/法兰偏移 |
| `trajectory.py` | **高层运动库** | MoveJ、MoveL、五次平滑轨迹和路径采样 |
| `advanced_controller.py` | **高层运动库** | MoveJ、MoveJP、MoveL、FollowP、抢占和安全调度 |
| `ros_node.py` | **本地服务入口** | ROS 2 状态话题、FK/IK/Move* 与安全服务 |
| `ros_ws/src/lbot_arm_interfaces/` | ROS 2 SDK | 从控制器接口恢复并可在 x86_64 构建的消息和服务定义 |
| `a7_v2_left.json` / `a7_v2_right.json` | 配置 | 关节软限位、模型和 Motorevo 参数 |
| `commission.py` | 验收工具 | 状态、失能、回零和逐关节小范围验收 |
| `tests/` | 回归测试 | 协议、FK/IK、轨迹和安全门禁测试 |

旧版网络 SDK 已清理。当前 USB-CAN 直连路径只使用本目录。

## 3. 已实现和已验收能力

ROS 2 右臂命名空间为 `/right`，左臂对应 `/left`：

| 类别 | 接口 | 状态 |
|---|---|---|
| 状态 | `arm_state`、`robot_run_state`、`joint_states` | 已实现 |
| 计算 | `fk`、`ik`、`get_joint_limit` | 已实现并验收 |
| 关节运动 | `movej` | 七关节 20°往返已验收 |
| 位姿运动 | `movejp` | 已真机验收 |
| 直线运动 | `movel` | 10 mm 往返已真机验收 |
| 单点跟随 | `followp` | 已真机验收；不做轨迹插值 |
| 圆弧运动 | `movec` | 原厂当前版本是失败空桩，本机保持相同行为 |
| 安全 | `enable_arm`、`emergency_stop` | 已真机验收 |
| 直接话题 | `joint_cmd`、`teach_cmd` | 已实现安全门禁；高频使用需上层控制权互锁 |

规划层保留原厂 100 Hz 时间律；直连电机的 CAN 位置帧按真机验证结果安全限频为
50 Hz，并始终发送精确终点。七个关节以 50 Hz 主动查询状态，反馈超过 250 ms
时运动会中止，不通过放宽门限掩盖掉帧。

这里没有 MoveIt、环境碰撞检测或绕障规划。IK 成功只说明目标可解，不说明路径没有
碰撞；VLA 输出仍需工作空间、速度、单步变化、超时和控制权安全层。

## 4. 首次构建

```bash
cd /home/descfly/linkagent/botSDK/lbot_arm_api-master
tools/local_arm_can/build_ros_interfaces.sh
python3 -m unittest discover -s tools/local_arm_can/tests -v
```

当前依赖 ROS 2 Jazzy、Python 3、SciPy、SocketCAN 和 `can-utils`。

## 5. 核对并配置 USB-CAN

先查看适配器和接口，不发送运动命令：

```bash
cd /home/descfly/linkagent/botSDK/lbot_arm_api-master
python3 -m tools.local_arm_can.service doctor
ip -details -statistics link show can1
```

确认 `can1` 确实连接右机械臂后配置 1 Mbit/s：

```bash
sudo python3 -m tools.local_arm_can.service configure can1
```

不要在没有核对实物时配置 `can0`。底层默认保护 `can0`；双臂脚本会显式解除该
保护，因此更需要现场确认 `can0` 当前确实属于机械臂，而不是其他 CAN 设备。

## 6. 本地 ROS 2 服务启动方式

每个终端先加载环境。域 `121` 与树莓派域隔离，`LOCALHOST` 禁止服务泄漏到局域网：

```bash
cd /home/descfly/linkagent/botSDK/lbot_arm_api-master
source /opt/ros/jazzy/setup.bash
source tools/local_arm_can/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=121
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

### 6.1 纯只读计算

```bash
python3 tools/local_arm_can/ros_node.py --interface can1 --arm right
```

该模式不发 CAN，可调用 FK/IK；由于 J6/J7 需要主机查询，不能把它作为持续真机状态
服务。

### 6.2 遥测待机（推荐常驻形态）

```bash
python3 tools/local_arm_can/ros_node.py \
  --interface can1 --arm right \
  --allow-tx --poll-telemetry
```

此模式只允许必要的状态查询和自动上报配置。因为没有
`--physical-confirmation RIGHT_ARM_AREA_CLEAR`，`enable_arm` 和所有运动接口都会
被物理确认门禁拒绝。它是验收结束后的推荐待机状态。

### 6.3 有人监护的单次运动会话

只有在机械臂区域清空、物理急停复位、接口映射正确、没有其他控制进程且急停可随时
触达时，才能停止待机节点并启动：

```bash
python3 tools/local_arm_can/ros_node.py \
  --interface can1 --arm right \
  --allow-tx --poll-telemetry \
  --physical-confirmation RIGHT_ARM_AREA_CLEAR
```

启动节点本身不会使能。现场再次确认后，在另一个同环境终端显式使能：

```bash
ros2 service call /right/enable_arm std_srvs/srv/SetBool '{data: true}'
```

结束后先失能：

```bash
ros2 service call /right/enable_arm std_srvs/srv/SetBool '{data: false}'
```

不要把物理确认词写入 systemd、开机脚本、容器、VLA 配置或常驻服务。危险时使用
物理急停；软件服务不能代替物理急停。

## 7. ROS 2 接口示例

查看本机服务，确认没有树莓派同名服务混入：

```bash
ros2 service list -t | grep '^/right/'
ros2 topic echo /right/arm_state --once
```

只计算 FK，不运动：

```bash
ros2 service call /right/fk lbot_arm_interfaces/srv/FK \
  '{joints: [0, 0, 0, 0, 0, 0, 0]}'
```

IK 请求包含种子关节、目标位置和 RPY。完整字段以本机构建的接口为准：

```bash
ros2 interface show lbot_arm_interfaces/srv/IK
ros2 interface show lbot_arm_interfaces/srv/MoveJP
ros2 interface show lbot_arm_interfaces/srv/MoveL
```

应用或 VLA 推荐调用 ROS 2 服务，不要绕过 `ros_node.py` 直接裸发 CAN。这样才能保留
软限位、反馈新鲜度、使能状态、确认词、轨迹互斥、调度超时和急停门禁。

## 8. Python 高层库离线使用

FK/IK 可以直接作为 Python 库使用，不启动 CAN：

```python
from tools.local_arm_can.kinematics import ArmKinematics

kinematics = ArmKinematics("right")
pose = kinematics.forward([0.0] * 7)
solution = kinematics.inverse(pose.position, pose.rpy, [0.0] * 7)
print(pose)
print(solution)
```

轨迹生成位于 `trajectory.py`，高层执行器位于 `advanced_controller.py`。实际真机应用
仍应经 ROS 2 服务进入，避免不同进程争抢同一 CAN 控制权。

## 9. HTTP/SSE 低层服务

`service.py` 也可提供仅绑定本机的状态服务，主要用于诊断和 `commission.py`：

```bash
python3 -m tools.local_arm_can.service serve \
  --interface can1 --arm right \
  --allow-tx --poll-telemetry \
  --bind 127.0.0.1 --port 8766

# JSON 状态
curl http://127.0.0.1:8766/state
```

HTTP 服务和 ROS 节点各自拥有 CAN 运行时。运动时只保留一个明确的命令所有者；不要
同时从 HTTP、ROS、网页控制台和自定义脚本发送位置命令。

## 10. 双臂脚本边界

`start_stack.sh` 会配置 `can0` 和 `can1`，并启动左右臂 HTTP 遥测服务；它是明确的
双臂工具，不是右臂单独启动的默认入口。只有确认两只适配器当前都连接机械臂时才可
运行：

```bash
cd /home/descfly/linkagent/botSDK/lbot_arm_api-master/tools/local_arm_can
./start_stack.sh
./status.sh
```

停止脚本会先尝试失能双臂：

```bash
./stop_stack.sh
```

如果 `can0` 正被其他设备或调试链路使用，不要运行上述双臂脚本；按第 6 节只启动
已核对的单臂节点。

## 11. 强制安全门禁

1. 默认不自动使能，也不在构造节点时运动。
2. CAN 发帧需要显式 `--allow-tx`。
3. 使能和位置命令需要对应机械臂的物理确认词。
4. 任一关节反馈缺失或超过 250 ms，拒绝或中止运动。
5. 每个目标和每个轨迹点都检查当前软限位。
6. 相邻直接关节命令默认限制为 `0.08 rad`。
7. 轨迹调度落后超过 50 ms 时中止，禁止瞬间补发积压帧。
8. 同一机械臂一次只能有一个轨迹和一个命令所有者。
9. 失能和软件急停不依赖区域确认词；它们始终是故障安全动作。
10. 软件急停不代替物理急停和断电措施。

## 12. 验证与进一步文档

```bash
python3 -m unittest discover -s tools/local_arm_can/tests -v
python3 -m py_compile tools/local_arm_can/*.py
```

- 高层实现、恢复依据和 2026-08-04 真机验收：
  [`../../docs/本机直连机械臂高级控制器.md`](../../docs/本机直连机械臂高级控制器.md)
- 初期双臂 CAN 验收：[`VALIDATION_2026-08-01.md`](VALIDATION_2026-08-01.md)
- 协议逆向边界：[`REVERSE_ENGINEERING.md`](REVERSE_ENGINEERING.md)
