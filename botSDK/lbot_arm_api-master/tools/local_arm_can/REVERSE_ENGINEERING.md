# A7_V2 CAN HAL 逆向重建说明

## 来源

核心协议从树莓派当前控制器的未剥离 ARM64 二进制恢复：

`install/lbot_lkls_i1_hal/lib/lbot_lkls_i1_hal/lbot_arm_driver`

同时使用以下内容交叉校验：

- 树莓派启动日志和 ROS 话题/服务边界；
- 树莓派 `data/lsi1_*_joint_limit.json`、`data/lsi1_pid.json`；
- 左右臂真机 `candump -x` 帧；
- 本机 x86_64 ROS Jazzy 消息编译与真机状态回读。

## 已恢复的 CAN 边界

- 11 位标准 CAN，1 Mbit/s。
- 右臂基址 `0x33`，左臂基址 `0x3D`，每臂 7 个连续关节 ID。
- `0x38/0x39/0x42/0x43` 为 Motorevo，其余为经典电机。
- 经典状态帧：`0x96 + pos16 + vel12 + effort12 + status`。
- 经典位置命令：`0x0A + float32 little-endian`。
- 经典使能/失能：`2A01 / 2A00`。
- 经典控制模式：`07 + mode`。
- 经典温度查询：`5F`；自动上报：`95 + float32(period_ms)`。
- Motorevo 状态查询：`FFFFFFFFFFFFFFFA`。
- Motorevo 使能/失能：末字节 `FC / FD`。
- Motorevo 模式 4：在 `motor_id + 0x600` 发送 `670B010000001576`。
- Motorevo 位置命令：`position16 + 6×uint8`，六参数取自 `lsi1_pid.json`。

完整编码/解码实现在 `protocol.py`，SocketCAN 传输在 `socketcan.py`，持续遥测、
安全门禁和 HTTP/SSE 服务在 `service.py`。

## 已恢复的上层能力

在 CAN HAL 上进一步分析了树莓派当前未剥离的
`lbot_api_server/lib/lbot_api_server/lbot_api_server`，并重建：

- 精确 URDF 几何链和 FK；
- 与官方种子/限位/偏移语义一致的数值 IK；
- 100 Hz MoveJ 独立关节梯形轨迹；
- MoveJP、KDL 等效 MoveL 和 FollowP；
- 与树莓派同名同字段的 ROS 2 服务。

实现位于 `kinematics.py`、`trajectory.py`、`advanced_controller.py` 和 `ros_node.py`。
提取证据、验证误差、使用方式及没有伪造的边界见
`docs/本机直连机械臂高级控制器.md`。

树莓派当前 `MoveC` 是失败桩，本机保持失败；碰撞检测和绕障规划不属于该官方
控制器，未伪装成已恢复能力。
