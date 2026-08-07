# LinkBot SDK 双臂双手模型

这是双机械臂 + 左右 L20_8 灵巧手的正式组合 description package。

## 主文件

- `urdf/linkbot_bimanual_sdk_description.urdf`
- `config/joint_groups.yaml`
- `config/mounts.json`
- `config/simulation_baseline.yaml`
- `config/left_g20_motion_calibration_20260726.yaml`

## 坐标与安装

- 机械臂 link、joint、axis、origin、limit、质量和惯量直接来自随设备交付的
  SDK 控制平台模型 `A7V2_50_4.27`，没有采用在线 `linker-sim` 的关节符号重定义。
- 右手：`R8_Link -> right_hand_base_link`，
  `xyz=(0,0,0)`，`rpy=(π,0,+π/2)`。
- 左手：`L8_Link -> left_hand_base_link`，
  `xyz=(0,0,0)`，`rpy=(π,0,-π/2)`，按用户修正与右手镜像。

安装方向的唯一机器可读来源是 `config/mounts.json`；正式 URDF、仿真配置和训练任务
必须与它逐值一致。

## 仿真参数边界

机械臂 14 个运动关节的 `<dynamics damping="20" friction="0">` 来自厂商
公开仿真项目，只是启动基线，不是实机辨识结果。`friction=0` 不能解释成真实无摩擦。

已连接左手的安全 raw 范围、双向死区和速度已完成实机运动标定。正式 URDF 的左手
`velocity` 使用 80% 热稳态实测速度再保留 15% 余量；连续命令硬上限为 raw 204。
独立横摆只允许使用标定覆盖层中的窄范围，更宽姿态必须使用带自碰撞检查的多指
协同轨迹。右手仍保留原厂占位速度，不能直接继承左手标定值。

双手仍没有可信的官方关节阻尼、摩擦、真实输出转矩或控制增益，因此没有写入猜测值。
两手各包含 16 个独立指令关节和 5 个 mimic 关节。

详细训练判断见：
`../../reports/BIMANUAL_SDK_TRAINING_READINESS_20260726.md`。
