# G20 实机上下文

本文件记录当前仍有效的灵巧手事实。旧 L20 误分类、右手树莓派转发和网页控制链路
已经清理，不再作为实现依据。

## 当前硬件

- 左手，G20 控制协议；
- `can0`，1,000,000 bit/s，标准 CAN ID `0x28`；
- 嵌入式版本 `1.0.10`；
- 第二代矩阵压感，类型码 `2`；
- 五指各 `12 × 6`，共 360 个原始触觉点；
- 官方 20 维逻辑姿态，经 G20 驱动映射为五组串联位置帧。

实物外观和序列号可能包含 T20/L20 标识，但动作帧已经通过实机验证应选择 SDK 的
`G20` 驱动。配置文件为 `config/hand_left.yaml`。

## 已验证协议

- `0x01..0x06`：六组并联位置；
- `0x41..0x45`：五指串联位置；
- `0x49..0x4D`：五指速度；
- `0x51..0x55`：五指扭矩；
- `0x83`：清故障。

完整确认记录见 `notes/left-g20-official-gui-2026-07-19.md`。

## 关键工具与数据

- `scripts/preflight.py`：无动作环境检查；
- `scripts/read_only_connect.py`：只读状态与触觉类型确认；
- `scripts/read_pressure_snapshot.py`：五指矩阵压力快照；
- `scripts/start_left_g20_official_gui.sh`：官方 G20 GUI；
- `scripts/calibrate_left_g20_motion.py`：运动标定；
- `scripts/calibrate_left_g20_high_speed.py`：高速可靠性标定；
- `scripts/analyze_left_g20_motion.py`：标定结果分析；
- `logs/left_g20_connection_20260726.json`：只读连接证据；
- `logs/left_g20_pressure_baseline_20260726.json`：无接触压力基线；
- `logs/motion_calibration/`：G20 运动标定证据。

## 触觉数据边界

SDK 返回的矩阵点值范围为 `0..255`，当前没有可信的牛顿换算、温漂补偿或逐点标定。
PPO 可以把它作为相对接触、接触分布和滑移特征，但不能直接解释为物理力值。

## 安全边界

- 默认配置必须保持 `armed: false`；
- 只读检查不得发送位置、速度或扭矩命令；
- 官方 GUI 启动后可能立即发布姿态，启动前必须保持手周围净空；
- PPO 和 Agent 必须经过上层限幅、超时、控制权和急停门禁；
- USB 重插或系统重启后必须重新确认 `can0` 对应的实际设备。
