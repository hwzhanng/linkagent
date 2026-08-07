# G20 灵巧手控制工作区

本目录只保留当前 G20 灵巧手的直接控制、压力读取、运动标定和实验记录。相邻的
`../linkerhand-ros2-sdk-main/` 是只读上游 SDK；禁止在其中写入项目配置、日志或缓存。

## 当前设备

- 型号/协议：G20；
- 当前已验证设备：左手；
- CAN：`can0`，1 Mbit/s，CAN ID `0x28`；
- 触觉：五指第二代矩阵压感，每指 `12 × 6`；
- 当前配置：`config/hand_left.yaml`；
- 动作锁默认值：`armed: false`。

外观或序列号可能包含 T20/L20 字样，但当前设备的动作协议已经通过实机确认是 G20。
旧 L20 网页控制、预设和探测脚本已清理，禁止重新使用其帧解释。

## 目录

- `scripts/`：G20 连接、压力读取、标定和官方 GUI 启动；
- `config/`：当前 G20 配置和配置模板；
- `logs/`：G20 连接、压力基线和运动标定证据；
- `notes/`：接口、安全边界和实机结论；
- `environment.yml`：可重建的最小 Python 环境。

## 重建 Python 环境

`.conda/` 不保存在工作区。首次使用时重建：

```bash
cd /home/descfly/linkagent/botSDK/linkerhand-control-workspace
conda env create --prefix .conda/envs/linkerhand --file environment.yml
source scripts/activate_env.sh
python3 scripts/preflight.py
```

只读连接和压力读取默认使用 `config/hand_left.yaml`，不会发送执行器动作：

```bash
python3 scripts/read_only_connect.py
python3 scripts/read_pressure_snapshot.py
```

## 重建官方 ROS 2 工作区

`ros2_official/build`、`install` 和 `log` 是生成目录，已经清理。需要官方 GUI 时先执行：

```bash
cd /home/descfly/linkagent/botSDK
source /opt/ros/jazzy/setup.bash
colcon --log-base linkerhand-control-workspace/ros2_official/log build \
  --base-paths linkerhand-ros2-sdk-main \
  --build-base linkerhand-control-workspace/ros2_official/build \
  --install-base linkerhand-control-workspace/ros2_official/install
```

构建完成后，确认手周围净空再运行：

```bash
linkerhand-control-workspace/scripts/start_left_g20_official_gui.sh
```

官方 GUI 启动后会发布当前页面姿态，因此这是可能产生动作的入口。

## 安全边界

1. 首次连接只使用 `preflight.py`、`read_only_connect.py` 和压力读取；
2. 执行动作前核对左右手、CAN 接口、CAN ID 和工作区净空；
3. 项目脚本不得绕过 `armed: false` 和人工确认；
4. PPO/Agent 不直接实例化厂商 SDK，应通过冻结的灵巧手或 Pick Skill 适配层调用；
5. 压力矩阵原始值 `0..255` 没有可信牛顿换算，未标定前只作为相对接触信号。

详细设备结论见 [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)。
