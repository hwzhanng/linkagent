# 初始审阅记录

日期：2026-07-18

## SDK 入口

- 直接 Python API：`LinkerHandApi`
- 构造参数：`hand_type`、`hand_joint`、`modbus`、`can`
- 连接构造阶段会尝试拉起 CAN，并读取固件版本和序列号。
- 位置控制：`finger_move(pose)`，每个值为 `0..255`。
- 状态读取：`get_state()`；速度和力矩分别通过 `set_speed()`、`set_torque()` 设置。
- 支持的动作向量长度：O6/L6 为 6，L7 为 7，L10 为 10，L20/G20 为 20，L21/L25 为 25。

## 首次连接为何不用 ROS 2 主节点

`linker_hand.py` 的 `_init_hand()` 不只是建连：它会按型号立即设置较高的默认速度和力矩，并调用 `finger_move()` 发送默认姿态。首次调试应使用直接 Python API，并在外层加入型号校验、只读连接模式和动作解锁。

## 已发现的实现注意点

- SDK 的 `setting.yaml` 位于 SDK 内部，含用于自动配置 CAN 的明文 sudo 密码字段。工作区不修改该文件，也不保存密码；CAN 由操作者在外部显式拉起。
- `LinkerHandApi.close_can()` 引用了未保存在实例上的 `modbus` 名称，直接调用可能触发 `NameError`。后续控制器不依赖该方法关闭接口。
- 型号或左右手字符串无效时，API 构造过程缺少明确校验。工作区程序必须在实例化前校验。
- 仓库自带 `find_linker_hand.sh` 会发送 CAN 探测帧，不属于只读检查，当前未运行。

## 本机初始状态

- Ubuntu 24.04，Python 3.12.3。
- 当前 shell 未发现 `ros2`；系统 Python 未发现 `rclpy`。
- 已安装 `colcon`，但未发现 ROS 2 环境脚本。
- 系统 Python 未发现 `python-can`；`PyYAML` 和 `pyserial` 可用。
- 未发现 CAN 网络接口、`/dev/ttyUSB*`、`/dev/ttyACM*` 或 USB-CAN 设备。
- 未发现 `cansend`/`candump`，说明 `can-utils` 当前不可用。

