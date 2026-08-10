# System Self Check 与 Orbbec Gemini Pro 边界

## 结论

`system_self_check` 已作为 `HarvestCycleTask` 的第一个工具和 `INIT -> OBSERVE` 唯一准入
门禁。当前运行模式是 `offline_mock`，不连接深度相机、机械臂、G20、触觉设备或 CAN，
不会枚举 USB，也不会构造 Orbbec `Pipeline`。

用户提供的本地 SDK 位于：

```text
botSDK/orbbec-gemini-pro-viewer
```

该查看器的稳定能力边界是：

- 深度流由项目内 Orbbec SDK v1/`pyorbbecsdk` 读取；
- RGB 流由 Linux V4L2/UVC 读取；
- `GET /api/status` 返回设备信息、RGB/深度 live 状态、分辨率、FPS、来源和错误；
- viewer 自己拥有相机采集生命周期，外部自检不应再次打开设备。

因此 LinkAgent 的实时相机健康适配器只读 `/api/status`，不会导入 viewer 的 OpenCV/NumPy，
不会调用 `CameraManager.start()`，也不会调用 SDK `Pipeline.start()`。

## 当前两种 CameraHealthPort

### `mock`

默认离线配置。它可校验以下 SDK 文件是否存在，但所有 live 状态都明确标记为
`simulated=true`：

- `camera/capture.py`；
- `vendor/pyorbbecsdk/stubs/pyorbbecsdk.pyi`；
- `vendor/pyorbbecsdk/install/lib/libOrbbecSDK.so`；
- `pyorbbecsdk.cpython-*.so`。

### `orbbec_viewer`

为后续 shadow/真机阶段预留。默认 `live_probe=false`，只检查 SDK 布局并返回相机未连接；
只有显式配置 `live_probe=true` 才会对配置的 `/api/status` 地址执行只读 HTTP GET。即使
开启 live probe，它也不会直接访问 `/dev/video*` 或 USB。

## 自检项目

每项结果为 `PASS`、`FAIL` 或 `UNKNOWN`，所有项目当前都是关键门禁：

1. `camera_sdk`：Orbbec SDK 布局是否完整；
2. `camera_connection`：设备状态是否连接且健康快照新鲜；
3. `camera_rgb`：RGB 是否 live；
4. `camera_depth`：深度是否 live；
5. `camera_calibration`：相机标定 ID 是否匹配；
6. `camera_identity`：相机序列号是否绑定到配置的标定 profile；
7. `vision_worker`：隔离视觉 Worker 是否可达、身份匹配、backend ready 且快照新鲜；
8. `execution_mode`：shadow/hardware 是否错误使用 simulated 相机、Worker 或急停证据；
9. `arm_state`：机械臂是否使能、有效且新鲜；
10. `hand_state`：手部是否有效、新鲜且无故障；
11. `tactile_state`：五指 force/matrix 是否有效、新鲜且标定匹配；
12. `emergency_stop`：急停链是否可用、新鲜且健康；
13. `control_lease`：同侧 arm/hand 控制租约是否空闲；
14. `container`：容器是否可用且未满；
15. `policy_manifest`：66D actor、83D critic、11D action 和 hand mapper 合同是否匹配。

任一关键项不是 `PASS` 时，`admission_allowed=false`，工作流停留在 `INIT`，不能调用
目标检测或任何运动能力。

## 当前离线运行

```bash
python -m script.run --config config/harvest_cycle_mock.yaml
```

该配置会检查 SDK 文件布局，但相机、机器人和急停健康证据仍来自 mock。不能把这次通过
解释成真机验收，也不能据此把任何 AtomicSkill 的 `hardware_approved` 改为 `true`。

使用 `config/harvest_cycle_worker_replay.yaml` 时，`vision_worker_health` 改为 loopback live
probe；自检会实际读取 `/health` 并验证 worker ID。该 Worker 的 replay backend 仍标记为
`simulated=true`，因此只能通过 `offline_mock`，不能用于 shadow/hardware 准入。
