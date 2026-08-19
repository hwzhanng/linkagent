# Gemini Pro 视觉台

面向旧款 Orbbec Gemini / Gemini Pro（OpenNI 协议）的本地网页查看器，同时展示 RGB 与深度伪彩画面。

本机已验证的设备信息：

- 深度：`2bc5:0614`，Orbbec SDK v1，最高分辨率档 `1280 × 800 @ 30 FPS`
- RGB：`2bc5:0511`，Linux UVC，目标 `1280 × 720 @ 30 FPS`
- 设备：`SV1301S_U3`
- 序列号：`AY2PB4200YY`
- 固件：`RD3013`
- 总线：USB 2.0

## 启动

```bash
cd /home/cle/linkbot/orbbec-gemini-pro-viewer
./start.sh
```

打开 <http://127.0.0.1:8080>。服务监听 `0.0.0.0`，同一局域网设备也可用终端打印的局域网地址访问。

首次在另一台机器上运行时，`start.sh` 会调用 `scripts/install.sh`：创建 Python 3.11 虚拟环境、安装网页依赖，并从官方 `v1.3.4` 标签编译 Orbbec Python SDK。系统 Python 不会被修改。

如需为本地 `video` 组开放相机 USB 节点，可审阅并安装仓库提供的最小 udev 规则：

```bash
sudo install -m 0644 config/99-orbbec.rules /etc/udev/rules.d/99-orbbec.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

该规则只匹配 Orbbec USB 厂商 ID，不授予机器人或 CAN 控制权限。

## 工作方式

- 深度线程通过项目内编译的 Orbbec SDK v1 读取传感器。
- RGB 线程通过 Linux V4L2/UVC 读取相机，按 USB 厂商和产品 ID 定位，不依赖固定的 `/dev/videoN` 编号。
- 两路各自维护最新 JPEG 帧，Flask 通过 MJPEG 推送给浏览器；多个浏览器不会重复打开相机。
- `/api/status` 提供分辨率、FPS、时间戳、设备和错误状态。
- 页面设置可实时调整 `1000–10000 mm` 深度色标与 JPEG 质量。

相机采集只允许一个进程占用，因此 Gunicorn 固定为一个 worker、八个线程。

## 热插拔与排错

网页会自动等待并重连两路设备。若 RGB 区域提示“未找到 Gemini RGB UVC 节点”，把相机 USB 拔下再插一次即可；无需重启网页服务。

运行诊断：

```bash
./scripts/diagnose.sh
```

正常情况下应同时看到：

```text
2bc5:0511  USB 2.0 Camera
2bc5:0614  ORBBEC Depth Sensor
```

并在视频节点列表中看到 Orbbec 的 `USB 2.0 Camera`。如果设备被其他程序占用，请先关闭 Orbbec Viewer、相机预览软件或另一份本服务。

## 接口

- `GET /`：监控页面
- `GET /stream/rgb.mjpg`：RGB MJPEG
- `GET /stream/depth.mjpg`：深度 MJPEG
- `GET /api/status`：设备与视频流状态
- `GET|PUT /api/settings`：显示与编码设置
- `POST /api/retry`：立即清理错误状态并重试
- `GET /healthz`：网页服务健康检查

## 测试

```bash
./.venv/bin/pytest -q
```

视觉稿保存在 `design/gemini-pro-console-concept.png`。
