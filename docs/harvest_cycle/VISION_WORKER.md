# 隔离视觉 Worker 与原始 RGB-D 采集包

## 1. 进程边界

`HarvestCycleTask` 不直接导入 Ultralytics、NumPy、Pillow、VLM SDK 或 Orbbec SDK。
`TomatoVisionWorkerClient` 只连接带显式端口的 loopback HTTP Worker，并且只把位于配置
`capture_root` 内的相对 sidecar 路径发给 Worker。请求和响应均有大小限制、schema version、
request ID 和 worker ID；响应返回后还会完整重跑同步、标定、mask、track、深度和三维拟合
门禁。因此 Worker 不能仅靠返回一个三维坐标绕过核心校验。

`system_self_check` 现在单独检查 Worker 的 reachability、worker ID、backend ID、ready 状态
和是否 simulated。检测之前 Worker 不可用会阻止采摘准入。loopback Worker、相机健康和
急停健康是三个独立证据源，互不替代。

## 2. 可重复的跨进程运行

终端一启动冻结 replay Worker：

```bash
python -m script.tomato_vision_worker \
  --config config/tomato_vision_worker_replay.yaml
```

终端二运行通过 HTTP Worker 完成的离线采摘闭环：

```bash
python -m script.run --config config/harvest_cycle_worker_replay.yaml
```

只验证 Worker 配置和 backend 能否构造，不进入服务循环：

```bash
python -m script.tomato_vision_worker \
  --config config/tomato_vision_worker_replay.yaml --check
```

replay bundle 位于 `artifacts/vision_worker/replay/tomato_001.json`。它不声称运行了真实
YOLO/VLM，只用于证明核心和 Worker 已通过真实进程边界、HTTP 协议、身份检查和二次校验。

## 3. 原始 RGB-D sidecar 与原子 spool v1

未来的 Orbbec capture adapter 必须产出同一 FrameSet 中的 RGB、D2C-aligned 16-bit depth、
设备时间戳和已冻结标定。`SynchronizedRgbdSource` 是唯一硬件插槽，只有
`start/read_synchronized_frame/close` 三个生命周期方法；当前仓库没有厂商实现，也不会自动
打开相机。SDK adapter 返回 `RgbdSpoolFrame` 后，由 `TomatoVisionSpoolWriter` 负责下面的
持久化合同：

```text
capture_root/
  .writer.lock
  latest.json
  captures/
    00000000001001000000-<uuid>/
      rgb.png
      aligned-depth.png
      capture.json
```

writer 在进程锁内写入并 fsync 临时目录，校验完整 sidecar 后原子 rename 成不可变 capture，
最后原子替换 `latest.json`。只有 `latest.json` 更新成功的新帧才对 Worker 可见；generation 和
设备采样时间必须严格增加。pointer 保存 sidecar SHA-256、设备/标定身份和两个流的时间戳，
reader 会再次交叉校验，不能跳过已损坏的 latest 继续发布。RGB 必须是 8-bit RGB/RGBA PNG，
depth 必须是与 RGB 同尺寸的 16-bit 灰度 PNG，单位只能由 sidecar 的 scale 给出。

每个不可变 capture 内的 sidecar 形状如下：

```json
{
  "schema_version": 1,
  "capture_id": "orbbec-serial/frame-000001",
  "rgb_path": "frame-000001.rgb.png",
  "aligned_depth_path": "frame-000001.depth-d2c.png",
  "depth_scale_m_per_unit": 0.001,
  "camera": {
    "intrinsics": {
      "width_px": 1280,
      "height_px": 720,
      "fx_px": 900.0,
      "fy_px": 900.0,
      "cx_px": 640.0,
      "cy_px": 360.0,
      "calibration_id": "gemini-pro-cell-a-2026-08"
    },
    "camera_pose_in_world": {
      "position_m": [0.0, 0.0, 0.0],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "frame_id": "world"
    },
    "rgb_sample_monotonic_ns": 1000000000,
    "depth_sample_monotonic_ns": 1001000000,
    "depth_aligned_to_rgb": true,
    "device_id": "orbbec-serial",
    "evidence_sources": ["orbbec:rgb", "orbbec:frame-sync", "orbbec:d2c-depth"]
  }
}
```

sidecar、RGB 和 depth 都必须留在 `capture_root` 内；绝对路径、`..` 或 symlink 越界会失败。
sidecar 限制为 2 MB，Worker 配置还可限制请求、sidecar、图像文件和总像素数。RGB/depth
尺寸必须等于内参尺寸，depth 必须是单通道整数传感器单位，尺度只能来自 sidecar。

持续采集时，Env 的 Observation 固定引用 pointer，而不是某一帧：

```json
{
  "tomato_vision_worker": {
    "capture_latest": "latest.json"
  }
}
```

首次检测读取当前 latest；`target_reobserve` 默认把上次 `sample_monotonic_ns` 作为下界，
Worker 最多等待配置的 `new_frame_wait_timeout_s`，只接受严格大于下界的 capture。等待超时返回
HTTP 409，绝不复用旧帧。冻结 replay 仍可用 `capture_bundle`，但这种 direct bundle 模式禁止
携带新帧等待参数。

## 4. 录制导入、审计与恢复

随仓 [`tomato_vision_spool_recorded_shadow.yaml`](../../config/tomato_vision_spool_recorded_shadow.yaml)
是离线 profile：`hardware_approved=false`，绑定回放设备/标定身份，至少要求 100 帧、60% 有效
深度覆盖和 1 GiB 写入后磁盘余量。它不是实机标定；真实单元必须复制成新的 profile，并替换为
真实 serial 和 calibration ID。

把已有 raw sidecar 录制逐帧导入 spool：

```bash
python -m script.import_tomato_rgbd_capture \
  --source-capture-root /data/orbbec-recording \
  --source-bundle frames/frame-000001/capture.json \
  --destination-profile config/tomato_vision_spool_recorded_shadow.yaml
```

导入保留原始设备时间戳和证据来源，重新验证格式、同步、身份和磁盘门禁，不会把旧录制伪装成
当前硬件帧。生成 recorded-shadow 质量清单：

```bash
python -m script.audit_tomato_vision_spool \
  --profile config/tomato_vision_spool_recorded_shadow.yaml \
  --output artifacts/vision_spool/recorded-shadow-audit.json
```

审计逐帧验证 sidecar、PNG 解码、hash、时间顺序、RGB/depth skew、设备/标定一致性、深度范围
覆盖率和 latest 指向，并在发现未完成 staging 时失败。该报告只证明原始 RGB-D 数据面通过，
不证明 YOLO 召回率、VLM 选择质量、三维拟合成功率或真机采摘效果；后四项必须在真实权重和
标注数据到位后追加 shadow 指标。

把一份已通过同一 profile 审计的 source spool 按顺序重新发布到另一个 `latest.json`，可用于
驱动 Worker recorded-shadow 回放：

```bash
python -m script.replay_tomato_rgbd_recording \
  --source-spool-root /data/orbbec-recording-spool \
  --destination-profile config/tomato_vision_spool_recorded_shadow.yaml \
  --publish-interval-s 0.2 \
  --output artifacts/vision_spool/recorded-shadow-replay.json
```

replay 不改写原设备时间戳，source 与 destination 必须不同，并且先执行 profile 全量审计。
输出清单只证明各帧按序原子发布，不声称 Worker 来得及处理了每一帧；模型级 shadow 需要另外
记录 request/capture ID 对应关系和逐帧输出指标。

崩溃恢复和保留策略通过 `TomatoVisionSpoolWriter.recover_staging()` 与 `prune()` 显式调用；
两者默认 `dry_run=true`，不会在启动时静默删除证据。必须先保存 audit/运行清单，再由操作员
明确执行清理。

## 5. 真实 YOLO-seg + VLM backend

真实 backend 配置形状如下。`model_path` 必须指向已经存在的本地 segmentation 权重；模型
别名和隐式下载不被接受。VLM 凭据只从环境变量读取：

```json
{
  "schema_version": 1,
  "worker_id": "tomato-vision-cell-a",
  "host": "127.0.0.1",
  "port": 18765,
  "capture_root": "/data/tomato-rgbd-spool",
  "max_request_bytes": 64000,
  "max_bundle_bytes": 2000000,
  "backend": {
    "type": "ultralytics_vlm",
    "model_path": "/models/tomato-yolo-seg.pt",
    "confidence": 0.5,
    "tomato_class_names": ["tomato"],
    "max_instances": 20,
    "max_depth_samples": 512,
    "min_depth_m": 0.1,
    "max_depth_m": 3.0,
    "track_iou_threshold": 0.3,
    "track_iou_ambiguity_margin": 0.05,
    "vlm": {
      "type": "http_json",
      "endpoint": "https://vlm-gateway.example/v1/tomato-selection",
      "model_id": "configured-vlm-version",
      "api_key_env": "TOMATO_VLM_API_KEY",
      "timeout_s": 15.0
    }
  }
}
```

Worker 先用 YOLO instance segmentation 生成 `som-1...som-N` 和编号图，再要求 VLM 只返回：

```json
{
  "schema_version": 1,
  "selected_instance_id": "som-1",
  "confidence": 0.93,
  "reason": "ripe and minimally occluded",
  "point_xy": [620, 350],
  "stem_point_xy": [620, 315],
  "ripeness_score": 0.91,
  "occlusion_ratio": 0.08
}
```

VLM 不能返回深度、三维点、机械臂姿态或 PPO 输入。`point_xy` 必须只属于一个且正是选中
的 YOLO mask；stem 像素的三维位置由 aligned depth 和内参反投影。固定 re-observation
期间若 VLM 切换 track，Worker 和核心各拒绝一次。

真实 Worker 还发布整张 mask 的 `mask_area_px` 与其中有效深度的
`valid_depth_pixel_count`。核心用二者计算深度覆盖率，而不是用 Worker 已筛过的有效采样
点计算，避免把深度黑洞隐藏成 100% 有效。

`target_reobserve` 默认 `require_new_sample=true`：重观察目标的相机采样时间必须严格晚于
首次检测，且年龄不能超过 request 上限。持续采集服务可以原子更新同一个 spool sidecar，
也可以让 Env 提供新的相对路径；两者都必须带新的设备时间戳。随仓 replay 配置为了重复
使用冻结帧而显式设为 `false`，这一例外只允许 `offline_mock`，真实配置不得复制。

编号图会用不同颜色半透明叠加每个 YOLO instance mask，再绘制编号和 bbox。Worker 会把
RGB、aligned depth 和本地 YOLO 权重的 SHA-256 加入核心证据链。相邻帧暂用保守 IoU track；
当两个旧 track 的匹配分数落入歧义 margin 时不复用任何旧 ID，而是分配新 ID，使 pinned
reobserve 明确失败。它不会在歧义时猜测身份；真实动态枝叶验收后仍可替换成 3D/外观 tracker。

## 6. Ark box 临时后端与仿真闭环

`config/tomato_vision_worker_ark_box.yaml` 使用 Coding Plan Responses endpoint 和
`doubao-seed-2.0-pro`。它要求 Ark 返回 0–1000 坐标系中的番茄 box、中心点、果梗连接点、
置信度、成熟度和遮挡率；本地把坐标映射回原图，并用椭圆、box 内 aligned-depth 有效像素及
中心深度带构造保守 pseudo-mask。track 仍由本地 IoU 分配，三维点、stem 三维位置和椭球均
由深度与标定计算。该 pseudo-mask 不等价于 YOLO segmentation，适合现在的闭环开发与数据
采集，不能作为最终像素精度结论。

可复现软件仿真使用：

```bash
python -m script.generate_tomato_vision_sim_capture \
  --output-root artifacts/vision_spool/ark-synthetic-fixed-clock \
  --sample-monotonic-ns 999000000
set -a; . ./.env.local; set +a
python -m script.tomato_vision_worker \
  --config config/tomato_vision_worker_ark_box.yaml
# 第二终端同样载入 ARK_API_KEY
python -m script.run --config config/harvest_cycle_worker_ark.yaml
```

已验证的路径是：真实 Ark 图片识别、归一化坐标转换、aligned-depth pseudo-mask、512 点三维
采样、椭球拟合、跨进程 Worker、目标选择、准备位、PPO mock、回撤、放置和最终验收。
`thinking_type=disabled` 用于低延迟结构化检测；凭据只从 `ARK_API_KEY` 读取。

## 7. 当前未关闭的硬件门禁

本阶段已经实现 SDK-neutral 原子 spool、latest 新帧协议、录制导入、profile 和原始数据审计，
但没有实现 Orbbec SDK adapter，也没有随仓发布 tomato YOLO-seg 权重、VLM endpoint 或真实
内外参。因此真实 backend 只能在显式提供这些资源后构造；缺权重、缺依赖、缺凭据、VLM
不可用、相机 sidecar 不完整、Worker 身份错误或目标 track 丢失都会 fail closed。任何一种
失败都不会回退到 Observation 中旧的 `tomato_vision` metadata。

接入深度相机后的下一步是：实现 `SynchronizedRgbdSource` 的 Orbbec frame-sync + D2C adapter，
写入真实静态标定物数据，采集叶片/反光/多果遮挡 shadow 数据集，冻结 YOLO/VLM 版本并追加
模型质量报告，然后才是单命令低速硬件验收。现有 viewer 的独立 UVC RGB 与 depth preview
线程不是同步原始 FrameSet，不能作为这一 adapter 的度量输入。
