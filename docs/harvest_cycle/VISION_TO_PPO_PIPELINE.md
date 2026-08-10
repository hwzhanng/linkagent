# 视觉选择、三维目标与 PPO 准备位闭环

## 1. 已冻结的执行顺序

```text
read-only system gate
  -> synchronized RGB-D frame
  -> instance candidates (YOLO-seg, or temporary Ark boxes)
  -> VLM semantic selection / image points
  -> selected pixel support + aligned depth + calibration
  -> robust 3D ellipsoid and away-from-plant pull axis
  -> deterministic harvestability/reachability-profile gates and ranking
       occlusion-only failure -> bounded active-view MoveL -> new RGB-D -> reselect
  -> pinned TomatoTarget track + strict new-frame re-observation
  -> training-distribution grasp plan
  -> read-only world point-cloud swept-volume clearance check
  -> move_to_pick_ready + safety/distribution proof
  -> final target-state update
  -> bounded PPO contact/grasp/detach/hold
  -> verified retreat -> retention-monitored transport -> bound pre-place -> release
  -> verified cycle -> optional reset to observation pose -> new observation sequence
```

这条顺序将四种权威来源分开：VLM 只负责在编号候选中做成熟度、遮挡、损伤和可采性语义
选择；正式路径中 YOLO instance mask 是二维像素归属权威；临时 Ark 路径只把 box 与对齐深度
交集形成的保守区域当作像素支持，不声称它等价于 segmentation。对齐深度、相机内参与相机到 world 外参是
尺度和三维位置权威；确定性规划器、安全端口和控制租约是运动权威。VLM 的自由坐标不能
直接进入机械臂或 PPO。VLM 可返回一个辅助点，但该点必须落在且只落在选中的 YOLO mask
内，否则整帧失败。

VLM 语义判断与原子目标选择不是同一权限层：VLM 可以表达“更成熟/更少损伤”等语义偏好，
但 `select_harvest_target` 会在三维几何生成后重新执行确定性硬门禁。高语义分不能覆盖陈旧、
低置信、重遮挡、尺寸越界或 workspace profile 外的候选；成功结果也只授权 pinned
re-observation，明确返回 `motion_authorized=false`。

## 2. 当前可运行实现

核心代码不导入 NumPy、OpenCV、Ultralytics、相机 SDK 或模型 SDK。注册的
`tomato_vision_replay` 后端从冻结观测中重放真实接口形状；新增的
`tomato_vision_worker` 则通过 loopback HTTP 调用独立模型进程，收到结果后仍执行同一套
核心校验。两条路径依次执行：

1. `ReplayRgbdCapturePort`：校验设备 ID、内参标定 ID、camera-to-world 位姿、D2C 对齐和
   RGB/depth 时间戳；
2. `ReplayYoloSegmentationPort`：输出带稳定 `instance_id/track_id` 的 mask、bbox、置信度
   和 mask 内深度采样；
3. `ReplayVlmTargetSelectorPort`：重放 Set-of-Mark 候选选择，并校验候选 ID、pinned track
   和辅助点；
4. `RobustMaskedDepthGeometryPort`：按 pinhole 模型反投影、执行 median/MAD 离群过滤、
   拟合有向椭球、用 stem-to-fruit 方向定义 z+（远离植株）并变换到 world；
5. `TomatoVisionPipeline`：统一执行同步、对齐、检测置信度、选择置信度、深度有效率、点数、
   拟合残差和尺寸范围门禁。

部署配置还会把 `expected_calibration_id` 和可选 `expected_device_id` 绑定到 pipeline；帧中
任一身份不符都会在检测/三维拟合前失败，防止健康检查、回放和部署外参混用。

回放后端不是“假装已经部署 YOLO/VLM”。它是可复现的接口与几何验收层。隔离 Worker 已有
可运行的 replay backend、可选 `ultralytics_vlm` backend，以及临时 `ark_vlm_box` backend。
Ark 使用真实多模态 Responses 调用返回归一化 box/点，三维仍完全来自 raw RGB-D sidecar；
YOLO 权重可用后应切回 instance mask。协议、配置和运行
方式见 [`VISION_WORKER.md`](./VISION_WORKER.md)。

pipeline 的 DTO 已允许一个 `TomatoDetectionResult` 携带多个 `TomatoTarget`，选择原子 Skill
会对多候选产生完整稳定排序。当前 replay/Worker 视觉 adapter 仍按一次 VLM 选择输出一个
三维候选，因此当前闭环主要验证安全筛选、pin 身份和状态机；把所有 YOLO 实例逐个拟合为
多三维候选属于后续感知吞吐扩展，不在本次交付中伪装为已完成。

## 3. 原子目标选择合同

`select_harvest_target` 位于首次三维检测与严格复观测之间，无任何机器人端口，也不提交
动作。每个候选依次经过：候选集/目标时间戳、成熟度、pick confidence、遮挡率、果实半轴、
绑定左右臂的 workspace profile。只有硬门禁全通过的候选才参与最终选择；软分由成熟度、
置信度、可见度和 profile 边界裕量构成，同分时用 margin 和 `target_id` 确定性打破平局。

输出包括选择 revision、完整有序候选、每项 hard gate、分数组成、拒绝原因与证据引用。
若只因遮挡失败，下一步为 `active_view_required`；缺少 workspace bounds 时为
`reachability_context_required`；陈旧、空集合或没有可采候选返回稳定失败码。当前 workspace
AABB 是粗范围过滤，不是 FK/IK、路径碰撞或枝叶净空证明，结果强制携带
`requires_pregrasp_clearance_check=true`。

`active_view_reposition` 只接受该遮挡恢复状态。视角来自绑定左右臂的有限 profile，不接收 Agent
自由位姿；相机固定安装在双臂中间顶部并斜向前下看，技能不移动相机，而是通过
`robotics.arm.move_linear` 把机械臂移到有限的去遮挡位。读取实际末端位姿、确认固定相机外参
未漂移后，强制抓取新样本并回到选择状态。shadow 只返回计划且不推进观测。视角耗尽、IK 不可达、到位
误差或新帧失败都保持可恢复失败，不会进入 PPO。

规划后必须调用 `pregrasp_clearance_check`。该工具只接收 `grasp_plan_id`；当前 observation、
机械臂末端位姿和场景点云均由 Runtime/Port 提供，Agent 不能注入位姿或点云。Skill 检查
当前位置到 pregrasp、pregrasp 到 contact、contact 到果心三段扫掠体，并用带 margin 的目标
椭球剔除果实自身点。结果绑定 target/plan/scene revision；未通过时 `move_to_pick_ready` 保持锁定。
当前 mock 与 observation metadata 端口已经闭环验证，真实深度相机端口尚待硬件到位。

## 4. 视觉到 PPO 的精确数据

当前 PPO actor 共 66 维，其中视觉/几何链只提供 9 维：

|字段|维度|来源|
|---|---:|---|
|`fruit_center_in_palm`|3|`TomatoTarget.center_pose` 与实时 palm pose|
|`fruit_semi_axes`|3|选中 mask 的对齐深度椭球|
|`pull_axis_in_palm`|3|grasp frame z+，约定为远离植株|

其余 57 维来自机械臂末端速度、手指位置/速度、五指触觉、重力方向和上一动作，不由 VLM、
YOLO 或点云生成。`move_to_pick_ready` 的成功结果会原样报告这 9 个值及每项准入判断，便于
在启动 PPO 前审计。

## 5. PPO 准备位不是普通 pregrasp

`TrainingDistributionTomatoGraspPlanner` 使用训练时相同的手掌局部约定：闭合抓持中心为
`(0.148, -0.0607, 0.0535)m`；准备位沿 approach 方向额外保持 55–75 mm，并保持 5–20 mm
横向偏置；姿态误差不超过 7°；果实半轴必须位于 25–35 mm 的策略训练范围。

`move_to_pick_ready` 现在是正式 `AtomicSkill`：

- `offline_mock` 仅在 bundle 明确提供 `mock_world` 时提交受租约保护的位姿命令；命令后
  重新读取实际 palm pose，不能把该配置用于真实机器人；
- `shadow` 只执行规划、安全和分布检查，不提交命令，并明确返回
  `ppo_execution_allowed=false`；
- 两种模式都要求 plan 未过期、视觉目标不超过 0.5 秒、pregrasp 和 PPO residual envelope
  两次安全评估通过；
- descriptor 的 `hardware_approved=false`，不存在自动硬件 fallback。

## 6. PPO 执行期间的目标新鲜度

一次性视觉目标在 0.5 秒后会变陈旧，而策略通常执行更久。当前实现没有放宽这个门禁，而是
增加 `TargetStatePort`：

- CONTACT/GRASP/DETACH 前，使用经过标定的 world-frame 静态植株目标进行显式世界模型传播；
- 独立 verifier 确认脱离并进入 HOLD 后，冻结 fruit-in-palm 和 pull-axis-in-palm，改用手部
  附着运动学传播；
- `target_id`、`observation_sequence` 与 plan revision 始终不变，防止执行中换果。

真实部署使用顶部中置固定 RGB-D 相机 5–15 Hz 更新；相机帧间才使用世界传播。视觉 track 丢失、
外参改变、机器人底座移动、枝条/果实明显运动或 verifier 状态未知时必须停止，不得继续把
静态模型当测量值。

## 7. 真实 YOLO、VLM 与深度相机接入门禁

真实硬件适配器需要同时满足：

1. Orbbec 原始 RGB 与 depth frame 可访问，不能从 MJPEG 深度预览反推距离；
2. 使用 SDK D2C alignment 和 frame sync，并保留设备时间戳；
3. 内参、深度 scale、RGB-depth 外参、camera-to-base/world 外参都有不可混用的 calibration ID；
4. YOLO 必须输出 instance mask，不以 bbox 中值深度替代 mask 点云；
5. VLM 使用编号候选图，只返回结构化候选 ID、置信度、原因和可选 mask 内点；
6. 记录原始帧 hash、mask/selection/geometry evidence，能够离线复跑；
7. 先通过 replay、recorded shadow、静止标定物和动态枝叶数据集，再进行单命令低速验收；
8. 相机或模型进程不可用时 fail closed，不能退化成旧 metadata 或默认目标。

当前 Worker 已实现第 4、5、6、8 项的软件边界；SDK-neutral spool 已实现第 1～3 项的数据
合同、原子发布、严格新帧读取和 recorded-shadow 原始数据审计。尚未完成的是 Orbbec 厂商
capture adapter、真实权重、真实 VLM endpoint 和真实标定数据，它们必须由后续硬件阶段补齐。

当前 `orbbec-gemini-pro-viewer` 仍只用于设备健康和预览；它没有同步原始深度、D2C、完整标定
与点云输出，因此不会被注册成 `RgbdCapturePort`。

## 8. 已覆盖的失败场景

自动测试覆盖：未对齐深度、RGB/depth 超时差、VLM 点离开 mask、VLM reobserve 换 track、
mask 深度不足、dense mask 深度覆盖率、椭球尺寸/残差不合格、Worker 路径越界、Worker
身份错误、固定 track 不匹配、缺本地权重、真正跨 Python 进程的 HTTP 调用、自检阻止未
ready Worker、spool 并发发布/篡改/漂移/残留 staging、严格等待新帧、准备位训练分布、
目标选择硬门禁优先级、稳定排序/revision、左右臂 profile 绑定、陈旧/空集合/遮挡/未知可达
失败、选择器零运动授权、shadow 零运动、shadow 禁止 PPO、目标超过 0.5 秒后的
phase-aware 状态传播，以及完整采摘周期回归。

仍需真实数据才能关闭的风险包括：叶片造成的 mixed pixels、反光/黑洞深度、细果梗方向误判、
多果接触 mask、相机温漂、机械臂遮挡、base/world 外参漂移、果实随风运动和真机碰撞场景。

## 9. 外部资料依据

- Set-of-Mark Visual Prompting: <https://arxiv.org/abs/2310.11441>
- Ultralytics instance segmentation: <https://docs.ultralytics.com/tasks/segment/>
- Orbbec SDK D2C、frame sync 与 point cloud: <https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/source/3_Application_Guide/ApplicationGuide.html>
- Grounding DINO（未来开放词汇候选补充，不替代 mask/深度权威）: <https://arxiv.org/abs/2303.05499>
