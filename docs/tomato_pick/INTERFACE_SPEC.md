# 番茄采摘接口规范

状态：C0/C2-0 冻结；C1 Mock、C2-1 环境骨架与 C4-O 离线适配完成；Isaac 物理/vendor gateway/真机/PPO 尚未实现  
版本：1.0.1  
日期：2026-08-04  
依赖基线：[SDK_CAPABILITY_BASELINE.md](./SDK_CAPABILITY_BASELINE.md)

本规范冻结 RealAgent、番茄 Task、PPO Skill、规划/验证服务和 LinkBot SDK adapter 之间
的边界。C1 已按本规范实现离线 mock，本文继续作为 C2 PPO 环境和 C3/C4 adapter 的
输入。除修正错误外，字段、单位、扁平顺序
和成功语义的改动都必须升版本，并同步修改 mock、训练环境、adapter 和契约测试。

文中的“必须”是契约要求，“建议”允许实现替换。C1 完成不表示真机已经接入或 PPO
已经训练。

## 1. 总体边界

```text
TopAgent
  -> AgriPickTask 的 5 个任务级 AgentTool
      -> TomatoPickService（业务编排）
          -> Perception / TomatoGraspPlanner / MotionSafety / PickVerifier
          -> TomatoPickSkillRunner（PPO + 相位/安全监督器）
              -> ArmControlPort / HandControlPort / TactilePort
                  -> Mock、Isaac 或 LinkBot/G20 adapter
```

约束如下：

1. Agent 只传 `target_id`、`grasp_plan_id`、`execution_id` 和 `arm_side` 等小型语义参数。
2. SDK client、关节数组、轨迹、图像、点云、触觉矩阵和 PPO tensor 不进入模型消息。
3. Task 决定工具白名单和完成条件；PPO 不调用 LLM，SDK adapter 不导入 Agent/Task。
4. `TomatoPickSkillRunner` 从预抓取位开始，负责接触、抓取、拉脱和稳定持有；到预抓取
   位的全局运动由 Task 编排，不属于 PPO。
5. 一个执行期只有一个对象持有目标臂和手的控制租约。Agent 不能在 Skill 运行中穿插
   原始运动命令。

## 2. 公共数据约定

### 2.1 标识与时间

- `target_id`：一次感知结果内稳定的目标 ID；目标重识别后可以生成新 ID。
- `grasp_plan_id`：绑定目标、观测序号、手臂侧和规划版本的不可变 ID。
- `execution_id`：Task handler 在首次执行时生成的唯一 ID；同一计划的重复调用返回同一
  个进行中/最终执行，不重复下发物理动作。
- `command_id`：Service 内部每条可变状态命令的唯一 ID，不由模型生成。
- 所有状态同时带 `source_sequence: int`、`sample_monotonic_ns: int`、
  `received_monotonic_ns: int`、`valid: bool` 和 `error_code: str | null`。
- 过期判断只使用单调时钟；墙上时间仅用于日志。

### 2.2 坐标、顺序与单位

- 长度为 m，角度为 rad，速度为 m/s 或 rad/s，力为 N。
- 中立位姿为 `Pose(position_m[3], quaternion_wxyz[4], frame_id)`。
- 四元数必须归一化，顺序固定为 `w,x,y,z`；RPY 只允许出现在 SDK adapter 内部。
- `ArmSide = LEFT | RIGHT`；每臂关节顺序固定为 `1..7`。
- 手指顺序固定为 `thumb,index,middle,ring,little`。
- 数组必须在构造时校验长度、有限数和范围；不允许静默截断、补零或交换左右侧。

### 2.3 执行状态

所有可变状态命令共享：

```text
ExecutionState = QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELLED | TIMED_OUT

ExecutionHandle:
  command_id: str
  owner_id: str
  state: ExecutionState
  submitted_monotonic_ns: int

CommandResult:
  command_id: str
  state: ExecutionState
  error_code: str | null
  message: str                 # 有界、可脱敏，不作为成功真值
  started_monotonic_ns: int | null
  finished_monotonic_ns: int | null
```

命令提交只说明已接收。只有查询到 `SUCCEEDED` 才表示控制器完成；语义成功仍由上层
Task/Verifier 判断。

## 3. 设备与服务接口

以下签名已用 Python `Protocol`/不可变 dataclass 落地；真实 adapter 仍须满足同一语义。

### 3.1 `ArmControlPort`

```text
get_state(side) -> ArmState
fk(side, joint_positions_rad[7]) -> FKResult
ik(side, target_pose, seed_joint_positions_rad[7]) -> IKResult
move_joint(lease, JointMoveCommand) -> ExecutionHandle
move_pose(lease, PoseMoveCommand) -> ExecutionHandle
move_linear(lease, LinearMoveCommand) -> ExecutionHandle
follow_pose(lease, FollowPoseCommand) -> ExecutionHandle
get_result(command_id) -> CommandResult
cancel(lease, command_id) -> CommandResult
emergency_stop(side, reason) -> CommandResult
close() -> None
```

`ArmState` 至少包含 7 维 `joint_positions_rad`、7 维 `joint_velocities_rad_s`、当前
`ee_pose`、SDK 使能/运行状态、反馈时间和有效性。若 SDK 没有可靠速度，可由带版本的
滤波器估算并标记 `velocity_source=ESTIMATED`，不能伪装成硬件测量。

映射固定为：

- `move_joint` -> MoveJ；
- `move_pose` -> MoveJP；
- `move_linear` -> MoveL；
- `follow_pose` -> FollowP；
- `fk/ik` -> 现有 FK/IK。

接口不定义 MoveC。`move_linear` 逐点 IK 成功仍不等于环境无碰撞。

### 3.2 `HandControlPort`

```text
open(read_only: bool) -> HandSession
get_state(side) -> HandState
command_synergy(lease, HandSynergyCommand) -> ExecutionHandle
hold(lease, side) -> CommandResult
stop(lease, side, reason) -> CommandResult
get_result(command_id) -> CommandResult
close() -> None
```

构造对象不得连接设备或发命令。`open(read_only=True)` 不得写姿态、速度或 torque。

公共手部目标固定为：

```text
HandSynergy:
  closure: float             # [0,1]，0=标定张开，1=标定闭合
  thumb_opposition: float    # [-1,1]，物理方向由 mapper 标定
  finger_spread: float       # [-1,1]，-1=窄，+1=宽
```

`HandSynergyMapper` 是版本化纯函数：

```text
to_g20_logical(side, synergy, calibration_id) -> logical_raw[20]
to_sim_independent(side, synergy, calibration_id) -> joint_targets_rad[16]
from_g20_state(side, logical_raw[20], calibration_id) -> HandSynergyEstimate
```

仿真的 5 个 mimic 关节由模型约束产生，不是 mapper 的额外动作。G20 20 维 raw 命令只
存在于 adapter 内部；11～14 保持标定文件规定的安全保留值。mapper 输出还必须通过
逐通道范围、死区、速率和连续速度上限检查。

`HandState` 包含硬件逻辑状态的安全内部快照、协同量估计、当前协同目标、跟踪误差、
温度、故障和新鲜度。torque limit 单独记为配置，不放入 measured effort。

### 3.3 `TactilePort`

```text
get_frame(side) -> TactileFrame
close() -> None
```

```text
FingerTactile:
  normal_force_n: float | null
  tangential_force_n: float | null
  tangential_direction_raw: float | null       # [0,255]
  matrix_weight_raw: uint8[12][6] | null
  approach_raw: float | null
  force_sample_monotonic_ns: int | null
  matrix_sample_monotonic_ns: int | null
  force_valid: bool
  matrix_valid: bool

TactileFrame:
  side: ArmSide
  fingers: FingerTactile[5]
  calibration_id: str
  feature_extractor_id: str
  received_monotonic_ns: int
  source_sequence: int
```

matrix 是 raw 空间权重，不是 N。每指力和矩阵分别判断新鲜度；某一指缺失不能用上一
帧数据冒充当前帧。`TactileFeatureExtractor` 由这些信号产生 contact、CoP 和 slip
score；阈值属于版本化配置，不写死在 PPO 网络或 Agent prompt 中。

`normal_force_n/tangential_force_n` 只接受经过逐指标定的实机合力或明确标记的仿真
代理。当前 G20 矩阵 raw 没有可信的 N 换算：真实 gateway 在完成标定前必须令
`force_valid=false` 且两个 N 字段为 `null`。名义 `20 N` 量程不能替代标定。若未来
决定让 actor 直接消费 raw 矩阵派生量，必须升级 observation/feature extractor manifest
并重新训练，不能在 V1 同一 layout ID 下偷偷改语义。

### 3.4 感知与抓取位姿

```text
TomatoPerceptionPort.detect(observation) -> TomatoDetectionResult
TomatoGraspPlanner.plan(target, arm_state, hand_state) -> TomatoGraspPlan
```

`TomatoDetectionResult` 返回小型目标摘要和后端持有的大数据引用。每个目标至少有
`target_id`、3D 中心/尺度、成熟度/可采置信度、遮挡度、估计果梗方向、观测序号和
不确定度；mask、RGB-D 和点云不进入 Agent。

```text
TomatoGraspPlan:
  grasp_plan_id: str
  target_id: str
  source_observation_sequence: int
  arm_side: ArmSide
  grasp_frame: Pose
  pregrasp_pose: Pose
  contact_pose: Pose
  retreat_pose: Pose
  pull_axis_in_grasp_frame: [0,0,1]
  max_residual_translation_m: float[3]
  max_residual_rotation_rad: tuple[float, float, float]
  mapping_calibration_id: str
  planner_version: str
  expires_monotonic_ns: int
```

抓取坐标系 `z+` 固定为远离果梗/植株的计划拉脱方向；从外侧预抓取向果实接触通常沿
`z-`，拉脱沿 `z+`。这个约定进入训练数据和动作监督器，不能由 adapter 改符号。

### 3.5 `MotionSafetyPort`

```text
assess_pregrasp(plan, arm_state) -> SafetyAssessment
assess_skill_envelope(plan, arm_state) -> SafetyAssessment
```

C1 的 scene 信息由 mock 配置内部持有；真实安全 adapter 可在内部读取版本化 scene
snapshot，但不能改变返回 DTO 或把场景大数组送入 Agent。

`SafetyAssessment` 包含 `assessment_id`、绑定的 plan/观测/臂状态序号、
`reachable`、`joint_limits_ok`、`path_sampled_ok`、`collision_status`、允许的局部工作包络、
到期时间和失败原因。过期或序号不匹配时不能复用。

首版不强制引入 MoveIt。可先实现适合固定采摘工位的轻量安全层：IK、软限位、采样
路径、预验证自碰撞姿态、工作空间盒/管道和感知障碍物距离。若目标场景超出该受限
工作空间，再把同一 Port 替换为 MoveIt 或其他碰撞规划器，Task/PPO 接口不变。

### 3.6 `PickVerifier`

```text
begin(execution_id, target, plan) -> VerifierSession
update(session, perception: TomatoDetectionResult | null, arm, hand, tactile)
  -> TomatoPickVerification
finish(session) -> TomatoPickVerification
```

```text
EvidenceState = CONFIRMED | REJECTED | UNKNOWN

TomatoPickVerification:
  execution_id: str
  detached: EvidenceState
  retained_without_slip: EvidenceState
  dropped: bool
  stable_window_s: float
  evidence_sources: str[]
  verifier_version: str
  valid: bool
  reason: str | null
```

实机 detached 证据由部署时可用的视觉果梗分离/目标随手运动和拉脱瞬态特征组合；
retained 由视觉持有关系、手部状态和触觉滑移趋势组合。具体阈值要用番茄实验数据标定，
但输出三态和成功不变量现在冻结。仿真果梗约束断裂可用于 reward、critic 和 mock 验证，
必须标记 `SIM_ONLY`，不能悄悄成为实机 actor 输入。

## 4. `TomatoPickSkill`（PPO）契约

### 4.1 调用边界

```text
TomatoPickSkillRequest:
  execution_id: str
  target_id: str
  grasp_plan_id: str
  arm_side: ArmSide
  policy_id: str
  max_duration_s: float

TomatoPickSkill.run(request, context) -> TomatoPickSkillResult
TomatoPickSkill.cancel(execution_id, reason) -> TomatoPickSkillResult
```

`context` 由 Service 注入，包含计划仓库、Ports、控制租约、安全评估和 Verifier；它不
序列化进 Agent 消息。Skill 开始前必须已经到达预抓取位。未满足条件直接返回明确失败，
不自行做一次未知的全局移动。

### 4.2 相位

策略活动相位固定为：

```text
CONTACT -> GRASP -> DETACH -> HOLD
```

- `CONTACT`：尚未建立有效接触；
- `GRASP`：已接触但尚未确认稳定承重包络；
- `DETACH`：已形成合格抓持但果梗尚未脱离；
- `HOLD`：果实已脱离，等待 Verifier 确认稳定保持。

这些相位是确定性诊断标签，不进入 Actor，也不屏蔽、改写或许可动作，不设独立相位
超时。PPO 在整个回合始终控制相同 22 维连续动作；只有全局时限和安全终止有效。

### 4.3 Actor 观测 V2

Actor 不接收 raw RGB、depth、mask、点云、360 taxel 矩阵、果实仿真真值或果梗约束
状态。`TomatoPickObservation` 固定为 113 个 `float32`，按下表扁平化。旧 114D
协同/phase 布局已经原位删除，不保留兼容层，也不加载按旧布局生成的 checkpoint。

|索引|字段|维度|说明|
|---|---|---:|---|
|0..5|`target_pose_error`|6|抓取坐标系中位置 3 + rotvec 3|
|6..11|`ee_twist`|6|抓取坐标系中线速度 3 + 角速度 3|
|12..18|`arm_joint_position_norm`|7|按当前侧 1..7，按软限位归一化|
|19..25|`arm_joint_velocity_norm`|7|按 manifest 速度尺度归一化|
|26..41|`hand_joint_position_norm`|16|G20 独立关节角，按冻结顺序和软限位归一化|
|42..57|`hand_joint_velocity_norm`|16|同顺序关节速度|
|58..62|`finger_pressure_norm`|5|拇指到小指的法向压力|
|63..67|`finger_pressure_rate_norm`|5|五指压力单周期变化|
|68..70|`gravity_direction`|3|抓取坐标系中的单位重力方向|
|71..92|`previous_action`|22|上一周期实际归一化动作|
|93..95|`target_arm_hand_valid`|3|目标跟踪、臂、手有效标记|
|96..100|`force_valid`|5|逐指压力有效标记|
|101..103|`target_arm_hand_age_norm`|3|按 manifest 最大年龄归一化|
|104..108|`force_age_norm`|5|逐指，过期裁剪为 1|
|109..111|`fruit_axes_norm`|3|番茄 `x/y/z` 半轴尺寸；目标无效时填 0|
|112|`episode_progress`|1|全局回合时间进度，裁剪到 `[0,1]`|

无效连续值填 0，同时对应 valid=0；年龄不能因填充值而清零。V2 对有效、已标定的
法向压力按 manifest 的 `20 N` 归一化上限裁剪；这只是策略归一化尺度，不声明
实机 raw 到 N 的换算关系。C2 的 N 值是明确标记的仿真代理；真机逐指标定完成前
`force_valid=0`。训练时允许 153 维 critic 额外读取仿真私有状态，但 actor 的 113 维布局在
仿真、shadow 和真机完全一致。

### 4.4 Actor 动作 V2

Actor 输出 22 个 `float32`，每项先裁剪到 `[-1,1]`：

|索引|字段|作用|
|---|---|---|
|0|`delta_x`|抓取坐标系横向小步|
|1|`delta_y`|抓取坐标系横向小步|
|2|`delta_z`|抓取坐标系轴向小步，正负均由 PPO 选择|
|3|`delta_rotation_vector_x`|抓取坐标系绕 x 小转动|
|4|`delta_rotation_vector_y`|抓取坐标系绕 y 小转动|
|5|`delta_rotation_vector_z`|抓取坐标系绕 z 小转动；这就是绕轴旋转，不再增加重复维度|
|6..21|`delta_g20_joint[0..15]`|16 个独立 G20 关节目标增量|

G20 顺序严格为 `thumb_joint2,index_joint1,middle_joint1,ring_joint1,little_joint1,`
`thumb_joint0,index_joint0,middle_joint0,ring_joint0,little_joint0,thumb_joint1,`
`thumb_joint3,index_joint2,middle_joint2,ring_joint2,little_joint2`。原始槽位 11～14
保留，5 个机械 follower 不进入动作。仿真把 6D 局部增量经 7 自由度 IK 执行；部署由
FollowP 执行，不能把旋转维度错误映射到第 7 关节。

动作在全回合保持同一语义，不按相位改变正负号或可用维度。只执行全局工作空间、硬件
关节限位、速度/增量上限和过力安全保护。实际裁剪后动作回写到下一帧 `previous_action`。

### 4.5 频率与 manifest

V2 在仿真、shadow 和初始真机统一为 20 Hz。Arm adapter 内部仍可按现有 50 Hz 调度；
G20 命令不超过其 30 Hz 上游限制。矩阵触觉允许较慢异步更新，由逐指 age/valid 表达。

每个 checkpoint 必须配套不可缺失的 `PolicyManifest`：

```text
policy_id, interface_version, checkpoint_sha256,
observation_layout_id, action_layout_id, policy_rate_hz,
action_scales, phase_step_limits, normalization_stats,
fruit_axis_lower_m, fruit_axis_upper_m,
arm_joint_order, g20_joint_order, finger_order, hand_joint_mapper_id,
tactile_feature_extractor_id, grasp_frame_convention,
training_asset_id, domain_randomization_id
```

manifest 不匹配时 runner 拒绝加载，不能靠数组长度碰巧一致来运行。

### 4.6 结果与失败

```text
SkillStatus = SUCCEEDED | RETRYABLE_FAILURE | SAFETY_ABORT |
              CANCELLED | TIMED_OUT | INFRASTRUCTURE_ERROR

TomatoPickSkillResult:
  execution_id: str
  target_id: str
  grasp_plan_id: str
  status: SkillStatus
  final_phase: str
  success: bool
  verification: TomatoPickVerification
  failure_code: str | null
  duration_s: float
  peak_normal_force_n: (float | null)[5]
  peak_tangential_force_n: (float | null)[5]
  final_arm_sequence: int | null
  final_hand_sequence: int | null
  final_tactile_sequence: int | null
  policy_id: str
```

冻结失败码：

```text
PRECONDITION_FAILED, TARGET_STALE, PLAN_STALE, NO_CONTACT,
UNSTABLE_GRASP, DETACH_NOT_CONFIRMED, SLIP_DETECTED, DROPPED,
EXCESS_FORCE, ARM_STATE_STALE, HAND_STATE_STALE, TACTILE_STALE,
OUT_OF_WORKSPACE, JOINT_LIMIT, COLLISION_UNSAFE, CONTROL_BUSY,
DEVICE_FAULT, COMMAND_REJECTED, CANCELLED, TIMEOUT, INTERNAL_ERROR
```

唯一成功不变量：

```text
success == True
iff status == SUCCEEDED
and verification.valid
and verification.detached == CONFIRMED
and verification.retained_without_slip == CONFIRMED
and verification.dropped == False
```

其他所有组合的 `success` 必须为 `False`。PPO、Agent 文本或 SDK 命令成功不能覆盖该
规则。

## 5. `AgriPickTask` 与 Agent Tools

V1 Task 只暴露以下 5 个工具：

|工具|模型可见参数|作用|是否改变世界|
|---|---|---|---|
|`detect_tomatoes`|无|返回小型候选摘要|否|
|`plan_tomato_grasp`|`target_id`, `arm_side=auto\|left\|right`|生成冻结的 grasp plan|否|
|`move_to_pregrasp`|`grasp_plan_id`|安全评估后到达预抓取位|是|
|`execute_tomato_pick`|`grasp_plan_id`|运行接触→抓取→拉脱→持有 Skill|是|
|`verify_tomato_pick`|`execution_id`|读取/完成独立验证|否|

handler 内部生成 command/execution ID、检查状态机和去重。模型不能提交速度、关节角、
位姿、策略动作、force 阈值或 timeout。

Task 状态机固定为：

```text
OBSERVE -> PLAN -> PREGRASP -> PICK -> VERIFY -> COMPLETE
   ^         |         |         |         |
   +---------+---------+---------+---------+
       可恢复失败按原因回到 OBSERVE 或 PLAN（受 retry budget 限制）

任意状态 -> ABORTED（安全失败、设备故障、取消或预算耗尽）
```

Task 只有在 `verify_tomato_pick` 返回符合第 4.6 节成功不变量的结果时完成。放置果实、
换筐和连续多果循环不属于 V1 工具白名单；它们在单果采摘闭环稳定后作为下一版本增加。

## 6. 控制权、重试和恢复

- `ControlLease` 绑定 `owner_id`、arm side、hand side、有效期和 controller epoch。
- SkillRunner 执行时持有臂+手租约；租约丢失立即停止/保持并返回 `CONTROL_BUSY` 或
  `SAFETY_ABORT`。
- `execute_tomato_pick` 是 at-most-once：同一 `grasp_plan_id` 在有效期内重复调用只返回
  既有 `execution_id` 和状态，不重复动作。
- 只读查询可重试；可变状态命令不做透明自动重试。通信结果不确定时先读状态和
  command ID，再决定恢复。
- `RETRYABLE_FAILURE` 可以由 Task 退回重新感知/规划；`SAFETY_ABORT`、设备故障和未知
  物理状态不能由 LLM 自行清除。
- 取消顺序为：停止产生新动作 -> Arm cancel/hold -> Hand hold/stop -> 释放租约 ->
  完成 Verifier 快照。软件急停保留为任何时刻可调用的故障安全操作。

## 7. Factory、配置和依赖规则

当前 C1 显式 Registry 都只注册 `mock`。以下其余名称是后续候选，不代表已实现：

```text
ARM_PORT_REGISTRY: mock | linkbot
HAND_PORT_REGISTRY: mock | g20
TACTILE_PORT_REGISTRY: mock | g20
PLANNER_REGISTRY: mock | tomato_geometry
SAFETY_REGISTRY: mock | constrained_workspace | moveit_future
SKILL_REGISTRY: mock | ppo
VERIFIER_REGISTRY: mock | tomato_multimodal
```

默认配置必须继续使用 mock，且不
导入 ROS、CAN、Isaac 或 torch。真实 adapter 放在稳定 Port 后面，不能让 `botSDK`
类型进入 `proto/`、Runtime、Agent 或 Task 的模型输入。

部署配置必须显式选择左右侧、CAN/ROS endpoint、标定 ID、工作空间、安全阈值、policy
manifest 和传感新鲜度。配置不保存凭据，也不在 import/构造时启动外部进程。

C4-O 已实现 `ArmSdkGateway`/`G20SdkGateway` 注入契约和冻结 Port 的转换 adapter，但未
注册 `linkbot`/`g20` 后端。只有 vendor gateway、部署 endpoint 和真机 shadow 验收完成
后才能修改 Registry；离线 adapter 类存在不等于硬件 profile 可用。

## 8. C1 实现验收

以下 C1 验收已经完成：

1. DTO 长度、单位、有限数、枚举、时间和不可变性测试；
2. 113 维 observation 和 22 维 action 的逐索引 golden test；
3. 实机 20 维与仿真 16 独立 + 5 mimic 不混用的 mapper 测试；
4. Task 只暴露 5 个工具，模型参数不含低层数组；
5. Mock 正常跑通 CONTACT→GRASP→DETACH→HOLD→VERIFY；
6. 注入无接触、拉脱失败、滑移、掉落、过力、陈旧传感、控制权冲突和超时；
7. 证明只有“脱离 + 未滑落稳定持有”能完成 Task；
8. 重复 `execute_tomato_pick` 不重复动作；取消会停止后续 command；
9. 默认离线测试不导入或启动 ROS/CAN/Isaac/torch；
10. 规范、`preview.md` 和 `IMPLEMENTATION_PLAN.md` 与实际类名和测试数同步。

完成证据：Isaac 托管 Python 通过 145 项 `unittest`；其中一个 soak 测试循环执行 100
个确定性成功/故障场景，另有 8 项 C4-O fake gateway、7 项 C2-0 跨文件测试和 12 项
C2-1 环境/训练门禁测试；`config/agri_pick_mock.yaml` 完成一次
CONTACT→GRASP→DETACH→HOLD→VERIFY→COMPLETE；默认导入不包含 ROS/CAN/Isaac/torch。
