# 采摘闭环原子能力路线

## 当前交付边界

当前代码把已经存在的番茄 PPO pick 作为一个受保护的中间步骤，不修改其 66D actor
observation、83D privileged critic state、11D action、checkpoint 或训练配置，也不启动
Isaac/PPO 训练。新 `HarvestCycleTask` 只在 Task/Service 层组合下面的执行链：

```text
system_self_check -> synchronized RGB-D -> instance support + VLM semantics -> 3D targets
                  -> select_harvest_target
                         occluded -> active_view_reposition -> new RGB-D -> reselect
                         visible  -> pinned reobserve -> plan
                  -> pregrasp_clearance_check -> move_to_pick_ready -> existing PPO pick
                  -> verify pick -> retreat -> transport_payload -> place -> verify cycle
                  -> optional reset_for_next_target -> new observation -> next cycle
                                                 failure -> safe retract
```

当前首批原子 Skill 是：

|Skill|当前实现|真实任务效果|成功证据|
|---|---|---|---|
|`system_self_check`|Orbbec SDK 布局校验、离线 mock、Isaac 顶置 RGB-D/机器人/触觉健康端口|运动前检查相机、机器人、触觉、急停、容器和策略合同|逐项 PASS/FAIL/UNKNOWN 与只读状态引用|
|`select_harvest_target`|无动作确定性原子 Skill + 配置化离线 workspace profile|先按新鲜度、成熟度、置信度、遮挡、尺寸和粗可达范围硬筛选，再按收益排序并 pin 目标|逐候选门禁、分数组成、拒绝原因、profile ID、选择 revision；始终 `motion_authorized=false`|
|`active_view_reposition`|受限离线 MoveL + 零动作 shadow|固定顶置相机不动；目标只因机械臂遮挡失败时，把机械臂移到有限去遮挡位并强制获取新帧|view profile/index、请求/实到位姿、位置姿态误差、固定相机位姿零漂移、新样本和可见度变化|
|`target_reobserve`|回放或隔离 Worker + 原子 RGB-D latest|对 pinned track 严格等待并取得更新的 mask-depth 三维观测|新设备时间戳、目标身份、遮挡、深度率、点数、拟合残差和标定引用|
|`pregrasp_clearance_check`|无动作 swept-capsule 点云检查 + mock/observation/Isaac 顶置深度端口|在准备位移动前排除枝叶、支架和相邻果实侵入路径|plan/target/scene 绑定、三段路径、最近障碍、所需净空、覆盖率和标定引用|
|`move_to_pick_ready`|离线、Isaac 动作 + 零动作 shadow|进入 PPO 训练分布内的准备位|两级安全评估、实际 palm 误差、9 维视觉输入与分布检查|
|`payload_retreat`|mock 或 Isaac arm/hand/tactile|果实脱离后低速回撤并保持抓持|机械臂命令、手状态、配置数量的有效接触指|
|`transport_payload`|mock 或 Isaac 两段低速 MoveL + 分段保持采样|从回撤点抬升并运输到容器前置位，运输中掉果则禁止放置|绑定容器前置位、两条命令、起点/抬升/终点三组手部与触觉证据|
|`payload_place`|mock 或 Isaac arm/hand + 独立果实刚体验证|进入配置容器位姿并松手|机械臂命令、手部打开、放置 receipt、容器内刚体位置|
|`localize_dropped_payload`|固定顶置相机端口、确定性 mock 与 Isaac 刚体真值验收端口|掉果后确认已入容器、位于专用回收区或必须人工处理；不自动复用悬挂果实 PPO|新鲜 world-frame 果实中心、置信度、容器归属、处置分类、`motion_authorized=false`|
|`reset_for_next_target`|空手确认 + 受限 MoveJP + 事务清理|验收后回到观测位并开始下一颗，不复用旧目标/计划/PPO execution|完成周期 ID、实际到位、空手状态、保留的周期历史与新观测要求|
|`safe_retract`|mock arm/hand|失败后保持当前手部状态并回到审核过的安全位姿|控制命令终态|

每个原子 Skill 都有稳定 `execution_id`、幂等终态、取消/关闭语义、失败码和按序 trace。
trace 只保存有界摘要和 `evidence_refs`；RGB、深度图、点云等二进制载荷不会进入日志。
所有新 descriptor 的 `hardware_approved` 当前均为 `false`。视觉到 PPO 的精确权威边界、
坐标约定、目标状态传播和硬件准入见
[`VISION_TO_PPO_PIPELINE.md`](./VISION_TO_PPO_PIPELINE.md)。

## 完成条件

模型工具调用成功不等于完成。`HarvestCycleService.completed` 只在以下条件同时成立时为真：

1. 原 PPO pick 的独立 verifier 确认果实已脱离并稳定持有，没有滑落；
2. 带载回撤完成，回撤后手部无故障且触觉仍确认抓持；
3. 两段运输完成，三个采样点均确认至少三指保持且没有掉果；
4. 容器放置 verifier 独立确认果实位于目标容器；
5. 松手已确认、手为空且没有掉落证据。

Agent 返回的文本或伪造工具 transcript 不参与完成判定。容器不可用、目标重观察失败、
带载不稳和放置证据未知都 fail closed。

`select_harvest_target` 只做“选哪个目标进入下一次严格观测”的决策，不做 IK、碰撞证明或
运动授权。当前 AABB profile 只说明目标中心是否位于某一条显式配置、绑定左右臂的离线粗
范围；当前 `pregrasp_clearance_check` 和安全规划仍必须独立通过。未配置 bounds、检测陈旧、
左右臂 profile 不匹配或全部候选不合格时都不会自动选择最接近的果实。

## 深度相机与真实硬件接入点

未来增加硬件时不改 Task 工具和原子 Skill 请求/结果合同，只替换显式注册的端口：

- `TargetObservationPort`：接入标定后的 RGB-D 同步采集、时间戳、内外参和目标重检测；
- `ClearanceScenePort`：把对齐深度或融合点云转换成有标定 ID、覆盖率和单调时间戳的 world-frame 场景；
- `PlacementVerifierPort`：使用固定顶置相机的容器 ROI、深度/实例证据确认放置；必要时再增加独立容器相机；
- `RoboticsBundle`：继续承载机械臂、G20、触觉、控制租约、急停和 SDK gateway；
- 配置注册表：硬件实现必须显式注册，不能目录扫描或自动 fallback 到真机。

真实接入前至少需要通过：相机内外参/时间同步、坐标系审计、容器位姿标定、只读 shadow、
单命令低速验收、触觉阈值标定、急停和掉果恢复演练。缺少任一项时保持 mock/shadow，不能
把 `hardware_approved` 改为 `true`。

当前已经提供 `SynchronizedRgbdSource -> TomatoVisionSpoolWriter -> latest.json -> Worker`
这一 SDK-neutral 接缝，以及录制导入和 recorded-shadow 原始数据门禁。后续硬件工作只允许在
该 source 接缝内增加 Orbbec SDK；不得把现有异步 MJPEG viewer、伪造标定或旧帧 fallback
注册成真实 `TargetObservationPort`。

## 从真实采摘任务推导的下一批能力

建议按下面顺序扩展，而不是继续放大单一 pick 策略：

1. `recover_dropped_payload`：只接受 `localize_dropped_payload` 的专用回收区证据，使用另行训练/
   验收的地面或托盘回收策略；不得复用悬挂果实 PPO；
2. `assess_fruit_damage`：放置后检测挤压、裂果和不可接受碰伤，作为质量 guardrail。

`localize_dropped_payload` 的合同和任务状态机现已完成：`payload_retreat`、`transport_payload`
或 `payload_place` 报告 `payload_dropped` 后，周期进入 `dropped_payload_recovery`，只能先定位。
落入容器时下一步为独立放置确认；落在配置回收区时只生成专用回收需求；区外或不可信位置
进入人工处理。任何分支都不会授权模型自由位姿或调用冻结的悬挂果实 PPO。

`pregrasp_clearance_check` 的合同、状态机和闭环已完成；当前 mock/metadata 点云只验证执行逻辑，
真实枝叶净空效果需要深度相机点云、外参标定和现场阈值数据到位后再验收。
`transport_payload` 的合同、分段动作、保持证据和放置绑定也已完成；真机阶段需要由轨迹控制器
提供运动过程中的高频触觉采样，当前同步端口在每个分段边界采样。
`reset_for_next_target` 已验证两个 observation sequence 的连续采摘；启动自检不会在每颗果实间
重复执行，但新一周期不能复用旧目标、计划、净空、PPO execution 或运输授权。

## ASPIRE 的借鉴边界

NVIDIA ASPIRE 的可取之处是以原子 primitive 为单位保存多模态执行轨迹，并从真实失败中
提炼可复用修复模式。本项目采用了“每个 primitive 有明确输入、终态、证据和失败分类”
这一原则，但不复制其多智能体 coordinator/actor 结构，也不会让模型自动修改并立即运行
真实硬件代码。这里仍由单 `TopAgent` 调用审核过的工具，真实执行必须经过确定性的安全门禁。

参考：<https://research.nvidia.com/labs/gear/aspire/>、<https://arxiv.org/abs/2607.00272>。
