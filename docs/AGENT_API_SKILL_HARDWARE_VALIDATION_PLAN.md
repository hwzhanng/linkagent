# Agent API → Skill → 真机能力验证计划

> 文档状态：持续更新（Living Document）  
> 初始版本：2026-08-13  
> 当前阶段：P7 已完成（Agent 臂 + RGB-D 相机只读组合；灵巧手跳过）
> 适用项目：`/home/miobarnacle/linkbot/linkagent`  
> 目标设备：LinkBot 左右机械臂、左右灵巧手、Orbbec Gemini Pro 深度相机  

## 1. 验证目标

本计划验证的不是“模型能够返回一段看似正确的文字”，而是确认接入真实模型 API 后，
Agent 能够在明确权限边界内完成以下真实调用链：

```text
模型 API
  → TopAgent 解析 tool call
  → Task 工具白名单
  → Atomic Skill / Service
  → 中立 Port
  → LinkBot / G20 / Orbbec Adapter
  → 底层 SDK、控制器或相机 Viewer
  → 结构化结果和证据回传模型
  → 模型根据真实结果给出终态回答
```

首轮重点验证两类能力：

1. **机械臂 IK 能力**：Agent 能读取当前关节状态、请求 IK、获得解算结果并明确区分
   “解算成功”和“已执行运动”。
2. **深度相机能力**：Agent 能读取相机健康状态、获取新鲜 RGB/Depth 帧信息，并调出
   受控网页预览供用户查看。

验证分为离线、真实 API、真机只读、影子计算、低速单命令五层。未经上一层验收，
不得进入下一层；“API 能调用工具”本身不构成实机运动授权。

## 2. 当前实现基线

### 2.1 已确认能力

- 系统为单 `TopAgent` 架构；`AgentRuntime` 管理外层控制步与资源释放。
- Task 持有完整工具白名单，模型无法访问任意 Python、Shell、SDK 或进程级注册表。
- 已实现 Ark OpenAI Responses 和 Ark Anthropic Messages 两类 API 协议适配。
- 已实现工具 JSON Schema、tool-call ID、结果回填、调用预算、异常边界和日志脱敏。
- `ArmControlPort` 已包含 `fk(...)` 和 `ik(...)`。
- `BotSdkLinkBotGateway` 已封装高级控制器的 FK/IK 和机械臂状态读取。
- Orbbec Viewer 已提供：
  - `GET /api/status`；
  - `GET /stream/rgb.mjpg`；
  - `GET /stream/depth.mjpg`；
  - 网页根路径 `/`。
- `linkbot_g20_shadow` 会打开真实设备的只读状态通道，并拒绝控制租约。
- 现有 40 项 Agent、Provider、Runtime、fake gateway 测试全部通过。
- `locate_target` 和 `agri_pick` 离线完整运行均已通过。
- 已新增隔离的 `hardware_capability_validation` Task：只提供能力清单、确定性离线 IK
  和离线相机回放元数据探针；所有结果显式标记 `simulated=true`、
  `hardware_accessed=false`、`motion_executed=false`。
- 已新增离线配置和 P0/P1 验收测试，覆盖组装/关闭、错误参数边界以及脚本化
  OpenAI Responses → Tool → Skill → 最终回答闭环。
- 已新增 `hardware_read_only_validation` Task；P5.5/P7 已通过真实 Ark API 调用真实
  机械臂状态、Shadow IK 和 Orbbec Viewer 状态，其中 P7 在一次 Agent 调用内完成右臂
  + RGB-D 组合读取。

### 2.2 当前缺口

- P7 已完成右臂 + RGB-D 的真实在线只读调用；后续若要执行运动，必须另开 R3 单次授权
  阶段，不能由本阶段只读结果推导运动许可。
- 灵巧手尚未连接，手部状态、触觉和手部控制 Skill 仍未接入；当前结果不包含手部能力。
- 当前 `task/factory.py` 尚未注册远端新增的 `harvest_cycle`，本地目录是保留差异后的混合树，
  正式验证前需要完成组合一致性检查。
- `linkbot_g20_shadow` 只能读状态与做 FK/IK，不能发送运动命令。
- 现有所有 Harvest Atomic Skill 的 `hardware_approved` 默认仍为 `false`。
- 实机触觉力标定、完整急停证据和真机 verifier 尚未达到采摘动作放行条件。

## 3. 验证范围与明确非目标

本轮范围：

- API tool call 是否能到达真实本地 Skill handler；
- Skill 是否确实调用底层 Port/Gateway；
- IK 是否使用真实当前状态作为 seed 并返回可审计结果；
- 相机是否返回真实、连续、新鲜的 RGB/Depth 状态与预览；
- Agent 是否能根据失败结果停止、改用只读能力或请求人工处理；
- 每次调用是否具有唯一 ID、时间戳、输入摘要、输出摘要和证据引用。

本轮不直接包含：

- 自动番茄采摘；
- PPO 策略上实机；
- Agent 自主使能机械臂；
- Agent 自主解除急停；
- 任意关节目标、任意笛卡尔目标或连续轨迹控制；
- Agent 修改相机、机械臂或灵巧手系统配置；
- 将 API 返回文本当作设备动作成功证据。

## 4. 待新增的验证专用 Task 与 Skills

已新增独立 `hardware_capability_validation` Task 和 `hardware_read_only_validation` Task。
它们与正式 `agri_pick`、`harvest_cycle` 隔离，避免验证工具意外进入生产采摘流程。P0/P1
使用确定性离线探针；P2/P3/P4/P5/P5.5/P7 的结果另外标记真实 API、真实 Port/Gateway
或真实 Viewer 来源，不能混用离线证据冒充真机证据。

### 4.0 P0/P1 已实现的离线探针

| Agent 工具名 | 返回内容 | 是否有物理副作用 |
|---|---|---:|
| `get_validation_capabilities` | 能力白名单、阻断动作和门禁状态 | 否 |
| `probe_inverse_kinematics` | 确定性候选关节向量和零运动残差 | 否 |
| `probe_camera_preview` | 离线回放帧元数据，不打开流 | 否 |

### 4.1 P2+ 计划中的生产级无副作用能力探针

| Agent 工具名 | 对应能力 | 是否有物理副作用 |
|---|---|---:|
| `get_validation_capabilities` | 返回本次允许的工具、模式和门禁状态 | 否 |
| `get_robot_health` | 读取臂、手、触觉、急停和连接状态摘要 | 否 |
| `get_arm_state` | 读取指定侧 7 关节、使能、故障和时间戳 | 否 |
| `get_hand_state` | 读取指定侧手部关节和会话状态 | 否 |
| `get_tactile_state` | 读取触觉原始有效性、新鲜度和标定 ID | 否 |
| `compute_forward_kinematics` | 对明确关节向量执行 FK | 否 |
| `compute_inverse_kinematics` | 对目标位姿和当前 seed 执行 IK | 否 |
| `get_camera_status` | 读取 `/api/status` | 否 |
| `get_camera_preview` | 返回受控 Viewer URL、流状态与帧序号 | 否 |
| `capture_rgbd_evidence` | 固化一组新鲜 RGB/Depth 证据引用 | 仅数据写入 |

### 4.2 IK Skill 输出要求

`compute_inverse_kinematics` 的结果至少包含：

- `execution_id`；
- `side`；
- `target_pose` 与 `frame_id`；
- `seed_source=current_feedback`；
- 7 维 seed 关节角；
- `solution_found`；
- 7 维候选关节角；
- 正解回代末端位姿；
- 位置与姿态残差；
- 关节限位余量；
- 与当前关节的最大/总变化量；
- `collision_checked` 和检查版本；
- `motion_executed=false`；
- 失败代码和有界错误摘要；
- 对应 Skill trace/evidence 路径。

仅返回关节解不算通过。必须做 FK 回代、限位检查，并明确没有执行运动。

### 4.3 相机 Skill 输出要求

`get_camera_preview` 不把 MJPEG 二进制塞进模型上下文，只返回：

- Viewer 页面 URL；
- RGB/Depth 流 URL；
- 设备 ID/序列号；
- 标定 ID；
- RGB/Depth 是否 live；
- 分辨率、FPS、最近帧时间与帧龄；
- 帧序号或 capture ID；
- 数据来源是否为真实设备；
- `preview_opened`（由本地 UI 启动器确认，而不是由模型声称）；
- 错误摘要。

模型可以请求“调出画面”，本地 UI adapter 负责打开 Viewer；模型不能拼接任意 URL，
也不能直接访问任意网络地址。

## 5. 分阶段执行计划

### P0：冻结基线与组合一致性

状态：`PASS`

工作项：

- [x] 确认 Agent、Runtime、Provider Adapter 和 fake gateway 主链路。
- [x] 运行相关 40 项离线测试。
- [x] 运行 `locate_target` 离线闭环。
- [x] 运行 `agri_pick + mock` 离线闭环。
- [x] 记录当前本地提交、远端提交和工作区差异清单。
- [x] 核对远端新增文件与本地旧 Composition Root 的兼容性。
- [x] 在不覆盖本地差异的前提下注册验证专用 Task。
- [x] 将本计划涉及的配置、测试和证据目录固定下来。
- [x] 运行验证 Task，确认资源关闭且运行时未导入 Isaac/CAN/Orbbec 模块。

通过条件：验证 Task 能在纯 mock 下组装和关闭，且不导入 Isaac、CAN 或相机 SDK。

### P1：离线 API 协议到 Skill 探针

状态：`PASS（脚本化 Provider；未访问真实 API）`

工作项：

- [x] 用 Mock Provider 配置调用 `get_validation_capabilities`，并用脚本化 OpenAI
  Responses 调用 `probe_inverse_kinematics`。
- [x] 验证工具 schema 不包含 handler、凭据、任意 URL 或路径参数。
- [x] 验证每个 tool-call message 只有一个离线 Skill `execution_id`，且执行计数为 1。
- [x] 验证 handler 只执行一次。
- [x] 验证 Skill 结果回填模型后，模型产生最终回答。
- [x] 未知工具、错误参数、重复 ID、超预算和传输异常均由既有 TopAgent/Provider 回归
  测试覆盖；本阶段没有网络超时或真实设备副作用。

通过条件：所有离线调用和错误都有结构化证据，失败时不触发其他 Skill；真实 API 和真实
设备仍保留在 P2 以后，未被本阶段冒充验收。

### P2：真实模型 API → 无副作用 Skill

状态：`PASS`

工作项：

- [x] 通过 `ARK_API_KEY` 环境变量加载 API key，配置文件和 P2 证据中不出现明文凭据。
- [x] 运行 5 次 `get_validation_capabilities`。
- [x] 运行 5 次 `probe_nonce`，确认实际 handler 执行且 nonce 完整回传。
- [x] 验证 API 请求、tool call、tool result、最终回答的两轮协议。
- [x] 使用中文自然语言和明确工具请求；10/10 工具选择正确。
- [x] 每个 Task 实例仅暴露一个工具，未授权工具无法进入白名单。

通过条件：真实 API 完成 10 次独立调用，工具选择正确率 100%，handler 重复执行为 0，
凭据泄露为 0。证据目录：`logs/agent_skill_validation/p2-20260813T064304Z/`。

### P3：API → Fake Gateway → IK/相机回放

状态：`PASS`

工作项：

- [x] 将 `probe_inverse_kinematics` 接到只读 Fake Gateway Shadow。
- [x] 使用固定关节状态和固定目标位姿验证确定性。
- [x] 对解算结果做 FK 回代、软限位裕量和位置残差检查。
- [x] 将相机探针接到固定 RGB-D replay 元数据，返回设备/标定身份和证据引用。
- [x] 让真实 API 分别调用能力、IK 和相机 replay 工具；每个 Task 实例只暴露一个工具。
- [x] 验证模型不会把 `solution_found=true` 描述成“机械臂已经运动”。

通过条件：API → Agent → Skill → Port → fake gateway 的调用链均有 provider `tool_call_id`
和 Skill `execution_id` 的审计映射；FK 回代结果与直接 Fake Gateway 结果一致。
本阶段没有真实碰撞模型，因此 `collision_checked=false` 被明确保留，不能推导真机安全。

### P4：真机连接前只读自检

状态：`PASS（双臂 + RGB-D；手部按用户说明未连接并跳过）`

前置条件：

- [x] 双臂物理急停状态未由本次脚本改变；软件使能均为 false。
- [x] 当前验证未发送运动/使能命令；手部未连接，手部检查按用户说明跳过。
- [x] 左臂 `can0`、右臂 `can1`，均为 1 Mbps、ERROR-ACTIVE，反馈完整。
- [x] Orbbec Viewer 独占相机并正常启动；RGB/Depth 均 live。
- [x] 未发现其他相机占用进程；CAN 遥测由本次只读审计独占。
- [x] P4 审计没有 Agent 运动工具或写控制权限。

验证顺序：

1. 启动 Orbbec Viewer；读取 RGB/Depth 状态和新帧。
2. 读取左右臂七关节 CAN 反馈，并计算当前 FK 姿态。
3. 保持软件失能和写权限关闭；未打开手部 CAN。
4. 验证 P4 只读审计没有运动工具、目标写入和控制租约。

通过条件：双臂状态来自真实硬件且七关节新鲜；相机页面/流可见且 RGB/Depth 新鲜；
手部未连接被明确记录；整个阶段零使能变化、零目标写入、零运动命令。允许的 CAN
状态查询/遥测帧单独计数，不计为运动写入。最新证据目录：
`logs/agent_skill_validation/p4-20260813T072431Z/`。

### P5：真机 Shadow IK

状态：`PASS（实时反馈 seed；Z−方向记录局部工作空间边界）`

工作项：

- [x] 读取指定机械臂的当前 7 关节状态作为唯一 seed；未假设机械臂处于零位。
- [x] 先请求“当前位置 IK”，验证返回解接近当前关节。
- [x] 请求当前位置附近的目标：X/Y/Z 正负 5 mm，每次只偏移一个轴。
- [x] 每个目标只做 IK + FK 回代，不发送运动；不可达目标不被强行改写。
- [x] 使用左右臂各自的高级控制器配置和方向映射。
- [x] 检查 J6/J7 的解是否出现跳支、绕圈或越限倾向。
- [x] 对同一目标重复调用，确认解算确定性与幂等。
- [x] 生成与 URDF 控制台相同的 Smooth MoveJ/FK 预演采样；控制台保持仿真只读。

建议初始偏移仅用于解算：`±5 mm`，姿态保持不变。此数值不构成运动许可。

通过条件：

- 当前位姿 IK 成功；
- FK 回代位置误差和姿态误差满足高级控制器既有验收阈值；
- 所有关节位于软限位内；
- J6/J7 不发生不连续跳变；
- 日志明确记录 `motion_executed=false`。

若固定姿态下的某个 5 mm 邻域不可达，必须沿同一方向缩小位移并记录边界；这类结果
不允许被包装成“目标已达到”，但在实时反馈、当前位置 IK、其余邻域、限位、J6/J7
连续性和零运动证据均通过时，可作为 P5 的已审计工作空间边界结束本阶段。

### P6：人工确认后的低速单命令验收

状态：`PASS（右臂单次低速 z+5mm；未开放连续动作）`

本阶段必须由用户在当次测试窗口明确授权后才能解锁。API 验证本身不自动授权 P6。
当前先执行 P6-预检：读取最新反馈、生成一个单臂小位移候选轨迹、在 URDF/运动学中
检查轨迹和门禁；预检不会发送任何实机目标。

约束：

- 一次只允许一侧机械臂；
- 一次只允许一个已预演目标；
- 必须从当前真实反馈生成轨迹；
- 使用已验证的平滑轨迹控制，不允许瞬时关节目标；
- 限制速度、加速度、jerk 和最大位移；
- J6/J7 使用连续角和跳变检查；
- 任意轨迹点失败时保持使能并停止，不得自动失能掉落；
- 工具返回后不自动请求下一个运动；
- 每条命令都需要新的人工确认 token；
- 急停优先级高于 Agent、Skill、SDK 和网页控制台。

首条运动应是从真实当前位置出发、末端单轴小位移的低速 MoveL/受控轨迹，不做抓取、
不闭合灵巧手。

通过条件：实机、URDF、反馈轨迹和目标方向一致；停止行为安全；一次授权只产生一次命令。

P6-预检通过条件：

- [x] 选定右臂和唯一 `z_plus_5mm` 目标；
- [x] 目标由执行前最新实时反馈生成，未复用旧 P5/P5.5 seed；
- [x] 轨迹时长 5 s，最大关节速度不超过 `0.12 rad/s`；
- [x] 轨迹所有点在当前软件限位内，J6/J7 无跳变，起点与执行前反馈一致；
- [x] URDF/FK 预演完成后才执行；
- [x] 工作区、物理急停、预演确认和右臂单次授权已由现场操作者确认。

### Run `p6-preflight-20260813T075653Z`

- 日期：2026-08-13
- 阶段：P6 无运动安全预检
- 目标：左右臂分别生成当前反馈到 `z_plus_5mm` 的候选轨迹；尚未选定实际执行侧。
- 结果：左右臂反馈均 7/7 健康、软件未使能、IK/FK 通过；每条 Smooth MoveJ 时长 5.0 s、
  501 点，最大关节速度分别约 `0.0779 rad/s`（左）和 `0.0875 rad/s`（右），均低于
  `0.12 rad/s` 限制；轨迹起点和终点精确匹配。
- 安全结果：`motion_commands_sent=0`、`target_writes=0`、`enable_changes=0`，
  `ready_for_real_execution=false`；工作区、物理急停、URDF 人工确认和单次授权 token
  尚未提供，因此本次不能执行实机动作。
- 证据目录：`logs/agent_skill_validation/p6-preflight-20260813T075653Z/manifest.json`
- 结论：预检 PASS；P6 实机动作仍 BLOCKED_BY_DESIGN。
- 下一步：由操作者选择一侧，确认现场安全条件和 URDF 预演，再提供一次性授权。

### Run `p6-right-execute-20260813T080115Z`

- 日期：2026-08-13
- 阶段：P6 右臂单次低速实机动作
- 操作者授权：用户已确认工作区清空、物理急停复位且可触达，并指定右臂；使用一次性
  `RIGHT_ARM_AREA_CLEAR` 确认字符串，本次无重试、无第二目标。
- 执行目标：执行前重新读取右臂反馈，以最新 seed 计算末端 `z_plus_5mm` Shadow IK。
- 轨迹：Smooth MoveJ，5.0 s、501 个规划点，实际调度 251 个目标采样；最大关节速度
  `0.0875 rad/s`，低于 `0.12 rad/s`，IK 位置误差约 `1.27e-13 m`。
- 结果：轨迹成功；最终 7/7 反馈健康，最大关节目标误差小于 `0.03 rad`，无 CAN 错误。
- 安全结果：未重试、未复位、未执行第二动作；右臂当前保持软件使能以保持最终姿态，
  后续动作前必须重新读取反馈并再次预演。
- 证据目录：`logs/agent_skill_validation/p6-right-execute-20260813T080115Z/manifest.json`
- 结论：PASS（仅证明一次授权的单臂低速动作；不代表连续控制或抓取许可）
- 下一步：右臂已安全失能；P7 改为先验证 Agent 臂+相机只读组合，灵巧手跳过。

### Run `right-disable-20260813T080617Z`

- 日期：2026-08-13
- 阶段：P6 完成后的右臂安全失能
- 操作者：用户确认右臂可以安全失能
- 结果：发送 7 个右臂禁用帧；失能后 7/7 反馈健康、无 CAN 错误、无运动目标写入。
- 证据目录：`logs/agent_skill_validation/right-disable-20260813T080617Z/manifest.json`
- 结论：PASS；右臂当前保持失能。

### P7：Agent 臂 + 相机只读组合能力（灵巧手跳过）

状态：`PASS（真实 Ark API；右臂已失能；手部跳过）`

工作项：

- [x] 真实 Ark API 在同一次 Agent 调用中调用 `get_live_arm_state` 和
  `get_live_camera_preview`。
- [x] Agent 正确汇总右臂 7/7 状态、软件失能状态和 RGB/Depth live 状态。
- [x] Agent 明确声明本阶段只读，不执行 IK 目标、轨迹、使能或急停。
- [x] 保存 `tool_call_id`、`execution_id` 和设备/帧状态证据。
- [x] 灵巧手未连接，明确跳过手部和触觉，不伪造手部结果。
- [x] 不执行 PPO 抓取，不闭合手指。

通过条件：模型在同一次受限 Agent 调用中编排机械臂状态和相机状态 Skill；两者均来自真实
设备；右臂保持失能；没有任何运动或写入；手部跳过被明确记录。

### Run `p7-20260813T080839Z`

- 日期：2026-08-13
- 阶段：P7 Agent 臂 + RGB-D 相机只读组合
- 操作者授权：用户明确允许在进入本阶段前安全失能右臂；右臂失能证据为
  `right-disable-20260813T080617Z`。
- Agent/API：一次真实 Ark OpenAI Responses 调用（`doubao-seed-2.0-pro`），同一调用内
  严格完成两个工具调用：`get_live_arm_state(side=right)`、`get_live_camera_preview()`。
- 实际结果：2/2 工具成功；右臂 `can1` 反馈 7/7 健康、软件未使能；Orbbec 序列号
  `AY2PB4200YY`，RGB/Depth 均为 live；Agent 最终回答正确区分
  `hardware_accessed=true`、`motion_executed=false`、`write_control_authorized=false`。
- 工具白名单：未暴露 IK、轨迹、目标写入、使能、复位、急停或灵巧手工具；灵巧手和触觉
  按当前连接状态明确跳过。
- 安全结果：`agent_invocations=1`、`motion_commands_sent=0`、`target_writes=0`、
  `enable_changes=0`；未执行 PPO 或手指动作。
- 证据目录：`logs/agent_skill_validation/p7-20260813T080839Z/manifest.json`
- 运行脚本：`script/run_p7_agent_arm_camera_validation.py`
- 结论：PASS；P7 验收完成。后续若接入实机运动，必须另开 R3 单次人工授权阶段；手部
  接入与触觉验证另行规划。

## 6. 安全与权限规则

### 6.1 工具分级

| 等级 | 示例 | 默认权限 |
|---|---|---|
| R0 | capability、配置摘要 | 自动允许 |
| R1 | 状态、相机健康、FK、IK | 自动允许但必须审计 |
| R2 | 保存帧、打开本地 Viewer | 允许，限制路径和 URL |
| R3 | 使能、复位、MoveJ/MoveL、手指动作 | 默认禁止，单次人工授权 |
| R4 | 连续策略、抓取、双臂协同 | 本计划不授权 |

### 6.2 强制约束

- API 模型永远不能接收或输出 API key、sudo 密码、CAN 设备写权限令牌。
- 工具白名单按 Task 实例生成，不使用进程级万能工具表。
- URL 只能来自配置的 loopback Orbbec Viewer，不接受模型提供的任意 URL。
- 文件证据只能写入项目内固定目录，拒绝绝对路径和路径穿越。
- IK Skill 与运动 Skill 必须是两个不同工具。
- `compute_inverse_kinematics` 永远不能隐式执行运动。
- 模型说“已完成”不能作为设备成功依据；必须读取 SDK 状态或独立 verifier。
- 任何异常默认保持当前安全状态，不自动复位、不自动失能、不自动重试动作。

## 7. 证据与验收记录

建议每次验证创建：

```text
logs/agent_skill_validation/<run_id>/
  manifest.json
  redacted_transcript.json
  tool_calls.jsonl
  skill_trace.jsonl
  device_state_before.json
  device_state_after.json
  camera_status.json
  captures/
  summary.md
```

`manifest.json` 至少记录：

- 当前 Git HEAD、远端 HEAD 和工作区差异摘要；
- 配置文件 SHA-256；
- 模型 provider、协议、模型名，不记录 key；
- Agent/Task/Skill 版本；
- 设备 ID、标定 ID、endpoint 摘要；
- 是否连接真机；
- 是否允许写控制；
- 开始/结束时间；
- 人工确认者与确认步骤编号；
- 最终 PASS/FAIL/BLOCKED。

## 8. 每次验证的更新模板

### Run `<run_id>`

- 日期：
- 阶段：
- 操作者：
- 模型/API：
- 配置：
- 设备连接：无 / 相机 / 臂 / 手 / 全部
- 控制权限：只读 / 单命令 / 禁止
- 执行步骤：
- 预期结果：
- 实际结果：
- 证据目录：
- 安全事件：
- 失败与根因：
- 修复提交：
- 回归结果：
- 结论：PASS / FAIL / BLOCKED
- 下一步：

## 9. 总体验收标准

只有同时满足以下条件，才能声称“Agent 接入 API 后能够调用相关 Skill”：

1. tool call 来自真实 API 响应；
2. 工具名属于当次 Task 白名单；
3. handler 被实际执行且仅执行一次；
4. Skill trace 证明调用到目标 Port/Gateway；
5. 结果来自真实底层能力，不是模型文本或 mock 冒充；
6. 工具结果回传模型后，模型能正确区分成功、失败、只计算和已执行；
7. 失败时没有越权调用、重复动作或安全状态恶化；
8. 日志、设备状态和相机证据能够独立复核。

对于本轮两个核心样例：

- **IK 通过**：Agent 调用真实机械臂高级控制器 IK，完成 FK 回代和限位检查，并明确
  `motion_executed=false`；后续单命令运动另行验收。
- **相机通过**：Agent 读取真实 Gemini Pro 状态并调出可见的 RGB/Depth Viewer，帧状态
  新鲜且设备/标定身份匹配。

## 10. 启动下一步前需要的信息

- API 提供商与协议：现有 Ark OpenAI Responses / Ark Anthropic Messages，或新增其他提供商；
- 模型名或 endpoint ID；
- API key 对应的环境变量名，凭据不得写入本文或聊天记录；
- Orbbec Viewer 计划使用的本地端口；
- 双臂和双手当前连接 endpoint；
- 首轮真机验证只读时间窗口；
- P6 低速单命令阶段是否另行开放，默认不开放。

## 11. 变更记录

### Run `p0-20260813-offline-capability-validation`

- 日期：2026-08-13
- 阶段：P0
- 操作者：Codex（离线自动化）
- 模型/API：`MockChatModel`；无网络
- 配置：`config/hardware_capability_validation_offline.yaml`
- 设备连接：无
- 控制权限：禁止
- 执行步骤：组装验证 Task；执行能力清单、离线 IK、离线相机元数据；关闭 Task/Env；检查导入模块。
- 实际结果：三项工具均成功；`hardware_accessed=false`、`motion_executed=false`；Task/Env 均关闭。
- 证据：`tests/test_api_skill_validation.py`、CLI 日志 `logs/run_20260813T061614_862521Z_pid22943.log`。
- 安全事件：无
- 回归结果：P0 专项 + 原有 Provider/Runtime/Fake Gateway 共 43 项通过。
- 结论：PASS
- 下一步：进入 P1 脚本化 Provider 协议闭环。

### Run `p1-20260813-scripted-openai-ik-skill`

- 日期：2026-08-13
- 阶段：P1
- 操作者：Codex（脚本化 Provider）
- 模型/API：`ArkOpenAIResponsesModel` + `RecordingTransport`；未连接公网
- 配置：测试内确定性响应
- 设备连接：无
- 控制权限：禁止
- 执行步骤：脚本化 function call → `probe_inverse_kinematics` → 结构化结果回填 → 最终回答。
- 实际结果：两轮协议成功；handler 执行 1 次；`execution_id` 唯一；未执行运动。
- 证据：`tests/test_api_skill_validation.py` 与既有 `tests/test_provider_models.py`。
- 安全事件：无
- 回归结果：P0/P1 专项及既有 40 项回归共 43 项通过。
- 结论：PASS（仅证明离线 API 适配与 Skill 边界，不证明真实 API/硬件）
- 下一步：按用户确认后再规划 P2；不自动连接实机。

### Run `p2-20260813T064304Z`

- 日期：2026-08-13
- 阶段：P2
- 操作者：Codex（用户已授权真实 API 调用）
- 模型/API：Ark OpenAI Responses / `doubao-seed-2.0-pro`
- 配置：`config/_model.yaml` 的 `ark_openai_doubao`
- 设备连接：无
- 控制权限：禁止
- 执行步骤：5 次能力清单 + 5 次 nonce 探针；每次独立两轮 tool-call 闭环。
- 实际结果：10/10 成功；工具选择正确率 100%；每次 handler 1 次；nonce 全部匹配。
- 证据目录：`logs/agent_skill_validation/p2-20260813T064304Z/`
- 安全事件：无；API key 未写入配置、日志或 manifest。
- 回归结果：P2/P3 专项及既有 Provider/Runtime/Fake Gateway 共 44 项通过。
- 结论：PASS
- 下一步：进入 P3 Fake Gateway/replay。

### Run `p3-20260813T064658Z`

- 日期：2026-08-13
- 阶段：P3
- 操作者：Codex（用户已授权真实 API 调用）
- 模型/API：Ark OpenAI Responses / `doubao-seed-2.0-pro`
- 配置：`config/_model.yaml` 的 `ark_openai_doubao`
- 设备连接：无；Fake Gateway Shadow + RGB-D replay
- 控制权限：禁止
- 执行步骤：能力清单、右臂 Fake Gateway IK/FK、wrist RGB-D replay 三个独立调用。
- 实际结果：3/3 成功；IK 最大位置残差 `0.0 m`；解来自 fake gateway 当前反馈；
  相机设备 `orbbec-replay-001`、标定 `orbbec_gemini_pro_mock_calibration_v1` 匹配。
- 证据目录：`logs/agent_skill_validation/p3-20260813T064658Z/`
- 安全事件：无；未打开真实相机、未发送控制命令、未获取控制租约。
- 限制：Fake Gateway 无碰撞模型，结果明确标记 `collision_checked=false`。
- 回归结果：P2/P3 专项及既有 Provider/Runtime/Fake Gateway 共 44 项通过。
- 结论：PASS（仅 Fake/Replay，不代表真机验收）
- 下一步：P4 真机连接前只读自检，仍需用户明确授权并满足设备前置条件。

### Run `p4-20260813T072431Z`

- 日期：2026-08-13
- 阶段：P4 真机连接前只读自检
- 操作者：Codex（用户确认摄像头连接已恢复）
- 设备连接：左/右机械臂 + Orbbec RGB-D；灵巧手未连接，按用户说明跳过
- 控制权限：只读；未获取可写控制租约
- 相机结果：`SV1301S_U3`，序列号 `AY2PB4200YY`，固件 `RD3013`，SDK `1.10.16`；
  RGB `1280×720` live、Depth `1280×800` live，Viewer `http://127.0.0.1:8080/`。
- 机械臂结果：`can0`/`can1` 均为 1 Mbps、`ERROR-ACTIVE`；左右臂各 7/7 关节反馈新鲜，
  `software_enabled=false`，并记录了当前关节状态和 FK 姿态。
- 安全结果：`motion_commands_sent=0`、`target_writes=0`、`enable_changes=0`；仅发送了
  有界状态查询/温度遥测帧，共 `1395` 帧，未发送使能、模式、目标、轨迹或急停帧；
  只读租约拒绝验证通过。
- 证据目录：`logs/agent_skill_validation/p4-20260813T072431Z/manifest.json`
- 安全事件：无；验证脚本结束后关闭 CAN 遥测运行时，未改变硬件使能状态。
- 结论：PASS（双臂 + RGB-D 可用；手部未连接故按说明跳过）
- 下一步：P5 真机 Shadow IK，仅计算 IK/FK 回代，不发送运动。

### Run `p5-20260813T073408Z`

- 日期：2026-08-13
- 阶段：P5 真机 Shadow IK
- 操作者：Codex（用户说明机械臂当前不在零位；未执行回零）
- 设备连接：左右机械臂；灵巧手未连接；相机不参与本阶段
- 控制权限：只读；IK seed 来自本次实时七关节反馈 `seed_source=current_feedback`
- 实际结果：左右臂当前位置 IK/FK 回代通过；X/Y ±5 mm、Z +5 mm 目标全部通过；
  每个目标重复解算的最大关节差为 `0`，所有候选均在软件限位内，J6/J7 未发生跳变。
- 边界结果：保持测得姿态不变时，Z −5 mm 请求在两臂均不可达。沿同一方向缩小探测后，
  左臂约 `−0.3125 mm`、右臂约 `−0.078125 mm` 为最近可收敛探针，因此记录为局部
  工作空间边界，而非修改 seed、回零或强行放宽限位。
- URDF 预演：每个可达目标及最近可达边界探针均生成 2.0 s、201 点 Smooth MoveJ/FK
  仿真预演；双臂控制台 `http://127.0.0.1:8770/` 已加载本次实时 seed，未开放 TX。
- 安全结果：`motion_commands_sent=0`、`target_writes=0`、`enable_changes=0`；仅发送
  `1921` 个有界遥测查询帧；软件使能仍为 false。
- 证据目录：`logs/agent_skill_validation/p5-20260813T073408Z/manifest.json`
- 限制：P5 未做碰撞检测，也未执行实机动作；这两项只能在后续授权阶段单独验收。
- 结论：PASS（含已审计的 Z− 局部工作空间边界）
- 下一步：完成 P5.5 的真实 API 凭据验收；P6 仍需用户当次明确授权。

### P5.5：Agent → 真机只读 Skill

状态：`PASS（真实 Ark API → Agent → 真机只读 Skill）`

新增独立 Task：`hardware_read_only_validation`。它只暴露以下工具：

- `get_live_arm_state`：读取左右臂七关节、反馈新鲜度、软件使能和 FK；
- `compute_live_shadow_ik`：使用实时反馈 seed，对固定的 P5.5 邻近目标做 IK/FK 回代；
- `get_live_camera_preview`：读取 loopback Orbbec Viewer 状态并返回固定预览 URL。

所有工具均显式返回 `hardware_accessed=true`、`motion_executed=false`、
`write_control_authorized=false`。Task 内部只允许有限 CAN 遥测查询帧，不暴露命令、轨迹、
使能、模式或急停方法；灵巧手因未连接不开放。

当前证据：

- 本地 `TopAgent → live Skill → 真实双臂/相机` 运行通过，三次工具调用均成功，证据目录：
  `logs/agent_skill_validation/p5.5-local-20260813T074659Z/`。
- 真实 Ark API 首次使用旧凭据时返回 HTTP 401；更新凭据后已完成真实闭环，旧失败记录保留在：
  `logs/agent_skill_validation/p5.5-20260813T074554Z/`。

通过条件：真实 API 的每个工具调用都到达对应 handler；状态与 P5 实时结果一致；IK/FK、
限位和相机 live 状态正确回传；所有调用仍保持 `motion_executed=false`。已满足。

### Run `p5.5-20260813T075210Z`

- 日期：2026-08-13
- 阶段：P5.5 真实 Ark API → Agent → 真机只读 Skill
- 操作者：Codex（用户提供新凭据；仅当前进程使用）
- API/模型：Ark OpenAI Responses / `doubao-seed-2.0-pro`
- 工具调用：5/5 成功，分别读取左右臂状态、计算左右臂 `z_plus_5mm` Shadow IK、读取
  Orbbec RGB-D Viewer；每次都有唯一 `tool_call_id` 和 `execution_id`。
- 机械臂结果：左右臂 7/7 关节健康、软件未使能；左右 Shadow IK 均成功并通过 FK 残差检查；
  J6/J7 变化被结构化回传，未执行运动。
- 相机结果：序列号 `AY2PB4200YY`，RGB/Depth 均 live。
- 安全结果：`motion_commands_sent=0`、`target_writes=0`、`enable_changes=0`；所有结果
  均为 `hardware_accessed=true`、`motion_executed=false`、`write_control_authorized=false`。
- 状态差异说明：右臂当前反馈相对 P5 基线最大关节差约 `0.4218 rad`；本次及复核均未发送
  写控制指令，因此仅记录为两次只读快照之间的状态差异，进入 P6 前应重新以当前反馈预演。
- 证据目录：`logs/agent_skill_validation/p5.5-20260813T075210Z/manifest.json`
- 结论：PASS
- 下一步：P6 仍需用户当次明确授权；首条动作必须重新基于当前反馈生成并预演。

| 日期 | 版本 | 变更 | 结论 |
|---|---|---|---|
| 2026-08-13 | 0.1 | 建立 API → Skill → 真机能力验证计划；固定 IK、相机和分级门禁 | P0 进行中 |
| 2026-08-13 | 0.2 | 新增隔离离线验证 Task、配置和 43 项 P0/P1 回归；确认无硬件访问 | P0/P1 PASS |
| 2026-08-13 | 0.3 | 完成 10 次真实 API P2 验证与 3 项 Fake Gateway/replay P3 验证 | P2/P3 PASS |
| 2026-08-13 | 0.4 | 完成双臂 + Orbbec RGB-D 真机只读自检；修复 CAN 反馈统计并固化遥测/零运动证据 | P4 PASS |
| 2026-08-13 | 0.5 | 完成实时反馈 seed 的双臂 Shadow IK、FK 回代、限位/J6-J7/幂等性和 URDF 预演；记录 Z− 局部边界 | P5 PASS |
| 2026-08-13 | 0.6 | 新增 P5.5 真机只读 Task、状态/Shadow IK/相机 Skills；真实 Ark API 首次闭环通过，旧凭据 401 记录保留 | P5.5 PASS |
| 2026-08-13 | 0.7 | 更新 P5.5 真实 API 验收与右臂状态差异说明；冻结 P6 低速单命令预检门禁，仍未开放实机动作 | P6 预检进行中 |
| 2026-08-13 | 0.8 | 完成 P6 无运动预检：左右臂当前反馈生成 5 s/501 点 z+5 mm 候选轨迹，速度和端点门禁通过 | P6 实机动作待授权 |
| 2026-08-13 | 0.9 | 经用户确认执行右臂一次 z+5 mm、5 s 低速轨迹；到达并完成反馈核验，保持使能，禁止自动追加动作 | P6 PASS |
| 2026-08-13 | 1.0 | 右臂安全失能；将下一阶段改为同一次 Agent 调用读取右臂与 RGB-D 相机，灵巧手跳过；真实 Ark API 2/2 只读工具通过 | P7 PASS |
