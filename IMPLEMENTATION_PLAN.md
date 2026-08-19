# RealAgent 分阶段实施计划

更新日期：2026-08-06  
对照基线：[preview.md](./preview.md)

第 1～9 节保留已经完成的离线首版设计与证据；第 10 节记录获得明确模型、协议地址和
测试授权后制定并完成的 Ark 真实模型接入计划；第 11 节记录番茄采摘阶段的设备审计、
接口冻结和后续开发计划。第三阶段同样不改变单 `TopAgent`、两层有界循环和中立 DTO
约束；旧的多阶段 7D/22D PPO 试验实现已归档，当前唯一有效训练边界是 C2 最小 V3：
66D Actor、83D privileged Critic、11D 连续动作。Isaac 物理门禁和短 PPO 数值门禁均已
通过；正式 PPO 长训、真实感知、vendor gateway 和真机仍未启动。

## 1. 目标与不可变约束

第一阶段只实现可测试的 Agent 层级、运行时生命周期和组件边界。首版必须在
Python 3.11 及以上版本中仅依赖标准库运行，不访问公网，不加载模型权重，不连接
仿真器或机器人，也不执行真实下游动作。

以下约束在实现期间保持不变：

1. 系统只有一个 `TopAgent`。Task、Env、感知和 manipulation 都是职责组件，不是
   第二层 Agent，也不自行发起模型调用。
2. 系统有两层有界循环：外层由 `AgentRuntime` 控制环境观测和任务完成条件；内层由
   `TopAgent` 在单次 `invoke()` 中控制推理/工具往返。
3. Task 是任务语义的唯一所有者，负责 prompt、工具白名单、输入构建和可验证的完成
   条件；Runtime 不包含具体任务分支。
4. Env 只表达世界能力，所有跨边界数据使用中立 DTO；具体 SDK 对象不能泄漏到
   Runtime、Task 或 Agent。
5. Factory 采用显式 Registry，不扫描目录，不在 import 阶段连接外部系统。
6. 第一阶段所有模型、感知、manipulation 和 Env 行为都使用确定性的离线 mock；
   第二阶段仅把模型替换为显式选择的 Ark adapter，其他外部能力继续使用 mock。
7. 到达新的真实 SDK/API 联调点时暂停，不猜测凭证、地址或协议，向用户索要所需
   信息后才继续对应测试。Ark 模型所需信息和授权已经由用户提供。

## 2. 本阶段范围

### 2.1 范围内

- `Observation`、`Action`、`Result` 及感知、位姿、抓取候选等后端无关 DTO；
- `RuntimeContext` 与 `AgentRuntime` 的有界外层控制循环；
- 单一 `TopAgent` 的有界内层工具循环；
- `BaseTask`、一个可完成的示例 Task 及显式 Task Factory；
- `BaseEnv`、`BasePerceptionTool`、`BaseManipulation` 及对应 mock 和 Factory；
- JSON 兼容 YAML 配置、composition root 和命令行入口；
- 生命周期、刷新、失败传播、工具白名单和边界条件的离线单元测试；
- 标准库日志；日志不参与业务数据传递。

### 2.2 明确不在范围内

- LangChain、供应商模型客户端及真实 tool-calling 协议；
- PyYAML 或其他配置依赖；
- RoboTwin、真机、相机、YOLO、SAM、GraspGenX 等 SDK/权重/服务；
- GPU、网络请求、付费 API、真实 API key；
- 真实运动规划、抓取、末端执行器或其他下游动作；
- 自动启动第三方进程、修改 `cwd`/`sys.path`、下载资源；
- 以仿真 ground truth 代替任务要求的视觉感知。

配置加载器首先用标准库 `json` 解析；当输入不是 JSON 兼容 YAML 且环境中恰好已
安装 PyYAML 时，可以使用它做可选回退。LangChain 仍是后续适配点。二者都不是离线
核心的硬依赖，默认测试也不依赖它们。

## 3. 目标结构与依赖方向

```text
script/run.py                  composition root / CLI
        |
        +--> module/runtime.py --------> task/_base_task.py
        |           |                   module/tools/env/_base.py
        |           +-----------------> proto DTO
        |
        +--> module/agent/              单一 TopAgent、离线模型策略
        +--> task/factory.py            显式 Task Registry
        +--> module/tools/*/factory.py  显式能力 Registry
        +--> config/default.yaml        JSON 语法的 YAML 子集
```

计划目录：

```text
module/
├── runtime.py
├── agent/
└── tools/
    ├── env/
    ├── perception/
    └── manipulation/
proto/
task/
config/
script/
tests/
```

允许的核心依赖方向是 `script -> factories/runtime`、`runtime -> BaseTask/BaseEnv/DTO`、
`concrete task -> base contracts/context/capability APIs/DTO`、`concrete adapter -> base
contract/DTO`。禁止 Runtime 导入具体 Task 或后端，禁止 Env 导入 Task/Agent，禁止
Agent 导入具体 Env。

## 4. 两层循环设计

### 4.1 外层：`AgentRuntime`

一次默认运行遵循下列顺序：

```text
env.reset() -> 首帧写入 RuntimeContext
task.run(runtime)
  -> 对每个控制步：
       task.build_input(current_observation)
       top_agent.invoke(...)
         -> 内层工具循环
       task.is_complete(...)
       未完成且仍有步数：env.get_obs() -> 原子替换当前观测
finally:
  task.close()
  env.close()
```

- `runtime.max_steps` 必须是正整数；外层最多执行该数量的 Agent 调用。
- `reset()` 返回的观测就是控制步 1 的首帧，不能立即重复调用 `get_obs()`。
- 如果 tool 没有自行推进环境，只有当前步未完成且还有下一控制步时才刷新一次观测。
- `runtime.step(..., refresh_observation=True)` 可在动作后立即刷新；外层通过观测版本避免
  在控制步边界重复采集。即使动作返回失败 Result 也会刷新，因为被拒绝或部分执行的
  物理命令仍可能改变世界。
- 完成条件或步数上限已经确定后，外层不再做无消费者的额外刷新；完成判断前由动作
  tool 触发的刷新不属于“完成后刷新”。
- Task tool 必须从 `RuntimeContext.require_observation()` 读取最新帧，不能闭包捕获旧帧。
- Task 完成由可验证工具结果决定，不能只依据模型生成的“已完成”。

### 4.2 内层：`TopAgent`

单次 `TopAgent.invoke()` 可执行多轮“模型决策 -> 白名单工具 -> 工具结果”，但必须受
配置项 `agent.max_iterations` 限制。离线模型使用确定性脚本/策略模拟决策，不模仿
任何供应商 SDK 对象。

- `TopAgent` 只接收 Task 提供的工具列表；没有全局工具仓库。
- 工具名必须唯一，未知工具或越权工具立即失败。
- 工具调用成功只表示函数运行成功；语义失败（例如空检测）必须显式失败，不能伪装
  成成功结果。
- 达到内层上限时返回明确的未完成/耗尽结果或抛出明确边界异常，不能无限循环或静默
  成功。
- Agent message 只保存小型摘要和结构化结果，不复制 RGB bytes、mask、点云或密钥。

## 5. 关键语义

### 5.1 失败语义

1. 配置、Registry 名称和参数在资源获取前尽早校验；错误信息包含字段名和期望。
2. `env.reset()`、Task、Agent、tool 或完成条件抛出的不可恢复异常必须继续向调用者
   传播；不得捕获后返回伪成功。
3. 未找到目标、无抓取候选或动作 mock 明确拒绝等属于语义失败；Task 的完成条件必须
   拒绝这些结果。
4. 外层或内层耗尽要有可区分结果，测试必须证明循环次数不超过配置上限。
5. 日志可补充阶段、组件名和异常堆栈，但不能改变返回值或作为模块通信通道。

### 5.2 观测刷新语义

1. `reset()` 产生首帧并立即更新上下文。
2. 当前控制步内的所有工具读取同一个当前观测。
3. 动作 tool 可通过 `runtime.step()` 立即更新观测；若本步没有 tool 更新，则仅在未
   完成并准备进入下一步时由外层调用一次 `get_obs()`。
4. 新观测完整替换旧观测；不允许只更新部分字段造成跨帧混用。
5. 上下文尚无观测时，`require_observation()` 给出明确错误。

### 5.3 清理语义

1. 资源由创建者管理：Env 释放自身资源，Task 释放其初始化的 manipulation 等后端。
2. Runtime 在 `finally` 中依次调用 `task.close()`、`env.close()`，覆盖正常完成、循环
   耗尽、reset 失败、Agent/tool 失败和完成条件失败。
3. 所有 `close()` 都采用单次尝试语义：首次调用先标记已关闭，再尝试释放；无论释放
   成功或失败，重复调用都不再重复释放，也不产生新错误。
4. 构造阶段获得部分资源后失败的组件，必须先释放自己已获得的资源再继续抛错。
5. 某个清理动作失败时仍要尝试其余清理动作；清理错误不能让先前的主异常悄然丢失。
   具体聚合方式仅使用 Python 3.11 标准异常机制，并通过测试固定行为。

## 6. 分阶段实施

|阶段|内容|退出条件|状态|
|---|---|---|---|
|P0|对照 `preview.md`，冻结离线范围、目录和两层循环契约|计划、入口和验收项无歧义|完成|
|P1|实现 DTO、Base 契约、显式 Registry 和 mock 能力|能力测试不需要网络/GPU/SDK|完成|
|P2|实现单一 `TopAgent`、Task 和 Runtime 两层循环|正常、失败、耗尽、刷新和清理路径可测|完成|
|P3|实现 JSON 兼容配置、composition root、CLI 和日志|默认配置可离线运行并输出确定性摘要|完成|
|P4|执行单元测试、编译检查和文档-实现回查|验收矩阵全部通过或明确记录差异|完成|
|P5|Ark OpenAI Responses / Anthropic 兼容协议适配|双协议离线契约测试与显式真实冒烟通过|完成|
|P6|真实 Env/感知/manipulation/下游动作|获得对应 SDK、硬件与动作授权后另行规划|范围外|

## 7. 验收矩阵

|ID|场景|预期证据|
|---|---|---|
|A01|纯标准库启动|全新 Python 3.11+ 环境无需安装依赖即可运行默认 CLI|
|A02|默认组装|配置能显式构建 mock model/Env/Task/感知/manipulation|
|A03|首帧|`reset()` 返回值直接用于第一个控制步，未多调 `get_obs()`|
|A04|刷新|未完成后上下文恰好前进一帧；动作已刷新时外层不重复取帧|
|A05|完成|完成判断后外层不再刷新，Runtime 返回明确成功结果|
|A06|工具白名单|TopAgent 只能调用 Task 暴露的工具；模型只见不可执行的 `AgentToolSpec`，未知/重复名称失败|
|A07|内层多轮|单个外层控制步可发生多次 mock 决策/工具往返|
|A08|内层有界|同时执行 `max_iterations`、`max_tool_calls`、`max_tool_calls_per_turn` 预算，超限轮不执行 handler|
|A09|外层有界|Agent 调用次数不超过 `max_steps`，耗尽状态可区分|
|A10|语义失败|参数错误和显式 `RecoverableToolError` 可由模型修正；空检测/无候选/拒绝动作不满足完成条件|
|A11|异常传播|reset、tool、Agent、Task 异常经过清理后仍可观察|
|A12|完整清理|正常与所有失败路径都尝试 `task.close()` 和 `env.close()`|
|A13|幂等清理|重复 close 不重复释放，部分构造失败自行回收|
|A14|Factory 边界|未知类型/profile 给出包含合法选项的明确错误|
|A15|离线隔离|默认测试不联网、不读取 API key；真实冒烟必须显式启动并从环境变量取 Key|
|A16|DTO 边界|跨模块不传递供应商对象，大体积数据不写 Agent message/log|
|A17|静态可导入|`python -m compileall` 通过|
|A18|基线回查|每轮修改后按第 8 节核对 `preview.md` 并修正文档/实现漂移|
|A19|配置原子性|根/Agent/Runtime/Registry/构造参数/Task/profile 在创建资源前完成预校验|
|A20|READY 释放|已组装但未运行的 Runtime 可显式 `close()`，并拒绝非法 Task owner|
|A21|边界脱敏|消息与完整日志共享脱敏逻辑，覆盖引号/转义/多行值且不执行陌生 key 的字符串化|

## 8. 与 `preview.md` 的持续校对规则

每个实现回合结束前执行一次回查：

1. 对照第 1～4 节确认 Runtime、Task、Env、Agent 的职责和依赖方向；
2. 对照第 5 节确认没有扩大公共契约或引入禁止做法；
3. 对照第 6～12 节确认 Task、tool、Env、能力 Factory 和配置的行为；
4. 对照第 13 节确认类型、日志、异常和清理语义；
5. 运行验收测试并用实际类名、入口和行为修正本计划与 README；
6. 若离线实现与文档冲突，优先在本阶段范围内修正实现或说明差异，不能让文档宣称
   尚不存在的能力；
7. `preview.md` 是用户指定的持续对照基线；当实现证据表明其中的示例、现状陈述或
   契约已经错误时，同步校正该指南并保留范围外能力的明确标记。

当前相对基线的有意简化只有三项：LangChain 内层循环改为标准库确定性 mock；所有
外部后端改为 mock；内置 YAML 文件使用 JSON 兼容子集并由 `json` 解析（若用户另行
安装 PyYAML，加载器可选择性解析普通 YAML）。这些简化不改变两层循环、Task 语义
所有权、DTO 边界、工具白名单和资源生命周期。

## 9. 章节到实现的追踪矩阵

|`preview.md` 章节|当前实现证据|主要测试|状态|
|---|---|---|---|
|§1～4 组装、两层循环、依赖方向|`script/run.py`、`module/runtime.py`、`module/agent/`|`test_agent_runtime.py`、`test_architecture.py`|符合|
|§5 修改边界与禁止项|lazy public exports、显式 Registry、无第三方 SDK；网络实现隔离在 provider transport|`test_architecture.py`、`test_security.py`、`test_provider_models.py`|符合|
|§6 Task|`task/_base_task.py`、`task/locate_target/`、`task/factory.py`|`test_task_composition.py`|符合|
|§7 Agent tool|带只读参数 schema 的 `AgentToolSpec`、handler 隔离、`TopAgent` 白名单、预算和可恢复错误|`test_agent_runtime.py`、`test_security.py`、`test_provider_models.py`|符合|
|§8 Env|`BaseEnv`、`MockEnv`、`build_env()`|`test_proto_capabilities.py`|符合（真实适配器延期）|
|§9 perception|`BasePerceptionTool`、`MockPerceptionTool`|`test_proto_capabilities.py`、`test_task_composition.py`|符合（SDK detector/segmenter 延期）|
|§10 manipulation|`BaseManipulation`、`MockManipulation`、抓取 DTO|`test_proto_capabilities.py`|符合（真实规划服务延期）|
|§11 model|`BaseChatModel`、mock、Ark Responses/Messages adapter、有界 HTTPS transport、显式 Factory|`test_agent_runtime.py`、`test_provider_models.py`|符合（双协议真实冒烟通过）|
|§12 配置|离线默认、两个 Ark 根配置、无秘密 profile、资源创建前完整预校验|`test_task_composition.py`、`test_provider_models.py`|符合|
|§13 类型、日志、异常|严格 `RuntimeReport`、异常聚合、有界消息、共享完整日志脱敏|全套测试，重点 `test_security.py`|符合|

第二阶段完成时，`python -m unittest discover -s tests -v` 共 84 项通过，并已分别在
Python 3.12 与当前解释器执行；默认 CLI 和 `python -m compileall module proto task
script tests` 通过。当前有意偏离旧稿的地方已写回 `preview.md`：Ark 真实协议 adapter
已落地但不引入供应商 SDK；perception 增加幂等生命周期；manipulation 大数组不直接
作为 Agent 参数；内置配置使用 JSON 兼容 YAML。

## 10. 第二阶段：Ark 双协议模型接入计划

第二阶段目标：在不改变 Agent 层级和真实下游 mock 边界的前提下，让现有
`TopAgent` 能通过两种 Ark 兼容协议安全调用 `doubao-seed-2.0-pro`，完成真实
tool-calling 往返，并保持默认运行、测试和凭据处理可控。

### 10.1 已确认输入与实现选择

- 模型标识：`doubao-seed-2.0-pro`；
- Anthropic 兼容 Base URL：`https://ark.cn-beijing.volces.com/api/coding`；
- OpenAI 兼容 Base URL：`https://ark.cn-beijing.volces.com/api/coding/v3`，主路径使用
  Responses API；
- 凭据只从 `ARK_API_KEY` 环境变量读取，不进入 YAML、源码、对象 repr、异常正文或
  日志；
- 使用 Python 标准库 HTTPS 直接实现兼容协议，不增加 OpenAI/Anthropic SDK 硬依赖。
  这样仍执行真实 API，但不会把供应商 SDK 类型带入中立模型契约。

Responses 工具往返采用 `function_call`、`call_id`、`previous_response_id` 和
`function_call_output`；Anthropic 工具往返采用 `tool_use` 和 `tool_result`。两种协议
都只在模型适配器内转换，`TopAgent`、Task、Runtime 和本地 handler 不感知供应商格式。

### 10.2 开发顺序

1. 扩展 `AgentToolSpec`，加入只读、JSON-compatible 的 object 参数 schema；本地
   `AgentTool` 仍独占可执行 handler。
2. 实现有响应体上限、HTTPS 校验、超时和安全错误摘要的 JSON transport。
3. 实现 OpenAI Responses 适配器：首轮传消息与 tools，后续只回传尚未提交的 tool
   output，并解析文本与 `function_call`。
4. 实现 Anthropic Messages 适配器：把中立 transcript 转换成交替 message blocks，
   并解析文本与 `tool_use`。
5. 在显式 `MODEL_REGISTRY` 注册两种 adapter，提供无秘密的 profile 和两个独立根配置；
   默认配置继续使用 mock，确保无 Key 时仍可完整离线运行。
6. 用 fake transport 覆盖请求转换、响应解析、工具往返、协议畸形、HTTP 错误、超时和
   凭据脱敏，再显式执行两个真实协议的最小 tool-call 冒烟。
7. 对照实现回写 `preview.md`、README、本计划与测试基线。

### 10.3 第二阶段验收矩阵

|ID|场景|退出证据|
|---|---|---|
|B01|工具 schema|模型只能看到 name/description/parameters，永远拿不到 handler|
|B02|Responses 首轮|请求使用 `/responses`、Bearer 鉴权、model/input/tools/store|
|B03|Responses 工具续轮|只回传匹配 call_id 的 `function_call_output` 并关联 previous response|
|B04|Anthropic 首轮|请求使用 `/v1/messages`、Anthropic version、system/messages/tools|
|B05|Anthropic 工具续轮|assistant `tool_use` 与 user `tool_result` 正确配对且合并并行结果|
|B06|严格解析|未知/缺失/错误类型的关键响应字段产生供应商协议异常，不伪造终态|
|B07|传输边界|仅 HTTPS、有限 timeout/响应体、非 2xx 与网络错误给出无秘密摘要|
|B08|配置与凭据|模型 profile 无 Key；缺少 `ARK_API_KEY` 时在获得 Env/Task 前失败|
|B09|离线回归|默认测试不联网、无需 Key，mock 默认 CLI 行为不变|
|B10|真实 OpenAI 冒烟|完整模型→tool→模型往返并返回终态|
|B11|真实 Anthropic 冒烟|完整模型→tool→模型往返并返回终态|
|B12|文档校正|`preview.md` 不再宣称“没有 Ark/API/schema”，且准确标记未接入的下游|

### 10.4 仍不在第二阶段范围内

- 真实 Env、相机、感知模型、抓取规划、机器人 SDK 或下游动作；
- LangChain、供应商 SDK 类型或自动发现 adapter；
- 在配置文件保存 Key、自动把 Key 写入 shell profile、`.env` 或日志；
- 流式输出、内置联网搜索/MCP、自动重试和长期会话持久化。这些能力需分别评估重复
  计费、状态恢复、数据合规和资源生命周期后再扩展。

### 10.5 完成证据与计划对齐

|验收项|实现/测试证据|状态|
|---|---|---|
|B01|`AgentToolSpec.parameters` 深度冻结、detached `as_dict()`、签名推断与非法/敏感 schema 测试|完成|
|B02～B03|`ArkOpenAIResponsesModel` 请求形状、`call_id`、`previous_response_id` 和续轮 fake transport 断言|完成|
|B04～B05|`ArkAnthropicMessagesModel` messages/tool blocks、并行 `tool_result` 合并测试|完成|
|B06|两种 adapter 的缺失字段、错误类型和畸形 JSON 协议测试|完成|
|B07|HTTPS/URL 凭据、timeout、请求/响应大小、HTTP/网络安全错误测试|完成|
|B08|无 Key 配置失败、模型先于 Env/Task 构造、profile 不含凭据、对象不保留 Key|完成|
|B09|当时 84 项默认测试均使用 mock/fake transport；无 Key 默认 CLI 通过|完成|
|B10|OpenAI-compatible 配置完成模型→`locate_target`→模型的真实往返，CLI 返回 0|完成|
|B11|Anthropic-compatible 配置完成同样的真实往返，CLI 返回 0|完成|
|B12|README、`preview.md` 与本计划已按真实类名、配置、范围和测试基线修正|完成|

第二阶段未将任何真实凭据写入仓库；真实冒烟日志经过凭据模式扫描，匹配数为 0。
当时保持范围外的 P6 已在用户提供设备仓和任务定义后，由第 11 节重新拆分为 C0～C5；
这不改变第一、二阶段的完成证据，也不表示真实下游已经接入。

## 11. 第三阶段：番茄采摘 Task 与 PPO Skill

第三阶段目标：在不继续扩展 Agent 框架的前提下，实现单果番茄采摘闭环。PPO 负责从
预抓取位开始的接触、抓取、拉脱和稳定持有；Task 只有在独立 Verifier 确认“果实已
脱离且没有滑落”时完成。

本阶段设计依据：

- [设备能力基线](./docs/tomato_pick/SDK_CAPABILITY_BASELINE.md)
- [冻结接口规范](./docs/tomato_pick/INTERFACE_SPEC.md)
- [全局架构复核](./docs/tomato_pick/ARCHITECTURE_REVIEW.md)
- [C2-0 番茄仿真开工规范](./docs/tomato_pick/C2_ZERO_SPEC.md)
- [C2-1 仿真与 PPO 环境规范](./docs/tomato_pick/C2_ONE_ENVIRONMENT.md)
- [C2 Isaac 运行环境](./docs/tomato_pick/C2_ISAAC_RUNTIME.md)
- [C2-2 资产物理决策](./docs/tomato_pick/C2_TWO_ASSET_PHYSICS_PROPOSAL.md)
- [C2-2 Sim2Real 与训练准备](./docs/tomato_pick/C2_SIM2REAL_TRAINING_PREP.md)

### 11.1 已确认设备与范围

- 双机械臂，每臂 7 个关节；左右 G20 灵巧手在训练资产中各有 16 个独立关节；
- 机械臂已有 FK、IK、MoveJ、MoveJP、MoveL、FollowP；MoveC 不进入系统；
- 正式训练前不连接 SDK、ROS、CAN 或真机；仿真只使用已固定和验证的 USD/URDF；
- G20 真机触觉是五指 `12×6` 原始矩阵。牛顿值和果梗断裂阈值仍需在真机发布前标定；
- 官方旧 Fruit-Grasp baseline 不复用。此前 LinkAgent 多阶段 PPO 代码、审计器和试验脚本已
  从当前工作树删除；如需审计，只从 Git 历史追溯，不再属于活动训练路径。

### 11.2 唯一冻结的最小 PPO V3

1. Agent 仍只看到 5 个任务级工具：`detect_tomatoes`、`plan_tomato_grasp`、
   `move_to_pregrasp`、`execute_tomato_pick`、`verify_tomato_pick`。
2. Actor 输入是 66 维、全部可部署：果实中心/坐标轴/拉脱轴、末端速度、16 个手关节角和
   速度、五指压力、重力方向、上一动作；所有量统一表达在手掌或抓取坐标系。
3. Critic 输入是 83 维：Actor 66 维加仿真专用的精确相对位姿、果实线/角速度、真压力、
   果梗载荷、脱离标志和质量。部署时不使用 privileged 部分。
4. Action 是连续 11 维：抓取坐标系 6D 末端增量与 5D G20 绝对手型协同；5D 统一映射为
   16 个独立关节目标。零动作保持已验证的预抓取手型。
5. 仿真中 6D 增量由阻尼最小二乘 IK 转为 7 轴目标；部署时由 FollowP/IK adapter 执行，
   PPO 不直接学习机械臂关节控制器。
6. 环境不把阶段放入 Actor，不屏蔽动作、不注入规则策略。阶段只用于日志诊断。
7. 成功只有一个定义：果实已经脱离，并连续保持 20 个策略步（1 秒），期间无掉落和过力。
8. 奖励只包含有上界的最佳包络进展、脱离后保持、成功、物理失败和时间成本；没有可循环
   刷取的逐帧阶段奖励。
9. 算法使用 RSL-RL 标准 recurrent PPO（LSTM + Gaussian），不使用自定义动作分布、BC、
   warm-start、跳阶段 reset 或课程规则。

### 11.3 开发阶段

|阶段|内容|退出条件|状态|
|---|---|---|---|
|C0|设备、资产与接口审计|设备事实和 Agent/Skill 边界冻结|完成|
|C1|Agent Task、Ports、mock 与 verifier|离线任务闭环通过|完成|
|C2|最小 V3 仿真环境和 PPO 训练前准备|单元、物理、短数值门禁全部通过|完成|
|C2-3|正式 PPO 训练、独立评估和 manifest 导出|多 seed 评估达到约定成功率且可复现|下一阶段，尚未启动|
|C3|真实 RGB-D 感知、抓取位姿和 verifier 数据|离线数据可区分脱离、保持、滑移和掉落|待开始|
|C4-O|SDK 离线 adapter 与 fake gateway|坐标、命令、取消、急停契约通过|完成|
|C4-H|真机只读、shadow、单命令与分级闭环|逐级通过安全验收|连接前暂停确认|

### 11.4 正式训练前门禁

|门禁|证据|结论|
|---|---|---|
|活动代码单元/回归|`101 passed, 76 subtests passed`|通过|
|Isaac 物理门禁|64 环境 × 32 步；66/83/11 张量有限、reset 无接触、最小间隙 5.148 mm、动作和断梗语义可达|通过|
|PPO 数值门禁|256 环境 × 32 步 × 4 更新；32768 transition，loss/std 全有限，无 NaN/Inf/overflow|通过|
|训练计划预检|8192 环境、W&B 已认证、路径和资产解析成功，`ready=true`|通过|
|真机发布标定|压力换算、果梗阈值和实机延迟分布|正式 sim-to-real 前待做|

### 11.5 C2-3 正式 PPO 计划

- 从零训练，不加载旧 checkpoint；
- 8192 并行环境，32 步 rollout，最多 1500 次更新；
- 固定学习率 `3e-4`、entropy coefficient `0.002`、初始标准差 `0.60`；
- 约 1000 万 transition 做第一次趋势判断，2000 万做硬判断；
- 除非发现明确代码、物理或数值错误，不在正式 run 中修改 MDP；
- 训练结束后从多个 checkpoint 做固定多 seed、无探索噪声评估，再生成 manifest；
- 只有正式训练启动后才恢复 10 分钟监控，避免空闲监控继续修改环境。

### 11.6 实际 SDK/API 暂停点

C2-3 不需要真实 SDK。只有进入 C4-H 的 ROS/CAN endpoint、真实状态读取或运动命令测试
时，才暂停并向用户确认运行环境、连接方式和安全测试窗口。

### 11.7 当前交付和唯一下一步

- PPO 冻结契约：`module/skills/tomato_pick/contracts.py`；
- G20 5D 手型映射：`module/skills/tomato_pick/hand_synergy.py`；
- 最小物理环境：`simulation/tomato_pick/isaac/env.py`；
- 标准 PPO 配置：`simulation/tomato_pick/isaac/agents/rsl_rl_ppo_cfg.py`；
- 可审阅训练配置：`simulation/config/ppo_v1.yaml`；
- 历史复杂实现已从当前工作树删除，仅保留在 Git 历史中。

当前状态是“正式 PPO 长训前已就绪”，不是“已经训练完成”。唯一下一步是在用户明确启动后
执行 C2-3 正式 run；本阶段不再增加 PPO 版本、审计脚本或规则补丁。
