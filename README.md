# RealAgent

RealAgent 是一个可运行的单 `TopAgent` 系统：外层 Runtime 管理观测、控制步和资源生命
周期，内层 Agent 管理有界的模型/工具往返。默认配置完全离线；第二阶段已经加入
`doubao-seed-2.0-pro` 的 Ark OpenAI Responses 与 Anthropic Messages 兼容协议适配。

真实相机、感知模型和真机动作仍未接入；默认 Agent 示例继续使用 mock。第三阶段 C1
已加入可运行的离线番茄采摘闭环，C2 已实现 Sim 6 中的番茄椭球、可断果梗和
DirectRLEnv，并完成 Clean-v5 正式 PPO 训练。当前推荐的 `model_1350.pt` 在三个固定
种子、26,077 个 episode 上取得 **96.18%** 汇总成功率，最差种子仍为 96.06%；模型、
训练配置、续训 checkpoint、固定评测证据和可直接加载的推理导出均随仓库发布。C4-O
已加入不连接设备的 LinkBot/G20 适配边界。

番茄采摘第三阶段已经完成 C0 接口冻结和 C1 离线实现：设备按双臂、每臂 7 关节及
左右 G20 建模；Skill 使用 114 维部署观测和 7 维“末端残差 + 手部协同量”动作；
`AgriPickTask` 只暴露 5 个任务级工具，并由独立 Verifier 确认果实脱离、稳定持有且
没有滑落。默认 C1 policy、感知、规划、安全、设备和验证器仍是确定性 mock；它与已
训练的 C2 Isaac PPO skill 保持隔离。SDK 中旧 Fruit-Grasp baseline 已清理，审计结论
只保留在文档中。

当前正式 minimal-v3 策略使用 66D actor observation、83D privileged critic state 和
11D action。训练前的旧布局不进入当前 checkpoint 兼容边界。

## PPO Clean-v5 model 1350

可恢复训练的 checkpoint、精确 resolved config、TensorBoard history、确定性推理导出、
推理 canary 和固定三种子评估结果均随仓库发布。模型 SHA-256 为
`c4ca9ac27aaa20806b630496362bcb9154d1e8076605f9d7026b58f719d124d0`。

安装环境、复现 96.18% 结果以及从 iteration 1350 继续训练的完整命令见
[`PPO_CLEAN_V5_MODEL1350_REPRODUCTION.md`](./docs/tomato_pick/PPO_CLEAN_V5_MODEL1350_REPRODUCTION.md)。
旧版 62.57% minimal-v3 基线继续保留在
[`PPO_MODEL1450_REPRODUCTION.md`](./docs/tomato_pick/PPO_MODEL1450_REPRODUCTION.md)，但不再是
推荐模型。

## Physics-v1 source baseline（不含 checkpoint）

Physics-v1 将用户标定的新版连续 5D 手部流形、派生碰撞 USD、相关物理/随机化配置、
奖励信用门控和训练前门禁证据作为独立源码基线发布。本次源码交接不包含任何新的
Physics-v1 checkpoint、W&B 运行文件或 TensorBoard 事件。完整复现和队友修改边界见
[`PHYSICS_V1_SOURCE_REPRODUCTION.md`](./docs/tomato_pick/PHYSICS_V1_SOURCE_REPRODUCTION.md)。

## 快速开始

要求 Python 3.11 或更高版本。项目没有第三方运行时硬依赖。

默认离线运行不需要 API key：

```bash
python -m script.run --config config/default.yaml
```

运行完整单元测试和编译检查（完整测试需要记录的 Isaac/PyTorch 环境）：

```bash
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
"$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python" -m pytest -q
python -m compileall module proto task script tests
```

测试只使用 mock/fake gateway/无物理契约环境，不联网、不连接 CAN、不读取真实凭据；
它们不会启动 Isaac 仿真或 PPO 训练。

运行 C1 离线番茄采摘闭环：

```bash
python -m script.run --config config/agri_pick_mock.yaml
```

运行无物理契约门禁（不会启动 Isaac 或 PPO）：

```bash
python3.12 -m script.sim_env preflight --role contract_smoke
python3.12 -m script.sim_env smoke --episodes 100 --seed 0
python3.12 -m script.train_tomato_ppo
```

前两条应通过；第三条会输出训练计划，并在缺少 Isaac Lab 路径或 W&B 登录时列出阻塞项。

隔离的 Isaac 主运行时、重复 smoke 命令和已知依赖说明见
[`docs/tomato_pick/C2_ISAAC_RUNTIME.md`](./docs/tomato_pick/C2_ISAAC_RUNTIME.md)；已经执行的
资产决策和训练前唯一阻塞见
[`docs/tomato_pick/C2_SIM2REAL_TRAINING_PREP.md`](./docs/tomato_pick/C2_SIM2REAL_TRAINING_PREP.md)。

## Ark 真实模型

凭据只从进程环境读取，不写入 YAML、源码或日志。设置 `ARK_API_KEY` 后，可分别运行：

```bash
export ARK_API_KEY="<your-key>"
python -m script.run --config config/ark_openai.yaml
python -m script.run --config config/ark_anthropic.yaml
unset ARK_API_KEY
```

随仓 profile 使用：

|配置|协议|Base URL|
|---|---|---|
|`config/ark_openai.yaml`|OpenAI Responses|`https://ark.cn-beijing.volces.com/api/coding/v3`|
|`config/ark_anthropic.yaml`|Anthropic Messages|`https://ark.cn-beijing.volces.com/api/coding`|

两条真实冒烟路径均已验证完整的“模型请求工具 → 本地执行 `locate_target` → 把工具结果
交回模型 → 返回终态”流程。接入使用标准库 HTTPS 和中立内部契约，没有引入 OpenAI、
Anthropic 或 LangChain SDK 类型。

## 番茄采摘与离线设备适配

### 2026-08-19 掌托式五指包络抓取增量

本次增量同步了掌部支撑、五指包络、断梗后保持和触觉/接触诊断的全部可审计结果，
包括通过、失败和被否决的实验，不将失败结果隐藏或改写。材料位于
[掌托式包络抓取实验资料](./docs/tomato_pick/enveloping_grasp_v2/) 和
[实验源码与结果索引](./research/palm_enveloping_grasp_v2/)。

当前结论：核心 nominal 35--40 mm 半轴域的三种子严格成功率约 39.17%，尚未达到
80% 门槛；轻度随机化约 23.82%，宽域压力测试约 0.9%。主要失败模式是摘取后掉落和
断梗后保持不足。所有结果来自离线 PhysX，不能直接用于实机，也不能把代理接触当作
真实触觉标定。本次通过 Git LFS 发布了精选的 checkpoint/teacher dataset（当前/最佳模型、教师数据及关键物理阶段权重）；大量重复的周期性评估快照仍保留在本地实验工作区，后续可按需要再补充或清理。GitHub 中同步了源码、报告、JSON/JSONL 诊断和失败证据。

- [设备能力基线](./docs/tomato_pick/SDK_CAPABILITY_BASELINE.md)：7 轴机械臂、G20、触觉、
  仿真资产和旧 PPO 的审计结论；
- [接口规范](./docs/tomato_pick/INTERFACE_SPEC.md)：Ports、Task tools、114D/7D PPO、
  状态机、成功和失败语义；
- [C2-0 开工规范](./docs/tomato_pick/C2_ZERO_SPEC.md)：正式模型、触觉单位、番茄/果梗
  环境边界和规则策略训练门禁；
- [C2-1 环境规范](./docs/tomato_pick/C2_ONE_ENVIRONMENT.md)：三套运行环境、无物理
  114D/7D 环境、PPO 配置和受控启动门禁；
- [全局架构复核](./docs/tomato_pick/ARCHITECTURE_REVIEW.md)：当前完成度、代码落点和
  C1～C5 路线。

C1 已完成完全离线的 `AgriPickTask + TomatoPickSkillRunner` 闭环；C2 已运行契约 smoke、
Isaac 物理门禁、PPO 数值探针和正式训练，并把 Agent、契约测试、Isaac/PPO 三套环境
隔离。下一阶段是 C3 真实感知、触觉标定和 sim-to-real 验收。

C4-O 已完成 SDK-neutral gateway 契约、LinkBot/G20 DTO 转换、双臂双手生命周期、控制
租约、只读门禁、命令幂等、全系统急停以及四设备部署模板。它没有导入供应商 SDK，
也没有注册成默认后端。真实连接时仍需补齐 vendor gateway、最终 CAN endpoint、右手
标定和 shadow/单命令验收；在此之前不会误进入硬件模式。

## 架构摘要

系统只有一个 `TopAgent`，其他对象都是职责组件：

- `AgentRuntime` 管理外层控制循环、最新观测和生命周期；
- `TopAgent` 管理单次调用内的模型决策/工具往返；
- Task 拥有 prompt、工具白名单、输入构建和可验证完成条件；
- Env 负责 reset、取观测、通用动作入口和环境资源释放；
- perception/manipulation 通过中立 DTO 和显式 Factory 提供能力；
- composition root 读取配置并按显式顺序组装。

```text
AgentRuntime.run()
  -> env.reset()                         # 直接取得首帧
  -> task.run(runtime)
       -> 外层控制步（最多 max_steps 次）
            -> TopAgent.invoke(...)
                 -> 模型/工具循环
                    （iterations、总 tool-call、单轮 tool-call 三重预算）
            -> task.is_complete(...)
            -> 必要时 env.get_obs()
  -> task.close()
  -> env.close()
```

Task 给出的工具列表就是完整权限白名单。模型只能看到只读的
`AgentToolSpec(name, description, parameters)`；`parameters` 是受大小和结构约束的
object JSON schema，实际 handler 始终只由 `TopAgent` 持有。schema 可显式提供，也可
从可信 Python handler 签名保守推断。

## 关键边界

- `env.reset()` 的返回值直接作为第一步观测，不重复采集首帧。
- 参数绑定错误和显式 `RecoverableToolError` 会作为 error tool message 交还模型；
  未知基础设施异常继续向外传播。
- Runtime 总会尝试按 `task.close()`、`env.close()` 顺序清理；组装阶段也遵守资源回收
  边界。
- 消息与日志共享脱敏器，并限制二进制、集合、嵌套和字符串体积。
- Provider transport 只接受无 URL 凭据的 HTTPS Base URL，并限制超时、请求体和响应
  体；错误只暴露有界且脱敏的摘要。
- 缺少 `ARK_API_KEY` 时，在获取 Env/Task 资源之前失败；默认 mock 配置不要求凭据。
- 自动重试、流式输出、内置联网工具和长期会话尚未实现；避免在未设计幂等与状态恢复
  前引入重复调用或计费风险。

配置文件采用 JSON-compatible YAML，因此标准库 `json` 即可解析；若环境中已有
PyYAML，加载器可选择性解析普通 YAML。

完整开发契约见 [preview.md](./preview.md)，阶段目标、范围和逐项证据见
[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)。两份文档均应在每轮实现后与代码
和测试重新对齐。
