# 小果 PPO：A 阶段交付与审阅记录

日期：2026-09-22（Asia/Shanghai）。范围：冻结基线与评估口径、堵住假成功、保存可追溯证据。完成后停止，等待审阅；本次未进行 PPO 训练，也未进入后续可达性、控制器或奖励重构阶段。

## 1. 本次结论

历史高成功率不能作为“撤去果梗支撑后仍抓牢”的证据。旧 Verifier 的接触获取状态会锁存，随后即使三指受力连续离开软带，仍可能满足最终成功；最终门也没有要求实际撤支撑、实际施扰和对向接触几何证据。A 阶段修正的是这套验收口径，不代表已解决物理抓取。

旧 `model_150.pt` 已重新校验 SHA-256 并通过当前 14D 动作、114D Actor、175D Critic 接口加载。当前合同下的两环境短评估中，两个回合均未取得合格接触，脱支撑保持成功为 0。这只是入口、失败拒绝及落盘回归；使用的控制参数不同于旧 B0 完整发布合同，样本量也不足以给出策略成功率估计，不能把差异全部归因于本次 Verifier 修正。

## 2. 冻结的语义

任务版本：`small_fruit_pinch_v1.2.0`；评估合同：`support_retention_v1`。网络输入输出维度保持不变。

| 输出 | 本版本含义 |
|---|---|
| `contact_acquired` / 内部 `success` | 连续至少 80 个策略步满足三指合格接触；只用于控制阶段切换 |
| `support_removed` | 仿真果梗约束已解除；允许审核程序主动释放 |
| `support_retained` / `final_success` / `strict_success_rate` | 撤支撑后连续至少 120 步合格保持，并有当前保持窗内实际扰动和至少 40 步扰动后恢复证据 |
| `natural_detached` | 环境原果梗断裂逻辑发生，未由审核主动释放冒充 |
| `harvest_success` | `natural_detached` 与最终保持同时成立；仅表示仿真自然脱果后保持，不包含机械臂运输、放置或真机采摘 |

策略频率 20 Hz：接触获取至少 4 秒，脱支撑保持至少 6 秒，扰动后保持至少 2 秒。脱支撑连续保持到第 40 步后请求扰动。扰动通过实际物理对象速度写入注入动量：`Δv = 0.25 × 9.81 × 0.05 = 0.122625 m/s`，`J = m × Δv`；方向按环境编号轮换 `+X/-X/+Y/-Y`。只有后续物理步执行后才确认，不允许将 `disturbance_due` 请求当成执行证明。这是理想瞬时动量扰动，不是持续外力模型。

每帧必须满足三指软带、稳定性、真实传感器和对向几何门；软带越界、接触丢失、支撑恢复或几何失效均清空最终保持窗。恢复需要重新保持并接收新的扰动，旧扰动不能复用。硬过载、非有限传感器值、禁触、关节越限和自碰撞输入使本回合失效。旧的覆盖率配置字段保留用于兼容，但本版本逐帧条件比旧 85% 接触/75% 带内统计要求更严格。

`--support-mode audit_release` 为默认审核模式：接触获取后主动撤去果梗；`natural_detachment` 模式不主动释放。正式评估拒绝几何接触代理和掌面碰撞过滤；只有显式 `--diagnostic-only` 才允许启动这些诊断组合，且这些组合不能通过最终物理有效门。

## 3. 几何门及证据边界

使用按果实过滤的拇指、食指和中指**平均接触位置**，并非声称拿到了每个原始接触点。冻结门槛：拇指对向角 ≤35°、拇指到食中指支撑中心的夹持线段偏心 ≤0.45 倍等效半径、食中指支撑跨度 ≥0.40 倍等效半径、现有拇食指 OBB 表面间隙 ≥2 mm。无接触、重合、同侧或非有限位置不能通过。

这些几何条件不等于摩擦锥力闭合。全链节自碰撞、精细掌面碰撞体、摩擦分解和脚本正向抓取轨迹尚未在本阶段取得完整物理证明；现有自碰撞代理也不构成完整安全认证。原路线图 P1 的全部退出门仍未闭合，不能以本次 A 阶段完成替代它。

## 4. 文件与可复现性

- `source/verifier.py`：接触获取/脱支撑保持分离、逐帧有效性、实际扰动边沿确认、失败锁存和向量环境隔离。
- `source/native_env.py`：实际撤支撑、动量写入、物理步确认、保持窗失效后重新施扰、终止前状态快照。自动 reset 后评估仍读取该回合的快照。
- `source/verification_geometry.py`：纯张量接触几何门。
- `source/run_manifest.py`、`source/native_run_manifest.py`：训练与评估入口均冻结完整命令、解析配置、环境控制参数、关节/接触体映射、源码/资产/权重 SHA-256、运行时版本和实际场景 USD；已有清单拒绝覆盖。
- `source/native_evaluate.py`：独立接触获取、脱支撑保持、自然脱果保持计数，逐回合结果关联运行清单哈希。
- `source/legacy_bridge.py` 与 `source/vendor/hand_manifold.py`：将迁移后缺失的动作映射按归档字节原样纳入项目，不再依赖旧空目录。
- `source/env.py`：旧入口不再将接触获取 latch 计为最终成功；未提供新合同的必要物理证据时默认失败。

以上相对路径均以 `research/small_fruit_pinch_v1/` 为基准。工作树原先已有大量几何、探针及日志迁移改动；不能把整个 `git diff` 视为本阶段新增内容。

动作映射来自只读归档 `/media/miobarnacle/Seagate Expansion Drive/Codex整理/ppo_tomato_grasp_stage01_work/simulation.tar.zst` 的 `simulation/ppo_tomato_grasp/hand_manifold.py`。恢复文件 SHA-256：`7782539e65b5a80e6a50647680292dd84c3a94e8767102a38fe0d614e6fb79bf`。供应副本保留原文与注释以维持字节身份。

## 5. 验证证据

证据目录：`research/small_fruit_pinch_v1/evidence/20260922_phase_a/`。

| 检查 | 结果与解释 |
|---|---|
| A 阶段相关测试 | `pytest_phase_a.log/xml`：86 passed；源码 compileall 与所改文件 diff 格式检查通过 |
| Verifier 反例 | 覆盖软带越界、缺指、掌托、缺支撑/扰动/几何/物理证据、禁触、越限、自碰撞、NaN/Inf、错误形状、跨环境隔离和新扰动恢复 |
| 正向张量轨迹 | 满足所有输入条件的完整序列可通过；这是状态机回归，不是物理抓取成功 |
| 两环境 PhysX | `physx_events.json`：实际 Δv 为 `0.1226250008 m/s`，示例冲量 `0.0154265724 N·s`；确认实际物理步后施扰、主动释放不冒充自然断梗、零接触不能成功 |
| PhysX 测试边界 | 为单独检验物理事件接线，测试显式注入阶段标志，记录 `probe_injected_stage_flags=true`；不声称 PPO 自主达到这些阶段 |
| 旧权重短评估 | `eval_model150/` 为首次接线评估；`eval_model150_frozen/` 为场景快照版，两个回合均超时、接触获取/保持/自然脱果保持均为 0，原结果不覆盖 |
| 全项目回归 | `pytest_full.log/xml`：394 passed、58 failed，另有 89 subtests passed；主要失败在未注册 harvest_cycle/model profile、迁移缺失文件和旧策略导出约 1e-6 数值差异。不是全仓测试通过声明 |

全仓失败详情已保留，相关文件和注册表均不属于本次 A 改动。首次全仓测试忘设项目要求的 `LINKAGENT_ISAACLAB_ROOT`，发生收集错误；补齐后才得到上表结果，两个日志均保留。环境未安装 basedpyright 且此前已拒绝安装，本次以运行测试和编译检查验证，不声称通过 LSP。

最终评估运行身份：`e28e1a6d3dc0663b4942976e4e314e4b7c9ecd96867ccc9d4befb44620a186c4`。已逐文件复核清单的 95 个源码、资产及权重条目，并复核回合结果引用同一运行身份。`frozen_run.tar.gz` 保存该次运行所需源码、测试、资产、场景和权重字节，以及测试/历史日志；系统 Python、CUDA 和 Isaac 运行环境仍需按清单版本准备，归档不是可直接部署的独立环境。

独立代码审查代理因账户额度限制中断，未给出最终审查结论，不能声称独立审查通过。主代理完成代码复核及上述实测。代理初步提出的并行回合数量与源码重建疑问已核对：请求回合数为完成下限，成功率分母采用实际完成数，逐回合记录全部保留；源码重建由实际字节归档和文件哈希支持，不依赖本次创建 Git 提交。本阶段未创建提交。

两环境 smoke 可从项目根目录运行：

```bash
export LINKAGENT_ISAACLAB_ROOT=/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1
export PYTHONPATH=/home/miobarnacle/linkbot/linkagent
$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python -B -m research.small_fruit_pinch_v1.source.native_physx_smoke --num-envs 2 --device cuda:0 --verification-events --evidence-json /tmp/phase_a_physx_new.json
$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python -m pytest -q research/small_fruit_pinch_v1/source/tests
```

评估完整命令在 `run_manifest.json` 的 `cli_argv` 中；重跑时换用新的 `--run-manifest` 目录。Kit 曾出现打印 traceback 但进程返回 0 的行为，因此验收必须同时检查结果 JSON、断言和日志，不能只看退出码。

## 6. 保留的历史基线

旧 `model_150.pt` SHA-256：`e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d`。文件位于挂载盘 `Codex整理/linkbot/linkagent/logs/rsl_rl/small_fruit_pinch_v1_structural_retrain_anchor040_actorinit_resume50_to200_20260905/`。

历史三种子成功率为 99.80%、99.41%、99.61%，属于旧 B0 合同；原始评估文本保留作历史证据。此前审计记录的零残差 100%、撤果梗成功 0% / 掉落 99.22% 只核到审计报告记录，未据此声称找回了对应完整逐回合原始证据。两类证据不得混写为本次复跑结论。

## 7. 审阅后再决定的工作

请先审阅成功语义、逐帧阈值、理想扰动强度和证据边界。后续阶段应先证明在真实碰撞与新最终门下存在可重复的无 PPO 抓取保持轨迹，再讨论动作解耦、奖励及 PPO 训练；本阶段结束后暂停执行。
