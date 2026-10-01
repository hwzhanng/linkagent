# 小果 PPO A 阶段代码质量审查

## 结论

- `codeQualityStatus`: `WATCH`
- `recommendation`: `APPROVE`
- 阻塞项：无。
- 审查范围：A 阶段 verifier 合同、运行清单、评估计数和用户点名测试；未重复 PhysX 扰动恢复验证，未审查 B 阶段或训练效果。

## Findings

### CRITICAL

无。

### HIGH

无。

### MEDIUM

1. **入口模块仍然过大，后续修改的回归风险偏高。** `research/small_fruit_pinch_v1/source/native_evaluate.py` 约 1350 行纯代码，`research/small_fruit_pinch_v1/source/native_train.py` 约 992 行纯代码，超过 `programming` / `remove-ai-slops` 视角的 250 LOC 阈值。此次 A 阶段行为和证据未因此失效，不要求为交付 A 扩大重构；后续若继续改这些入口，应先按参数解析、环境配置、统计聚合和证据写入拆分。

### LOW

1. `verifier.py:167-170` 假定 `center_drift_m` 与 `relative_speed_m_s` 是调用方计算出的非负范数，只检查上限和有限性。当前 `native_env` 调用链满足该假定，因此不构成 A 阶段假成功；若未来把 verifier 变成外部输入边界，应在边界拒绝负值。
2. 冻结归档保存了本次运行输入字节，但 Python/CUDA/Isaac 运行环境仍需按 manifest 中的版本另行准备；归档不是可直接部署的独立环境。这一点已在 `PHASE_A_REVIEW_20260922.md` 明示。

## 已核验证据

- `verifier.py:193-206` 仅在脱支撑保持达到第 40 步后接受新的扰动上升沿，并要求扰动后保持窗；早期脉冲最小复现得到 `final_success=False`。
- `native_evaluate.py:1088-1104` 的 `--episodes` 是至少完成数；同步 vector 可越过下限，但 `native_evaluate.py:1256` 使用实际 `events["episodes"]` 作分母，`native_evaluate.py:1115-1122` 保存每个已完成回合。该行为不是计数错误。
- `eval_model150_frozen/run_manifest.json` 的 `run_sha256=e28e1a6d...186c4` 可重算匹配；`evaluation.json` 引用同一运行身份；当前 `native_env.py`、`native_run_manifest.py`、`verifier.py`、`native_evaluate.py` 哈希与清单一致；清单包含 `scene.usda`。
- `frozen_run.tar.gz` 存在，SHA-256 为 `6f3bf6a86f1978637f970446f8265fea716cd1e609c3a987d9aa7bbbc943190e`，包含权重、源码、清单、评估 JSON 和冻结场景。
- 独立复跑点名测试：77 passed；已检查项目 A 阶段日志结尾为 86 passed。短评估为 2 个完成回合、2 条逐回合记录，接触获取和脱支撑保持均为 0；没有把接口加载成功或零成功短评估写成物理抓取成功。
- 全项目失败未作为 A 阶段通过证据；报告明确保留摩擦锥、完整自碰撞和物理正例未验收限制。

## 技能视角检查

- `remove-ai-slops`：已运行。未发现删除型测试、只验证“移除”的测试、输出自比、常量镜像造成的假覆盖，或 A 目标不需要的生产解析/归一化。相关测试针对可观察合同，具备实际回归价值。
- `programming`：已运行。未发现 `Any`、`cast`、类型忽略或实现镜像式脆弱测试。入口模块尺寸违反该视角的维护性要求，已列为 MEDIUM；不影响本次 A 阶段正确性结论。

## 剩余验收边界

A 阶段只证明 verifier/证据链能够拒绝缺失物理证据，并能冻结与复核本次运行；没有证明旧 `model_150.pt` 在新合同下成功，也没有关闭摩擦锥力闭合、完整链节自碰撞或原 P1 的全部退出门。完成 A 后停止、不进入 B 或训练，与现有证据一致。
