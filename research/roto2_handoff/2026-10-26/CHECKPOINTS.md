# 模型权重与选择规则

成熟源模型是续接工作的默认基线，不是已完成研究目标的最终模型。V17已明确失败，不要因为目录更新就选它。每个完整agent检查点含模型和优化器/归一化状态；仅actor导出或诊断张量不能代替完整续训检查点。检查点接续不保存完整PhysX与随机数运行态，属于恢复优化器后重新初始化环境，不是逐位重放原进程。

|别名|用途|包内路径|SHA-256|
|---|---|---|---|
|mature_baseline|默认基线；650,281,216累计交互，持续触觉旋转/压力优势尚未完成|`logs/roto2_pressure_fruit/plain_seed7401/2026-10-21_16-16-13/checkpoints/final_agent.pt`|`6108ba5deaa7f535d02bcfc48f6cd2931b9007de5f3d9fc9c1b4841ca314f216`|
|native24_diagnostic|V16原探索24秒短训，V18已补评；不是替代基线|`logs/roto2_pressure_fruit/plain_seed9811/2026-10-23_13-56-45/checkpoints/final_agent.pt`|`4df800228e2ace2b0982ab50480d2ead7906a5c1969ff54369c9a20a5d978261`|
|native48_diagnostic|V16原探索48秒短训，V18已补评；不是替代基线|`logs/roto2_pressure_fruit/plain_seed9811/2026-10-23_14-03-06/checkpoints/final_agent.pt`|`253158a0156387658840a7dcdc2077979f0332980d28ff84062c00906fcc71f0`|
|v17_masked_negative|退化对照；不建议扩训；critic输入257维|`logs/roto2_pressure_fruit/plain_seed9880/2026-10-23_14-28-38/checkpoints/final_agent.pt`|`96b980726e854578000079f04cd4e133401fadbecad180fe982a9422b5bd94d3`|
|v17_validity_negative|退化实验；不建议扩训；critic奖励历史位；actor仍原接口|`logs/roto2_pressure_fruit/plain_seed9880/2026-10-23_15-16-34/checkpoints/final_agent.pt`|`d71e3d65cde9a5917510b553cf9bb869caeb0315815c060dc81f23aaf936b3e5`|

所有其他历史权重均在 `ALL_WEIGHTS.json`。Stage3的plain/scalar/Jacobian各3种子也完整保留，通过 `ALL_RUNS.json` 的seed8301/8302/8303筛选。`best_agent.pt` 是当时训练器的选择，不能直接理解成通过独立物理验收。
