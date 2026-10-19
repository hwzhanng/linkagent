# PPO Firm Grasp v2.2 本机留存包

生成日期：2026-10-18。此包用于本机留存和查阅。

F4 是在冻结任务和新评估条件下对 actor-only 检查点的比较。旧 v1.6 baseline 通过当前新条件适配后评估；这不声称复现它的原始 v1.6 条件或结果。

下表按每个矩阵的真实 `episodes.jsonl` 分母统计，且已与对应 `summary.json` 的状态和汇总结果核对。计数只描述本次观测；不构成高可靠性或泛化结论，也不自动说明 PPO 优于 initial。

| 矩阵 | quiet 分母（成功/失败） | force 分母（成功/失败） | torque 分母（成功/失败） | 总成功/失败 |
|---|---:|---:|---:|---:|
| F4_matrix_update000001 | 8（1/7） | 6（2/4） | 6（0/6） | 3/17 |
| F4_matrix_initial | 8（2/6） | 6（1/5） | 6（0/6） | 3/17 |
| F4_matrix_legacy_v16 | 8（0/8） | 6（0/6） | 6（0/6） | 0/20 |

`F4_open_quiet` 的 summary 为 `EVALUATION_COMPLETE`，并有 4 个非 UNKNOWN 的实际 quiet 结果。

两段录像分别来自 `F4_video_front` 和 `F4_video_side`；收据与录像文件均已纳入包内并列入 MANIFEST。

原始运行数据没有重复复制。`RAW_EVIDENCE_INDEX.json` 记录 N/evidence 下各 F4 run 的绝对路径，以及可用 raw trace、episodes、metadata、summary 和录像文件的 SHA-256 与大小。读取这些证据仍依赖原 N 目录。

source 以 gzip level 1 保存为 `source-f3-v2_2.tar.gz`，排除了 `.git`、`.omo`、`__pycache__` 和 `.pytest_cache`。此为本机留存包，不是便携安装包；复核或运行仍依赖原始证据和本机 Isaac 安装。

完整包内文件清单和 SHA-256 见 `MANIFEST.json`。

逐窗入口：`SUCCESS_WINDOWS/index.json`。完整执行说明见 `reports/EXECUTION_REPORT.md`，录像元数据继承字段说明见 `reports/VIDEO_REVIEW.json`。
