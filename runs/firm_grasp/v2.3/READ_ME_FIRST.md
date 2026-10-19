# 先读这里：F3/F4 续作交付索引

状态：**理想边界已达到，候选已冻结，停止训练**。日期 2026-10-19。

## 一句话结果

在固定 67 mm 果、真实拇指+食指+中指指肚、零实际支撑、严格 360 子步 G2 下：

- 新准备条件（支撑最后 40 策略步渐退、第 120 步起严格为零）把继承 v2.2 actor 的 quiet 从 1/4 提升到 3/4；
- 从该 actor 做 10 次受约束 PPO 更新后，新 F4 预登记 20 项为 **15/20**，同条件未更新初始化为 **10/20**；
- quiet 6/8、force 4/6、torque 5/6；15 个成功窗全部通过独立数值 G2 和原始完整网格 PAD 重分类；
- 独立重载（同清单 seed111）逐个 case 复现 15/20 vs 10/20；旧 v1.6 适配 0/20、开掌 0/4。

## 必读文件

|文件|内容|
|---|---|
|[reports/UNIFIED_REPORT.md](reports/UNIFIED_REPORT.md)|统一报告：结果、比较、限制、复现命令、资源账|
|[reports/FINAL_COMPLETION_AUDIT.json](reports/FINAL_COMPLETION_AUDIT.json)|逐项完成审计与未完成项|
|[reports/F4_CONTINUATION_COMPARE.json](reports/F4_CONTINUATION_COMPARE.json)|新旧条件、初始化/候选/旧版/开掌对比表|
|[reports/RESOURCE_LEDGER_CONTINUATION.json](reports/RESOURCE_LEDGER_CONTINUATION.json)|25 次启动、墙钟、transitions、更新数、失败账|
|[CANDIDATE_FREEZE.json](CANDIDATE_FREEZE.json)|候选冻结时的权重/源码/条件/开发结果|
|[POST_FREEZE_RELOAD.json](POST_FREEZE_RELOAD.json)|冻结后的独立重载、负对照和录像结果|
|[RAW_EVIDENCE_INDEX.json](RAW_EVIDENCE_INDEX.json)|每个 run 的原始 summary/episodes/pad_trace/权重索引|
|[delivery/DELIVERY_MANIFEST.json](delivery/DELIVERY_MANIFEST.json)|可移植交付包及全部 SHA256|

## 权重

- 候选（best=last）：`evidence/C01_train10/final.pt`，SHA256 `ffbf091722a8693676f28b5f4d8cce24e003016081a1ccf151b69560c29fa163`
- 更新前：`evidence/C01_train10/initial.pt`（actor 来自 v2.2 `update_000001`）
- v2.2 更新/初始化、F2、v1.6 参考权重均在 `delivery/weights/`。

## 源码身份

- 主源码：`source`，commit `5faf919712d9a7f13d03dcabde6cbdcc0c312e30`，任务版本 `firm_grasp_v2.3.0`
- C01 训练同身份冻结副本：`source-v2.3.0-c01`，commit `04444652423ab9f440ed049e0633ad7ad7a7ff48`
- **strict reload 必须使用 C01 冻结副本**（主源码在训练后增加了 v2.3 actor seed 兼容，critical hash 已变）；跨身份矩阵使用显式 `--actor-seed-checkpoint`。

## 不要做的事

- 不要把 15/20 写成高成功率、统计显著或分布外泛化；它只换 seed 未换初态分布。
- 不要把 ramp40 的准备条件收益写成 PPO 学习收益。
- 不要修改旧 v1.6/F2/v2.2 或原始失败；不要再在最终清单上调参。
- 不要开启 CAN/真机；本任务从未使能硬件。
