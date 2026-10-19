# PPO 单手果实旋转抓取同步说明（2026-10-13 ~ 2026-10-19 v2.3）

## 同步范围

- **时间区间**：2026年10月13日 ~ 2026年10月19日（截止 v2.3 交付）
- **同步内容**：核心研究代码 + 权重 + 交付报告
- **已排除**：
  - 外部 agent a/b/c 三线正在进行的工作（ppo-firm-grasp-next-{a,b,c}）
  - v2.4 及之后的本地实验
  - 完整原始仿真 trace（window.jsonl / pad_trace.jsonl）
  - 源码归档压缩包（与 research/ 目录冗余）
  - 训练缓存与中间 checkpoint

## 目录结构

### 研究代码（research/）

| 目录 | 说明 |
|---|---|
| research/firm_grasp_v1/ | 主力紧握 PPO 管线（v2.3 对应源码） |
| research/thumb_pad_grasp_v1/ | 拇指指垫抓取变体 |
| research/fixed_fruit_grasp_v1/ | 固定果实抓取基线 |

### 运行结果（runs/）

| 目录 | 日期 | 说明 | 权重大小 |
|---|---|---|---|
| runs/grasp_first_result/ | 10-13 | 首次真实接触抓取结果 | ~7.5MB |
| runs/grasp_learning/ | 10-13 | 抓取学习探索 | ~7.5MB |
| runs/thumb_pad_v1/ | 10-16 | 拇指指垫 v1.6 交付 | ~8MB 权重 |
| runs/firm_grasp/v1.0_exploration/ | 10-17 | F0-F2 紧握探索阶段 | ~8MB |
| runs/firm_grasp/v1.8_f2_handover/ | 10-18 | F2 v1.8 交接包 | - |
| runs/firm_grasp/v2.2/ | 10-18 | F3/F4 v2.2 交付 | ~8MB |
| runs/firm_grasp/v2.3/ | 10-19 | **F3/F4 v2.3 最终交付** | ~35MB |

## v2.3 关键信息

- **状态**：理想边界已达到，候选已冻结，停止训练
- **核心权重**：runs/firm_grasp/v2.3/weights/C01_final.pt
  - SHA256: ffbf091722a8693676f28b5f4d8cce24e003016081a1ccf151b69560c29fa163
- **结果摘要**：固定 67mm 果、真实多指指肚接触、严格 360 子步 G2 下，F4 预登记 20 项为 15/20
- **必读文档**：
  - runs/firm_grasp/v2.3/READ_ME_FIRST.md
  - runs/firm_grasp/v2.3/reports/UNIFIED_REPORT.md
  - runs/firm_grasp/v2.3/DELIVERY_MANIFEST.json

## Git LFS

以下大文件类型使用 Git LFS 追踪：
- *.pt, *.pth（模型权重）
- *.onnx（ONNX 模型）
- *.mp4（视频）
- *.usda（USD 资源）
- *.bin（二进制数据）

## 源码提交身份

- 主源码 commit: 5faf919712d9a7f13d03dcabde6cbdcc0c312e30
- C01 训练冻结副本 commit: 04444652423ab9f440ed049e0633ad7ad7a7ff48

## 注意事项

1. 本同步不包含真机硬件相关代码，所有实验均在仿真环境中完成
2. v2.3 strict reload 必须使用 C01 冻结副本源码
3. 原始完整 trace 数据保存在 Seagate 扩展盘 linkbot-runs/ 目录
4. 请勿在最终清单上调参，请勿修改旧版本基线结果
