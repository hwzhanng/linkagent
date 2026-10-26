# 接触可行性优先搜索结果

完成 2/2 轮、8192 条完整试验。有效指腹接触优先且仅使用可行精英更新，仍无达到连续三周期各 20 度净转动的安全候选。

[
  {
    "file": "SEARCH_round00.npz",
    "passed": 0,
    "safe": 161,
    "contact_feasible": 27,
    "best_contact_feasible_min_late_deg": 1.1239418983459473
  },
  {
    "file": "SEARCH_round01.npz",
    "passed": 0,
    "safe": 1494,
    "contact_feasible": 639,
    "best_contact_feasible_min_late_deg": 2.926363229751587
  }
]

诊断搜索不再追加预算。此次固定四路点、6秒周期、8个起点的失败不证明一般反馈策略无法持续转动。没有新增 PPO 样本，也没有改动冻结的9个核心文件。
