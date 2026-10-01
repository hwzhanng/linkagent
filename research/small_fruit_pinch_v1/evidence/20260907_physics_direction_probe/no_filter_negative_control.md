# 无掌部过滤负对照（2026-09-07）

目的：确认 provisional 权重的三指接触不是由掌部过滤之外的隐藏臂/右手或固定终姿造成。

保持以下条件不变：单左手 USD、64–70 mm、同一 `model_final.pt`、同一中心
`(-52.42,-16.09,-177.48) mm`、显式手部 PD；只去掉
`--palm-fruit-collision-filter`。

```text
policy_trace_summary peak_force_n=[[0.0, 0.0, 0.0, 0.0, 0.0]]
required_contact_steps=[0]
first_contact_step=[[-1, -1, -1, -1, -1]]
contact_order=[[]]
```

该对照没有把失败当作训练崩溃：当前孤立 USD 的 `base_link` 粗糙碰撞 AABB 会在 reset
阶段把果实沿接近射线推出工作区，因此它暴露的是碰撞几何问题。正式发布前应以精确掌面
碰撞体替代过滤开关，并重新跑同一负对照；在此之前本权重仅用于动作方向人工检查。
