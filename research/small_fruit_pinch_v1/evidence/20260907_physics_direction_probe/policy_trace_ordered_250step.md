# 250 步动作顺序审计（同一 provisional 权重）

轨迹使用 `policy_trace_probe.py`，仅左手、64–70 mm、显式手部 PD、同一 y-shift
果心和掌部诊断过滤。输出经过新增的首次接触记录字段解析。

```text
step=000 action=[0.006, 0.021, -0.0063, -0.0222, 0.0315, 0.0111, 0.0476, 0.0335, 0.017, 0.012, 0.0002, 0.0263, 0.0222, 0.0054]
step=125 action=[0.0119, -0.0006, -0.0142, 0.0247, 0.0053, -0.0124, 0.0039, -0.0125, 0.0089, -0.0029, 0.0267, 0.0319, 0.0167, -0.0172]
policy_geometry_summary source=contact_points best_step=227
  opposition_angle_deg=43.47 thumb_to_index_middle_chord_offset_mm=11.59
policy_trace_summary
  peak_force_n=[[0.47404, 0.51160, 0.59083, 0.0, 0.0]]
  required_contact_steps=[17]
  first_contact_step=[[14, 205, 211, -1, -1]]
  contact_order=[[0, 1, 2]]
  terminated=0
```

索引 `0/1/2` 分别为拇指、食指、中指；`-1` 表示无名指/小指未达到
`0.08 N` 接触阈值。该证据支持“拇指先进入、食指随后、中指最后补齐”的动作顺序，
但不等价于断梗和摘后保持成功。
