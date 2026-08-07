# 2026-07-27 LinkBot Sim2Real 实验目录

目标是取得能够修正双臂双手 Isaac 模型的真实数据，不把未标定的 SDK 原始量误写成
SI 物理量。

## 文件

- `EXPERIMENT_PLAN.md`：现场执行顺序、安全条件和验收标准。
- `session_manifest.template.yaml`：设备、固件、工具和采样条件记录。
- `hand_joint_system_id.template.csv`：灵巧手关节响应、摩擦和迟滞。
- `arm_joint_system_id.template.csv`：机械臂关节响应、摩擦、延迟和温度。
- `tactile_contact_calibration.template.csv`：五指法向、切向、方向和滑移标定。
- `rigid_body_measurements.template.csv`：手、转接件、线束、工具的质量和重心。
- `fit_equivalent_joint_dynamics.py`：从恒速正反向数据拟合第一版粘性/库仑摩擦；无力矩标定时只输出电流域系数。
- `SIM2REAL_ACCEPTANCE.md`：数据回来后的模型更新与训练准入条件。
- `PARAMETER_MATRIX.md`：机械臂、灵巧手、触觉、安装和农业对象的缺口—实测—模型落点对照。

实验时复制模板，不要直接覆盖模板，例如：

```bash
cp hand_joint_system_id.template.csv hand_left_trial_001.csv
```

所有时间序列必须同时记录主机单调时钟，不能仅使用收到数据的序号。
