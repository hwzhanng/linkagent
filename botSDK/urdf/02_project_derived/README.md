# LinkBot 正式双臂模型与仿真资产

本目录只保留当前双臂 7 关节机械臂和左右 G20 灵巧手需要的正式模型、仿真基线、
辨识资料与验证报告。

## 唯一 ROS 2 模型

```text
ros2_packages/linkbot_bimanual_sdk_description/
└── urdf/linkbot_bimanual_sdk_description.urdf
```

模型包含左右各 7 个机械臂关节和左右灵巧手，直接使用 SDK 控制坐标系。关节分组、
安装变换、仿真基线和左手 G20 标定覆盖层位于该包的 `config/`。

## 正式 Isaac Sim 资产

- `isaac_assets/linkbot_bimanual_sdk_description/`：标准双臂双手资产；
- `isaac_assets/linkbot_bimanual_sdk_training_mass1kg/`：质量修正训练资产。

保留的生成和验证工具位于
`../../lbot_arm_api-master/tools/isaac/`，包括双臂转换、质量修正、训练准入检查、
交互检查和 PhysX 冒烟测试。

## 其他关键内容

- `simulation_baselines/tomato_pick_v1.yaml`：C2 当前唯一番茄/果梗训练入口；
- `simulation_baselines/agricultural_grasp_no_equipment_proxy.yaml`：只保留旧物理先验；
- `system_identification/20260727_lab/`：实机辨识方案、数据模板和拟合工具；
- `reports/`：正式双臂模型、Isaac 资产和训练准入验证；
- `../01_vendor_original/`：原始机械臂/灵巧手模型和来源证据。

## 清理边界

历史压缩包、旧版本 USD、单臂/单手模型、网页预览模型和旧 PPO 专用场景均已移除。
这些内容不是当前控制或训练入口，不应被新的 Skill 或 PPO 代码引用。
