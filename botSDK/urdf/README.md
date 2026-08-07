# LinkBot 模型资产

本目录只保留两类内容：

- `01_vendor_original/`：厂商原始模型、下载包和官方仓库快照，仅用于溯源；
- `02_project_derived/`：当前正式双臂模型、Isaac 资产、仿真基线、辨识资料和报告。

## 当前唯一模型入口

- ROS 2/URDF：
  `02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/`
- 标准 Isaac Sim：
  `02_project_derived/isaac_assets/linkbot_bimanual_sdk_description/`
- 质量修正训练资产：
  `02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/`

历史单臂、单手、网页预览、旧 PPO 场景、版本归档和交付压缩包均已清理。

## 管理规则

1. 不修改 `01_vendor_original/` 中的原始文件；
2. 项目修改只进入 `02_project_derived/`；
3. 仿真默认值与实机辨识值必须分开标记；
4. 新 PPO/Skill 只引用正式双臂入口，不复制模型目录；
5. 生成结果必须可由 `lbot_arm_api-master/tools/isaac/` 中保留的工具重建。

详细准入和验证结论见 `02_project_derived/reports/`。
