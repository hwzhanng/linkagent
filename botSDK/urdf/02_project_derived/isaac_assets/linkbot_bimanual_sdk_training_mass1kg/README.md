# LinkBot 双臂双手 Isaac 训练质量基线

本目录由官方/项目基准 USD 派生。左右 L20 整手总质量均补偿为 `1.000 kg`。

补偿只作用于左右 `hand_base_link`：官方指节质量保持不变，缺失质量加入掌部，掌部主惯量按质量同比缩放，质心和主轴保持不变。这是缺少实物质心/转动惯量测量时的训练起始估算，不是最终 Sim2Real 辨识结果。

入口资产：`linkbot_bimanual_sdk_training_mass1kg.usda`

详细数值和限制见 `mass_correction_manifest.json`。
