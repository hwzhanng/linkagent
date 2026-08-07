# LinkBot 双臂双手 Isaac USD 验证报告

日期：2026-07-26  
运行环境：Isaac Sim 6.0.1.0、Isaac Lab 6.1.14、CUDA PhysX

## 结论

正式 USD 已生成到独立资产目录，并完成真实 PhysX 加载与小幅运动测试。模型可以进入 Isaac Sim 的场景集成、控制接口开发、观测/动作空间设计及运动学主导的初期强化学习验证。

当前不能把它视为完成 Sim2Real 的动力学模型。左手运动标定已经实测，右手当前只把
左手参数作为仿真先验并保持待验收状态；仍缺灵巧手真实质量分布、阻尼、摩擦、关节/电机等效
惯量、驱动增益、力矩限制和果实接触材料参数。

## 资产

- 主入口：`../isaac_assets/linkbot_bimanual_sdk_description/linkbot_bimanual_sdk_description.usda`
- 资产清单：同目录 `linkbot_asset_manifest.json`
- 动态测试：同目录 `tests/physx_smoke_report.json`
- 源 URDF SHA-256：`5bb770c58f8af9fd03e92a8c147eabeffefa5424dbec6ec0a095c0406b38865b`
- 主 USD SHA-256：`4a5be25b3ed31ea59d7b0dad66a9b05003a17b73ddb85090293d0fbfad720bb2`

资产是分层结构，包含独立的几何、实例、材质、通用 Physics、PhysX 和 MuJoCo payload。所有 9 个 USD 文件的大小与 SHA-256 都已写入 manifest。

## 静态检查

- 默认 prim：`/linkbot_bimanual_sdk`
- Articulation root：`/linkbot_bimanual_sdk/Geometry`
- 转动关节：56
- 固定关节：15，其中 1 个是固定底座
- 刚体：71
- 碰撞网格：71，全部为 `convexDecomposition`
- mimic：左右手各 5 个，共 10 个
- USD 所有已用文件层均解析到本资产目录内，无外部网格路径依赖

手部安装变换与已确认的项目姿态一致：

- 右手：`rpy=(pi, 0, +pi/2)`，以正式 URDF 和 `config/mounts.json` 为准
- 左手：`rpy=(pi, 0, -pi/2)`，以正式 URDF 和 `config/mounts.json` 为准

## Isaac importer 修正

URDF mimic offset 使用弧度，USD/Newton 的转动 mimic offset 使用度。Isaac Sim 6.0.1 importer 未完成该单位换算。本转换脚本对 10 个 `newton:mimicCoef0` 执行 rad→degree，并在 manifest 中逐项记录。

全零手指状态会与负 mimic offset 和从关节下限冲突。正式资产写入满足 mimic 方程的安全张开初态，避免刚加载时产生强约束冲击。这个初态是仿真稳定初值，不是实机零位标定。

## CUDA PhysX 冒烟测试

测试时长 3.0 秒，物理步长 1/120 秒：

- PhysX articulation 创建：通过
- 固定底座：通过
- 根部平移漂移：0 m
- 根部旋转分量漂移：0
- 数值有限性：通过，无 NaN/Inf
- L1 目标约 0.0500 rad，实际约 0.04649 rad，误差约 0.00352 rad
- R1 目标约 -0.0500 rad，实际约 -0.04649 rad，误差约 0.00351 rad
- 10 个 mimic 动态关系：通过
- 左手 21 个关节实测速度上限：逐项通过
- 右手 21 个仿真关节在当前先验速度上限下：逐项通过（不等于真机继承验收）
- mimic 最大绝对残差：约 `3.19e-5 rad`

测试只运行仿真，没有连接或控制真机。

## 进入农业抓取训练前的剩余工作

1. 实测一只代表手的总质量、各主要连杆质量/质心和近似惯量，并由用户确认的
   同批次一致性同步到右手，修复当前 URDF 手部总质量明显偏低的问题。
2. 对左手代表样机的每类关节辨识静摩擦、动摩擦、粘性阻尼、回差和力矩饱和，
   结果同步到右手；速度、死区和温升运动标定已经完成。
3. 标定真机零位与本资产安全张开初态之间的映射。
4. 标定控制链延迟、状态采样抖动和命令保持行为，并在训练中随机化。
5. 为指尖与果实建立接触材料、柔性/顺应性和触觉阵列模型；采摘还需加入果梗断裂或约束释放模型。
6. 在单环境低速接触稳定后，再开启必要的自碰撞对并扩展到并行强化学习环境。
