# 官方资产查找与模型补齐报告

日期：2026-07-26

## 结论

厂商公开资料中已经存在本机 `LKLS73 / A7V2_50_4.27` 机械臂的官方仿真资产，
而且官方 `linker-sim` 仓库已经包含 Isaac Sim/Isaac Lab 加载后端。当前组合模型
的机械臂质量、重心和惯量没有缺失，并与官方在线仿真资产的 17 个刚体逐项一致。

仍然不能从官方资料中得到真实关节摩擦、齿隙、传动效率、电机转子惯量、控制延迟
和接触材料。官方仓库给出的阻尼、刚度和 armature 是仿真起始值，不是实机辨识值。

## 已检查的来源

1. 用户官网下载包 `L20_8_Urdf文件.zip`，内部左右手包日期为 2024-12-07。
2. 当前工作目录随设备交付的 SDK：
   - `机械臂控制接口文档v1.0.5.pdf`
   - `机器人控制平台说明文档v1.1.1.pdf`
   - C/C++/Python API、ROS 2 driver 和控制平台模型资源。
3. 官方 LinkerHand URDF 仓库：
   https://github.com/linker-bot/linkerhand-urdf
   ，快照 commit `0b2a80ff68b09485e6e77474a58a0e79fddfc00b`。
4. 官方仿真仓库：
   https://github.com/linker-bot/linker-sim
   ，快照 commit `ed1b8ea26bb1161326916934e2ceb596ea01f7a6`。
5. 厂商官网产品页和 FAQ：
   https://www.linkerbot.cn/product?page=J
   和 https://www.linkerbot.cn/faq 。

## 机械臂查找结果

官方 `linker-sim` 中存在：

- `assets/components/arms/lkls73_arm/`
- `assets/components/bases/lkls73_torso/`
- `assets/workstations/lkls73_i1_*`
- Isaac 后端 `UrdfFileCfg + ArticulationCfg + ImplicitActuatorCfg`
- MuJoCo MJCF 版本

与随设备交付模型的对照结果：

- 17/17 个刚体的质量、重心和惯量数值完全一致。
- 16/17 个 STL 与官方 Git LFS 对象 SHA-256 完全一致。
- 唯一不同的是在线仿真仓库的右侧 `R8_Link.STL`：
  - 本机交付版约 `34 × 34 × 1 mm`，312 三角面；
  - 在线仿真版约 `120 × 60 × 6 mm`，1000 三角面。
  在线版本显然包含了不同的右侧法兰/适配板，不能在未核对实物前替换当前模型。
- 官方仿真组件对多处关节轴方向做了符号重定义，L2/R2 的非对称限位也随之取反。
  它与 SDK 实时关节值之间需要建立显式符号映射，不能直接替换后继续回放现有录制轨迹。

官方仿真起始参数已经整理到
`../simulation_baselines/lkls73_official_sim_baseline.yaml`：

- 关节位置控制：`stiffness=2000`、`damping=20`
- OSC：`stiffness=150`、`damping=8`
- MuJoCo：`damping=20`、`armature=0.01`、`frictionloss=0`
- Isaac 导入：`convex_decomposition`
- Isaac articulation：关闭自碰撞、位置求解迭代 8

这些参数解决了“没有任何可用起始驱动配置”的问题，但没有解决实机摩擦辨识。

## L20 手查找结果

官网下载的 2024 版模型：

- 左右各 27 links / 26 joints；
- 21 个 revolute，其中 16 个独立关节 + 5 个 mimic；
- URDF 总质量分别约 0.1978 kg 和 0.1987 kg；
- 所有关节没有 `<dynamics>`；
- effort 写成 100，velocity 写成 1；
- visual STL 同时用作 collision。

官方在线 `linkerhand-urdf` 当前 L20 模型：

- 左右各 22 links / 21 revolute joints；
- 同样为 16 个独立关节 + 5 个 mimic；
- 去掉了旧版的 5 个固定末端占位 link，命名改为人体解剖式名称；
- URDF 总质量分别约 0.1805 kg 和 0.2043 kg；
- effort 改为 1，velocity 仍为 1；
- 仍无 damping/friction，仍使用视觉 STL 作为 collision。

官网 FAQ 给出的 L20 整手重量为 1 kg、手指速度最高 1.86 rad/s、单指约 10 N。
两个版本 URDF 的总质量都只有约 0.2 kg，因此 URDF 惯量显然没有覆盖整机电机、
线束、外壳或传感器的全部实物质量。不能把在线新 URDF 视为已经修正了动力学质量。

当前项目暂不切换到在线新 L20，原因是：

1. 当前 2024 L20_8 模型已经与真机页面姿态和安装方向确认一致；
2. 在线新模型的关节命名和结构不同，需要先完成 16 通道映射；
3. 两个版本都需要实物总质量/重心和关节动力学标定。

## SDK PDF 能补齐什么

交付 SDK 文档确认：

- 支持 A7 Lite、A7 V1、A7 V2、H75 V1 等机械臂；
- 本项目通信结构为双臂通过 CAN 接树莓派，电脑通过网线访问控制器；
- 普通状态监控 50 Hz，关节/位姿跟随接口 50–200 Hz；
- API 运动速度和加速度输入范围为 0–20 rad/s、0–20 rad/s²；
- 工具坐标系可以设置末端工具重量和重心；
- 可读取关节位置、速度、力矩/电流和温度。

PDF 没有给出：

- 各关节真实 damping/friction；
- 电机转子惯量、减速比、齿隙和效率；
- 电流到关节力矩标定系数；
- 碰撞材料、指尖柔度或接触模型；
- 各关节控制器实际 PD 增益。

因此 SDK 文档能补齐控制频率、动作接口、工具载荷和状态观测定义，不能直接补齐
完整刚体/执行器动力学。

## 当前已完成的补齐

- 原始与派生资产已分目录，不再混放。
- 官网压缩包已完整原样解压，保留外层和两个内层 ZIP。
- 随设备 SDK 的两份 PDF 和 LKLS73 原始模型已单独归档。
- 官方最新 L20 左右 URDF/mesh 已归档并通过 XML、树结构、惯量和 ROS `check_urdf`。
- 官方 LKLS73 Isaac/MuJoCo 起始参数已形成独立 baseline，不污染原始 URDF。
- 派生五个 ROS 2 包已在新路径重新生成并再次通过验证。

## 仍需实测或后续实现

1. 在实机静止和慢速扫频数据上辨识每个机械臂关节的阻尼、库仑摩擦和控制延迟。
2. 称量一只安装了触觉传感器和线束的同批次代表手并至少测量整手重心；右手按
   用户确认的同型号同批次高一致性继承。
3. 为 L20 指尖制作独立低面数、高曲率精度 collision；其他手指 link 使用凸分解。
4. 建立官方在线 L20 16 通道与当前 SDK 通道的映射，并做至少三个姿态的视觉核对。
5. 对官方 `linker-sim` 与 SDK 的机械臂关节符号做实机单关节小角度核验。
6. 农业采摘训练时开启物体/指尖接触，但不要直接照搬官方“全机器人关闭自碰撞”
   作为最终设置，应改成经过测试的自碰撞过滤表。
