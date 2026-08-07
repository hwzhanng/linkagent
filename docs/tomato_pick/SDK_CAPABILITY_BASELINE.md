# 番茄采摘设备能力基线

状态：C0 冻结基线；C2-0/C2-1 事实校正与 C4-O 离线适配已对齐  
版本：1.0.1  
日期：2026-08-04  
适用设备：LinkBot 双臂（每臂 7 关节）+ 左右 G20 灵巧手

本文记录 `botSDK/` 的静态审计结论，作为番茄采摘接口设计的设备事实来源。审计期间
没有启动 ROS、CAN、仿真或真机运动。用户最新确认的“每臂 7 关节”优先于更早的
口头描述；旧 Fruit-Grasp PPO 已从 `botSDK/` 清理，只保留本文件中的历史审计结论。

本文中的“设备已实现”只表示能力存在于 `botSDK/`。RealAgent 已有 C4-O 离线转换
adapter，但默认 Registry 仍只运行 mock；vendor gateway 和真机链路尚未接入。

## 1. 结论

1. 机械臂不需要重新开发 FK、IK 或基础高级控制器。现有主链路已经提供 FK、带种子
   和软限位的 IK、MoveJ、MoveJP、MoveL、FollowP、取消、失能和软件急停。
2. MoveC 不可用，正式接口中不声明、不模拟、不用折线伪装。
3. 现有机械臂 SDK 没有环境碰撞检测、自碰撞规划或绕障。IK 成功只表示目标位姿有
   解，不能作为路径安全证明；采摘系统仍需要独立的运动安全接口。
4. G20 的外部硬件姿态命令是 20 维 `0..255` 逻辑向量，其中 11～14 是保留通道；
   仿真手则是每手 16 个独立关节和 5 个 mimic 关节。PPO 不能直接把任一维度当成另一
   维度，必须经过版本化的手部映射器。
5. 番茄采摘 PPO 不直接输出 7 个机械臂关节角，也不沿用旧 baseline 的 27 维动作。
   PPO 只输出抓取坐标系中的小幅末端残差和 3 个手部协同量；安全监督器再映射到
   FollowP 和 G20 逻辑命令。
6. 采摘成功必须由独立验证器确认“果实已经脱离且仍被稳定持有、没有滑落”。PPO 的
   自报、单次抬升或仿真私有状态不能单独构成实机成功。
7. 当前已经验证的 G20 触觉是五指 `12×6`、`0..255` 的相对矩阵权重。名义 `20 N`
   量程不是换算曲线；逐指标定前不能生成有效的 N 值。

## 2. 机械臂事实

主要证据：

- [本机直连机械臂 README](../../botSDK/lbot_arm_api-master/tools/local_arm_can/README.md)
- [高级控制器说明](../../botSDK/lbot_arm_api-master/docs/本机直连机械臂高级控制器.md)
- [高级控制器实现](../../botSDK/lbot_arm_api-master/tools/local_arm_can/advanced_controller.py)
- [ROS 2 服务入口](../../botSDK/lbot_arm_api-master/tools/local_arm_can/ros_node.py)
- [ROS 2 接口定义](../../botSDK/lbot_arm_api-master/tools/local_arm_can/ros_ws/src/lbot_arm_interfaces)

|项目|冻结事实|接口影响|
|---|---|---|
|结构|左右臂各 7 个独立转动关节，命名为 `L1..L7`、`R1..R7`|所有臂状态、命令、种子和解都严格为 7 维|
|单位|关节为 rad；位置为 m；SDK FK/IK 姿态为 RPY rad|中立接口统一使用 m、rad 和 `quaternion_wxyz`，只在 adapter 转 RPY|
|离线几何|FK、带种子/软限位/可达性判断的 IK|可在不发 CAN 时用于规划和测试|
|轨迹能力|MoveJ、MoveJP、MoveL、FollowP|分别映射关节轨迹、位姿到关节轨迹、直线末端轨迹和单点位姿跟随|
|缺失能力|MoveC 是官方失败空桩|公共接口不出现 `move_c`|
|轨迹/总线频率|官方兼容时间律按 100 Hz 采样；已验证的本机 CAN 调度为 50 Hz|adapter 内部保持 SDK 调度；首版 PPO 外环固定 20 Hz|
|反馈门禁|七关节反馈超过 250 ms 时拒绝或中止|Skill 使用更严格的可配置新鲜度上限，SDK 250 ms 看门狗继续保留|
|软限位|目标和每个轨迹点检查软限位；直接相邻关节命令默认不超过 0.08 rad|上层安全监督器不能绕过 SDK 限位|
|调度门禁|调度落后超过 50 ms 时中止|超时属于安全失败，不能自动补发积压动作|
|抢占|取消、失能和软件急停可阻止后续帧；一个高级轨迹独占控制权|Skill 执行期必须持有唯一控制租约|
|碰撞|没有 MoveIt、环境碰撞检测、自碰撞规划或绕障|预抓取运动前必须取得独立 `SafetyAssessment`|

底层接口虽然有阻塞参数，番茄采摘公共契约使用执行句柄、状态查询和显式取消。这样
Task 不会因一次长运动失去超时、取消和故障可见性。

## 3. G20 手部与触觉事实

主要证据：

- [G20 官方接口包装](../../botSDK/linkerhand-ros2-sdk-main/linker_hand_ros2_sdk/linker_hand_ros2_sdk/LinkerHand/linker_hand_api.py)
- [G20 CAN 实现](../../botSDK/linkerhand-ros2-sdk-main/linker_hand_ros2_sdk/linker_hand_ros2_sdk/LinkerHand/core/can/linker_hand_g20_can.py)
- [G20 ROS 2 节点](../../botSDK/linkerhand-ros2-sdk-main/linker_hand_ros2_sdk/linker_hand_ros2_sdk/linker_hand_advanced_g20.py)
- [G20 实机标定](../../botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/config/left_g20_motion_calibration_20260726.yaml)
- [当前 G20 实机上下文](../../botSDK/linkerhand-control-workspace/PROJECT_CONTEXT.md)
- [左手 G20 工作区](../../botSDK/linkerhand-control-workspace/README.md)

### 3.1 手部命令

|项目|冻结事实|接口影响|
|---|---|---|
|硬件姿态|20 维逻辑值，范围 `0..255`|仅 G20 adapter 可见；不直接暴露给 Agent 或 PPO|
|有效通道|0～10、15～19 共 16 个；11～14 保留|mapper 必须保持保留通道的安全值，不能把其当作自由关节|
|串行帧|SDK 把 20 维重排为五指各 6 个协议槽|30 个协议槽不是 30 个独立物理关节|
|速度|五指速度限制，可配置|必须使用经设备标定的连续速度上限|
|torque|SDK 设置/读取的是最大 torque 限制|不能作为实测关节力矩或 PPO 力反馈|
|电流|G20 接口返回不可用占位值|状态契约不提供虚假的测量电流/力矩|
|状态|可读 20 维逻辑状态、20 维温度和 20 维故障|状态必须带时间戳、序号和有效性|

上游 G20 ROS 节点的初始化会设置速度、最大 torque 并发送默认姿态。这种构造副作用
不能进入生产 adapter。新 adapter 必须采用两阶段生命周期：构造对象不连接、不运动；
显式 `open(read_only=...)` 后才能读；只有通过人工门禁和控制租约后才能写。

左右手的 CAN 口和 ID 属于部署配置。已有记录中的右手 `0x27`、左手 `0x28` 是审计
证据，不固化在设备无关接口中。当前离线部署模板记录已验证过的左/右臂 `can0/can1`
和左右 G20 `0x28/0x27`，同时把右手直连与双臂双手同时拓扑标为未验收；完成这两个
验收标记后即可进入硬件 profile，不需要重新设计接口。

### 3.2 触觉

|信号|形状/单位|冻结解释|
|---|---|---|
|法向/切向合力|DTO 为五指各 1 个可选 N 值|只有逐指标定或仿真代理可以置有效；当前实机 gateway 必须置无效|
|切向方向|DTO 为五指各 1 个可选 `0..255` 方向码|只有对应力通道经过验证时有效；进入策略前转为 `sin/cos`|
|矩阵触觉|每指 `12×6`，整手 360 taxel，raw `0..255`|只表示接触空间分布权重，不是逐 taxel 牛顿值|
|接近量|五指设备原始量，可选|首版策略不依赖，存在时作为诊断信息|

当前没有逐指 raw→N 标定曲线、换算系数和温漂补偿。真实 `G20SdkGateway` 在完成标定
前必须返回 `force_valid=false`，`normal_force_n/tangential_force_n=null`；不能用线性
`raw/255*20N` 填充。C2 仿真可以输出 N 域接触合力，但必须标记为 simulation proxy。
如果未来显示逐 taxel 估算力，只能把已经标定的单指合力按矩阵权重归一分配，并清楚
标记为估算值。

官方 ROS 节点名义定时器为 60 Hz，但它按 11 个周期轮询五根手指矩阵，因此一套完整
新矩阵约每 183 ms 才更新一次，发布的数据可能混合不同年龄的手指帧。正式
`TactileFrame` 必须记录每根手指各自的采样/接收时间、新鲜度和有效性；单一整手时间戳
不够。

滑移可以融合标定后的切向/法向比、方向变化、矩阵中心移动和总响应下降。当前只有
矩阵分布证据，因此旧 SlipDetector 只作特征工程参考，不是冻结真值。新的 PPO/Verifier
必须使用版本化特征提取器，并通过番茄数据重新标定阈值。

## 4. 仿真资产事实

主要证据：

- [双臂训练就绪报告](../../botSDK/urdf/02_project_derived/reports/BIMANUAL_SDK_TRAINING_READINESS_20260726.md)
- [Isaac USD 验证](../../botSDK/urdf/02_project_derived/reports/ISAAC_USD_VALIDATION_20260726.md)
- [正式关节组](../../botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/config/joint_groups.yaml)

冻结结构为双臂 14 个独立关节；每手 16 个独立关节加 5 个 mimic 关节。正式组合资产
为 71 links / 70 joints，Isaac 验证中有 56 个转动 DOF 和 10 个 mimic 关系。

仿真当前可用于场景、任务代码、弱接触原型和带随机化的农业接触 RL 原型，但不能直接
宣称 sim-to-real。进入真机前至少还缺：

1. G20 20 维逻辑命令到 16 个独立仿真关节的双向标定映射；
2. 手部真实质量、质心、惯量、阻尼、摩擦、迟滞和输出能力；
3. 指尖柔度、接触材料以及触觉到仿真接触信号的映射；
4. 番茄果实柔度、摩擦、尺寸/质量分布和果梗断裂力/扭矩模型；
5. 自碰撞过滤、环境碰撞、零位、手眼标定、延迟、丢包和动作限幅随机化。

## 5. 旧 Fruit-Grasp PPO 的隔离规则

旧 baseline 的代码、checkpoint、训练日志和环境已经从 `botSDK/` 清理。本节只保留
清理前的审计结论：它的任务是从支撑物上抓起并稳定保持球形果实，策略动作为
“6 维末端增量 + 21 维仿真手关节”，成功基于抬升/保持；它没有果梗、拉脱和当前定义
的“脱离且未滑落”。

以下内容禁止继承到新番茄采摘策略：

- checkpoint、curriculum 进度和 reward 结论；
- 27 维 action head 和 21 维手部直接关节动作；
- 旧 actor/critic 观测布局；
- 球体支撑、抬升/保持成功定义；
- 将仿真活动关节直接等同于 G20 20 维硬件逻辑命令的做法。

允许复用的只有通用审计经验，例如相机噪声、延迟、动作限幅、接触参数随机化和
privileged critic 的训练方式；任何复用都要在新环境中重新实现和验证。

## 6. 冻结决定

|ID|决定|
|---|---|
|D01|设备基线为双臂、每臂 7 关节；任何 6 自由度描述均作废。|
|D02|MoveC 不进入接口、工具、mock 或验收。|
|D03|中立位姿使用 m + `quaternion_wxyz`；SDK RPY 只存在于 adapter 内。|
|D04|Agent 永远看不到关节数组、轨迹、20 维 G20 向量、触觉矩阵或 SDK client。|
|D05|PPO 不直接控制 7 关节；小幅末端残差由安全监督器映射到 FollowP。|
|D06|PPO 手部动作固定为 3 个协同量；mapper 输出实机 20 维或仿真 16 个独立关节，5 个 mimic 由仿真模型产生。|
|D07|Skill 执行前必须取得唯一控制租约；执行期禁止其他臂/手命令发布者。|
|D08|IK、MoveJP、MoveL 均不能替代碰撞安全判断。|
|D09|G20 adapter 构造和只读打开不得发送姿态、速度或 torque 命令。|
|D10|G20 torque 是限制值，不是测得力矩；矩阵 raw 和名义量程都不能充当 N 标定。|
|D11|所有传感与动作结果必须带单调时间、序号、有效性和来源。|
|D12|PPO 只能提出动作；相位监督器、安全层和 Verifier 有权拒绝、裁剪或终止。|
|D13|成功唯一真值为独立 Verifier 确认果实脱离且稳定持有、未滑落。|
|D14|旧 Fruit-Grasp PPO 不进入新策略依赖图。|
|D15|adapter 的离线实现可直接推进；实际连接设备或动作测试需确认测试窗口，并执行有人监护的分级验收。|
|D16|逐指力未标定时 gateway 必须置 `force_valid=false`；若改用 raw-matrix actor，必须升接口/特征版本并重新训练。|

冻结后的完整 DTO、Port、PPO 和 Task 规则见
[INTERFACE_SPEC.md](./INTERFACE_SPEC.md)。全局实现边界与后续顺序见
[ARCHITECTURE_REVIEW.md](./ARCHITECTURE_REVIEW.md)。
