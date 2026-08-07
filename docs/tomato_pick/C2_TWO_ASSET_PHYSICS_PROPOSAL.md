# C2-2 番茄、果梗、指尖碰撞与自碰撞决策记录

状态：番茄/果梗方案已实现并完成短 CUDA smoke；指尖/自碰撞验收未完成，PPO 未开放  
日期：2026-08-04

## 1. 先给结论

当前唯一实现路线：

```text
刚性番茄 + 软接触材质
        ↓
固定 attachment frame — 向量化 Kelvin–Voigt 6D 弹簧/阻尼 — 番茄
        ↓
G20 五个定制圆滑指尖 collider
        ↓
指尖 contact-reporter overlay + 明确的自碰撞 pair/group 表（训练前待完成）
```

具体选择是：

1. 番茄第一版不用高成本 deformable body，使用低面数刚体碰撞和 compliant contact；
2. 果梗断裂点放在离层 `F_abscission`；并行环境中采用 tensor 化 6D 弹簧/阻尼和一次性
   断裂，避免每次 reset 重建 USD joint；
3. `F_abscission` 和抓取坐标系统一规定 `z+` 为“植株 → 果实/远离植株”；
4. 五指末端换成专门的圆滑凸碰撞体，原末端碰撞体禁用，visual 不变；
5. 当前物理 smoke 暂时关闭全机器人自碰撞；训练前通过姿态 sweep 生成明确过滤矩阵后
   再启用，不能直接把 71-body 所有 pair 全打开。

实现证据位于 `simulation/tomato_pick/isaac/`。早期讨论过的 breakable D6 仍保留在本文
作为设计依据，但没有形成第二套运行环境；当前代码只使用 Kelvin–Voigt 路线。

## 2. 已知事实与需要测量的参数

### 2.1 文献能支持的事实

- 番茄果梗由近端、离层和连接花萼的远端组成，离层直径在一项实验中主要为
  `5–8 mm`。对应三组纯拉伸平均断裂力约为 `40.262 N`、`44.781 N`、`72.003 N`，
  强度约为 `1.44–1.69 MPa`；论文同时强调品种、成熟度和缺陷会造成差异：
  [Tomato Pedicel Physical Characterization](https://www.mdpi.com/2073-4395/14/10/2274)。
- 国内番茄力学试验发现压缩抗性有方向性且随成熟度改变，果梗在离层断裂，弯折比
  纯拉伸更适合脱果：
  [面向机器人采摘的番茄力学特性试验](https://www.aeeisp.com/nygcxb/article/id/20081214)。
- 一组番茄/橡胶静摩擦试验均值为 `0.305`，范围约 `0.284–0.344`。这只能作为材料
  初值，不能替代本项目硅胶指尖实测：
  [Evaluation of engineering properties for waste control of tomato](https://pmc.ncbi.nlm.nih.gov/articles/PMC6475739/)。
- 一项人工采摘研究给出的组织损伤临界压缩参考约为 `42–45 N`、`138–146 kPa`；它只
  适用于论文中的品种、成熟度和加载方式：
  [Tomato Fruit Quality as Affected by Ergonomic Conditions](https://www.mdpi.com/2077-0472/13/9/1831)。

### 2.2 这些值现在不能假装知道

- 我们实际番茄的半径、质量、成熟度、压缩刚度和损伤阈值；
- G20 指尖硅胶/番茄的静摩擦、动摩擦、接触刚度和阻尼；
- 实际采摘品种在“拉 + 弯 + 扭”组合载荷下的断裂力/扭矩与疲劳规律；
- 指尖 STL 与真实触觉片表面的几何偏差；
- 右手和双手同时拓扑的真机触觉/控制证据。

因此下面的数值分为“文献种子”“训练搜索范围”“待实测量”，不会写成真机标定值。

## 3. 番茄资产方案

### 3.1 几何层级

建议每个环境的番茄至少有三个逻辑 prim：

```text
/World/envs/env_.*/Tomato
├── Visual              # 只渲染，可做颜色/纹理/形状随机化
├── Collision           # 物理碰撞，不复用高面数 visual
└── Frames/fruit_com    # 质量中心/观测参考
```

- visual：低面数球状/超椭球或轻微分瓣模型；视觉训练以后可以引用更丰富网格；
- collision：优先一个圆滑 convex hull；单凸包不能覆盖形状时最多拆成 `2–4` 个凸体；
- 首轮并行 RL 不使用 dynamic triangle mesh 或 SDF，也不让 visual 网格直接参与碰撞。

Isaac Sim 官方说明，dynamic rigid body 上 triangle mesh 会退化为 convex hull；SDF 虽可
用于动态三角网格，但成本更高。contact offset 越大约束越多、计算越贵，太小又可能
漏接触；rigid material 的 compliant contact 可以近似软体：
[Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html)。

### 3.2 材质与损伤代理

- 文献种子：tomato/rubber 静摩擦名义值先取 `0.30`；第一轮可在 `0.20–0.60` 做均匀
  随机化，项目旧宽先验 `0.20–1.20` 只保留作搜索边界，不直接当均匀分布；
- 果实仍是 rigid body，但 tomato 或 fingertip material 使用 compliant contact；刚度、
  阻尼通过单指压缩曲线拟合；
- actor 只接收可部署的触觉特征；仿真内部记录五指法向力、切向力、接触点、相对滑移、
  峰值总夹持力和接触冲量供 reward/critic/termination；
- 参考 `42–45 N` 可先设“软惩罚开始约 30 N、候选 hard gate 约 40–45 N”，但在本项目
  番茄压缩测试前不能冻结，更不能直接解释成单指阈值。

## 4. 果梗和 attachment frame

### 4.1 物理拓扑

最初候选拓扑是 breakable D6；为适配大批量 reset，最终实现等价的 tensor 约束：

```text
Plant/StemAnchor（固定 frame）
        │ 6D Kelvin–Voigt force/torque，超过 force 或 torque 阈值后一次性关闭
Tomato（断后自由刚体）
```

- 近端/远端几何先用 capsule；文献中的近端/远端直径 `3–5 mm`、离层直径 `5–8 mm`
  可作为初始几何随机化种子；长度仍需样本测量；
- 线位移和转角分别计算弹簧/阻尼 force/torque，使断裂前可表达拉、弯、扭；
- 约束完全位于任务 tensor 中，不进入机器人 articulation，也不修改正式 USD；
- 每个 episode 只有 `INTACT -> BROKEN` 一次状态转移，断后不能重新连接；只有 reset
  才重建连接；
- success 仍由独立 Verifier 判断 `broken + retained_without_slip + not_dropped`，不能只
  看 joint broken。

OpenUSD joint 原生定义 `body0/body1`、两侧 local frame、`breakForce`、`breakTorque` 和
`excludeFromArticulation`；PhysX D6 可分别限制 twist/swing，关节超过 force/torque
阈值会断裂：
[UsdPhysicsJoint](https://openusd.org/release/api/class_usd_physics_joint.html)、
[PhysX Joints](https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/Joints.html)。

### 4.2 frame 约定

统一使用右手坐标系：

|frame|原点|`z+`|`x+` / `y+`|
|---|---|---|---|
|`F_stem_anchor`|果梗与植株/主梗连接中心|沿果梗中心线，植株指向果实|`x+` 取世界上方向在 `z` 法平面的投影，`y=z×x`|
|`F_abscission`|离层几何中心|近端指向远端，即植株指向果实|与 `F_stem_anchor` 连续，避免随机翻转|
|`F_fruit_com`|番茄质量中心|与 `F_abscission.z+` 同向|用于质量、位姿和观测参考|
|`F_grasp`|规划器给出的抓取中心|冻结为远离果梗/植株|继续满足接口 `pull_axis=(0,0,1)`|

如果果梗几乎与世界上方向平行，`x+` 的投影会退化，此时使用环境 `x+` 投影作为确定性
fallback。所有 quaternion 继续使用现有 `wxyz`，单位米/弧度。

### 4.3 断裂参数如何设

当前冻结 baseline 中旧宽先验 `[1,20] N` 与文献中的纯轴向 `40–72 N` 平均值不一致，
不能无说明继续作为真实拉脱阈值。建议：

1. 纯拉伸 sanity test 用 `F = σ π d² / 4`，其中 `σ` 从约 `1.44–1.69 MPa`、`d` 从
   `5–8 mm` 采样，并用论文实测分布校验；
2. PPO 的最终动作是弯折/扭转/拉伸组合，不能只用轴向阈值；当前模型同时设 break
   force 和 break torque，必要时再加累计损伤变量；
3. 早期 curriculum 可以降低断裂门槛帮助探索，但评估环境必须回到实测分布；
4. break torque、弹簧和阻尼在采 20–30 个本项目样本做弯折/扭转试验前不冻结。

## 5. G20 指尖 collider

### 5.1 精确目标 link

每手 5 个触觉/指尖 link，共 10 个：

```text
{side}_hand_thumb_link5
{side}_hand_index_link4
{side}_hand_middle_link4
{side}_hand_ring_link4
{side}_hand_little_link4
```

`side = left | right`。

### 5.2 几何和接触设置

- 每个指尖制作一个圆角盒/椭球/短 capsule 形式的主 convex hull，必要时加一个支撑
  hull；单指不超过两个凸体；
- collider 贴合实际触觉接触面，可先比 visual 表面凸出 `0.3–0.8 mm`，再用实物尺寸
  和闭合测试修正；
- 禁用该末端 link 原碰撞 mesh，避免新旧 collider 双重接触；visual STL 保留；
- 初始候选 `contact_offset=0.5–1.5 mm`、`rest_offset=0–0.2 mm`；根据穿透、抖动和
  计算成本收敛；只有出现高速穿透才开启 CCD；
- contact sensor 第一版只挂在 active hand 的 5 个末端 link，输出每指法向/切向合力、
  接触中心和 slip proxy；另一只手进入训练前用相同生成流程镜像，但不混用 link 名。

其余指节用 capsule 或 `1–2` 个低面数 convex hull，手掌和机械臂用简化凸包。正式
asset 当前有 71 个 rigid bodies、49 个复用/实例化 collider 定义使用自动
`convexDecomposition`；任务 overlay 不修改源 USD，而是引用后覆写碰撞层。

## 6. 自碰撞过滤

### 6.1 原则

源资产目前 `newton:selfCollisionEnabled = 0`。任务 overlay 中先显式启用 articulation
self collision，再用 joint 默认过滤、`UsdPhysicsCollisionGroup` 和少量 pair filter
关闭不需要的组合。CollisionGroup 默认只能进一步禁用碰撞；`invertFilteredGroups`
可以表达“除列出的组外全部禁用”：
[UsdPhysicsCollisionGroup](https://openusd.org/release/api/class_usd_physics_collision_group.html)。

### 6.2 候选碰撞矩阵

|组合|候选策略|原因|
|---|---|---|
|同一手指相邻 link|过滤|关节相邻，几何常在边界接触|
|同一手指非相邻 link|启用；若 nominal 永久重叠再逐 pair 过滤|阻止手指反折穿透|
|不同手指 tip/distal|启用|阻止抓取时互穿，尤其 thumb 对四指|
|不同手指 proximal-proximal|默认启用，pose sweep 后只过滤永久重叠 pair|不提前牺牲真实性|
|tip 与 palm/hand base|启用|阻止手指卷入掌内|
|手指与本侧 wrist/arm|启用非相邻 pair|阻止极端动作穿过腕臂|
|同侧 arm 相邻 link|过滤|机械关节相邻|
|同侧 arm 非相邻、arm 与 torso|启用|补足 SDK 没有场景碰撞规划的风险|
|左臂与右臂/双手|启用|双臂协调时防止互穿|
|active hand 与番茄/果梗|启用|任务接触|
|inactive hand 与任务物体|首版可过滤并停在 parking pose|降低单手 PPO 成本；双手任务再打开|

不是按链节名称猜过滤表，而是由生成脚本产出稳定 pair matrix 和 hash，随资产 manifest
保存。首版 active arm 仍由 plan 决定，环境接口保留 `auto|left|right`；左右侧用同一规则
生成但分别验证。

## 7. C2-2 资产验收

冻结方案和进入 `DirectRLEnv` 前至少完成：

1. open / mid / closed 三组无果手势：没有非预期接触或初始穿透；
2. 多半径球/番茄闭合：接触主要发生在 5 个 tip，最大穿透在阈值内；
3. 极限跨指动作：碰撞阻止互穿，solver 无爆炸、NaN 或异常高速；
4. 果梗轴向、弯折、扭转三个单元测试：断裂只发生一次，阈值与 frame 方向正确；
5. 断梗后抓稳、滑移、掉落三种情况可被独立区分；
6. 1、16、128、512 环境逐级测 FPS/显存，不一开始就用 512 掩盖单环境问题；
7. 100 个 seed reset/规则策略 smoke：无 NaN、无持续初始接触、成功不变量正确；
8. 资产版本、frame、材料、碰撞 pair matrix 和随机化范围全部写入 manifest。

## 8. 决策结果与剩余验收

|决策|当前结果|剩余动作|
|---|---|---|
|番茄首版物理|相关采样三轴刚体椭球 + compliant-contact proxy|以后只增强 visual，不改 actor 接口|
|离层模型|tensor Kelvin–Voigt 6D + 一次性 force/torque 断裂|用本项目样本替换宽先验|
|frame 方向|`z+ = plant -> fruit`|已冻结|
|指尖 collider|源 collider 暂用；overlay 未完成|按 10 个精确 link 生成并验收|
|自碰撞|训练环境暂时关闭|生成 pair matrix，跑姿态 sweep 后启用|

`assets_v1.yaml` 现在允许 Isaac physics reset/step，但 `ready_for_ppo_training=false`。
下一步只完成指尖/contact reporter overlay、自碰撞姿态 sweep 和规则策略物理验收；
详细门禁见 [C2_SIM2REAL_TRAINING_PREP.md](./C2_SIM2REAL_TRAINING_PREP.md)。
