# C2-2 番茄采摘物理环境与 Sim2Real 训练准备

状态：环境主体已实现并通过 CUDA reset/step 冒烟；PPO 未启动，规则策略验收未通过  
日期：2026-08-04

## 1. 唯一路线

项目只保留一套番茄采摘 V1，不并行维护多个 observation、环境或训练方案：

```text
YOLO 实例分割 + 配准 RGB-D
        ↓ 椭球拟合：中心、姿态、x/y/z 半轴
114D actor observation（无仿真私有真值）
        ↓
7D policy action（末端局部残差 4 + G20 协同量 3）
        ↓
相位/安全 supervisor
        ↓
仿真：Differential IK + 关节目标
真机：FollowP + G20 mapper
```

原 111D 训练前草案已经原位替换为 114D；没有第二个兼容环境，也没有旧 checkpoint
需要迁移。Actor 不接 raw 点云，点云只在策略外用于拟合可部署的三维几何。

## 2. 已实现内容

- `LinkAgent-TomatoPick-Direct-v0` 已注册为 Isaac Lab `DirectRLEnv`；
- 正式双臂/双 G20 USD 只读加载，首版只激活右臂，另一臂停在 parking pose；
- 番茄为每环境相关采样的三轴椭球，质量由 `4/3*pi*a*b*c*density` 计算，不独立乱采；
- 果梗为向量化 Kelvin–Voigt 六自由度弹簧，只有稳定抓取后的 DETACH 阶段才允许一次性断裂；
- 物理频率 120 Hz、动作频率 20 Hz，actor 114D、privileged critic 33D、action 7D；
- 已实现 batch reset、四阶段、reward、termination、结构化 metrics 和 RSL-RL 配置；
- 4 个 CUDA 环境连续 16 步的创建/reset/step 冒烟通过，所有张量形状和有限性正确；
- PPO 启动器仍不生成可执行命令，且本阶段没有启动训练。

## 3. Sim2Real 冻结规则

机器可读配置是 `simulation/config/sim2real_v1.yaml`，核心规则如下：

1. 仿真真值必须先经过与真机相同的观测模型，actor 不得读取精确果实位姿、真实断裂
   阈值、精确质量或约束状态；这些只允许进入 critic。
2. 果实三轴、密度和质量相关采样；视觉测量另加中心/姿态/尺寸误差、标定偏置、延迟、
   dropout，避免策略依赖“完美坐标”。
3. 机械臂加入关节零偏、阻尼/摩擦、动作尺度和 0～2 个 policy step 的延迟；G20 加
   协同增益、死区和命令延迟。
4. 触觉加入增益/偏置、时延、逐指 dropout 和矩阵质心噪声；真机发布前必须完成逐指
   力标定，不能把 `raw/255*20N` 当真值。
5. 训练课程固定为：名义力学 → 尺寸/预抓取误差 → 接触/执行器/触觉 → 全随机化；
   固定评估集不与训练 seed 混用。
6. 仿真用微分 IK，真机用有界 FollowP；二者共享 7D 动作尺度、20 Hz、workspace/速度/
   过力 supervisor，而不是要求底层控制器完全相同。

## 4. 当前没有通过的门禁

短物理冒烟通过不等于动作可训练。当前规则策略诊断出现假过力终止，不能完成
`CONTACT → GRASP → DETACH → HOLD`。原因是正式 USD 的实例化 payload 还不能直接挂
逐指 ContactReport，当前临时使用的“基线扣除 + 低通的关节反力”会混入驱动瞬态，
不能作为最终触觉真值。

训练前只剩一条主线，按顺序完成：

1. 生成不修改正式 USD 的 G20 指尖 collider/contact-reporter overlay；
2. 用 open/mid/closed pose sweep 冻结碰撞体和明确的自碰撞过滤矩阵；
3. 用逐指 contact wrench 替换关节反力代理，跑通单环境规则策略四阶段；
4. 完成 1、16、128、512 环境扩展检查和 100-seed 规则策略物理 smoke；
5. 用实际番茄拉/弯/扭数据替换果梗宽先验，再将 PPO gate 改为 ready；
6. 最后才执行 PPO 训练。

任何一步失败都保持 `simulation/config/ppo_v1.yaml` 的
`blocked_until_physics_rule_policy_acceptance`，不得靠降低成功条件、删除过力终止或
把仿真真值塞给 actor 来绕过。

## 5. 训练完成后的 To-Real 顺序

训练得到 checkpoint 后也不直接运动真机：先校验 manifest/hash 和归一化，再做离线
录包回放、FollowP shadow、低速受限工作空间空载、软假果、带梗番茄。只有独立
Verifier 同时确认“已脱离、仍稳定持有、未掉落”才计为成功。
