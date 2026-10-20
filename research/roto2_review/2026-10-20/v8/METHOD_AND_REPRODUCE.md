# v8 协议、奖励及复现细节

[返回本轮报告](README.md) · [历史安装/复现说明](../REPRODUCE.md)

## 物理与数据协议

沿用PhysX 120Hz、控制20Hz，L20掌心朝上、固定腕部、16主动关节、64mm名义球形果实。校准使用plain、nominal；N−1验证密度固定950kg/m³，候选搜索仍保留850–1050kg/m³的逐环境密度采样，后期NPZ保留实际质量。没有改摩擦、资产形状或加入支撑力。环境和依赖版本见[历史环境清单](../environment.json)；本轮GPU RTX5090，32,607MiB显存。

每项候选创建6个匹配条件：保持、松拇指、松食指、松中指、松无名指、松小指。不在失败后自动复位。每个条件150控制步=7.5秒：

| 步索引 | 时间 | 操作 |
|---|---|---|
| 0–39 | 2秒 | 保持基线，末20步统计有效pad |
| 40–79 | 2秒 | 单指逐步卸载 |
| 80–89 | 0.5秒 | 必须整个手指真实无接触 |
| 90–149 | 3秒 | 向原控制目标恢复，继续检查物体稳定 |

初始门槛使用六个条件的共同有效pad集合：至少4指各在基线末20步≥80%有效，基线稳定。比“每个条件各自存在任意4指”更严格；原始`baseline_pad_fraction`矩阵保留，可供老师评估该门槛是否合理。

最新卸载观测覆盖26个手指刚体，每个物理子步更新。对每个连杆取6个子步接触力模长最大值，再对该指连杆求和，连续10控制步必须<0.05N。这是偏保守的整指法向接触指标，不包含完整摩擦力向量；不能称为真实触觉硬件量测。

稳定定义：距初始中心<25mm、高度下降<15mm、线速<0.12m/s、tilt<0.5；检查原物理终止、18kPa压力代理和单连杆8N力限。通过要求每个释放条件在释放及恢复全程稳定安全。v2仅把恢复末段4pad作为诊断；早期协议将其作为硬门槛。逐文件`gate_version`和协议是判断依据。

adaptive控制逐步降低指定屈曲关节的持续目标，确认卸载后保持实测关节位置；重新接触会继续调整。最初版本误用“每步实测q减固定小量”，没有累计足够目标偏移，已修正。7种路径为flexion、distal、proximal、spread_positive、spread_negative、thumb_roll_positive、thumb_roll_negative；准确标签以输入JSON及当前源码为准。输入可以覆盖每次增量，最新462项实际为0.01rad，协议中的`adaptive_increment_rad=0.015`是默认值，不是每行覆盖后的值。

每个NPZ字段的实际形状、dtype见[npz_schema.json](npz_schema.json)。N−1数组保存物体姿态、速度、位移、倾斜、稳定/失败、pad、整指力、压力、关节位置/目标等；搜索数组保存提案、分数及后期的实际沉降状态。读取时不得把所有版本强行当成同一schema；没有保存逐连杆全时序或每一个物理子步的原始向量，只保存对应聚合指标。

## baseline奖励：已实现，未训练

令ω=sign(goal)×真实材料轴姿态步增量/dt。正向项为min(max(ω,0),2|goal|)，仅在首秒后、路径有效且支持条件满足时给信用。反向项min(ω,0)保留未截断成本，以避免快速反向抵消被截断而反复刷正向奖励。

支持条件：(任意3pad力>0.15N **或** 位移/速度/高度稳定)且tilt<0.5。没有必需拇指条件；“稳定”不等于已经完成转动。

| 项 | 当前每控制步形式 |
|---|---|
| 旋转 | 上述有符号角速度项，系数1 |
| 归一化抓姿 | −0.01×mean(((q−q_ref)/joint_range)²) |
| 做功代理 | −0.002×Σabs(关节施加力矩×关节速度) |
| pad小额信用 | +0.005×有效pad比例 |
| 平移速度 | −0.05×speed |
| 偏心 | −0.02×min((displacement/0.05)²,9) |
| 倾斜 | −0.05×tilt² |
| 动作变化 | −0.001×mean((action−previous_action)²) |
| 时间 | −0.005 |
| 无效旋转路径 | 首秒后−0.20 |
| 物理失败 | 环境原有终止惩罚−60保留 |

baseline不额外罚区域外接触。压力/物理安全仍保留，plain仅关闭动作压力修正。奖励项数值、输入归一化与旧任务不完全相同，不能直接比较不同奖励版本的回报大小。4组CPU测试通过证明指定数值例子满足预期，不证明RL可收敛。已有8环境真实PhysX冒烟记录，也不等于阶段0通过。

## 下载和完整性校验

仓库为私有；具有权限并已登录`gh`后：

```bash
git clone --branch research/roto2-object-pressure-track https://github.com/hwzhanng/linkagent.git
cd linkagent
mkdir -p /tmp/roto2-v8-download
gh release download roto2-review-20261020-v8 \
  --repo hwzhanng/linkagent --dir /tmp/roto2-v8-download
cd /tmp/roto2-v8-download
sha256sum -c SHA256SUMS
```

压缩包以项目相对路径存放全部`artifacts/simulation/rotation_v8/`和`handoff_source/`。在干净checkout根目录解压以恢复原命令使用的路径；有本地同名数据时先核对manifest，不要覆盖不同实验。历史权重与资产需另外下载v7 Release，恢复安装步骤见旧包说明。不会因为下载了数据包就自动具备Isaac运行环境。

## 本轮关键命令

以下是在已安装项目运行时、仓库根目录使用的命令。输入原件在本补充的`evidence/`或完整归档，输出用`reproduced/`新目录，保护原证据。版本及硬件变化可影响接触结果，不保证逐比特重复。

```bash
mkdir -p artifacts/simulation/rotation_v8/reproduced
bash simulation/.runtime/run_roto2.sh evaluate \
  --num-envs 2772 --seed 2093 --pressure-variant plain \
  --domain-profile nominal --hand-posture palm_up \
  --reset-grasps artifacts/simulation/rotation_v8/stage0/settled_profile_sweep.json \
  --validate-redundant-reset --calibration-release-mode adaptive \
  --probe-output artifacts/simulation/rotation_v8/reproduced/settled_profile_sweep_n1.json

bash simulation/.runtime/run_roto2.sh evaluate \
  --num-envs 1024 --seed 2094 --pressure-variant plain \
  --domain-profile nominal --hand-posture palm_up \
  --reset-grasps artifacts/simulation/rotation_v8/stage0/palm_settled_2090_candidates.json \
  --search-redundant-reset --calibration-rounds 1 --calibration-horizon 40 \
  --calibration-spread 0.025 --calibration-region palm --calibration-replicas 4 \
  --probe-output artifacts/simulation/rotation_v8/reproduced/replica_smoke_2094.json

python3 script/summarize_roto2_n1.py \
  artifacts/simulation/rotation_v8/reproduced/settled_profile_sweep_n1.json
```

当前源码重放早期协议未必复现旧结果：老记录包含控制器修正前、不同采样率及更严格通过规则。需要精确审查历史时应使用各自保存的源码快照，缺失版本已在报告中明确，不能事后补造。

未来训练配置仅为计划，**未执行，不应绕过阶段0验收**：

```text
pressure_variant=plain, hand_posture=palm_up, reward_stage=baseline
num_envs=8192, rollout_steps=16, initial_action_std=0.6
goal_direction=positive, million_steps=300, fresh weights
reset_grasps=<尚不存在的独立验收合格库>
```

后续每次运行必须保存实际run_config、环境/质量、种子、输入库哈希、代码快照、TensorBoard、完成标记与逐回合评估。不能把上述计划当作已执行训练预算。
