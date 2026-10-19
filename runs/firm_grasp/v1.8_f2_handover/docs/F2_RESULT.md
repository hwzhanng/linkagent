# F2 结果：真实 PPO 更新后的三指指肚 G1

日期：2026-10-18。状态：**F2 完成**。达到完成边界后停止，未开始 F3/F4、3 秒保持、扰动、泛化或提举。

## 结论

- 真实 PPO 更新后的 checkpoint `update_000001.pt`（1 次完整更新，768 个真实交互 transition）和 `final.pt`（10 次完整更新，7680 个真实交互 transition）均已在新 PhysX 进程中严格重载，并实际驱动出 G1。
- 更新后 actor 的 4 回合确定性评估均为 **4/4 SUCCESS**；`update_000001.pt` 的完整 125 子步成功窗经原始 120Hz 数据重算和原完整网格 PAD 重分类，**PASS**。
- 初始化本身（零输出权重 + 声明动作偏置）已经 4/4 成功；`update_000001.pt` 也 4/4 成功。因此**不声称 PPO 带来成功率提升**，只声明：真实 PPO 更新后的权重在新进程严格重载下仍可复现三指指肚 G1。
- 训练 rollout 使用探索噪声，47 个随机回合全部失败（drop）；确定性更新后 actor 成功。该差异如实保留，见“初始化/更新对照”。

## 新 checkpoint

|用途|绝对路径|SHA256|
|---|---|---|
|**主交付：1 次真实 PPO 更新**|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018/evidence/F2_train_handover01/update_000001.pt`|`a5997ed42977b4cd66bcba021510a95445818153b6de5b0fcb219fa8c98154b7`|
|10 次更新 final|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018/evidence/F2_train_handover01/final.pt`|`cd49ac1778c11934d79a55cc64a5a873584758ec59a73d9da6b2cf8f8ea263a6`|
|初始化对照 initial|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018/evidence/F2_train_handover01/initial.pt`|`6c3c932e4e0770695acd640acc468a3e5ac99d33f73f6b02ec6a298854afdbad`|

actor/critic/action 维度：162/223/14；任务 `firm_grasp_v1.8.0`；`task_identity_sha256=7855b3f0033e7342c2ccc7079073d0dbc274e9f7102b093afd1b4e97b517838b`；`critical_source_sha256=a9c3971f96f3564d65c9e6ad436f403144d44fbf5a511cf3708544e14380a04e`。

## 源码与任务条件

- 源码 worktree：`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018/source-f2-handover-v1_8`
- 分支：`codex/ppo-firm-grasp-f2-handover-20261018`，提交 **3a9dab0**（父提交 c372803）。
- 新 run root：`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018`
- 初始化/释放条件：
  - `wrist_tilt_deg=41.68575721082819`，轴 `[0.7374624811217811, 0.6651274395898447, 0.11727991320922924]`
  - `reset_fruit_reference_hand_mm=[81.14439249, -15.65784216, 193.53402406]`
  - `reset_reference_offset_hand_mm=[0.2022, 0.8588, -0.4707]`（手系 +1 mm world-Y，仅 reset 写入）
  - `base_lift_m=0.03`（固定基座竖直抬升，仅 reset 写入，排除定向姿态下拇指碰撞网格与地面重叠）
  - `support_ramp_policy_steps=20`（释放前最后 1 s 线性退出；释放步 120，之后严格零支撑）
- 冻结内容未变：果实质量/摩擦、PD、关节安全范围、0.2 Nm 力矩上限、五指 PAD 区域与 0.15 N / 5% 判据、七类 sticky 硬故障、120 子步 G1 全部保持。基座抬升与 reference 偏移是显式初始化条件，不是 PPO 学习增益。

## 严格重载与 G1 结果

每个评估都是新的 PhysX 进程，`--policy checkpoint` 严格校验 `task_identity`、`critical_source_sha256`、162/223/14 维度和源码哈希，然后 `runner.load` 严格加载。

|Actor|4 回合结果|G1 峰值子步|完整复核窗|
|---|---|---|---|
|initial（零输出权重 + 动作偏置）|4/4 SUCCESS|124,125,125,120|125 子步 PASS|
|`update_000001.pt`（1 次 PPO 更新）|4/4 SUCCESS|122,120,125,125|125 子步 PASS|
|`final.pt`（10 次 PPO 更新）|4/4 SUCCESS|123,121,125,124|125 子步 PASS|

`update_000001.pt` 的成功窗（`F2_UPDATE01_SUCCESS_WINDOW.json`）：
- 同一组合 `thumb+index+middle` 连续 **125** 个真实 120Hz 子步（超过 120）。
- 重算最小 PAD 法向力：thumb **1.707 N**、index **0.162 N**、middle **0.200 N**（均 ≥0.15 N）。
- 重算最小承载份额：thumb **72.5%**、index **6.50%**、middle **9.53%**（均 ≥5%）。
- 最大同组合锚点漂移 **0.34 mm**（≤10 mm）；最大相对速度 **0.0118 m/s**（≤0.05）。
- 五指 NONPAD 与掌心承载均为零；UNKNOWN=0；硬故障=0；无效通道=0。
- 每个子步实际支撑力/力矩严格为零；果实底部净空最小 **92.97 mm**（>5 mm）。
- 原完整网格逐承载点重分类：thumb 125 点、index 125 点、middle 125 点，全部 PAD；最近三角面均在批准 PAD 集合内；最小边界距离 4.41/1.04/0.83 mm（均 >0.5 mm 边界不确定带）；最大源距离 0.194 mm（<0.5 mm 容差）；最大法向误差 9.65°（<25°/45° 容差）。
- 原始窗口：`F2_UPDATE01_SUCCESS_WINDOW_window.jsonl`（SHA256 见 receipt）。

## 初始化 / 更新后对照（同条件 4 回合）

条件完全相同：seed 110、4 env、20 步支撑渐退、41.686° 定向 reset、+30 mm 基座抬升、+1 mm 手系 reference 偏移。

|Actor|成功|失败|UNKNOWN|G1 峰值|复核|
|---|---:|---:|---:|---|---|
|initial prior（未训练）|4|0|0|124–125|PASS|
|update 1|4|0|0|120–125|PASS|
|update 10|4|0|0|121–125|PASS|

训练期间使用探索噪声（其余通道 σ=0.02，动作 8/9 σ=0.20）：47 个训练回合 0 成功，全部 drop。这说明成功对动作噪声敏感；确定性更新后 actor 仍成功。**如实归因**：本次 G1 主要来自声明的初始化条件与动作偏置先验；真实 PPO 更新后的权重保留了该行为，但没有证明 PPO 提高了成功率（初始与更新后均为 4/4）。

## 最小复现

```bash
ROOT="/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f2-continuation-20261018"
PY="/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python"
cd "$ROOT/source-f2-handover-v1_8"
BIAS='[0.43116165292000025,0.193433788076279,-0.04478002490162736,0.0025961911302571244,0.28408886015079765,0.05576175655382348,0.0,0.3,0.0,0.0,-0.18191298991496457,-0.1,-0.18191298991496457,-0.18191298991496457]'

# 训练（真实 PPO，10 次更新；本次已执行）
PYTHONPATH=. $PY -m research.firm_grasp_v1.launch --name F2_train_handover01 --module run --num-envs 4 -- \
  --mode train --policy fresh --iterations 10 --seed 110 --support-ramp-policy-steps 20 \
  --wrist-tilt-deg 41.68575721082819 \
  --wrist-tilt-axis-world-xyz 0.7374624811217811 0.6651274395898447 0.11727991320922924 \
  --reset-fruit-reference-hand-mm 81.14439249 -15.65784216 193.53402406 \
  --base-lift-m 0.03 --reset-reference-offset-hand-mm 0.2022 0.8588 -0.4707 \
  --initial-action-bias-json "$BIAS"

# 新进程严格重载更新后 checkpoint，4 回合确定性评估
PYTHONPATH=. $PY -m research.firm_grasp_v1.launch --name F2_eval_update01 --module run --num-envs 4 -- \
  --mode evaluate --policy checkpoint \
  --checkpoint "$ROOT/evidence/F2_train_handover01/update_000001.pt" --episodes 4 --seed 110 \
  --support-ramp-policy-steps 20 --wrist-tilt-deg 41.68575721082819 \
  --wrist-tilt-axis-world-xyz 0.7374624811217811 0.6651274395898447 0.11727991320922924 \
  --reset-fruit-reference-hand-mm 81.14439249 -15.65784216 193.53402406 \
  --base-lift-m 0.03 --reset-reference-offset-hand-mm 0.2022 0.8588 -0.4707

# 原始120Hz窗口重算 + 原完整网格复核
$PY "$ROOT/evidence/verify_success_window.py" \
  --eval-dir "$ROOT/evidence/F2_eval_update01" \
  --source-root "$ROOT/source-f2-handover-v1_8" \
  --original-classifier "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/verification/original_pad_geometry.py" \
  --output "$ROOT/evidence/F2_UPDATE01_SUCCESS_WINDOW.json"
```

## 证据索引

- 训练：`evidence/F2_train_handover01/`（summary、updates、episodes、initial/update_*/final）
- 评估：`evidence/F2_eval_initial_ckpt01/`、`evidence/F2_eval_update01/`、`evidence/F2_eval_update10/`
- 成功窗复核：`evidence/F2_INITIAL_CKPT_SUCCESS_WINDOW.json`、`evidence/F2_UPDATE01_SUCCESS_WINDOW.json`、`evidence/F2_UPDATE10_SUCCESS_WINDOW.json` 及对应 `_window.jsonl`
- 对照汇总：`evidence/F2_COMPARISON.json`
- 复核器：`evidence/verify_success_window.py`

## 旧版本保护

- 旧权重 SHA256：`0650ac4b35e742430f910319492d8ada621ee1c25f954e8aba6bd8163e59ddf6`（本次复核一致）。
- 旧留存 ZIP SHA256：`cfd88bb1b2a59e2578a2d9a508fcfe153967fd4d92baab636b6d2b4be0df423b`（一致）。
- 旧运行根 `/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-v1-20261017` 与旧 S8 worktree 未修改（S8 worktree 仍为 c372803、状态干净）；新实验全部在新 run root。

## 剩余限制

- G1 仅在同一声明初始化条件、seed 110、4 个评估回合、20 步支撑渐退下验证；未做统计显著性、泛化、3 秒保持或扰动测试（F3/F4 未授权）。
- 成功对动作噪声和 reference 偏移敏感：训练探索噪声下 47/47 失败；+0.5 mm 或更大的 reference 偏移会使 middle 接触丢失。因此“鲁棒抓牢”尚未证明。
- 本次成功主要归因于初始化先验；PPO 更新后的权重保留了成功，但没有展示成功率提升。不得把该结果表述为 PPO 学会了抓牢。
- 成功窗的独立复核使用运行时记录的真实接触点 + 原完整网格分类器；未重新物理仿真该窗口。
