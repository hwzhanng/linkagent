# F3/F4 续作执行计划（2026-10-19）

## 已确认事实

|项目|结果|
|---|---|
|新根|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-continuation-20261019`|
|新源码|`NEW/source`，分支 `codex/firm-grasp-f3-f4-continuation-20261019`，基线 commit `f60b912`|
|继承身份|critical source SHA256 `7460ecde...8167` 在新副本逐文件复算一致|
|D01 严格重载|v2.2 update_000001 在 ramp20 下 quiet 1/4，identity `2620b17d...`，与旧训练身份一致|
|D02 ramp40|同一 actor，quiet 3/4；3 个成功窗 `verify_g2` 数值 PASS，原始完整网格 PAD 重分类 PASS|
|D03 ramp40 force|2/4；D04 ramp40 torque|2/4|
|支撑语义|ramp40 仍在策略步 120 起严格为零，正式 G2 支撑为零不变|

## 当前瓶颈

1. 第一次 episode（env1/2/3）在 ramp40 下可建抓；同一 env 的第二次 reset 后多例在 267–316 子步处失稳，尚未到 360。
2. env0 在三种模式下均失败；其首 episode 就未建立 G2 窗。
3. 初始 trace 中第一次 episode 与后续 episode 的果速差异已确认是记录相位差（相差一个物理子步），不是 reset 写入的初始状态不同；不据此改 reset 代码。
4. 继承 actor 的 mean MLP 实质仍接近常量先验（输出层权重接近 0）；只做 1 update、LR 2e-6。需要真实多 update，同时防止破坏已有建抓。

## 实验设计

### C1：全 actor 低学习率续训（先验证学习是否有效）

- 条件：ramp40、goal g2、3 s、quiet 训练；actor-only seed 自 v2.2 update_000001（含 normalizer）。
- 超参：LR 2e-6（继承量级）、std 5e-4、每 update 4 env × 256 步、40 updates 一段。
- 目的：验证在 ramp40 准备条件下，真实 PPO 多 update 是否能在不破坏 3/4 quiet 的前提下提高保持余量。
- 开发检查：update 10/20/40 各用 4 quiet + 4 force(axis0+) + 4 torque(axis0+) 同条件评估；支持实际为零、数值与网格核验成功窗。
- 判据：quiet 不低于 3/4，force/torque 至少保持 2/4，且至少一个失败 env 转为成功；否则视为无进展。

### C2：受约束 PPO（行为信任域）

- 在 C1 基础上启用 `--freeze-actor-normalizer` 和 `--actor-mean-shift-max`。
- 每次 PPO update 后用同一 rollout 观测批比较当前 actor 与冻结 prior actor 的 mean 输出；若平均绝对偏移超过阈值，则把所有 actor 参数向 prior 线性插值，使偏移回到阈值。记录每 update 的 shift/alpha。
- 候选阈值：0.002（严格）与 0.005（较宽），先跑严格版；如果学习被压死再放宽。
- 目的：保留 v2.2 已建立的建抓行为，同时允许输出层/隐层在信任域内改进。

### 备选（有证据才做）

- F2 actor 作为独立起点（其输出层有非零观测依赖，可能比 v2.2 常量先验更适合学习）。
- ramp60（更慢退出）作为准备条件对照。
- 小范围 reset 随机化（果实 reference ±0.5 mm）训练鲁棒性；正式评估仍用固定 reference。
- 早期建抓奖励：对 released 后 ≤60 子步内的合格三指窗给有界 bonus；必须与正式 G2 分开记录。

## 正式验收（不因训练改变）

- 同一拇指+两个非拇指组合，连续 360 个真实 120 Hz 子步；各 PAD ≥0.15 N 且 ≥5% 总承载；NONPAD/掌为零；七类硬故障未触发；支撑实际为零；果底离地 >0.005 m；相对速度 ≤0.05 m/s；锚点漂移 ≤0.01 m；撤支撑后 ≤60 子步起窗；非拇指间距 ≥0.006 m；对向 dot ≤−0.3；相对角 ≤10°、角速度 ≤0.5 rad/s；正式力 0.10 mg、力矩 0.01 mgR，180–191 子步共 12 真实步；恢复段 252 子步起速度 ≤0.02 m/s。
- 新 F4：预登记 20 项（8 quiet + 6 force ±XYZ + 6 torque ±XYZ），新的交错 case 顺序，同一顺序用于所有 actor；另做独立重载小批（每类 4 回合）。
- 每个声称成功窗：`verify_g2` 数值复算 + `verify_g2_mesh_selected` 原始完整网格 PAD 重分类。
- 两视角录像；开掌 4 回合负对照；旧版权重哈希复验。

## 轮次与止损

- 每个“有效方案”= 冻结假设/条件 + 真实 PPO 训练 + 同条件开发对照；C1/C2 各算一轮。
- 默认最多 8 轮；连续 4 轮无客观物理进展提前暂停；基础设施问题最多 3 个有区别修复周期。
- 不以 reward、单次峰值、参数变化或准备期接触代替物理 G2 进展。
- 达到理想边界后封存停止；未达到则交付部分完成报告，不降低判据。
