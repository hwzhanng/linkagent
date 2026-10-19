# F3/F4 有限续接探索台账

日期：2026-10-18  
范围：仅记录有限探索安排与汇总方法；本文件创建时没有启动训练或复测。

## 当前证据与解释边界

`CONTINUATION_STATUS.json` 当前为 `F3_INCOMPLETE_F4_NOT_STARTED`。保留的初始常量先验 quiet 评估为 1/4 SUCCESS、3/4 FAILURE、0 UNKNOWN；随后一次 PPO 更新完成 1 update（1024 transitions），对应更新 actor 的 quiet 复测为 0/4 SUCCESS、4/4 FAILURE、0 UNKNOWN。现有训练摘要标记为 `WEIGHTS_UPDATED_NO_GRASP_SUCCESS_CLAIM`，评估摘要也标记 `NO_NEW_WEIGHT_SUCCESS_CLAIM`。因此这组证据**不能支持 PPO 改善声明**；奖励、权重发生变化或进程正常结束均不替代 strict G2 成功。

拟用先验为 `inputs/power4_index025_chosen.json`，SHA-256：`074d405f15136b716fc21519f95cdd696b61a56a5459f32dd4796025a6f4db5c`。其身份属于初始化条件；任何仅由该常量先验带来的表现不得归因为 PPO 学习。

## 本轮上限与顺序

总计最多 **3 轮训练，每轮最多 1 个 PPO update，并对该轮候选复测**。达到成功冻结条件可提前结束；不得追加第 4 轮或把诊断筛查计作正式复测。

| 轮次 | 训练与复测安排 | 继续条件 |
|---|---|---|
| R1 | 从上述常量初始先验开始；探索标准差 `std=0.00005`、学习率 `LR=2e-7`，仅训练 1 update。随后先做完整分母 strict G2 quiet 复测。 | quiet 未通过时，记录完整失败与 checkpoint 身份后，才可评估是否进入 R2。quiet 通过才进入该候选的 force、torque strict G2 复测。 |
| R2 | 仅在 R1 未形成可用候选且预算未用尽时进行；最多 1 update，再按相同顺序复测。启动前记录起始 checkpoint 的路径与 SHA-256，以及本轮 std、LR 和随机种子。 | 若 R2 仍未满足冻结条件，才可进入 R3。不得通过降低门槛或删减分母来取得通过。 |
| R3 | 仅在 R2 仍未形成可用候选时进行；最多 1 update，再按相同顺序复测。启动前同样登记起始 checkpoint 身份及本轮参数。 | 到此轮后按下方暂停条件结案，不再续跑。 |

每个候选依次接受 quiet、force、torque 复测；各场景必须沿用冻结协议的完整请求分母，并保留每条 SUCCESS、FAILURE、UNKNOWN、原因、checkpoint 和原始证据索引。只有同一 checkpoint 可重载，且 quiet、force、torque 均通过 strict G2，才可冻结为 F4 候选。冻结后若开展 F4，沿用既定 20 case 矩阵（quiet/force/torque = 8/6/6）。候选冻结资格与“PPO 已改善”的结论分开报告；改善声明必须有同条件、同口径证据支持。

## strict G2 不放宽

沿用 `F3_F4_EXECUTION.md` 和既有 source/task identity：actor/critic/动作维度 162/223/14、原初态及原 PAD/full-mesh 标签、几何与旋转门槛、物理参数和七项硬故障守卫均不变。正式 G2 继续要求释放后 0.5 秒内进入验收窗口，并满足连续 360 个物理子步及冻结的几何/旋转条件。force、torque 按原协议验证（训练摘要所载 ratio 为 force 0.1、torque 0.01）；小力和小力矩各 12 个物理子步，支撑扳手与测试扳手分别记账。UNKNOWN、缺失原始记录或身份不一致均不能记为通过；不得改阈值、质量、摩擦、PD、驱动力矩限制、ROI 或分母。

## 暂停条件与交付

最多三轮后，若仍无法得到可重载且 quiet/force/torque 均通过 strict G2 的稳定可用更新候选，或扰动验证至 R3 仍无进展，按用户条件**暂停并交付报告**。报告保留全分母、失败与 UNKNOWN、每轮训练参数和起始/输出 checkpoint 哈希、同条件比较、strict G2 原始证据位置及尚未通过项；不得用汇总分数覆盖失败项。出现 strict G2 证据不完整、身份漂移或硬故障时，当前候选立即判为未通过并停止其后续验收，不通过放宽条件续跑。

## 可复用汇总命令

以下命令仅供后续汇总时复制使用；本次没有执行它们。`summarize_evals.py` 读取 episode、summary 和 run metadata，检查实际分母、UNKNOWN、失败原因及 checkpoint 身份，并要求 `--output-dir` 指向尚不存在的新目录。它不复算 strict G2 原始物理接触证据，不能单独作为 G2 PASS。

汇总已有两组 quiet 结果：

```bash
N='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018'
python "$N/summarize_evals.py" \
  --eval initial_quiet="$N/evidence/F3_power4_initial_eval01" \
  --eval update1_quiet="$N/evidence/F3_power4_updated_quiet_eval01" \
  --expect-quiet initial_quiet=4 \
  --expect-quiet update1_quiet=4 \
  --output-dir "$N/reports/F3_quiet_baseline_summary_20261018"
```

后续单轮 quiet/force/torque 一并汇总的模板（替换评估目录和新输出目录；quiet 分母按冻结协议填写）：

```bash
N='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018'
python "$N/summarize_evals.py" \
  --eval r01_quiet="$N/evidence/<r01_quiet_eval>" \
  --expect-quiet r01_quiet=4 \
  --eval r01_force="$N/evidence/<r01_force_eval>" \
  --eval r01_torque="$N/evidence/<r01_torque_eval>" \
  --output-dir "$N/reports/F3_r01_strict_G2_summary"
```
