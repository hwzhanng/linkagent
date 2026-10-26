# 不同实验尝试索引

以下68项保留2026-10-22路线图的原始分组与判定；不把检查点、搜索粒子和评估种子重复算成独立尝试。后续V15–V18及完整训练运行表另列。证据路径均相对仓库根目录，恢复Release数据包后可直接访问。

## 01 工程与测量层

运行环境 / 转角计分 / 触觉映射；修复测量可信度 ≠ 学会旋转

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|A01|独立安装与 PPO 跑通|ok|工程链路可运行|仅证明能训练，不证明行为成功|artifacts/simulation/reward_correction/README.md|
|A02|扩大并行环境|ok|后续使用 8192 / 16384 环境|吞吐提升未解决动作策略瓶颈|artifacts/simulation/rotation_v14/fresh_long_horizon/PLAN.json|
|A03|最初 Jacobian 模型|fail|旧指标称 6.23 圈，实际净偏航约 −5°|原 72.4% 成功率已撤回|artifacts/simulation/README.md|
|A04|四元数计分替换角速度积分|ok|奖励、评估与视频按真实姿态核验|修复“假转动”；仍需学习真实动作|artifacts/simulation/reward_correction/README.md|
|A05|有效指腹区域映射修正|ok|区域外接触不再截入传感器阵列|壳体接触不能冒充有效触摸|artifacts/simulation/reward_correction/README.md|
|A06|真实视频与逐帧位姿复核|ok|补录及位姿、解码检查已完成|可视证据不等于所有回合均成功|artifacts/simulation/rotation_v8/stage3/results/SELECTED_VIDEO_REVIEW.json|

## 02 场景与初态层

手安装姿态 / 抓姿库 / 复位分布；静态冗余成立，动态换指未成立

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|B01|直立策略直接迁移到平放|fail|独立测试物理失败约 90.6%|原抓姿和策略不能直接迁移|artifacts/simulation/palm_up/README.md|
|B02|平放抓姿物理搜索|partial|4 组候选零动作保持无物理失败|只是复位校准，不是学习结果|artifacts/simulation/palm_up/README.md|
|B03|V8 原抓姿 N−1 检查|fail|松指时缺乏足够冗余|持续固定三指妨碍交换接触|artifacts/simulation/rotation_v8/stage0/old_grasp_n1.json|
|B04|V8 抓姿搜索与独立复验|partial|冻结 256 个静态验证姿态|仍来自同一祖先家族、同一物体朝向|artifacts/simulation/rotation_v8/stage0/FROZEN_LIBRARY_GATE.json|
|B05|V8 偏移抓姿 IK 探针|partial|偏移后恢复参考接触存在残差|几何求解不能证明动力学可行|artifacts/simulation/rotation_v8/stage1/NEXT_EXPERIMENT_REASONING.md|
|B06|V9 无碰撞约束 IK 候选|fail|未得到可接纳的新 N−1 抓姿|几何接近没有转成物理承托|artifacts/simulation/rotation_v9/sprint/new_n1.json|
|B07|V9 参考 / 碰撞约束 IK|fail|有安全姿态，仍未通过四指要求|稳定与有效多指接触不能兼得|artifacts/simulation/rotation_v9/sprint/reference_n1.json|
|B08|V9 物理 CEM 抓姿搜索|fail|新家族扩展不足|少量候选不足以建立广泛动态库|artifacts/simulation/rotation_v9/sprint/PLAN.md|
|B09|V9 恢复状态课程库|partial|256 锚点 + 204 恢复状态|覆盖扩大，但仍是同一祖先家族|artifacts/simulation/rotation_v9/sprint/CURRICULUM_VALIDATION.json|
|B10|V10 恢复后姿态替换|fail|安全整圈 17/64 → 7/64|旧策略不能直接利用恢复后的姿态|artifacts/simulation/rotation_v10/recovery_path_search/POLICY_COMPARISON.json|
|B11|V13 三指冗余候选替换|fail|安全整圈 54/64 → 5/64|静态释放通过仍不保证策略转动|artifacts/simulation/rotation_v13/redundant_three_pad_screen/REPORT.md|

## 03 任务与 PPO 配置层

奖励 / 课程 / 探索参数；网络与 PPO 不改；反复出现“单次推转 → 限位 → 停住”

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|C01|修正后旋转试训 seed44|fail|未通过真实旋转验收|旧抓姿试训仍未得到可用旋转策略|artifacts/simulation/reward_correction/README.md|
|C02|修正后接触试训 seed45|fail|32M：整圈 0/64，失败 25/64|部分指腹接触，尚不能稳定旋转|artifacts/simulation/reward_correction/README.md|
|C03|V5 直立接触课程续训|partial|两个独立评估的接触门槛通过|仅支持进入旋转课程|artifacts/simulation/reward_correction/curriculum_v5_seed45/status.json|
|C04|V5 直立旋转课程|fail|标称两组旋转成功率均为 0|接触学习没有转成旋转能力|artifacts/simulation/reward_correction/curriculum_v5_seed45/status.json|
|C05|平放接触课程 seed46|partial|三指承托 88.6%，失败 0/128|仅通过接触课程|artifacts/simulation/palm_up/README.md|
|C06|平放旋转课程 seed46|fail|三种条件合计整圈 0/256|标称平均仅约 8.7°|artifacts/simulation/palm_up/README.md|
|C07|V6 两指过渡 + 转角里程碑|fail|平均 45.3°；整圈 0/128|初期收益后停滞，食指接近限位|artifacts/simulation/rotation_v6/trial_seed47/REVIEW.md|
|C08|V7 即时接触代价 + 抓姿先验|fail|承托约 74.5%，平均 18.4°|接触改善，但减少动作更容易获益|artifacts/simulation/rotation_v7/trial_seed48/review_summary.json|
|C09|V8 plain 初始 300M|fail|两组整圈均 0/64|末段平均仅约 18–21°|artifacts/simulation/rotation_v8/stage1/final_validation/QUANTITATIVE_GATE.json|
|C10|V8 姿态恢复权重 0.3|fail|两组整圈均 0/64|仍丢失指腹接触，未形成循环|artifacts/simulation/rotation_v8/stage1_pose030/FINAL_ASSESSMENT.json|
|C11|V8 top4 接近奖励 0.025|fail|101.8M：整圈 0/64|接触略增；后改入组合方案|artifacts/simulation/rotation_v8/stage1_approach0025/milestone_100000000/ASSESSMENT.json|
|C12|V8 组合方案 warm|partial|独立安全整圈 19/128|末段推进未达到当时门槛|artifacts/simulation/rotation_v8/stage1_direct/warm/milestone_125000000/INDEPENDENT_ASSESSMENT.json|
|C13|V8 组合方案 fresh|unknown|约 15.6M 后让出资源|提前停止，未完成独立效果判定|artifacts/simulation/rotation_v8/stage1_direct/DIRECT_SEARCH_SPEC.json|
|C14|V8 失败惩罚提高至 180|partial|独立安全整圈 24/128；失败 30/128|中期转角门槛通过，仍需完整验证|artifacts/simulation/rotation_v8/stage1_direct/failure180/milestone_123562080/INDEPENDENT_ASSESSMENT.json|
|C15|V8 选定分支继续至 300M|partial|独立留出安全整圈 77/128|真实单圈成立；重复触觉换指未证实|artifacts/simulation/rotation_v8/stage1_direct/consolidated/final_validation/FINAL_ASSESSMENT.json|
|C16|V9 恢复课程 + 48 秒续训|fail|约 204M：标称整圈 38→5 /64|接触增加但转动退化，提前止损|artifacts/simulation/rotation_v9/sprint/check200m_pairs/EARLY_STOP_DECISION.json|
|C17|V10 运动优先奖励续训|fail|约 102M：标称失败 4→24 /64|减弱接触约束未解决回收动作|artifacts/simulation/rotation_v10/motion_first_recovery/EARLY_STOP_REPORT.md|
|C18|V14 随机初始化 + 48 秒|fail|约 102M：两组均整圈 0/64|先转约 60°；后 24 秒仅 0.037°|artifacts/simulation/rotation_v14/fresh_long_horizon/EARLY_STOP_REPORT.md|

## 04 触觉奖励层

7 个独立奖励分支 + 接触引导续训；接触改善伴随减速；旋转与接触未兼得

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|D01|对照：无新增触觉约束|fail|安全整圈 72/128；失败 30/128|末段任一有效指腹仅 1.9%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D02|指腹质量：低权重|fail|安全整圈 86/128；失败 29/128|末段任一有效指腹仅 2.1%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D03|指腹质量：高权重|fail|安全整圈 63/128；失败 42/128|末段任一有效指腹仅 2.7%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D04|区域外接触惩罚：低|fail|安全整圈 84/128；失败 21/128|末段任一有效指腹仅 3.0%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D05|区域外接触惩罚：高|fail|安全整圈 86/128；失败 19/128|末段任一有效指腹仅 3.2%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D06|压力惩罚：低权重|fail|安全整圈 88/128；失败 29/128|末段任一有效指腹仅 1.9%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D07|压力惩罚：高权重|fail|安全整圈 94/128；失败 32/128|末段任一有效指腹仅 2.1%|artifacts/simulation/rotation_v8/stage2/results/FINAL_BATCH_ASSESSMENT.json|
|D08|接触引导追加 300M|partial|匹配安全整圈 84→88 /128；失败 29→4|末段接触改善，净转角约 204→156°|artifacts/simulation/rotation_v8/stage2_contact_guided/run/MATCHED_ASSESSMENT.json|

## 05 恢复动作与学习层

物理路径探针 / 搜索 / 模仿学习；能恢复接触的局部路径尚未转成连续步态

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|E01|保持对照|partial|48/48 稳定|没有恢复新关节行程|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E02|joint2 回收再返回|fail|43/48 稳定|回到旧目标，未保留新行程|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E03|roll 内转再返回|fail|48/48 稳定|未同时获得行程与有效指腹|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E04|末端松开再返回|fail|48/48 稳定|可重接触，但没有新行程|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E05|组合退回再返回|fail|30/48 稳定|仍未通过稳定 + 行程 + 指腹|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E06|先打开再回收闭合|fail|14/48 稳定且恢复行程|0/48 同时恢复有效指腹|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E07|直接回收并保持|fail|30/48 稳定且恢复行程|0/48 同时恢复有效指腹|artifacts/simulation/rotation_v10/thumb_recovery_diagnostic/REPORT.md|
|E08|V10 CEM 接触恢复路径|partial|独立三条件交集 2/36 条候选|额外重放 1/2 通过；非持续净旋转|artifacts/simulation/rotation_v10/recovery_path_search/REPORT.md|
|E09|V11 恢复路径模仿 BC|fail|离线误差下降，标称整圈 0/64|闭环失败 42/64，离线拟合不等于控制|artifacts/simulation/rotation_v11/recovery_imitation/REPORT.md|
|E10|V11 补回终端保持标签|fail|修复后标称整圈仍 0/64|离线误差下降，闭环仍失败 44/64|artifacts/simulation/rotation_v11/recovery_imitation/REPORT.md|
|E11|短周期搜索 → 长时间重放|fail|1× 与 2× 速度重放均 0/64 通过|早期偶有推进，重复后衰减|artifacts/simulation/rotation_v12/net_cycle_search/REPORT.md|
|E12|全关节、8 周期搜索|fail|6 轮共 24,576 条完整试验，0 通过|末三周期最小净转角最高仅约 3.71°|artifacts/simulation/rotation_v12/steady_cycle_search/REPORT.md|
|E13|接触可行候选优先搜索|fail|追加 2 轮、8192 条，0 通过|接触可行性改善，持续推进仍不足|artifacts/simulation/rotation_v12/contact_feasible_cycle_search/REPORT.md|

## 06 压力方法验证层

plain / scalar / Jacobian；核心算法未改；模块参与动作，但独立降压收益未证实

|编号|尝试|原判定|结果|问题|证据路径|
|---|---|---|---|---|---|
|F01|plain · seed 8301|partial|安全整圈 34/128；失败 8/128|平均回合峰压 14.48 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F02|plain · seed 8302|partial|安全整圈 27/128；失败 6/128|平均回合峰压 13.87 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F03|plain · seed 8303|partial|安全整圈 57/128；失败 5/128|平均回合峰压 13.92 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F04|scalar · seed 8301|partial|安全整圈 41/128；失败 7/128|平均回合峰压 14.31 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F05|scalar · seed 8302|partial|安全整圈 60/128；失败 3/128|平均回合峰压 13.83 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F06|scalar · seed 8303|partial|安全整圈 57/128；失败 4/128|平均回合峰压 14.08 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F07|jacobian · seed 8301|partial|安全整圈 56/128；失败 5/128|平均回合峰压 13.75 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F08|jacobian · seed 8302|partial|安全整圈 34/128；失败 3/128|平均回合峰压 13.80 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F09|jacobian · seed 8303|partial|安全整圈 46/128；失败 9/128|平均回合峰压 13.62 kPa；重复换指未验收|artifacts/simulation/rotation_v8/stage3/results/seed_metrics.csv|
|F10|同模型开 / 关 Jacobian 修正|fail|384 回合：开启后平均峰压 +0.62%|训练组间 −2.61% 不能归因于修正|artifacts/simulation/rotation_v8/stage3/diagnosis/DIAGNOSIS_AND_NEXT_PLAN.md|
|F11|稀疏压力风险覆盖检查|fail|后段多数密集高压未激活稀疏风险|风险输入与密集评价信号不一致|artifacts/simulation/rotation_v8/stage3/diagnosis/DIAGNOSIS_AND_NEXT_PLAN.md|
|F12|局部压力预测重放|fail|总体 MSE 约为不变基线的 1.05–1.07 倍|预测尚无稳定优势；核心改动未实施|artifacts/simulation/rotation_v8/stage3/diagnosis/DIAGNOSIS_AND_NEXT_PLAN.md|

## 后续实验（不与上面的68项合并为统一计数）

|阶段|问题与设计|预算/数据|结论|证据|
|---|---|---|---|---|
|V15|actor观测之外的物理真值是否改善critic|256完整回合，24次离线拟合|物理真值未通过主要回报预测筛查；没有启动物理真值PPO|`artifacts/simulation/rotation_v15/critic_information/`|
|V16|价值裁剪、时间边界；24秒vs48秒|2×64更新，各8,372,224交互|边界冲击随时限移动；不是已证明的唯一停转根因|[报告](reports/V16_BOUNDARY_REPORT.md)|
|V17离线|时钟/奖励历史拆分|48次额外离线拟合|时钟收益小；奖励路径有效位有预测收益，但属适应性探索|[报告](reports/V17_DIAGNOSTIC_REPORT.md)|
|V17在线|masked vs critic奖励历史位；actor不接收该位|2×765更新，各100,074,240交互；576评估回合|两组整体退化，实验后段转角比对照更低；不扩训|[完整报告](reports/V17_FINAL_REVIEW.md)|
|V18|补评V16已保存的原探索强度检查点|6×64=384评估回合，新增训练0|无普遍立即崩坏；留出恢复抓姿仍退化；不能将探索重置或时限直接定为唯一原因|[数据](reports/V18_RESULT.md)、[解读](reports/V18_INTERPRETATION.md)|

## 全量运行与文件索引

- `ALL_RUNS.json`：每个保存的run_config及其完成状态，失败/缺少完成标记均保留。
- `ALL_WEIGHTS.json`：全部360个权重/张量文件，包含诊断张量；不等同于360个独立已训练策略。
- `ALL_VIDEOS.json`：46个原始视频，包括预览、失败和历史遗留计分视频，不能都当成功展示。
- `ALL_REPORTS.json`：1003个Markdown/PDF/HTML报告或渲染文档。
- `DATA_MANIFEST.json`：13,994个文件的路径、字节数和SHA-256。
