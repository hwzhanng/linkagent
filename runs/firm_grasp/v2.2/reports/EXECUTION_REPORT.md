# F3/F4 抓牢有限探索执行报告

状态：物理评估、逐窗核验及前/侧录像已完成；按用户止损要求，本轮交付后暂停进一步训练，等待下一步指示。

## 结论

已得到真实拇指—食指—中指参与的3秒抓牢更新候选，原批准指肚位置未改。无扰动、力扰动、力矩扰动在开发复测中各1/4成功，且相应能力在另一次重载中得到复现：quiet/force见最终矩阵，torque同协议独立重载为1/4。每个声称成功的窗口均需数值重算和原始完整网格重分类，不能用录像或训练奖励替代。

最终矩阵显示新任务初始先验与PPO更新策略总成绩相同，无法证明本轮PPO带来总体成功率提升；力矩最终矩阵仍0/6，说明建抓和重复reset稳定性不足。独立重载的力矩1/4不替换这6条失败。

## 最终固定20项对照

|策略（同一新评估条件）|无扰动|力扰动|力矩扰动|合计|FAILURE|UNKNOWN|
|---|---:|---:|---:|---:|---:|---:|
|只读v1.6 actor，恢复其原动作语义的适配|0/8|0/6|0/6|0/20|20|0|
|v2.2未更新初始化先验|2/8|1/6|0/6|3/20|17|0|
|v2.2 PPO update1|1/8|2/6|0/6|3/20|17|0|

开掌负对照：0/4 SUCCESS、4 FAILURE、0 UNKNOWN。另一次同协议力矩重载：1/4 SUCCESS、3 FAILURE、0 UNKNOWN。所有分母完整，三组矩阵case0..19唯一、quiet/force/torque=8/6/6。报告详见 `F4_FINAL_COMPARE/EVAL_TABLE.json` 与 `.md`。每个轴仅一次案例，固定球果实/初态不支持方向可靠性、尺寸/位置泛化或真实硬件结论。

## 本次恢复后的训练

上限预设为3轮；实际完成2轮，每轮1次PPO更新、1024 transitions，共2更新/2048 transitions。

- R1：保留v2.1先验，std=0.00005、LR=2e-7，独立quiet重载0/4。降低更新幅度没有恢复成功。
- R2：独立v2.2副本新增0.15—0.30 N区间的第三指真实指肚载荷余量奖励，最高2分，保留原过载等惩罚；使用std=0.0005、LR=2e-6。quiet开发重载1/4，正式力和力矩各1/4。此奖励变化与R1相比还伴随LR/std恢复，不将结果单独归因于奖励。
- R2出现候选后即冻结，再做最终20项测试；第三轮未启动，预拟ramp40条件没有执行。查看最终成绩后没有继续调模型或重复挑选同一测试清单。

这是有界探索，未穷尽PPO方案。后续若继续训练，应使用新版本和重新预登记的最终对照，不修改本次留存。

## 验证与诊断

v2.2 mean MLP参数确有变化，见 `PPO_MEAN_PARAMETER_CHANGE.json`。更新策略3个最终成功案例、初始化先验3个最终成功案例全部通过独立数值G2重算和原始完整网格PAD重分类。旧版20项无runtime成功，未伪造成功窗。F3开发三模式与力矩独立重载窗口也分别通过两种核验。

更新策略10个扰动失败案例均在脉冲180子步之前丢失合格连续窗，不能简单归因于扰动力矩过强。主要瓶颈为撤支撑后0.5秒内稳定建立夹持及重复reset一致性。参见 `F4_UPDATED_FAILURE_DIAGNOSIS.json`；`F4_REPEAT_RESET_DIAGNOSTIC.json`只是首记录子步差异诊断，尚未证明reset代码有错。

未更改三指≥0.15 N/≥5%承载份额、360真实子步、对向/分散几何、转动/速度、支撑为零、七故障等验收规则；未扩大批准PAD、改质量/摩擦/PD/力矩上限或在回合中写果实状态。新条件power4撤支撑和动作先验的收益不能冒称PPO学习收益。刚性球模型也不能证明真实果实损伤降低。

原v1.6和F2受保护权重哈希复验保持不变，见 `PROTECTED_WEIGHTS_AFTER.json`。新源码提交：f60b912ce85f1361cd02bc11da15a10e302b1cbe；候选冻结和模型SHA见上级 `F4_CANDIDATE_FREEZE.json`。

## 留存与后续

两视角均179帧、10 fps、17.9秒，各4回合中1成功/3失败；录像成功窗的数值与原网格核验通过。已解码检查前/侧第7秒画面，见 `VIDEO_REVIEW.json`。原录像回执沿用了旧thumb_pad schema/task_id字段；实际任务版本、冻结checkpoint与原轨迹以run_metadata及VIDEO_REVIEW为准，未改写原回执。

独立留存目录：`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/delivery/ppo-firm-grasp-v2_2-20261018`，含权重、源码归档、报告、录像和MANIFEST。它是本机留存包，原始大轨迹通过RAW_EVIDENCE_INDEX索引至N/evidence；复核仍依赖原目录和本机Isaac。资源明细见 `RESOURCE_LEDGER_CONTINUATION.json`。

前视录像：[evaluation.mp4](</media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F4_video_front/evaluation.mp4>)；侧视录像：[evaluation.mp4](</media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F4_video_side/evaluation.mp4>)。

最值得先做的两项未执行建议见 `NEXT_MINIMAL_STEPS_NOT_EXECUTED.md`：先查清重复reset后的建抓一致性，再做保留初始化先验能力的受约束PPO实验。无需重新人工标定已确认的五指指肚区域。
