# 200M 终止原因实测与最终诊断准备

冻结 checkpoint、seed5101、64 局真实 PhysX 重放，新增终止前最大指端接触力记录。
全部 30 个原有轨迹数组与原 200M 记录逐元素一致；新增力数据不改变策略行为。
28,775 个有效环境步上，重建的物理终止判据全部与 terminated 标志一致。64 局中 7 次物理失败、57 次时间上限结束。

| 环境 | 终止时间 s | 原因 | 最大指端力 N | 位移 mm | 压力代理 Pa |
|---|---:|---|---:|---:|---:|
| 7 | 9.80 | distal_contact_force | 8.4572 | 15.250 | 13379.31 |
| 12 | 5.05 | distal_contact_force | 8.2053 | 13.445 | 13923.84 |
| 20 | 10.20 | distal_contact_force | 8.6763 | 19.806 | 11247.58 |
| 26 | 15.20 | center_displacement | 0.0000 | 151.054 | 0.00 |
| 36 | 10.10 | distal_contact_force | 8.0935 | 16.679 | 12477.43 |
| 62 | 14.55 | distal_contact_force | 8.2662 | 23.643 | 16320.58 |
| 63 | 8.70 | center_displacement | 0.0000 | 131.399 | 0.00 |

结论：5 次指端力超过 8 N，2 次球心位移超过 105 mm；持续超压、世界高度和非有限位置均未触发。
这将此前“可能是接触力”的推测升级为该固定轨迹下的直接实测；不表示其他种子、checkpoint 或真实硬件具有同样分布。
下一候选的接触恢复项应同时检查接触力失败率，不能只看指腹接近质量。原 8 N / 105 mm / 18 kPa 三步持续等安全门槛不变。

## 验证和数据范围

新增 audit_roto2_terminal_causes.py 验证连续超压计数、物理终止与超时重叠、重置后数据排除及不完整轨迹拒绝；合成边界样例通过，上述真实 64 局也通过。
既有 200M 接触分析经扩展后的 summarize_roto2_recontact.py 复算，除新增 final_training_result 元数据外，全部值与旧报告完全一致。
完整新增 NPZ 保存在本地 artifacts/simulation/rotation_v8/stage1_pose030/final_diagnostic_preparation/trace_200M_seed5101.npz，哈希见 REPLAY_VERIFICATION.json；本次分支包含汇总、逐局终止数据和命令，NPZ 将随最终完整包冻结，不声称本次已上传。

## 最终诊断队列

训练结束后先运行原两种子独立评估和真实录像；随后 diagnose_roto2_final_run.py 重放同一最终 checkpoint 的 seed6101，核验轨迹一致性、终止原因、关节限位、有效指腹和后10秒运动收益。
服务 roto2-stage1-pose030-final-diagnostic20261021.service 当前等待既有验收进程结束。该后续任务尚未完成，不代替视频人工核验，不代表阶段1通过。
本次只修改诊断输出及离线审计；没有修改网络、PPO、观测动作、奖励、物理失败条件或成功判定。
