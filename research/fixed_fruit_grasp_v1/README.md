# 当前续训版本：v1.2.0

目标为撤支撑后真实多指抓握1秒；尚需同条件训练后20次reset验证。原USD保持不变，显式细化CAD凸分解，保留求解器全部碰撞；跨指自碰撞守护使用真实PhysX separation，包围盒只作诊断。奖励改为接近势差及真实保持门控。15N/指为已放宽的仿真工程上限，不代表柔和抓握或实机验收。旧失败源码/检查点保留，不直接兼容续训。

# 固定单果真实接触 PPO（首版保持课程）

隔离分支 `codex/ppo-grasp-first-result-20261013`，基于 `ed9cd84`。新任务使用 canonical isolated hand USD、实际自碰撞与掌面碰撞、14D 动作、21 个关节最终力矩裁剪。67 mm 球果质量约 141.7 g，材料固定；所有随机化关闭。固定初态由少量接线试验选定：`--pregrasp-distance-mm 5 --center-local-mm -61 -13 -225`，实际 reset 坐标另存。

本版只有 `contact`（6 秒撤支撑、保持 1 秒）和 `hold`（2 秒撤支撑、保持 3 秒）。**受驱动提举未实现，完整 30 mm + 3 秒工程目标未达到。** `lift` 直接拒绝，不逐帧移动手根或果实。课程成功和完整目标分开报告。果梗支撑在声明时刻撤除，最终保持必须由真实拇指与至少另一根手指参与。

真实穿透读取 PhysX packed contact separation，覆盖 27 个手部刚体、在 policy step 采样；AABB overlap 仅诊断。该读取不声称覆盖全部物理子步资格审计。驱动边界和非有限状态、过载、穿透仍终止回合。

## 实际运行目录

`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013`

源码在其 `source/` 工作树，证据在 `evidence/`。旧 canonical、wave2、Revised 的修改和记录保留。缓存、日志和临时文件都在外置盘；`supervisor` 串行锁定 GPU，结算每个进程实际墙钟时间，累计最多 12 小时。进程退出 0 不能替代有效摘要、优化更新、权重变化与独立重载验证。

## 已实际启动的短训命令

从本工作树运行：

```bash
python -m research.fixed_fruit_grasp_v1.supervisor \
  --run-root '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013' \
  --name short_contact_02 --timeout-seconds 3720 -- \
  /home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  -m research.fixed_fruit_grasp_v1.train --headless --device cuda:0 --num-envs 32 \
  --stage contact --pregrasp-distance-mm 5 --center-local-mm -61 -13 -225 \
  --resume '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013/initial_contact_v2/initial.pt' \
  --iterations 100000 --max-seconds 3600 \
  --log-dir '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013/short_contact_02'
```

该身份已经存在，复现需更换新身份/输出目录以保留原记录。初始化用 `--initialize-only`，必须先存初始权重并独立评估后才开始 PPO。`--resume` 必须显式传入同一固定姿态；合同不匹配会拒绝。

## 已实际执行的初始 actor 评估

```bash
python -m research.fixed_fruit_grasp_v1.supervisor \
  --run-root '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013' \
  --name eval_initial_contact_v2 --timeout-seconds 600 -- \
  /home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  -m research.fixed_fruit_grasp_v1.evaluate --headless --device cuda:0 --num-envs 4 \
  --episodes 20 --policy checkpoint \
  --checkpoint '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013/initial_contact_v2/initial.pt' \
  --output '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-first-result-20261013/evidence/eval_initial_contact_v2.json'
```

训练后评估只替换检查点、进程身份和输出路径；开掌对照用 `--policy open --stage contact` 并显式传入同一姿态。每个 reset 前完成的回合逐行保存，不删除失败。具体最终结果以运行目录的一页结论为准。

无 CAN、真实硬件、自然断梗、泛化或旧 R2 验收结论。
