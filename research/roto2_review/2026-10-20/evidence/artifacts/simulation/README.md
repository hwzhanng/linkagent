# 最终模型真实仿真视频

主视频：`final_ppo_simulation_verified.mp4`，1280×720、20 fps、25.95 秒。其中真实物理回合 23.95 秒，最后 2 秒为带文字的停留画面。seed2026、固定环境 0、64 个并行环境中的单视角；没有按结果挑选样本。

**该视频揭示了此前评估的严重问题。** 旧算法累计角速度声称转动 6.23 圈，实际姿态却主要为小幅摆动：偏航范围约 16.4°，首末净偏航 −5.0°。原“成功”标记已被更正；此前 72.4% 完整成功率结论暂时撤回。

RGB 帧来自 Isaac Sim RTX 渲染真实 PhysX 状态；最终模型直接输出动作，无脚本动作或生成式视频。彩色标记按实测姿态更新，只作视觉用途。视频文字由原有角速度指标改为逐帧四元数核算的偏航变化。原始像素场景未作替换。结束步发生自动复位，最后图像明确停留在复位前一帧，不把新回合姿态冒充终止状态。

- `video_verification.json`：视频参数、逐帧姿态核验结果。
- `diagnostics/raw_pose_trace.json`：479 步实际指标与非终止步水果位姿。
- `diagnostics/legacy_velocity_overlay.mp4`：保留排查用的错误旧指标版本，不应作为结果展示。
- `verified_frame100.jpg`、`verified_frame250.jpg`、`verified_frame478.jpg`：最终视频抽帧。

本机 RTX 启动兼容处理只作用于录制进程：取消 DISPLAY、禁用驱动磁盘着色器缓存、仅使用 GPU0。没有更换系统驱动，也没有改动训练核心。

复现录制入口：

```bash
cd /home/descfly/linkagent-ppo
bash simulation/.runtime/run_roto2.sh record \
  --evaluate-checkpoint /home/descfly/linkagent-ppo/logs/roto2_pressure_fruit/jacobian_seed42/2026-10-19_23-03-40/checkpoints/final_agent.pt \
  --record-video /absolute/path/to/diagnostic.mp4
```

录制器保留旧角速度累计，但明确标注为待核验，不能据此判定旋转成功。下一步需要逐物理子步核验并报告旋转奖励/评估核心修正后，再恢复训练。
