# 当前安装环境下复核

使用已有IsaacLab环境：`/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python`。冻结源码工作区为`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/source-low-noise-v1_6`，源码归档保存在本目录；运行仍依赖IsaacLab/PhysX及CUDA，不包含环境安装包。

```bash
cd '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/source-low-noise-v1_6'
LINKBOT_RECHECK_NAME="delivery_recheck_$(date +%Y%m%d_%H%M%S)"
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python -m research.thumb_pad_grasp_v1.launch_probe --name "$LINKBOT_RECHECK_NAME" --module run --num-envs 4 -- --mode evaluate --policy checkpoint --checkpoint '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/weights/thumb_pad_ppo_v1_6_update1.pt' --episodes 4 --seed 110 --thumb-mapping independent --center-local-mm -88.76065266544136 -11.325082731003075 -193.5340228574738 --pregrasp-distance-mm 5
```

单GPU启动器会锁定资源，输出目录必须尚不存在。新一轮物理仿真可能不同于已有小样本，不预先保证1/4重复率；已有原始成功窗保留以供独立复核。

## 只读离线复核

以下命令使用numpy/scipy，未启动仿真，也不修改权重。输出文件须不存在。

```bash
python '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/verification/recheck_success_full_mesh.py' --audit-dir '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/evidence/eval_low_noise_v1_6_update1_01_offline_audit' --run-dir '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/evidence/eval_low_noise_v1_6_update1_01' --source-root '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/source-low-noise-v1_6' --raw-window '/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/evidence/success_window.jsonl' --output /tmp/linkbot_delivery_full_mesh_recheck.json
```

移走项目时可把源码归档解压到新目录，并将上条命令的`--source-root`指向解压根目录；其余路径指向本交付包。脚本默认读取同目录原完整网格分类器，核对拷贝成功窗哈希，重新计算局部坐标与world/body转换后的PAD标签。它证明记录的几何接触一致性；PPO更新事实由训练/参数回执证明，运行时故障来源和手坐标漂移仍保留原守卫的证据边界。
