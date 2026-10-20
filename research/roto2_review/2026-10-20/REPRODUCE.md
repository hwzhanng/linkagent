# 下载、校验、恢复与复现

分支保存代码、报告、原始日志、配置、TensorBoard事件、轨迹和视频；81份模型检查点与39份派生物理资产放在[同仓库Release](https://github.com/hwzhanng/linkagent/releases/tag/roto2-review-20261020-v7)。仓库权限不变，访问者须有私有仓库读取权限。

## 1. 获取并校验完整证据

在有GitHub仓库权限的环境执行：

```bash
git clone --branch research/roto2-object-pressure-track https://github.com/hwzhanng/linkagent.git linkagent-review
cd linkagent-review
mkdir -p review-download
gh release download roto2-review-20261020-v7 --repo hwzhanng/linkagent \
  --pattern 'roto2-20261020-models-and-assets.tar.gz' --pattern SHA256SUMS \
  --dir review-download
(cd review-download && sha256sum -c SHA256SUMS)
python3 research/roto2_review/2026-10-20/restore_evidence.py \
  --archive review-download/roto2-20261020-models-and-assets.tar.gz
```

完整校验应报告`verified_files=2136`、`archive_verified=true`。归档1,049,734,404字节，SHA256为`4d03859a83d4bb9ec6013f4a5b039ad5fc21e542352ffb1e3f550995a413d2b5`。校验工具先校验归档整体，再逐成员校验；不需要PyTorch。

将证据恢复到原仓库相对路径，供程序直接读取：

```bash
python3 research/roto2_review/2026-10-20/restore_evidence.py \
  --archive review-download/roto2-20261020-models-and-assets.tar.gz \
  --restore-to "$PWD"
```

恢复器拒绝覆盖内容不同的现有文件，也拒绝符号链接和路径越界；建议使用新克隆目录。下载、解压、运行环境另外占空间，建议预留至少100GB。原始日志中`/home/descfly/linkagent-ppo/`为历史路径，不必在新机器建立同名用户名目录；启动命令使用当前仓库相对路径。不要批量改写证据文件。

## 2. 安装运行环境

原机软件/硬件见[environment.json](environment.json)，完整已安装依赖见[requirements-installed.txt](evidence/simulation/.runtime/setup/requirements-installed.txt)。这里没有打包约43GB的wheel/uv下载缓存及已安装运行时。

Ubuntu x86_64、兼容的NVIDIA驱动、Python 3.12、`uv`、`git-lfs`为基础条件。原机为RTX 5090 32GB。固定Isaac Lab源码提交、Sim版本和RoTO依赖的安装入口：

```bash
export LINKAGENT_ISAACLAB_ROOT="$PWD/simulation/.runtime/IsaacLab-3.0.0-beta2.patch1"
bash script/bootstrap_roto2_pressure.sh
```

脚本不安装系统驱动、不代接受NVIDIA许可。安装前阅读其要求。目录名称含3.0 beta；安装后发行包元数据的isaaclab版本为6.1.14，两者不是可以互换的版本定位依据，应核对源码提交。通用安装脚本的部分辅助依赖使用版本范围；需精确排错时以保存的实际依赖清单和安装日志为准。完整干净机器重新安装尚未在交付时重跑，不能承诺跨驱动逐位复现。

先运行8环境真实PhysX烟雾验证，再尝试大规模运行：

```bash
bash simulation/.runtime/run_roto2.sh smoke
```

## 3. 复查当前v7最终模型

以下评估默认64环境、每环境一个完整回合；2071/2072各运行一次，对应当前128回合结果。输出到新目录，避免覆盖归档证据。

```bash
mkdir -p artifacts/reproduce-v7
bash simulation/.runtime/run_roto2.sh evaluate \
  --seed 2071 --domain-profile nominal --hand-posture palm_up \
  --reward-stage continuous --goal-direction positive \
  --reset-grasps simulation/roto2_pressure/reset_pad_grasps_palm_up.json \
  --evaluate-checkpoint logs/roto2_pressure_fruit/jacobian_seed48/2026-10-20_17-27-37/checkpoints/final_agent.pt \
  --evaluation-output artifacts/reproduce-v7/nominal_seed2071.json
```

将种子和输出名都改为2072再运行一次。当前代码的`continuous`分支对应v7。历史v2–v6运行应使用对应运行目录中的`source_snapshot`和原始配置；不能拿当前奖励解释旧训练回报。个别早期运行缺少完整快照/配置，运行表已标注，不能保证精确重建。

录像同样传入以上手姿、课程、初始化与检查点参数，用`record`模式替代`evaluate`，并将`--evaluation-output ...`替换为`--record-video artifacts/reproduce-v7/nominal_seed2071_video.mp4`。视频文件名必须带`_video`，因为会写同名JSON旁文件；不要与评估JSON重名。该视频只代表选定env0，应同时读整体评估。

## 4. 重跑本轮32M训练

这段命令是已完成v7试验的复现入口，不表示该配方已达标，也不是建议立即继续扩展预算：

```bash
bash simulation/.runtime/run_roto2.sh train \
  --seed 48 --million-steps 32 --entropy-weight 0.005 \
  --domain-profile nominal --hand-posture palm_up \
  --reward-stage continuous --goal-direction positive \
  --reset-grasps simulation/roto2_pressure/reset_pad_grasps_palm_up.json \
  --resume-checkpoint logs/roto2_pressure_fruit/jacobian_seed46/2026-10-20_08-13-16/checkpoints/final_agent.pt
```

16,384为总环境数，其中16个用于训练内评估，16,368个训练环境；32M预算实际执行31,999,440条转移。原机平均约80k转移/秒，其他显卡不保证相同吞吐。显存不足时降低`--num-envs`可检查流程，但会改变批量与实际训练协议，不能称作同条件复现。

当前检查点保存并恢复注册的网络、优化器及归一化模块状态；没有恢复完整随机数状态、环境状态与rollout，因此不是逐位等价的断点续训。加载行为见固定版本上游PPO的`load`与本仓库检查点写入代码。

## 5. 训练日志与表格

```bash
simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  -m tensorboard.main --logdir logs/roto2_pressure_fruit --host 127.0.0.1 --port 6006
```

无需GPU也可查看[TensorBoard标量CSV](tables/tensorboard_scalars.csv)、[所有运行](TRAINING_RUNS.md)和[逐回合CSV](tables/episodes.csv)。CSV保留原tag、step、wall_time；不同运行的step可能从零重新计数，不能直接拼成单模型曲线。各段样本数以`training_result.json`为准，继承累计数不是本轮新增数。

保存范围和逐文件哈希见[manifest.json](manifest.json)。`source_snapshot`是各次运行当时的局部源码快照；分支代码是交付时最终实现。完整PPO逐步训练rollout没有全部落盘，无法从这些日志无损恢复为静态离线训练集。部分诊断NPZ保存的是评估轨迹，不是全部训练样本。
