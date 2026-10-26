# RoTO2 完整研究交接 · 2026-10-26

这份交付用于在另一台电脑恢复全部已保存实验材料、安装独立环境，并从明确的基线继续。**研究目标尚未完成：持续触觉旋转仍有停转，压力创新优势尚未证实。** 不能把“训练正常结束”理解为“方法成功”。

分支：`research/roto2-object-pressure-track`。全部大文件：[roto2-handoff-20261026-all Release](https://github.com/hwzhanng/linkagent/releases/tag/roto2-handoff-20261026-all)。仓库为私有，另一台电脑的GitHub账号需要访问权限。

## 从哪里看

- [不同实验尝试](EXPERIMENTS.md)：历史68项分支逐项列出结果和问题，另列V15–V18；完整运行配置见 `ALL_RUNS.json`。
- [权重及续训选择](CHECKPOINTS.md)：成熟基线、短训诊断模型和退化模型分开说明。
- [视频与可视化](VISUALS.md)：8个可单独下载的代表视频、V17抽帧对照；全部46个视频都在包内。
- [下一步计划](NEXT_PLAN.md)：已结合V18最新结果更新，尚未开启新的探索强度配对。
- [V17完整结果](reports/V17_FINAL_REVIEW.md)、[V18补测数据](reports/V18_RESULT.md)、[V18解读](reports/V18_INTERPRETATION.md)、[Stage3九轮结果](reports/STAGE3_NINE_RUN_RESULT.md)。
- `DATA_MANIFEST.json` / `PACKAGE.json`：逐文件和分卷校验清单。`ALL_WEIGHTS.json`、`ALL_VIDEOS.json`、`ALL_REPORTS.json`为全量索引。

## 交付范围

完整数据包含13,994个文件、11,772,101,540原始字节；zstd压缩9,010,490,781字节，分10卷（前9卷各1,000,000,000字节）。包含360个权重/张量文件、46个视频、1003个Markdown/PDF/HTML文件，318个带run_config的运行目录。360不是独立策略数量，1003不是人工撰写报告数量；诊断张量、历史副本、渲染文档和未完成运行均保留，不夸大实验数。

完整保存：`artifacts/`、`logs/`、早期`simulation/.runtime/setup/`实验记录、`literature/`、历次`review_upload/`交付副本、生成的手模型资产、三份固定上游源码bundle。训练输入库、全部新代码、配置和交接报告在Git分支；以前已提交的报告仍保留。少量绝对路径符号链接已在数据包中实体化；3个目录别名由恢复脚本重新建立相对链接。

不搬运：旧Python虚拟环境、可重下载的wheel/uv缓存、Git内部对象目录、Python字节码、个人凭据和systemd自动任务。旧环境不能通过目录拷贝可靠跨机迁移；提供全部263个已安装包的精确版本、14个editable源包信息、固定源码和重建入口。没有把个人GitHub凭据放进包。

## 在新电脑操作

目标为Linux x86_64 + NVIDIA GPU，Python 3.12、Isaac Sim 6.0.1.0、Isaac Lab固定提交、Torch 2.10.0+cu128。原训练使用RTX5090 32GB。其他GPU需要实际smoke验证；8192环境不保证适合显存更小的卡。建议预留150GB空间（数据恢复、下载缓存和模拟器环境）。NVIDIA驱动需能运行CUDA12.8和所固定的Isaac版本；原机驱动595.84。安装系统驱动不是本脚本的工作。

以下为Ubuntu准备命令，需在你有管理权限的机器运行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg zstd fonts-noto-cjk \
  build-essential cmake pkg-config python3-venv libegl1 libgl1 libglib2.0-0
```

另需GitHub CLI `gh`、`uv`、Python 3.12。`gh auth login`使用你自己的账号登录；uv安装参考 [官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。不要复制原电脑的凭据。

### 1. 拉分支、下载和校验全部实验数据

```bash
gh auth login
gh repo clone hwzhanng/linkagent linkagent-ppo -- \
  --branch research/roto2-object-pressure-track --single-branch
cd linkagent-ppo
git lfs pull
python3 script/restore_roto2_handoff.py download
python3 script/restore_roto2_handoff.py restore
```

运行恢复工具使用Python3.12或更新版本。下载会跳过已经存在的分卷，并核验全部SHA-256；坏分卷会报错，保留现场，移走对应坏文件后重下。恢复不会覆盖不同内容的本地研究文件。完成标志为根目录`HANDOFF_DATA_VERIFIED.json`中的`verified: true`及13,994个文件。恢复过程不启动训练。

### 2. 重建独立环境

```bash
python3 script/bootstrap_roto2_handoff.py --execute
bash script/run_roto2.sh smoke
```

默认环境位置为当前仓库内`simulation/.runtime/IsaacLab-3.0.0-beta2.patch1/env_isaaclab`。脚本优先使用包内固定源码bundle；Python依赖仍需访问PyPI、NVIDIA和PyTorch下载站。以精确版本和`--no-deps`恢复已验证组合，随后校验版本；这不是通用的“pip升级到最新版”。首次Isaac启动如提示NVIDIA许可证，需要新机器使用者按提示完成许可流程。

`environment/pip-check-known.txt`记录原环境已有元数据冲突，包括Isaac声明Torch2.11而本项目验证使用Torch2.10，以及未装的上游开发/调参额外依赖。不能将`pip check`不干净隐瞒成干净环境，也不要用无约束升级“修复”而改变实验栈。实际GPU/PhysX smoke与成熟策略评估才是本机可用性的最终检查。原始pip freeze仅供溯源，含原机file://路径；安装应使用`requirements-exact.txt`和新重建入口。

### 3. 复现成熟模型，再开始研究

```bash
# 不带--execute只打印可审阅命令
python3 script/continue_roto2_handoff.py evaluate --condition nominal48
python3 script/continue_roto2_handoff.py evaluate --condition nominal48 --execute
python3 script/continue_roto2_handoff.py record --condition nominal48 --execute
```

默认加载成熟源模型：`logs/roto2_pressure_fruit/plain_seed7401/2026-10-21_16-16-13/checkpoints/final_agent.pt`。新输出写入`artifacts/simulation/handoff_new/`的新时间戳目录，不覆盖交付材料。可选`--checkpoint /新路径/agent.pt`、`--condition nominal24|nominal48|heldout_gentle48`。V17两组actor均可原生评估，但它们不是推荐基线。

原机对应nominal48种子9882、64回合：安全整圈51/64，物理失败7/64，24–48秒全64回合平均净转角119.88°。跨GPU/驱动不承诺逐位相同，需要保存实际差异。只看最好视频不足以验收。

### 4. 有意启动续训时

先阅读 `NEXT_PLAN.md`。下面是恢复成熟模型、保留原探索状态与24秒配置的有限预算命令模板，不是已经启动的新研究批次，也不等于正式压力长训已就绪：

```bash
python3 script/continue_roto2_handoff.py train \
  --million-steps 8.372225 --seed 9901
# 确认该实验设计后，在同一命令末尾加 --execute 才执行。
```

模板要求明确预算，不默认无限训练。可用`--num-envs`适配小显存，但环境数变化会影响批次和可比性，需作为新配置记录。默认成熟模型完整检查点保留模型/优化器/归一化；重启环境后继续优化，不是原物理进程的逐位重放。新探索重置对照需另写配对计划，不能只把上面单组当完整对照。

## 路径与历史脚本

原报告、run_config和PLAN里保留`/home/descfly/linkagent-ppo`绝对路径，用于溯源，不做全局替换。新入口在构建命令时转换路径；不要直接执行历史systemd、runner或旧PLAN里的绝对命令。旧supervisor多数有“目标目录已存在就退出”的保护，这是避免覆盖实验的正确行为。首次恢复校验应在bootstrap/新运行之前进行，因为重建可能更新可再生资产及runtime入口。

本次验证范围与远端交付凭据见 `DELIVERY_VERIFICATION.json`（交付结束生成）。原生新机器全量依赖安装仍需在目标硬件执行，不把复用本机已安装依赖的独立目录测试称为新机器安装验证。
