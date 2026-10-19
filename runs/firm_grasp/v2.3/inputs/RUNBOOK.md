# 同机接续入口与已知陷阱

下文变量仅为文档缩写；在脚本中实际声明再使用。旧脚本不直接运行到旧输出名。本文命令为接续模板，**本任务包制作时未执行GPU命令或创建新训练根**。

## 1. 路径、身份与保护

```bash
CANON=/home/miobarnacle/linkbot/linkagent
EP=/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python
OLD='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018'
SOURCE="$OLD/source-f3-v2_2"
DELIVERY="$OLD/delivery/ppo-firm-grasp-v2_2-20261018"
NEW='/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-continuation-20261019'
```

|继承物|位置与身份|
|---|---|
|v2.2源码|SOURCE；commit `f60b912ce85f1361cd02bc11da15a10e302b1cbe`；task `firm_grasp_v2.2.0`|
|v2.2更新权重|OLD/evidence/F3_train_margin_bounded02/update_000001.pt；SHA256 `af961f8ea500943278de77bc2683eaa12ea49e4ce742e839b0182eee48a660d9`|
|v2.2初始化权重|同目录initial.pt；SHA256 `b5335e101993fa86d4a9fbad9b37b484f4e54db637f1b5a7b939d6a64af1b683`|
|原始F2接收|CANON/.omo/evidence/ppo-firm-grasp-f2-takeover-20261018/；ACCEPTANCE.json、TAKEOVER_REVIEW.md|
|F2权重|上述目录received/evidence/F2_train_handover01/update_000001.pt；SHA256 `a5997ed42977b4cd66bcba021510a95445818153b6de5b0fcb219fa8c98154b7`|
|v1.6留存|`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/baselines/ppo-thumb-pad-v1_6-frozen-20261017`|
|v1.6权重|留存根thumb_pad_ppo_v1_6_update1.pt；SHA256 `0650ac4b35e742430f910319492d8ada621ee1c25f954e8aba6bd8163e59ddf6`|

SOURCE的actor/critic/action为162/223/14，动作8/9是食/中指侧摆。旧v1.6为161维观测且动作8/9语义不同（无名/小指），必须使用显式legacy适配，不能因14维相同直接装载。v2.2 critical source SHA256为 `7460ecdec4fce9a0e258fbb667d8aa61f18dda155d32d4c080dd7b48f7398167`；更改关键源码后不再具有该身份。

全部OLD（含DELIVERY）只读，此外继承OLD/BASELINE_LINK.json中所有protected_paths并新增OLD本身。原v1.6、原外部F2根及本机F2接收根同样只读。canonical现有脏改动不清理、不reset。输出保护需比较resolve路径，并排除软链/硬链别名，不只看文件名。

## 2. 先读这些事实，不必遍历历史

1. DELIVERY/reports/EXECUTION_REPORT.md、reports/F4_FINAL_COMPARE/EVAL_TABLE.json。
2. OLD/reports/NEXT_MINIMAL_STEPS_NOT_EXECUTED.md、F4_UPDATED_FAILURE_DIAGNOSIS.json、F4_REPEAT_RESET_DIAGNOSTIC.json。
3. OLD/F4_CANDIDATE_FREEZE.json、SOURCE_FREEZE_LOAD_MARGIN.json、inputs/power4_index025_chosen.json。
4. 具体追溯时读OLD/evidence中的run_metadata.json、episodes.jsonl、pad_trace.jsonl；交付包RAW_EVIDENCE_INDEX.json给原始数据路径。

本任务包references有上述关键报告的轻量副本。注意：OLD/CONTINUATION_STATUS.json仍留有历史next_action和pending字段；最新最终表、执行报告、终态过程回执优先，不能据残留描述重跑已完成队列。OLD/EXPERIMENTS.json也不是完整资源账。最新完整资源账为reports/RESOURCE_LEDGER_CONTINUATION.json。

## 3. 创建独立源码，不复制训练缓存或旧证据树

确认NEW不存在后，可按以下方式建立独立Git对象与工作文件，避免shared对象依赖及LFS过滤造成的伪脏状态：

```bash
test ! -e "$NEW"
mkdir -p "$NEW/evidence" "$NEW/inputs" "$NEW/reports" "$NEW/tmp"
git clone --no-local --no-checkout "$SOURCE" "$NEW/source"
cp "$SOURCE/.git/info/attributes" "$NEW/source/.git/info/attributes"
git -C "$NEW/source" read-tree HEAD
rsync -a --exclude=.git --exclude=.omo --exclude=__pycache__ --exclude=.pytest_cache "$SOURCE/" "$NEW/source/"
git -C "$NEW/source" switch -c codex/firm-grasp-f3-f4-continuation-20261019
git -C "$NEW/source" status --short
```

逐条检查退出状态；test失败立即改新后缀，不继续mkdir到旧根。这里read-tree只用于**新建副本**的索引，不能对旧树执行reset/clean。独立副本仍需实际检查LFS资产是否为完整二进制；不把Git clone成功当物理资产齐全。

复制所需OLD根驱动与核验工具到NEW后，自主适配并审查以下问题：

- execute.py默认`source-f3-v2_0`，必须改为新源码source；FIRM_SOURCE_NAME是相对NEW的子目录名，不设置时不能指向旧默认。
- SOURCE/research/firm_grasp_v1/launch.py从源码位置推导run root；源码必须在NEW直接下一层。其/tmp短链固定为`/tmp/linkbot-ppo-firm-grasp-f3-f4-20261018`，新副本改唯一新名，不能移走旧短链。
- 创建新的NEW/BASELINE_LINK.json，更新new_root/new_source和来源，继承保护路径并加OLD；run.py读取该文件进行输出保护。不要照抄旧source-f3-v2_0字段当作新实际来源。
- bwrap保持根只读、仅NEW及必要缓存/tmp可写；不让OLD整个树变可写。全局锁仍`/tmp/linkagent-phase-b-cuda0.lock`，真实GPU/PhysX最多一个。
- 新版本号、实际任务/观测/动作/奖励身份与actor迁移兼容关系在训练前确定；提交并冻结关键源码。后加任意package内py/json/npz可能改变critical source哈希，导致strict reload不兼容；诊断工具放NEW根。不要通过忽略strict检查解决。
- verify_g2_mesh.py和verify_g2_mesh_selected.py固定从工具相邻`source-f3-v2_0`导入；复制后要显式指向新源码或原只读验证源码，记录选择，保留原PAD与分类器哈希；verify_f4_all.py要绑定同一组工具。
- run_f4_frozen.py、run_f4_media.py含旧模型和输出名，不能原样重跑。make_delivery.py也绑定旧版本，必要时改新副本或简化打包。

v2.2的6个载荷余量奖励CPU测试已有通过记录。新增控制/reset/迁移/评估逻辑只补必要测试；不能用CPU通过替代PhysX。初次渲染shader编译可能数分钟高CPU，不因短时无帧就重启；先看日志和进程状态。

## 4. 真实运行参数与迁移语义

OLD/execute.py封装以下共同条件：seed110、support ramp20策略步、wrist tilt41.68575721082819°、axis(.7374624811217811,.6651274395898447,.11727991320922924)、reset果心hand坐标(81.14439249,−15.65784216,193.53402406)mm、base lift .03m、reset reference offset(.2022,.8588,−.4707)mm。实际选中支撑退出`--support-release-power 4`；6秒后严格为零，退出形状改善属于准备条件收益。

质量约0.1417312324kg、半径约.0335000008m；原驱动effort上限.2Nm，原摩擦/PD不提高。参考准备位移选中(0,0,0)。probe中间pregrasp距离与正式run不同，不能据中间参数误判实际reset坐标；核对实际pose。

三种初始化方式互斥且有不同含义：

- `--policy checkpoint --checkpoint PATH`：严格同任务、同critical source及配置完整重载/续训。初始化验证与最终视频优先此方式。
- `--policy fresh --actor-seed-checkpoint PATH`：显式actor及normalizer迁移，fresh critic/optimizer，新任务身份。跨奖励/扰动条件常需此方式；适配器支持列表要在新版本中正确校验。
- `--policy fresh --initial-action-bias-json '[14个值]'`：全新编码器＋常量输出先验；输入来自OLD/inputs/power4_index025_chosen.json，确认实际JSON结构后取14D列表，不传整个对象。这不是完整F2/v2.2 actor迁移。

fresh若未显式给初始化，现有run.py可能自动选F2 seed。不要靠默认值猜实际起点。learning-rate/agent config也参与strict身份，重载评估保持训练配置；不能为了eval随意改LR。

**完成新launcher适配后**，可用如下模板开始；TRIAL_INITIAL是已校验兼容的只读起点，轮内update数量按诊断决定：

```bash
TRIAL_INITIAL="$OLD/evidence/F3_train_margin_bounded02/update_000001.pt"
FIRM_SOURCE_NAME=source "$EP" "$NEW/execute.py" C01_train --mode train --policy fresh --actor-seed-checkpoint "$TRIAL_INITIAL" --iterations 20 --goal g2 --hold-seconds 3 --support-release-power 4 --exploration-std .0005 --learning-rate 2e-6
FIRM_SOURCE_NAME=source "$EP" "$NEW/execute.py" C01_quiet_reload --mode evaluate --policy checkpoint --checkpoint "$NEW/evidence/C01_train/update_000020.pt" --episodes 4 --goal g2 --hold-seconds 3 --support-release-power 4 --learning-rate 2e-6
```

20 update及LR/std是便于接续的例子，不是承诺最佳或硬性训练限额；checkpoint实际名称、completed_updates和新版本参数以真实产物为准。若已改变控制/观测，不得直接使用该actor seed模板，先实现明确兼容映射和输出核对。

现有run.py用disturbance-mode/axis/sign决定身份；正式force/torque评价跨身份需显式actor-only seed后，在该身份保存并严格重载复测，不能直接忽略task mismatch。matrix模块也使用显式actor seed。旧v1.6需同时显式checkpoint路径和`--legacy-thumb-baseline`恢复原观测/动作语义。

现有matrix.py固定4env，每env5项，case_id=env_id+4*ordinal，总8quiet/6force/6torque。新正式调度必须实际实现并记录，不是只改JSON文件名。先少量开发验证全部case唯一、done/retired mask和终态记录正确，再冻结正式清单。

`--max-env-steps`提前截断可能产生UNKNOWN与failure.json、exit2，不能填成已完成失败；Kit退出0也可能吞Python异常，须看summary/回执。保留失败和未完成项，不无限补跑直到分数好看。

## 5. 核验入口与录像

OLD/verify_g2.py：纯stdlib数值重算；OLD/verify_g2_mesh_selected.py：指定env/组合/end步的原网格重分类；OLD/verify_f4_all.py：完整矩阵逐case数值＋同窗原网格检查；OLD/summarize_evals.py：真实episodes/summary/metadata对比表。复制适配后可执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$EP" "$NEW/verify_g2.py" "$NEW/evidence/C01_quiet_reload" --output "$NEW/reports/C01_numeric.json"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$EP" "$NEW/verify_f4_all.py" "$NEW/evidence/F4_new_matrix" --output-dir "$NEW/reports/F4_new_verification"
```

输出须是新路径；重试用新后缀。mesh工具还依赖原完整分类器：
`/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/verification/original_pad_geometry.py`。

参考SHA256：approved PAD `abd16607249509762ac07483f9542b6b460628beee5b13a464fca31217b04258`；source_geometry NPZ `76adf67d22ac05d6eb62723796181c14513d664816d348cbb1a5ac73e358fb4c`；完整分类器 `14d7412e72af348e04d56e3452c7f9b34a135a9f2777de623e8faf4d7eea58bd`。

summarize_evals旧错误表F4_COMPARE_BEFORE_REPORT_TOOL_FIX不要用；权威旧表F4_FINAL_COMPARE。旧分类错误曾把所有F4前缀都当matrix，现已按实际matrix输入区分，复制时保持该修复。

record_video.py支持front/side、frame-every2、view-env-index；新候选选择实际成功env并保留整个回合及终态。旧录像receipt继承了thumb_pad_grasp_v1任务字段，实际metadata是firm_grasp_v2.2.0；旧receipt不得改，已在VIDEO_REVIEW说明。新版本在训练前修正记录身份，避免继续传播错误字段。绿色stem为视觉AssetBaseCfg MeshCylinder，无rigid/collision配置；是否有支撑以真实support字段和场景配置为依据，不能只按画面推断。

## 6. 交付接收

报告逐项写明F3物理能力、PPO改进、F4复核的完成状态。停止时关闭自己创建的训练进程与子Agent，不杀其他任务；留下新根绝对路径、重载命令、最好/最后权重、完整结果与下一步。用户无须接手调关节或重新标定。

异机运行时，Python绝对路径、Kit安装、资产/LFS、原分类器、冻结PAD、外盘和GPU架构均需重新核对；不能只传START_PROMPT或一个checkpoint宣称已完成迁移。
