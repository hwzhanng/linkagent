# 52–80 mm 扩域 E0 工具实现记录

日期：2026-09-20（Asia/Shanghai）  
结论：**代码与单元门通过；PhysX/model150 门尚未通过；不得启动 E1**

## 已实现

- `--diameter-bins-mm`：以左闭右开的等效直径桶定义采样域；
- `--diameter-bin-weights`：用最大余数法生成总数严格守恒的环境配额；
- `--geometry-cohort-id`：与 seed 组合为跨进程稳定的独立几何 cohort；
- 先采 `D_eq`、再采 B0 形状比例并按几何平均半轴归一化，避免宽区间向中部聚集；
- prestartup 保存 `scenario_id/bin/D_eq/a/b/c/density/mass`；
- `--first-episode-evidence-json`：每个环境只收首个完整回合，输出互斥终止原因、逐桶
  成功率与 Wilson 95% 区间、checkpoint SHA-256 和场景清单 SHA-256；
- forbidden-first-contact 使用独立终止编码 `5`，不再混入 timeout。

首回合证据当前明确要求 `--guided-residual`，因为该环境保存 DirectRLEnv 自动 reset
前的终止力、保持年龄和验证比例快照。该限制会在参数检查阶段 fail closed。

## 验证结果

```text
pytest research/small_fruit_pinch_v1/source/tests
9 passed, 50 warnings

python -m compileall -q <E0 changed Python files>
PASS

git diff --check
PASS
```

覆盖的工程断言包括：14 桶等权时每桶精确 10/140、固定 seed 逐张量相同、不同 cohort
seed 分离、`D_eq` 不越桶、椭球质量公式正确、质量/轴比约束成立、重叠桶拒绝、首回合
终止原因互斥，以及 Wilson 区间计算。

## 正式运行模板

14 桶零样本证据的核心参数如下；恢复 checkpoint 后再补入与 B0 完全一致的 guided
控制参数，不得仅凭该片段直接宣称合格。

```bash
$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python \
  research/small_fruit_pinch_v1/source/native_evaluate.py \
  --checkpoint <restored-model-150.pt> \
  --guided-residual \
  --num-envs 280 \
  --episodes 280 \
  --seed <frozen-e0-seed> \
  --geometry-cohort-id e0-zero-shot-c01 \
  --diameter-bins-mm 52:54,54:56,56:58,58:60,60:62,62:64,64:66,66:68,68:70,70:72,72:74,74:76,76:78,78:80 \
  --first-episode-evidence-json <output.json> \
  --headless
```

## 尚未通过的硬门

1. 当前工作树没有 `logs/rsl_rl/`，历史 model150 文件不可用；
2. 因此未运行 2 环境 PhysX smoke、64–70 mm 新旧采样器成功率差异、B0 掉落/过载/
   平滑性回归和传感器负对照；
3. 原 checkpoint 恢复后必须先校验历史 SHA-256
   `e39cd14d18912da68638b21b2c16a9d48c3867b833d4af0296a9e03fd5ce0c8d`；
4. 上述门全部通过前，状态保持 `E0 pending`，禁止训练 E1。
