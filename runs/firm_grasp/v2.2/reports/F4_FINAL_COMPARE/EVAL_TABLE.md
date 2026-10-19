# F3/F4 Episode 汇总

本表逐行读取各评估目录的 `episodes.jsonl`，按实际 episode 分母汇总。它记录观察到的结果，不证明 force closure、学习收益或可靠性显著性。诊断 early-window_screen 不计作完整 G2 evaluation。

生成时间（UTC）：2026-10-18T12:50:22+00:00

## legacy_v16

类型：`F4_MATRIX`；完整性：**COMPLETE**

| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |
|---|---:|---:|---:|---:|
| quiet | 8 | 0 | 8 | 0 |
| force | 6 | 0 | 6 | 0 |
| torque | 6 | 0 | 6 | 0 |
| unclassified | 0 | 0 | 0 | 0 |
| 总计 | 20 | 0 | 20 | 0 |

F4 case 完整性：唯一 case 20/20；缺失 `[]`；重复 `[]`；超范围 `[]`；非整数或空 ID 行 0。
实际场景分布：`{"quiet": 8, "force": 6, "torque": 6}`（期望 quiet/force/torque = 8/6/6）。
scenario 与实际 f3_protocol 不一致/缺失：0 行；逐 case 的 scenario 与协议见 JSON。

Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：

```json
{
  "policy": "fresh",
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "actor_seed_checkpoint_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/baselines/ppo-thumb-pad-v1_6-frozen-20261017/thumb_pad_ppo_v1_6_update1.pt",
  "actor_warmstart_path": null,
  "warmstart_action_offset": null,
  "initialization": {
    "kind": "explicit_actor_prefix_transfer_from_frozen_v1_6_plus_clock",
    "classification": "frozen_legacy_thumb_actor_only_seed_new_identity",
    "new_task_identity": "30f13fe87d0bdcd549aab1fa28917250160135d78611d8cbd31b44f336707bb7"
  },
  "declared_checkpoint_files": [
    {
      "declared_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/baselines/ppo-thumb-pad-v1_6-frozen-20261017/thumb_pad_ppo_v1_6_update1.pt",
      "sha256": "0650ac4b35e742430f910319492d8ada621ee1c25f954e8aba6bd8163e59ddf6",
      "status": "HASHED_READ_ONLY"
    }
  ]
}
```

失败与未知 episode：

| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |
|---:|---:|---:|---|---|---|---|---|
| 1 | 0 | 0 | quiet | FAILURE | timeout | 2 | late |
| 2 | 1 | 1 | quiet | FAILURE | timeout | 2 | late |
| 3 | 2 | 2 | quiet | FAILURE | timeout | 2 | late |
| 4 | 3 | 3 | quiet | FAILURE | timeout | 2 | late |
| 5 | 4 | 0 | quiet | FAILURE | timeout | 2 | late |
| 6 | 5 | 1 | quiet | FAILURE | timeout | 2 | late |
| 7 | 6 | 2 | quiet | FAILURE | timeout | 2 | late |
| 8 | 7 | 3 | quiet | FAILURE | timeout | 2 | late |
| 9 | 9 | 1 | force | FAILURE | drop | 3 | late |
| 10 | 8 | 0 | force | FAILURE | timeout | 2 | late |
| 11 | 10 | 2 | force | FAILURE | timeout | 2 | late |
| 12 | 11 | 3 | force | FAILURE | timeout | 2 | late |
| 13 | 14 | 2 | torque | FAILURE | drop | 3 | late |
| 14 | 13 | 1 | force | FAILURE | timeout | 2 | late |
| 15 | 12 | 0 | force | FAILURE | timeout | 2 | late |
| 16 | 15 | 3 | torque | FAILURE | timeout | 2 | late |
| 17 | 18 | 2 | torque | FAILURE | timeout | 2 | late |
| 18 | 17 | 1 | torque | FAILURE | timeout | 2 | late |
| 19 | 16 | 0 | torque | FAILURE | timeout | 2 | late |
| 20 | 19 | 3 | torque | FAILURE | timeout | 2 | late |

JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。

## initial

类型：`F4_MATRIX`；完整性：**COMPLETE**

| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |
|---|---:|---:|---:|---:|
| quiet | 8 | 2 | 6 | 0 |
| force | 6 | 1 | 5 | 0 |
| torque | 6 | 0 | 6 | 0 |
| unclassified | 0 | 0 | 0 | 0 |
| 总计 | 20 | 3 | 17 | 0 |

F4 case 完整性：唯一 case 20/20；缺失 `[]`；重复 `[]`；超范围 `[]`；非整数或空 ID 行 0。
实际场景分布：`{"quiet": 8, "force": 6, "torque": 6}`（期望 quiet/force/torque = 8/6/6）。
scenario 与实际 f3_protocol 不一致/缺失：0 行；逐 case 的 scenario 与协议见 JSON。

Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：

```json
{
  "policy": "fresh",
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "actor_seed_checkpoint_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/initial.pt",
  "actor_warmstart_path": null,
  "warmstart_action_offset": null,
  "initialization": {
    "kind": "actor_only_seed_fresh_critic_optimizer_new_identity",
    "new_task_identity": "42e651b2e422eab546f230c34a7a5d60ebe16428685f85d84be8e6b88223ef73"
  },
  "declared_checkpoint_files": [
    {
      "declared_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/initial.pt",
      "sha256": "b5335e101993fa86d4a9fbad9b37b484f4e54db637f1b5a7b939d6a64af1b683",
      "status": "HASHED_READ_ONLY"
    }
  ]
}
```

失败与未知 episode：

| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |
|---:|---:|---:|---|---|---|---|---|
| 2 | 1 | 1 | quiet | FAILURE | timeout | 2 | late |
| 3 | 2 | 2 | quiet | FAILURE | timeout | 2 | late |
| 4 | 3 | 3 | quiet | FAILURE | timeout | 2 | late |
| 7 | 5 | 1 | quiet | FAILURE | timeout | 2 | late |
| 8 | 6 | 2 | quiet | FAILURE | timeout | 2 | late |
| 9 | 7 | 3 | quiet | FAILURE | timeout | 2 | late |
| 10 | 12 | 0 | force | FAILURE | timeout | 2 | late |
| 11 | 9 | 1 | force | FAILURE | timeout | 2 | late |
| 12 | 10 | 2 | force | FAILURE | timeout | 2 | late |
| 13 | 11 | 3 | force | FAILURE | timeout | 2 | late |
| 14 | 16 | 0 | torque | FAILURE | timeout | 2 | late |
| 15 | 13 | 1 | force | FAILURE | timeout | 2 | late |
| 16 | 14 | 2 | torque | FAILURE | timeout | 2 | late |
| 17 | 15 | 3 | torque | FAILURE | timeout | 2 | late |
| 18 | 17 | 1 | torque | FAILURE | timeout | 2 | late |
| 19 | 18 | 2 | torque | FAILURE | timeout | 2 | late |
| 20 | 19 | 3 | torque | FAILURE | timeout | 2 | late |

JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。

## updated

类型：`F4_MATRIX`；完整性：**COMPLETE**

| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |
|---|---:|---:|---:|---:|
| quiet | 8 | 1 | 7 | 0 |
| force | 6 | 2 | 4 | 0 |
| torque | 6 | 0 | 6 | 0 |
| unclassified | 0 | 0 | 0 | 0 |
| 总计 | 20 | 3 | 17 | 0 |

F4 case 完整性：唯一 case 20/20；缺失 `[]`；重复 `[]`；超范围 `[]`；非整数或空 ID 行 0。
实际场景分布：`{"quiet": 8, "force": 6, "torque": 6}`（期望 quiet/force/torque = 8/6/6）。
scenario 与实际 f3_protocol 不一致/缺失：0 行；逐 case 的 scenario 与协议见 JSON。

Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：

```json
{
  "policy": "fresh",
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "actor_seed_checkpoint_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/update_000001.pt",
  "actor_warmstart_path": null,
  "warmstart_action_offset": null,
  "initialization": {
    "kind": "actor_only_seed_fresh_critic_optimizer_new_identity",
    "new_task_identity": "42e651b2e422eab546f230c34a7a5d60ebe16428685f85d84be8e6b88223ef73"
  },
  "declared_checkpoint_files": [
    {
      "declared_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/update_000001.pt",
      "sha256": "af961f8ea500943278de77bc2683eaa12ea49e4ce742e839b0182eee48a660d9",
      "status": "HASHED_READ_ONLY"
    }
  ]
}
```

失败与未知 episode：

| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |
|---:|---:|---:|---|---|---|---|---|
| 2 | 0 | 0 | quiet | FAILURE | timeout | 2 | late |
| 3 | 1 | 1 | quiet | FAILURE | timeout | 2 | late |
| 4 | 3 | 3 | quiet | FAILURE | timeout | 2 | late |
| 5 | 6 | 2 | quiet | FAILURE | timeout | 2 | late |
| 6 | 4 | 0 | quiet | FAILURE | timeout | 2 | late |
| 7 | 5 | 1 | quiet | FAILURE | timeout | 2 | late |
| 8 | 7 | 3 | quiet | FAILURE | timeout | 2 | late |
| 11 | 8 | 0 | force | FAILURE | timeout | 2 | late |
| 12 | 9 | 1 | force | FAILURE | timeout | 2 | late |
| 13 | 14 | 2 | torque | FAILURE | timeout | 2 | late |
| 14 | 15 | 3 | torque | FAILURE | timeout | 2 | late |
| 15 | 12 | 0 | force | FAILURE | timeout | 2 | late |
| 16 | 13 | 1 | force | FAILURE | timeout | 2 | late |
| 17 | 18 | 2 | torque | FAILURE | timeout | 2 | late |
| 18 | 19 | 3 | torque | FAILURE | timeout | 2 | late |
| 19 | 16 | 0 | torque | FAILURE | timeout | 2 | late |
| 20 | 17 | 1 | torque | FAILURE | timeout | 2 | late |

JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。

## open

类型：`NON_MATRIX_EPISODES`；完整性：**COMPLETE**

| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |
|---|---:|---:|---:|---:|
| quiet | 4 | 0 | 4 | 0 |
| force | 0 | 0 | 0 | 0 |
| torque | 0 | 0 | 0 | 0 |
| unclassified | 0 | 0 | 0 | 0 |
| 总计 | 4 | 0 | 4 | 0 |

Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：

```json
{
  "policy": "open",
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "actor_seed_checkpoint_path": null,
  "actor_warmstart_path": null,
  "warmstart_action_offset": null,
  "initialization": {
    "kind": "fresh_rsl_rl_initialization",
    "new_task_identity": "2620b17d6ec3cc0b311200698ccc46516ce3a9b5e8aceb6fe8f5036bd317db8d"
  },
  "declared_checkpoint_files": []
}
```

失败与未知 episode：

| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |
|---:|---:|---:|---|---|---|---|---|
| 1 | null | 0 | quiet | FAILURE | drop | 3 | late |
| 2 | null | 1 | quiet | FAILURE | drop | 3 | late |
| 3 | null | 2 | quiet | FAILURE | drop | 3 | late |
| 4 | null | 3 | quiet | FAILURE | drop | 3 | late |

JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。

## torque_reload02

类型：`NON_MATRIX_EPISODES`；完整性：**COMPLETE**

| 场景 | 分母 | SUCCESS | FAILURE | UNKNOWN |
|---|---:|---:|---:|---:|
| quiet | 0 | 0 | 0 | 0 |
| force | 0 | 0 | 0 | 0 |
| torque | 4 | 1 | 3 | 0 |
| unclassified | 0 | 0 | 0 | 0 |
| 总计 | 4 | 1 | 3 | 0 |

Checkpoint identity（取自 run_metadata / initialization；缺失字段为 null）：

```json
{
  "policy": "fresh",
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "actor_seed_checkpoint_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/update_000001.pt",
  "actor_warmstart_path": null,
  "warmstart_action_offset": null,
  "initialization": {
    "kind": "actor_only_seed_fresh_critic_optimizer_new_identity",
    "new_task_identity": "03d9a1844b0c3c61014a8e3e08ba49b8df82c2db41a5bdc0813a1501999a6391"
  },
  "declared_checkpoint_files": [
    {
      "declared_path": "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-firm-grasp-f3-f4-20261018/evidence/F3_train_margin_bounded02/update_000001.pt",
      "sha256": "af961f8ea500943278de77bc2683eaa12ea49e4ce742e839b0182eee48a660d9",
      "status": "HASHED_READ_ONLY"
    }
  ]
}
```

失败与未知 episode：

| 行 | case_id | env_id | 场景 | outcome | reason | reason_code | failure_phase |
|---:|---:|---:|---|---|---|---|---|
| 2 | null | 0 | torque | FAILURE | timeout | 2 | late |
| 3 | null | 1 | torque | FAILURE | timeout | 2 | late |
| 4 | null | 3 | torque | FAILURE | timeout | 2 | late |

JSON 中的 `failures_and_unknowns` 保留对应 episode 原始记录。
