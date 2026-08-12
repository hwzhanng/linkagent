# Physics-v1 perf-v7 final reproducibility release (iter 5150)

This release freezes the complete accepted Physics-v1 PPO history through the
user-authorized stop at the verified full-state checkpoint `model_5150.pt`.
It extends the original iteration-1200 release without changing physics,
reward, MDP, PPO, seed, environment count, rollout length, thresholds, or
learning rate.

The result is **not** a claim that the original 90% success gate passed. The
selected checkpoint reached 85.9715% fixed deterministic multi-seed success.
The user explicitly stopped the run after reviewing the curve and checkpoint
plateau.

## Final result

- task: `LinkAgent-TomatoPick-Physics-v1`;
- accepted training labels: 0 through 5150, with resume-boundary duplicates
  replaced by the later full-state segment;
- selected checkpoint: `model_5150.pt`;
- checkpoint SHA-256:
  `655806f86a1bcf9e1472c064a5580523f6d8344d94a04c39f07112dcdcceacb0`;
- checkpoint bytes: 4,912,949;
- fixed evaluation: deterministic policy, seeds 17/42/2026, 1,024
  environments and 400 simulator steps per seed;
- completed episodes: 20,102;
- success: 17,282 / 20,102 = 85.9715%;
- 95% Wilson interval: 85.4846%-86.4448%;
- drop: 1,363 / 20,102 = 6.7804%;
- worst-seed success: 85.6218%;
- cross-seed spread: 0.9301 percentage points;
- runner-up: `model_4995.pt`, 85.9604% success and 7.0148% drop;
- original target: 90%, **not reached**;
- final curve comparison: latest 100 labels 77.4907% versus previous 100
  labels 76.1276%, a +1.3631 percentage-point gain;
- shutdown: user-authorized, not formal curve-and-target convergence.

The selection rule ranks aggregate fixed-evaluation success first. Candidates
within 0.1 percentage point are treated as tied, then ranked by lower drop,
higher worst-seed success, and later iteration. Iter 5150 therefore wins the
effective tie with iter 5000.

## Published artifacts

The native RSL-RL run directory contains the selected checkpoint and exact
resolved runner state needed for a future authorized full-state continuation:

```text
logs/rsl_rl/tomato_pick_physics_v1/
  2026-08-12_09-39-56_tomato_pick_physics_v1_resume5000_to5400_8192/
    model_5150.pt
    params/agent.yaml
    params/env.yaml
    events.out.tfevents.1786499033.ubuntu.56253.0
    git/IsaacLab-3.0.0-beta2.patch1.diff
```

The audit and visualization package is under:

```text
logs/qualification/tomato_physics_v1/convergence_report/
```

It includes the self-contained HTML report, portable report artifact, combined
5,151-row training curve, all fixed checkpoint comparisons, per-seed metrics,
selection manifest, source inventory, and an identical convenience copy of the
selected checkpoint. Git stores the two identical checkpoint paths as one blob.

All nine fixed checkpoint evaluations are preserved as aggregate and per-seed
JSON under `logs/qualification/tomato_physics_v1/convergence_eval_*`. Raw local
W&B directories and redundant intermediate checkpoints are intentionally not
versioned.

## Training and W&B lineage

Every transition restored Actor, Critic, actor and critic observation
normalizers, policy log standard deviation, and Adam optimizer state.

| Accepted labels | W&B run | Boundary checkpoint |
| --- | --- | --- |
| 0-1200 | `hs0bad5e` | `model_1200.pt` |
| 1200-1799 | `u5dgy4u1` | `model_1799.pt` |
| 1799-2198 | `e9wszrmu` | `model_2198.pt` |
| 2198-2597 | `s0hjif5g` | `model_2597.pt` |
| 2597-2996 | `grhe7qkz` | `model_2996.pt` |
| 2996-3250 | `bo3xjhiy` | `model_3250.pt` |
| 3250-3399 | `k4twbypd` | `model_3399.pt` |
| 3399-3798 | `ijfzmal7` | `model_3798.pt` |
| 3798-4197 | `qk2u8c52` | `model_4197.pt` |
| 4197-4596 | `8vgxdtza` | `model_4596.pt` |
| 4596-4995 | `ep6hnmsc` | `model_4995.pt` |
| 4995-5150 | `1c93cmsp` | `model_5150.pt` |

The immutable W&B continuation runs are linked from
`logs/qualification/tomato_physics_v1/reproducibility_manifest_iter5150.json`.
W&B created new auditable run IDs when an existing run rejected a changed
resume `train_cfg`; this did not change the restored PPO state.

## Environment

Use the managed Isaac Lab Python. System Python or an unrelated Conda
environment may not contain the required simulator, Gymnasium, or Torch build.

```bash
cd /absolute/path/to/linkagent
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CONDA_PREFIX
ISAAC_PYTHON="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
```

The recorded Isaac Lab checkout was clean at commit
`ffff603eafc6b74264a5261cc0183d6a65390d78`. No SDK, CAN, or real hardware
was connected.

## Verify the release

Verify the published file identities from the repository root:

```bash
sha256sum -c \
  logs/qualification/tomato_physics_v1/PHYSICS_V1_PERF_V7_ITER5150_SHA256SUMS
```

Verify the full-state checkpoint structure:

```bash
CHECKPOINT=logs/rsl_rl/tomato_pick_physics_v1/2026-08-12_09-39-56_tomato_pick_physics_v1_resume5000_to5400_8192/model_5150.pt
"$ISAAC_PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
missing = sorted(required.difference(payload))
assert not missing, missing
assert int(payload["iter"]) == 5150
actor = payload["actor_state_dict"]
critic = payload["critic_state_dict"]
for key in ("obs_normalizer._mean", "obs_normalizer._var", "distribution.log_std_param"):
    assert key in actor, key
for key in ("obs_normalizer._mean", "obs_normalizer._var"):
    assert key in critic, key
assert payload["optimizer_state_dict"]["state"]
print("verified full-state checkpoint model_5150.pt")
PY
```

Run the lightweight source tests with no other Isaac task active:

```bash
"$ISAAC_PYTHON" -m pytest -q tests/test_tomato_physics_v1.py
"$ISAAC_PYTHON" -m compileall -q \
  simulation/tomato_pick/isaac/physics_v1 \
  script/*tomato_physics_v1*.py
```

## Reproduce the fixed evaluation

Evaluate serially; never run these Isaac jobs concurrently:

```bash
CHECKPOINT=logs/rsl_rl/tomato_pick_physics_v1/2026-08-12_09-39-56_tomato_pick_physics_v1_resume5000_to5400_8192/model_5150.pt
export TOMATO_REWARD_ITERATION_OFFSET=5150
for seed in 17 42 2026; do
  "$ISAAC_PYTHON" script/evaluate_tomato_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --output "/tmp/model_5150_seed_${seed}.json" \
    --num_envs 1024 \
    --steps 400 \
    --seed "$seed" \
    --task LinkAgent-TomatoPick-Physics-v1 \
    --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
    --viz none
done
```

Compare the outputs with the published JSON under
`convergence_eval_5150_pause/`. The fixed evaluator is deterministic for the
recorded software and GPU stack, but different simulator/CUDA builds should be
reported rather than silently treated as byte-identical environments.

## Reproduce the report data

The published combined CSV is the authoritative accepted curve. It removes
uncheckpointed labels 3251-3263 and 5151-5153 and resolves overlapping resume
labels in favor of the later full-state segment. Rebuild the report artifact
without relying on ephemeral `/tmp` logs:

```bash
python3 script/build_tomato_physics_v1_convergence_report.py \
  --training-csv logs/qualification/tomato_physics_v1/convergence_report/combined_training_metrics.csv \
  --generated-at 2026-08-12T03:00:09.823076Z \
  --output-dir /tmp/physics_v1_convergence_report_rebuild
```

With the published inputs and timestamp this reproduces `artifact.json` and
`best_checkpoint_selection.json` byte-for-byte. The self-contained HTML is the
reviewed rendering of that artifact and has its own published SHA-256.

## Future full-state continuation contract

The release is stopped. Do not resume unless a user explicitly authorizes new
training. If authorized, restore from the native iter-5150 run and preserve all
frozen variables:

```bash
export TOMATO_REWARD_ITERATION_OFFSET=5150
unset WANDB_RUN_ID WANDB_RESUME
"$ISAAC_PYTHON" "$LINKAGENT_ISAACLAB_ROOT/scripts/reinforcement_learning/train.py" \
  --rl_library rsl_rl \
  --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
  --task LinkAgent-TomatoPick-Physics-v1 \
  --num_envs 8192 \
  --seed 42 \
  --max_iterations 400 \
  --logger wandb \
  --experiment_name tomato_pick_physics_v1 \
  --run_name tomato_pick_physics_v1_resume5150_to5550_8192 \
  --resume \
  --load_run '^2026-08-12_09-39-56_tomato_pick_physics_v1_resume5000_to5400_8192$' \
  --checkpoint '^model_5150\.pt$' \
  --viz none
```

Never load policy weights alone: the continuation contract requires Actor,
Critic, both normalizers, policy standard deviation, and Adam state.
