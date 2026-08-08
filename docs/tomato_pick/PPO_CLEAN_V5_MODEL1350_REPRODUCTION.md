# Tomato PPO Clean-v5 model 1350 reproduction guide

This guide freezes the source, robot asset, resolved training configuration,
resumable checkpoint, deterministic policy exports, and fixed evaluation
evidence needed to inspect the Clean-v5 result, rerun it, and continue PPO
training.

## Verified release result

The selected checkpoint is:

```text
logs/rsl_rl/tomato_pick_clean_v5/
  2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch/
  model_1350.pt
```

- Task: `LinkAgent-TomatoPick-Clean-v5`
- SHA-256: `c4ca9ac27aaa20806b630496362bcb9154d1e8076605f9d7026b58f719d124d0`
- Checkpoint iteration: 1350
- Contents: actor, critic, optimizer state, and current iteration
- Aggregate deterministic success: **96.1805%** (25,081 / 26,077 episodes)
- 95% Wilson confidence interval: **95.9410%–96.4065%**
- Detach: **99.7929%**; retained: **96.2036%**; drop: **2.1015%**

| seed | success | completed episodes |
|---:|---:|---:|
| 17 | 96.0623% | 8,736 |
| 42 | 96.2771% | 8,676 |
| 2026 | 96.2031% | 8,665 |

The seed spread is only 0.2148 percentage points. Evaluation covered checkpoints
1300 through 1600; model 1350 was selected by aggregate success. Model 1600
reached 95.6390%, so the final checkpoint was not silently substituted for the
best checkpoint.

The authoritative selection record is
`logs/evaluations/tomato_pick_clean_v5_2026-08-08_fixed/release_manifest.json`.
It records deterministic inference, full environment randomization, all three
seeds, every candidate checkpoint, confidence bounds, failure categories, and
the passing selection gate.

## Frozen runtime

The recorded successful environment was:

- Ubuntu 24.04.3 LTS
- NVIDIA GeForce RTX 5090, driver 595.58.03
- Python 3.12.3
- Isaac Lab `v3.0.0-beta2.patch1`, commit
  `ffff603eafc6b74264a5261cc0183d6a65390d78`
- Isaac Sim 6.0.1.0
- Torch 2.10.0+cu128
- RSL-RL 5.0.1
- W&B 0.28.1

The runtime manifest is `simulation/environments/isaaclab_runtime_v1.yaml`.
Install that Isaac Lab revision using its official installer and accept the
required NVIDIA Isaac Sim license. The 13 GiB virtual environment is not copied
into Git.

From the repository root:

```bash
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m script.sim_env preflight --role isaac_ppo
```

The resolved training configuration lives beside the checkpoint in
`params/agent.yaml` and `params/env.yaml`. The robot USD is checked in under
`botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/`;
the environment creates the tomato and compliant stem procedurally.

## Verify the exact files first

```bash
sha256sum \
  logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch/model_1350.pt \
  artifacts/simulation/tomato_pick_clean_v5/model_1350_best/source_checkpoint.pt
```

Both lines must print
`c4ca9ac27aaa20806b630496362bcb9154d1e8076605f9d7026b58f719d124d0`.
The packaged skill manifest additionally binds every inference artifact and
evaluation file to its SHA-256.

## Reproduce one fixed seed

```bash
export TOMATO_REWARD_ITERATION_OFFSET=1350
"$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python" \
  script/evaluate_tomato_checkpoint.py \
  --task LinkAgent-TomatoPick-Clean-v5 \
  --checkpoint logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch/model_1350.pt \
  --output logs/evaluations/reproduction/model_1350_seed_2026.json \
  --num_envs 1024 \
  --steps 400 \
  --seed 2026 \
  --viz none
```

The recorded seed-2026 success is 96.2031%. GPU physics may introduce small
platform-dependent variation; the checked-in JSON, environment manifest, and
checkpoint digest are the release evidence.

Rerun the complete fixed sweep:

```bash
python3.12 -m script.evaluate_tomato_sweep \
  --run-dir logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch \
  --output-dir logs/evaluations/reproduction \
  --iterations 1300 1350 1400 1450 1500 1550 1600 \
  --seeds 17 42 2026 \
  --num-envs 1024 \
  --steps 400 \
  --wandb-run-id 1vp9dp6x \
  --wandb-url https://wandb.ai/hwzclementine-china-agricultural-university/PPO/runs/1vp9dp6x
```

## Continue training from model 1350

Authenticate W&B in the Isaac Lab environment with `WANDB_API_KEY` or
`wandb login`. Clean-v5 schedules reward alignment by absolute PPO iteration,
so a resumed run must set its iteration offset. First review a 50-iteration
continuation plan:

```bash
export TOMATO_REWARD_ITERATION_OFFSET=1350
python3.12 -m script.train_tomato_ppo \
  --isaaclab-root "$LINKAGENT_ISAACLAB_ROOT" \
  --resume-run 2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch \
  --resume-checkpoint model_1350.pt \
  --max-iterations 50
```

When the plan reports `"ready": true`, add `--execute` to the same command.
`--max-iterations` is the number of additional PPO iterations. The frozen
profile uses 8,192 environments, 32 rollout steps per environment, and seed 42;
only lower `--num-envs` when the GPU cannot hold the recorded batch.

## Direct offline inference and deployment gate

The promoted package contains deterministic TorchScript and `torch.export`
actors, inference canaries, the resumable source checkpoint, and training
parameters:

```bash
python script/check_tomato_deployment.py \
  --skill-profile config/skills/tomato_pick_clean_v5_model1350_best.json \
  --mode simulation \
  --minimum-success-rate 0.90
```

This validates the package hashes, interface layouts, fixed three-seed release
gate, policy canaries, and independent-verifier requirement. Hardware remains
fail-closed until real G20 calibration, vendor shadow validation, and explicit
graded release are complete.

## Included release evidence

- resumable `model_1350.pt`, resolved `agent.yaml` and `env.yaml`;
- local TensorBoard events, deterministic exports, and an example play video;
- full fixed evaluation JSON/logs for seven checkpoints × three seeds;
- selection manifest, Wilson interval, and terminal failure classification;
- promoted model-1350 package plus unpromoted model-1600 comparison candidate;
- executable environment, reward, export, promotion, preflight, and verifier
  source with offline tests.

The original W&B run ID is `1vp9dp6x`. W&B is supplementary: the repository
contains the evidence and files required for inspection, evaluation, inference,
and training continuation without dashboard access.
