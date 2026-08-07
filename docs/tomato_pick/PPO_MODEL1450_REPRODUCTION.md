# Tomato PPO model 1450 reproduction guide

This guide freezes the code, assets, resolved configuration, checkpoint, and evaluation evidence
needed to inspect model 1450, reproduce its reported result, and continue PPO training.

## Verified result

The selected checkpoint is:

```text
logs/rsl_rl/tomato_pick_minimal_v3/
  2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192/
  model_1450.pt
```

- SHA-256: `7886845cfe59cac4ce08d787fca36cce40d909110ba3e26fd8e18a81d41e5589`
- Checkpoint iteration: 1450
- Contents: actor, critic, optimizer state, and current iteration
- Aggregate deterministic success: **62.57%** over 23,842 completed episodes
- 95% Wilson confidence interval: **61.95%–63.18%**
- Seed 2026 success: **63.65%** (the rounded 64% headline result)

| seed | success | completed episodes |
|---:|---:|---:|
| 17 | 61.18% | 7,991 |
| 42 | 62.88% | 7,953 |
| 2026 | 63.65% | 7,898 |

The aggregate results and per-seed JSON files are in
`logs/evaluations/tomato_minimal_v3_2026-08-07_fixed/`. The exact resolved training
configuration is stored beside the checkpoint under `params/agent.yaml` and `params/env.yaml`.

## Frozen environment

The recorded successful environment was:

- Ubuntu 24.04.3 LTS
- NVIDIA GeForce RTX 5090, driver 595.58.03
- Python 3.12.3
- Isaac Lab source tag `v3.0.0-beta2.patch1`, commit
  `ffff603eafc6b74264a5261cc0183d6a65390d78`
- Isaac Sim 6.0.1.0
- Torch 2.10.0+cu128
- RSL-RL 5.0.1
- W&B 0.28.1

The full recorded runtime manifest is `simulation/environments/isaaclab_runtime_v1.yaml`.
Install that Isaac Lab tag with its official installer, including NVIDIA's required Isaac Sim
license acceptance. Do not copy the original 13 GiB virtual environment into this repository.

From the repository root, point LinkAgent at the installation:

```bash
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m script.sim_env preflight --role isaac_ppo
```

The checked-in robot asset is
`botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/`.
The environment creates the tomato and compliant stem procedurally.

## Reproduce the 64% seed result

```bash
"$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python" \
  script/evaluate_tomato_checkpoint.py \
  --checkpoint logs/rsl_rl/tomato_pick_minimal_v3/2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192/model_1450.pt \
  --output logs/evaluations/reproduction/model_1450_seed_2026.json \
  --num_envs 1024 \
  --steps 400 \
  --seed 2026 \
  --viz none
```

Expected success is about 63.65%. GPU physics can introduce small platform-dependent variation;
the checked-in JSON and checkpoint digest are the release evidence.

To rerun all three fixed seeds:

```bash
python3.12 -m script.evaluate_tomato_sweep \
  --run-dir logs/rsl_rl/tomato_pick_minimal_v3/2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192 \
  --output-dir logs/evaluations/reproduction \
  --iterations 1450 \
  --seeds 17 42 2026 \
  --num-envs 1024 \
  --steps 400
```

## Continue training from model 1450

Authenticate W&B in the Isaac Lab environment, either with `WANDB_API_KEY` or its `wandb login`
command. First print and review a 50-iteration continuation plan:

```bash
python3.12 -m script.train_tomato_ppo \
  --isaaclab-root "$LINKAGENT_ISAACLAB_ROOT" \
  --resume-run 2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192 \
  --resume-checkpoint model_1450.pt \
  --max-iterations 50
```

When the printed plan reports `"ready": true`, run the same command with `--execute`:

```bash
python3.12 -m script.train_tomato_ppo \
  --isaaclab-root "$LINKAGENT_ISAACLAB_ROOT" \
  --resume-run 2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192 \
  --resume-checkpoint model_1450.pt \
  --max-iterations 50 \
  --execute
```

`--max-iterations` is the number of additional PPO iterations in a resumed run. The default formal
profile uses 8,192 parallel environments, a 32-step rollout, and seed 42; lower `--num-envs` only
when the GPU cannot hold the recorded batch.

## Release files

The model release includes:

- `model_1450.pt`: resumable RSL-RL checkpoint;
- `params/agent.yaml` and `params/env.yaml`: exact resolved training configuration;
- `events.out.tfevents.*`: local TensorBoard training history;
- `exported/policy.pt` and `exported/policy.onnx*`: inference exports;
- `logs/evaluations/tomato_minimal_v3_2026-08-07_fixed/`: fixed-seed evaluation evidence;
- `simulation/`, `script/`, `module/`, and the frozen USD asset: executable task source.

The original W&B run ID is `pmt88pns`. W&B is supplementary; the repository contains the files
needed to inspect, evaluate, and resume without access to that dashboard.
