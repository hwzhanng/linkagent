# Physics-v1 perf-v7 incomplete run: checkpoint 1200

> Historical handoff: this run was later resumed without changing the PPO or
> environment contract and stopped by user authorization at the best tested
> full-state checkpoint, iter 5150. See
> `PHYSICS_V1_PERF_V7_FINAL_5150.md` for the final reproducibility release.

This handoff deliberately publishes an **unfinished** formal training run. It
is a recovery point for continuing the same experiment, not a selected model,
fixed-seed result, deployable skill, or replacement for the released Clean-v5
model 1350.

## Durable recovery point

The formal run started from scratch with the perf-v7 Physics-v1 source:

```text
logs/rsl_rl/tomato_pick_physics_v1/
  2026-08-10_03-22-34_tomato_pick_physics_v1_from_scratch_8192/
  model_1200.pt
```

- task: `LinkAgent-TomatoPick-Physics-v1`;
- intended schedule: 1,800 PPO updates;
- last completed log entry: iteration 1208;
- last atomic checkpoint: iteration 1200;
- uncheckpointed iterations 1201–1208 are intentionally discarded;
- checkpoint SHA-256:
  `d757dcb6bd6e0b7328ebef53dd049620d5457c8b54857a95ad6083c41a2bd83c`;
- W&B run: `hs0bad5e`;
- hardware boundary: no SDK, CAN, or real robot was accessed.

The checkpoint contains actor, critic, both observation normalizers, learned
policy log standard deviation, optimizer state, and `iter=1200`. The resolved
`agent.yaml`, `env.yaml`, TensorBoard event history, and Isaac Lab Git state are
stored beside it. The machine-readable pause and continuation record is
`logs/qualification/tomato_physics_v1/formal_training_pause_iter1200.json`.

The training process is no longer running. Its output ended normally in the
middle of the schedule without a Python/CUDA exception or a clean 1800-update
completion record. Therefore only `model_1200.pt` is authoritative.

## Observed training signal, not a release evaluation

At TensorBoard step 1200 the online rollout metrics were:

| metric | value |
|---|---:|
| success rate | 40.7379% |
| retained rate | 41.6353% |
| detach rate | 88.0447% |
| drop rate | 33.8255% |
| mean hold steps | 8.2257 |

The log continued to step 1208, where success was 41.6606%. These are rolling
training metrics from the current randomized rollout distribution. They are
not deterministic fixed-seed evaluation results and must not be quoted as the
final model quality. After update 1800, checkpoint selection still requires a
separate multi-seed evaluation sweep.

## Frozen environment and source identity

- Ubuntu 24.04.3 LTS;
- NVIDIA GeForce RTX 5090;
- Python 3.12.3;
- Isaac Lab `v3.0.0-beta2.patch1`;
- Isaac Sim 6.0.1.0;
- Torch 2.10.0+cu128;
- RSL-RL 5.0.1;
- 8,192 environments, 32 rollout steps, seed 42;
- physics at 120 Hz, control decimation 6, solver iterations 8/4;
- actor/critic/action contract: 66D / 83D / 11D.

The updated perf-v7 qualification gates and exact source hashes are frozen in
`logs/qualification/tomato_physics_v1/formal_training_manifest.json`. They
cover the versioned USD payloads, collision/contact representation, stem-load
duration, 8192-environment performance gate, PPO numerical gate, and first-
success gate. Changing any physics, reward, observation, action, hand mapping,
or asset input invalidates this checkpoint and requires a new profile/run.

## Verify before resuming

From the repository root:

```bash
sha256sum \
  logs/rsl_rl/tomato_pick_physics_v1/2026-08-10_03-22-34_tomato_pick_physics_v1_from_scratch_8192/model_1200.pt
```

The digest must be
`d757dcb6bd6e0b7328ebef53dd049620d5457c8b54857a95ad6083c41a2bd83c`.
Then set up the recorded runtime and run the pure/unit checks:

```bash
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CONDA_PREFIX
ISAAC_PYTHON="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"

"$ISAAC_PYTHON" -m pytest -q tests/test_tomato_physics_v1.py
"$ISAAC_PYTHON" -m compileall -q \
  simulation/tomato_pick/isaac/physics_v1 \
  script/*tomato_physics_v1*.py
```

Before launching, confirm there is no other Isaac task or large CUDA process
and that W&B authentication is available. Credentials are never stored in the
repository.

## Continue exactly from iteration 1200

The reward schedule needs the absolute iteration offset, and W&B should append
to the original run:

```bash
export TOMATO_REWARD_ITERATION_OFFSET=1200
export WANDB_RUN_ID=hs0bad5e
export WANDB_RESUME=allow
export WANDB_INIT_TIMEOUT=300

"$ISAAC_PYTHON" \
  "$LINKAGENT_ISAACLAB_ROOT/scripts/reinforcement_learning/train.py" \
  --rl_library rsl_rl \
  --external_callback simulation.tomato_pick.isaac.physics_v1.register_task \
  --task LinkAgent-TomatoPick-Physics-v1 \
  --num_envs 8192 \
  --seed 42 \
  --max_iterations 600 \
  --logger wandb \
  --experiment_name tomato_pick_physics_v1 \
  --run_name tomato_pick_physics_v1_from_scratch_8192 \
  --resume \
  --load_run '^2026-08-10_03-22-34_tomato_pick_physics_v1_from_scratch_8192$' \
  --checkpoint '^model_1200\.pt$' \
  --viz none
```

`600` is the number of additional updates, taking the authoritative state from
1200 to the intended absolute update 1800. Immediately verify that the runner
reports loading `model_1200.pt`, its iteration advances beyond 1200, and the
same W&B history resumes. Do not resume from the log-only step 1208.

## Completion gate

After training reaches 1800:

1. retain checkpoints around the apparent plateau rather than selecting the
   final checkpoint automatically;
2. run deterministic fixed seeds with full environment randomization;
3. report aggregate and worst-seed success, Wilson confidence interval, drop,
   detach, and retained rates;
4. promote only the checkpoint that passes the same provenance/hash gate used
   by Clean-v5;
5. keep hardware deployment blocked until measured L20, G20, tomato contact,
   and cultivar-specific stem parameters replace the provisional priors.
