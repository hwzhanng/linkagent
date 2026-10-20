# RoTO 2.0 × Object Pressure Gradient: Single-Hand Fruit Rotation

## Scope

One right LinkerHand L20 (16 active and 5 passive mimic joints), fixed wrist, one 58–70 mm fruit, Isaac Lab 3.0, 20 Hz policy / 120 Hz physics, 24 s episode. Learn continuous rotation with minimal surface load and no drop. There is NO second hand, arm, custom PPO, recurrent belief network or dynamics ensemble.

**Current status (2026-10-20):** Real Isaac GPU/PhysX training and independent pose-based evaluation have been run. The latest 32M-transition v7 pilot achieved **0/128 full-rotation successes**, 74.5% strict active-pad support and 18.4° mean net rotation. Earlier velocity-integral success claims, including 72.4%, are **withdrawn**. No converged continuous-rotation or zero-damage controller has been demonstrated.

**Teacher/research review:** [Complete Chinese experiment report, run registry, raw logs, trajectories, videos and checkpoint download instructions](../../research/roto2_review/2026-10-20/README.md). Start there for current findings; older run snapshots preserve historical, sometimes invalid conclusions.

### Recent baseline

[RoTO 2.0 (2026-05)](https://arxiv.org/abs/2605.21429) and its authors' [roto](https://github.com/elle-miller/roto) benchmark provide 2026 blind tactile Baoding tasks across multiple hands. We reuse the exact original [multimodal_rl](https://github.com/elle-miller/multimodal_rl) PPO / GAE / Trainer / Encoder / Memory / checkpointing library, not the incompatible Allegro robot task or old pre-trained Allegro weights. The original *one-step improvement track* below is the only novel control component. No fork of the optimiser.

The source installation pins:
- roto SHA 0fa344e6e224f141cb6ab3ed4c2fbbbd86cf1cc2
- multimodal_rl SHA 4ac55e854b4c732534d443715a0322d4220b461c

### Original research design and current implementation

1. **Dual-sided observations.** On the hand, 5 tactile grids at 12×6 = 360 proxy taxels. On the fruit, 6 small square patches at 2×2 = 24 measured pressure channels, their first differences, and 24 validity bits. Actor also sees encoder-derived joint state, contact centroid, pressure-face moments, target spin and past action. It NEVER sees privileged fruit pose / angular velocity or all-surface pressure directly.
2. **Measured action sensitivity.** A damped rank-one secant update produces Jp = d(patch pressure) / d(hand action) of shape 24×16 from online physical transitions. A similar local spin Jacobian Jw is estimated from simulation spin. No differentiable simulator and no neural pressure predictor.
3. **Task-tangent pressure correction.** Compute g = Jp^T (d pressure-hotspot-risk / d pressure) and suppress components parallel to Jw before applying a *small* corrected action. A correction is rejected unless secure three-finger support is present, enough valid patches exist and the local counterfactual predicts a pressure decrease with little rotation change. Current corrected rewards use a pressure-risk penalty and **no positive pressure-drop bonus**, to avoid rewarding release/regrasp loops. Later reward curricula and pose/contact fixes are documented in the review.

No claim of first invention of Broyden updating or null-space projection is made; the research contribution is testing this simple, sparse fruit-mounted sensing + task-tangent closed-loop integration against an unchanged strong tactile PPO baseline.

## Setup

Requirements: Linux, NVIDIA GPU, project-specific Isaac Lab 3.0.0-beta2.patch1 environment and Git LFS vendor meshes. The install script pins upstream agents and does NOT modify RoTO 2.0 optimiser.

```bash
git clone -b research/roto2-object-pressure-track https://github.com/hwzhanng/linkagent.git
cd linkagent
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
bash script/install_roto2_pressure.sh

PY="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
# A real CUDA PhysX check (not just mock/unit tests).
"$PY" -m script.train_roto2_pressure --headless --smoke --num-envs 8
# Train original 2026 multimodal RL PPO, unchanged.
"$PY" -m script.train_roto2_pressure --headless --num-envs 256 \
  --million-steps 100 --seed 42 --pressure-variant jacobian
```

Compare with identical seed and training budget, changing ONLY pressure_variant:

```bash
"$PY" -m script.train_roto2_pressure --headless --num-envs 256 --million-steps 100 --seed 42 --pressure-variant plain
"$PY" -m script.train_roto2_pressure --headless --num-envs 256 --million-steps 100 --seed 42 --pressure-variant scalar
"$PY" -m script.train_roto2_pressure --headless --num-envs 256 --million-steps 100 --seed 42 --pressure-variant jacobian
```

plain: RoTO 2.0 PPO with identical pressure/tactile observations but no pressure-specific reward or action residual.
scalar: adds conventional pressure-risk penalty, no action residual.
jacobian: applies local action-pressure Jacobian descent with task-tangent projection and the same pressure-risk penalty as scalar; no positive pressure-relief reward.

Checkpoint: logs/roto2_pressure_fruit/<variant>_seed42/<date>/checkpoints/best_agent.pt. Metrics: TensorBoard scalars and independent post-training evaluation.

Evaluate a **trained** RoTO2 checkpoint using the exact same Actor/Encoder and 64 fresh episodes (one independent episode per environment):

```bash
"$PY" -m script.train_roto2_pressure --headless --num-envs 64 --seed 2026 \
  --pressure-variant jacobian \
  --evaluate-checkpoint /absolute/path/to/best_agent.pt \
  --evaluation-output logs/roto2_pressure_fruit/eval_jacobian_seed2026.json
```

The JSON includes directional rotation, successful 360-degree rotation rate, secure-grip fraction, first-done failure rate, mean/peak full-object pressure proxy and fraction of pressure-gradient corrections accepted. Run the equivalent command with the proper variant for plain/scalar baselines.

## Exact actor input and action contract

Single policy step (4 frames are stacked by RoTO 2.0 wrapper):

| Input | Dimensions |
| --- | ---: |
| Fingertip tactile surrogate, 5×12×6 | 360 |
| Fruit film sensor, 6×2×2 | 24 |
| Film pressure first differences | 24 |
| Film validity mask | 24 |
| 16 measured active joint positions + 16 velocities | 32 |
| Fingertip-FK inferred contact center + pressure face means | 9 |
| Contact center velocity + pressure face flux | 6 |
| Gravity, target rotation axis and target speed | 7 |
| Previous action | 16 |
| TOTAL per frame | 502 |
| TOTAL history stack | 2008 |

PPO outputs 16 dimensionless clipped joint-target increments within [-1,1]. Each control step bounds the target motion to 0.035 rad and target lead to 0.11 rad, then standard robot control enforces its actual joint limits. Five follower joint angles use native PhysX mimic joints, not extra RL action dimensions.

Physics oracles (96 pseudo pressure samples, exact fruit pose and motion) are only used for training reward/evaluation or the single simulated local spin-response estimator. **There is currently no tissue deformation, bruise predictor or measured damage threshold.** A contact-force Gaussian painted onto the fruit surface is a **proxy**, not real film data, and its pressure values must not be presented as measured Pa accuracy.

## What constitutes the Aha result?

Across at least three common random seeds and identical test distributions, at *matched rotation performance*:
- higher sustained 360-degree completion and/or less object loss,
- lower independent oracle upper-tail/peak pseudo pressure,
- successful load redistribution between contacting fingers visible during the same rotation,
- nonzero, interpretable fraction of proposed Jacobian corrections accepted.

Do not report improved training reward alone. If corrected actions are never accepted, or the pressure proxy remains all zeros, the hypothesis is NOT demonstrated and the sensor/contact mapping must be audited. Source robot geometry, effective contact dynamics and virtual taxel readouts require Isaac GPU validation. Real flexible-fruit damage will require film calibration and delayed bruise measurement.

Source references: RoTO 2.0 arXiv:2605.21429 (2026); [IsaacLab 2026 in-hand rolling PR](https://github.com/isaac-sim/IsaacLab/pull/6345) for stable grasp reset architecture.
