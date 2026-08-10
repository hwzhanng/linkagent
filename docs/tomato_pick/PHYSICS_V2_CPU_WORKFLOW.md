# Physics-v2 low-power CPU workflow

This workflow advances Physics-v2 without launching Isaac Sim, using CUDA,
training PPO, querying a GPU, opening a robot SDK, or connecting hardware.  It
does not replace integrated PhysX qualification; it prepares better-defined
inputs for the next powered run.

## One-command qualification

Use the already-installed IsaacLab Python only because it contains PyTorch.
CUDA is hidden from every child process by the qualification runner:

```bash
/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/run_tomato_physics_v2_cpu_qualification.py
```

The generated evidence is written to
`logs/qualification/tomato_physics_v2/cpu_offline/`. Runtime evidence remains
ignored by Git; the scripts, tests, configuration, and acceptance rules are
versioned.

The runner executes an explicit allow-list:

1. Static and analytic Physics-v2 gates 1 through 7.
2. A 50,000-sample correlated parameter sensitivity screen.
3. Shared tomato MDP reward-schedule audit.
4. Reward/success counterfactuals and curriculum validation.
5. Static success-contract parity and unresolved-semantic registration.
6. Physics-v2 and shared-MDP CPU unit/property tests.

It sets `CUDA_VISIBLE_DEVICES` to an empty value, disables W&B, rejects the
Omniverse EULA for child commands, and removes known robot/ROS endpoint
variables.

## Training artifact audit

After a training process has naturally finished, inventory it without policy
evaluation:

```bash
python script/audit_tomato_physics_v2_training_artifacts.py \
  --log /path/to/training.log \
  --run-dir /path/to/rsl_rl/run \
  --manifest /path/to/training_manifest.json \
  --output /path/to/artifact_audit.json
```

This hashes every checkpoint and nominates candidates using physical training
metrics before reward. It deliberately never calls a candidate "best": only a
later deterministic, multi-seed Isaac evaluation may do that.

## Current analytic findings

The fixed 50,000-sample seed `20260810` audit passes all CPU checks. Its main
screening relationships are:

- fruit mass is dominated by the sampled size latent (`r = 0.9896`);
- contact stiffness decreases with ripeness (`r = -0.9144`);
- static friction decreases with wetness (`r = -0.7350`);
- the learnability stem tensile threshold is driven mainly by size
  (`r = 0.7339`);
- the bend/twist bending threshold decreases with ripeness (`r = -0.8616`).

The learnability stem mode is conservatively feasible in translation, bending,
and torsion for all sampled cases. The literature-axial mode is infeasible in
translation under the gentle hand-capacity bound, and is intentionally not
weakened. The bend/twist prior is feasible through its moment paths but not its
translation path. These are analytic feasibility statements, not cultivar
calibration.

## Inactive curriculum draft

`simulation/config/tomato_pick_physics_v2_curriculum_draft.json` defines six
single-objective stages: reach, enclosure, gentle hold, stem manipulation,
detach-retain, and robustness. It explicitly forbids using the tennis-ball demo
pose, a fixed joint reference, or a fixed finger count as the tomato strategy.

The draft is fail-closed. Activation requires a new profile revision,
from-scratch training, CPU qualification, 64-environment integration,
256x4 numerical PPO probing, and explicit authorization.

## Success-contract parity

The Physics-v1/Physics-v2 inherited runtime thresholds and the offline oracle
now agree on two supported fingertips, 0.25 N minimum support, 6 cm retained
distance, 0.10 m/s relative speed, 20 policy steps, and an 8 N force ceiling.

One semantic difference remains intentionally visible: the conservative
offline oracle requires continuous thumb-opposed fingertip support, while the
integrated runtime allows load to transfer to the palm or between fingers after
a qualified detach. The CPU parity report therefore sets exact semantic parity
to false and blocks formal claims until palm contact evidence and GPU
requalification resolve the choice.

## Deferred until power or hardware returns

- Isaac/PhysX integration and contact trajectories;
- PPO training or checkpoint policy evaluation;
- non-Hookean or deformable tomato modeling;
- cultivar-specific damage probabilities;
- real tactile, actuator, joint, and stem calibration.
