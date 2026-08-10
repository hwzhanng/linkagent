# Physics-v1 source-only reproducibility handoff

> This is the historical source-only handoff for commit `edc4197`. The current
> perf-v7 code and the safely paused formal run now include a resumable
> iteration-1200 checkpoint; see `PHYSICS_V1_INCOMPLETE_RUN_1200.md`.

This release freezes the current Physics-v1 source baseline so another team
member can modify the hand manifold, physics, randomization, or reward credit
without inheriting an opaque local state. It intentionally contains **no new
Physics-v1 checkpoint**, W&B run directory, TensorBoard event file, exported
policy, or training video.

## Included identity

- task: `LinkAgent-TomatoPick-Physics-v1`;
- parent source commit: `9c4d71b`;
- actor/critic/action contract: 66D / 83D / 11D;
- PPO contract: 8,192 environments, 32 rollout steps, seed 42, fixed learning
  rate `3e-4`, entropy coefficient `0.002`, initial standard deviation `0.60`;
- calibrated hand mapping:
  `tomato_g20_user_calibrated_5d_v2_1_collision_safe`;
- frozen hand draft SHA-256:
  `75d8122975dac780110505596419a1a98ed13bb3a6e06c3c5206f844e2fb30a7`;
- derived Physics-v1 USD SHA-256:
  `6bde6855bcd0d6c2ab9e23c5588b516c684743cff980d7b37b5981834dc5aa35`;
- provisional detachment model:
  `abscission_zone_combined_wrench_duration_v1`;
- break credit: exact 120 Hz break substep must have secure thumb/opposing-
  finger support, center distance at most 0.06 m, and relative speed at most
  0.10 m/s.

The complete parameter profile is
`simulation/config/tomato_pick_physics_v1.yaml`. The pre-launch frozen input
hashes are in
`logs/qualification/tomato_physics_v1/formal_training_manifest.json`; its
`formal_training_started=false` field describes the frozen launch plan, not
the later dynamic W&B run, which is deliberately omitted.

## Environment

Use the same Isaac Lab 3.0 beta 2 patch 1 API contract and its managed Python.
All launchers fail closed unless the checkout is supplied explicitly:

```bash
cd /absolute/path/to/linkagent
export LINKAGENT_ISAACLAB_ROOT=/absolute/path/to/IsaacLab-3.0.0-beta2.patch1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CONDA_PREFIX
ISAAC_PYTHON="$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python"
```

No command in this handoff connects to SDK, CAN, or real hardware.

## New five-dimensional hand manifold

The production mapping and the user's two independently adjusted thumb-root
joints are preserved in all of the following files:

- `logs/hand_5d_viewer/latest_draft.json` — frozen user draft;
- `simulation/config/tomato_hand_5d_user_calibrated_v2.json` — machine-readable
  center and ten signed endpoints;
- `module/skills/tomato_pick/hand_synergy_calibrated_v2.py` — continuous mapper;
- `simulation/tomato_pick/isaac/physics_v1/env.py` — Physics-v1 integration;
- `script/serve_tomato_hand_5d_viewer.py` and
  `tools/tomato_hand_5d_viewer/` — hand-only URDF inspection tool.

Verify the frozen draft, endpoint isolation, joint limits, continuity, and
local rank:

```bash
"$ISAAC_PYTHON" -m pytest -q \
  tests/test_tomato_hand_synergy_calibrated_v2.py \
  tests/test_tomato_hand_5d_viewer.py
cd tools/tomato_hand_5d_viewer
npm ci
npm run check
cd ../..
"$ISAAC_PYTHON" script/serve_tomato_hand_5d_viewer.py --open
```

The viewer opens the calibrated v2 mapping by default. `node_modules/` is not
versioned; `package-lock.json` is the dependency identity.

## Physics and reward source

The reproducible physical state is carried by:

- `simulation/config/tomato_pick_physics_v1.yaml`;
- `simulation/tomato_pick/physics_v1.py` and
  `simulation/tomato_pick/reset_v1.py`;
- `simulation/tomato_pick/isaac/physics_v1/`;
- `botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda`;
- `script/build_tomato_physics_v1_robot_asset.py`;
- the Physics-v1 audit, gate, and training launchers under `script/`.

The source keeps Clean-v5 intact. The two hooks added to
`simulation/tomato_pick/isaac/env.py` allow the independent overlay to replace
stem-progress and break-credit semantics without changing Clean-v5 behavior.

## Validation order

The lightweight release checks are safe while no second Isaac task is running:

```bash
"$ISAAC_PYTHON" -m pytest -q \
  tests/test_tomato_hand_synergy_calibrated_v2.py \
  tests/test_tomato_hand_5d_viewer.py \
  tests/test_tomato_physics_v1.py
"$ISAAC_PYTHON" -m compileall -q \
  module/skills/tomato_pick/hand_synergy_calibrated_v2.py \
  simulation/tomato_pick/physics_v1.py \
  simulation/tomato_pick/reset_v1.py \
  simulation/tomato_pick/isaac/physics_v1 \
  script/*tomato_physics_v1*.py
"$ISAAC_PYTHON" script/audit_tomato_physics_v1_abscission_feasibility.py \
  --samples 200000 --seed 42 --output /tmp/abscission_envelope_feasibility.json
"$ISAAC_PYTHON" script/train_tomato_physics_v1.py \
  --num-envs 8192 --iterations 1800 --seed 42 \
  --write-manifest /tmp/tomato_physics_v1_launch_plan.json
```

The preserved qualification reports demonstrate the previously completed
heavy gates:

- `reward_math_audit.json`;
- `clean_v5_training_reward_comparison.json`;
- `abscission_envelope_feasibility.json`;
- `isolated_physics_gate_abscission_v1.json`;
- `ppo_256x4_secure_kinematic_credit_gate.json`;
- `first_success_8192x50_secure_kinematic_credit_gate.json`.

Before a changed profile can train, rerun the heavy gates sequentially with no
other Isaac process:

```bash
"$ISAAC_PYTHON" script/check_tomato_physics_v1_isolated.py \
  --output /tmp/isolated_physics_gate.json
"$ISAAC_PYTHON" script/probe_tomato_physics_v1_ppo.py \
  --num-envs 256 --iterations 4 --seed 42
"$ISAAC_PYTHON" script/gate_tomato_physics_v1_first_success.py \
  --num-envs 8192 --iterations 50 --seed 42 \
  --output /tmp/first_success_8192x50_gate.json \
  --log /tmp/first_success_8192x50.log
```

## Team modification boundary

Create a new profile ID for every physics or reward-semantic revision. Update
the derived USD and all input hashes, then repeat pure tests, isolated physics,
256x4 numerical PPO, and the from-scratch 8192x50 retention/success gate.
Changing physics or reward semantics invalidates every existing Physics-v1
checkpoint; never resume one. A7 gains and all literature contact/stem priors
remain provisional until measured hardware data replaces them.
