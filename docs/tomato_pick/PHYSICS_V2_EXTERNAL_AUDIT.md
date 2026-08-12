# Physics-v2 external-audit handoff

Generated: `2026-08-13T00:07:23+08:00`

Status: **qualified and training-ready; PPO not started; hardware not
connected**.

## Reproducible evidence

| Evidence | Result | SHA-256 |
|---|---:|---|
| CPU public-prior Q1-Q6 | pass | `c44cbde4a2f855098bd125140fa568eb05e69531ec26524c17b0083d57d97f97` |
| Isaac smoke 1 env x 32 steps | 22/22 pass | `d14d055d21e20265cdd124309414945cff51e4ab3d011832eed0034792a2247a` |
| Isaac vector 64 envs x 128 steps | 22/22 pass | `9853ac8e439974dd82cce85593dc96cc001758d85498e316c386cc3dfed4bc27` |
| Frozen training manifest | generated | `52efd59e5cc3bf0e828657645af811004a29255e783cb96208bf9636c0068997` |
| Managed-environment test suite | 291 tests + 103 subtests pass | command below |

The committed evidence paths are:

- `logs/qualification/tomato_physics_v2/public_prior_cpu_gate.json`
- `logs/qualification/tomato_physics_v2/isaac_smoke_gate_1x32.json`
- `logs/qualification/tomato_physics_v2/isaac_smoke_gate_64x128.json`
- `logs/qualification/tomato_physics_v2/training_manifest.json`

## Scientific acceptance signal

The evaluator records final success separately from a break-time candidate.
`PhysicsV2/paper_strategy_success_rate` is true only when all of these hold:

1. torque-utilization-weighted effective twist of at least 360 degrees or
   bending utilization of at least 0.5;
2. axial break force in the nontrivial `1-15.29 N` range;
3. real fingertip support, including thumb opposition;
4. break-time distance and relative-speed retention;
5. the unchanged final secure-retention success predicate.

This signal is diagnostic only. It is not added to reward and therefore cannot
directly reward a scripted paper-looking motion.

## Training controller audit points

- New V2 policy only; no Physics-v1 checkpoint or optimizer may be loaded.
- `8192` environments, `32` rollout steps, seed `42`.
- Every segment passes explicit `--max_iterations 400`; there is no inherited
  `1800`-update terminal cap.
- Every continuation verifies Actor, Critic, actor/critic normalizers, policy
  standard deviation, optimizer, iteration field, and checkpoint SHA-256.
- Every boundary evaluates deterministic seeds `17`, `42`, and `2026`, each
  with `1024` environments for `400` steps, before continuation.
- Stop requires two consecutive aggregate success results at least 90%, within
  one percentage point, plus two consecutive 100-vs-100 curve plateaus.
- A lock forbids duplicate controllers; process checks forbid concurrent V2
  train/evaluate jobs; less than 8 GiB available memory forbids a new segment.
- The controller is resumable from its own auditable state/history, but refuses
  to adopt an untracked pre-existing initial run.

The controller has an exact-confirmation dry stop. This handoff does not invoke
its execute path.

## Reproduction commands

```bash
LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1 \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  -m pytest -q

LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1 \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/qualify_tomato_physics_v2_public_priors.py

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/check_tomato_physics_v2.py --num-envs 64 --steps 128 \
  --output logs/qualification/tomato_physics_v2/isaac_smoke_gate_64x128.json \
  --viz none --device cuda:0
```

## Non-claims

Passing these gates means the simulator is finite, internally consistent,
source-bounded where public evidence exists, explicit where engineering priors
fill gaps, and ready for a from-scratch training audit. It does not demonstrate
that PPO will discover the desired strategy, and it does not qualify real A7 or
L20/G20 transfer. The deployed pad/friction, compliance, actuator response,
target cultivar, perception error, and L20/G20 alias still require hardware
measurements before any real-robot claim.
