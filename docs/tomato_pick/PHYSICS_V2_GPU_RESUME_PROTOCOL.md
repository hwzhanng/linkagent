# Physics-v2 powered-session resume protocol

This protocol prepares the next GPU session without consuming GPU power. It
does not authorize PPO training or hardware access.

## Static handoff

After the CPU qualification has passed and the worktree is committed, create a
source-identity handoff:

```bash
python script/prepare_tomato_physics_v2_gpu_resume_manifest.py \
  --cpu-manifest logs/qualification/tomato_physics_v2/cpu_offline/manifest.json \
  --output logs/qualification/tomato_physics_v2/gpu_resume_manifest.json
```

The handoff hashes the profile, curriculum draft, experiment matrix, analytic
models, and integrated environment sources. It remains fail-closed and lists
the external checks that cannot be established during a low-power session.

## Resource sequence

When stable high-power supply returns, use three deliberately separate tiers:

1. `64 env × 10 updates`: integrated finite-state and metric-schema smoke.
2. `256 env × 4 updates`: fixed numerical PPO/checkpoint probe.
3. `256 env × 500 updates`: bounded learning observation.

No tier automatically authorizes the next. Formal-scale training remains out
of scope until bounded-learning evidence and physical calibration justify it.

## Evaluation and selection

Training metrics may nominate checkpoints but may never label one as best.
Each candidate is evaluated deterministically with seeds `17`, `42`, and
`2026`, full randomization, 64 environments, and a 160-step episode horizon.

Selection is lexicographic:

1. Higher independent success rate.
2. Lower diagnostic damage proxy.
3. Lower drop rate.
4. Lower peak fingertip force.

The report must also include the worst robustness bucket across small/large,
soft/firm, dry/wet, actuator-delay, and tactile-delay/dropout conditions.

## Ablations

The inactive experiment matrix defines one unchanged reference and four
single-family ablations: tactile uncertainty, actuator uncertainty, fruit
contact randomization, and curriculum staging. Ablation overlays are design
artifacts only; each requires a revisioned implementation and fresh CPU/GPU
qualification before execution.

## Boundaries

- No checkpoint becomes deployment evidence from training reward alone.
- The tennis-ball three-finger demonstration is not a tomato policy prior.
- The curriculum remains inactive until explicitly authorized.
- Non-Hookean fruit, cultivar damage, tactile calibration, and actuator system
  identification remain deferred until appropriate measurements exist.
