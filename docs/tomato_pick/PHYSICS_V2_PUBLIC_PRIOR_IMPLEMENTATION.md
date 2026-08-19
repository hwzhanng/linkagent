# Physics-v2 public-prior implementation

Physics-v2 is a new task profile. It does not mutate Physics-v1, its USD,
reward profile, checkpoint, or optimizer state.

## Identity and boundaries

- Task: `LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1`
- Config: `simulation/config/tomato_pick_physics_v2_public_prior_v1.json`
- Evidence package: `research/physics_v2_public_priors/`
- Gap-fill contract:
  `research/physics_v2_public_priors/ENGINEERING_PRIOR_SUPPLEMENT.json`
- Training: from scratch only when entering V2; a Physics-v1 checkpoint is
  rejected. Later V2 continuation must restore the complete V2 state.
- Frozen parent interface: 66D actor observation, 83D critic state, 11D action
  with the existing 5D hand manifold. Physics-v2 extends only its from-scratch
  actor to 103D and critic to 120D with one previous perceived fruit-to-palm
  sample, a current/previous perceived fruit-to-fingertip-centroid pair, and
  the two newest issued 11D commands needed to make the public 0--2-step
  actuator queue observable. The current 6D command-to-actual end-effector
  tracking error is expressed in the local grasp basis and normalized by the existing
  per-action translation/rotation scales. The controller target and encoder
  forward kinematics are deployment inputs. The added fingertip-centroid pair
  uses the same noisy fruit perception plus robot forward kinematics. Exact
  velocity, sampled delay, stem wrench, damage, detachment, and acceptance
  thresholds remain hidden from the actor. An 8D hand ablation remains future
  work.
- Controlled-break credit is latched at the exact 120 Hz fracture substep.
  The later 20 Hz policy boundary no longer erases a valid break after the
  remainder of the issued action moves the fruit. The same grasp-quality,
  secure-support, distance, and relative-speed predicates remain required;
  reward weights and all outcome thresholds are unchanged.
- Hardware: no SDK, CAN, or real robot is used by the qualification scripts.
  Hardware transfer remains blocked on the measurements listed in the profile.

## Implemented crop physics

1. Fruit morphology samples a shared size latent across three dimensions and
   mass. Samples outside the public density, mass, or aspect gates are rejected.
2. Abscission diameter is sampled first. Intact tensile mean/SD is interpolated
   from the three public diameter nodes rather than using the Physics-v1 4-8 N
   effective-break range.
3. Bending is sampled as a moment from the published 12 mm three-point fixture.
4. AZ shear uses `Triangular(0.60, 0.7423, 0.90) * Ft*`. The mode is the
   adjacent-section ratio `33.241/44.781`; it remains a method prior, never a
   relabelled direct AZ measurement.
5. Torsion has an explicit capacity and elastoplastic yield. Plastic twist
   reduces integrity through the cross-study constrained exponential law.
6. Tension, shear, bending, and torsion enter the dimensionless interaction
   criterion. Duration damage is applied to non-torsional modes so twist
   integrity is not counted twice.
7. World orientation follows the 378-bunch categorical distribution. Each
   category receives a truncated `18 deg` azimuth jitter and a
   `Triangular(0, 8, 15 deg)` polar tilt. The actor remains in the local grasp
   frame and never receives the world category.
8. Non-torsional fatigue samples episode-fixed onset
   `Triangular(.45,.60,.75)`, exponent `Triangular(1.5,2,3)`, and time constant
   `Triangular(.20,.55,1.20) s`.
9. Contact samples an effective pair static coefficient
   `Triangular(.30,.45,.70)`, dynamic/static ratio `0.85`, stiffness
   `Triangular(3000,5385,12000) N/m`, and coupled damping `10-60 Ns/m`.
   The `5385 N/m` center is the whole-fruit secant `51.27 N / 9.52 mm`, not a
   claim about local pad tissue stiffness.
10. A7/L20 dynamics retain vendor URDF/manual velocity, effort, force and
    open-close envelopes plus the passed implicit-drive surrogate. Policy
    latency is randomized uniformly over `0-2` steps (`0-100 ms`). The current
    checked-in hand is L20; historical G20 naming is treated as an alias that
    must be resolved before hardware transfer.
11. The audited first-success curriculum preserves public tensile and shear
    capacities throughout. Bending and torsion start at 10% and smoothstep to
    100% at update 40, and that temporary assistance is active only in the
    unchanged secure and kinematically retained envelope. Evaluation and the
    complete update-40--49 gate tail use 100% public capacity for every mode.

## Qualification

CPU public-prior gate:

```bash
LINKAGENT_ISAACLAB_ROOT=/path/to/IsaacLab \
  /path/to/IsaacLab/env_isaaclab/bin/python \
  script/qualify_tomato_physics_v2_public_priors.py
```

This evaluates Q1 morphology, Q2 axial, Q3 bending, Q4 torsion, Q5 360-degree
twist then pull, and a step-integrated Q6 counterfactual strategy matrix. Q6
confirms that `11.66 N` does not instantly break an intact AZ, pure axial,
shear, and bend overloads do break, a half-axial/half-torsion load remains
subcritical, a `360 deg` yield twist leaves integrity `0.26038` while still
attached, and the residual `11.6601 N` pull breaks after 67 substeps
(`0.5583 s`). Continued twist breaks by the public `550 deg` bound. The
committed report is:

`logs/qualification/tomato_physics_v2/public_prior_cpu_gate.json`

Isaac vector smoke gate:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  "$LINKAGENT_ISAACLAB_ROOT/env_isaaclab/bin/python" \
  script/check_tomato_physics_v2.py \
  --num-envs 64 --steps 128 --viz none --device cuda:0
```

This does not train. Its 25 checks cover task registration, scene cloning,
material authoring, resets, finite rollout/reward/runtime tensors, disjoint
termination masks, V2 capacity state, force/torque clamps, continuous
orientation, episode-fixed fatigue, pair-friction mapping, compliance,
three-slot action history for the sampled `0-2` delay, deployable actor access
to its relative-motion history, normalized controller tracking error, and two
newest issued commands, material diversity,
the unchanged secure-retain predicate, hash pinning, and no runtime mutation of
the Physics-v1 profile. The additional paper-strategy audit check verifies that
effective twist never exceeds raw rotation, the pull interval is nontrivial,
and the metric remains diagnostic-only.

The environment separately records `twist_assisted_break_rate`,
`bend_assisted_break_rate`, `moderate_pull_break_rate`,
`paper_strategy_candidate_rate`, and `paper_strategy_success_rate`. The last
metric requires torque-utilization-weighted effective twist of at least
`360 deg` or bend utilization of at least `0.5`, a final axial pull in the
nontrivial `1-15.29 N` range, real substep fingertip support, kinematic
retention, and the unchanged final success predicate. Raw unloaded rotation
does not count as effective twist, and a zero-pull pure bend does not count as
moderate pull. These are measurements, not reward shortcuts.

The V2 actor uses a single-layer 128-unit GRU over the validated 103D
deployable observation, followed by the inherited `512-256-128` ELU policy
head. The critic remains the inherited feed-forward 120D model. This supplies
bounded memory across the partially observable fracture-to-retention sequence;
it does not expose detachment or privileged stem state and changes no reward,
physics, PPO optimization, threshold, or acceptance criterion. Episode-level
`ever_retained_rate` and `mean_max_hold_steps` are audit latches only.

V2 uses gate-scale outcome alignment from update 10 through update 40 for the
existing failure and stable-detach channels. Its exact-substep
`break_grasped` precursor remains available with the unchanged 40-point weight:
unlike the inherited coarse event, this one-shot score already requires grasp
quality, secure thumb opposition, and the unchanged distance and speed
retention predicates at fracture. The V2 dense time cost remains at its
unchanged discovery endpoint (`-0.01` per nonterminal step); terminal failure,
stable-detach, and success channels still follow the gate-scale alignment.
This avoids allowing an episode-length baseline to dominate the sparse
physical credit while preserving the anti-stall sign and every terminal
penalty. Clean-v5 and Physics-v1 keep their original aligned-endpoint blends.

## Preparing the next training run

The frozen next-gate candidate is
`physics_v2_retention_objective_v1`. The CPU and managed-Isaac prerequisites
record hashes for every shared runtime input; the 8192x50 gate rejects stale
reports or a different candidate ID. This preparation does not authorize or
start training.

The exact first-success command can be audited without starting Isaac or PPO:

```bash
LINKAGENT_ISAACLAB_ROOT=/path/to/IsaacLab PYTHONPATH="$PWD" \
  /path/to/IsaacLab/env_isaaclab/bin/python \
  script/gate_tomato_physics_v2_first_success.py --preflight-only
```

The preflight must report `training_started=false`, the frozen candidate ID,
and the exact `8192 x 32 x 50`, seed-42 command. The next authorized experiment
is that bounded from-scratch gate, not a curve-controller segment.

The planner is non-mutating unless both `--execute` and the exact confirmation
string are supplied:

```bash
LINKAGENT_ISAACLAB_ROOT=/path/to/IsaacLab \
  python script/train_tomato_physics_v2.py \
  --write-manifest logs/qualification/tomato_physics_v2/training_manifest.json
```

Only after that gate passes may an explicitly authorized long-run plan use:

```bash
LINKAGENT_ISAACLAB_ROOT=/path/to/IsaacLab \
  python script/train_tomato_physics_v2.py \
  --execute \
  --confirmation START_PHYSICS_V2_PUBLIC_PRIOR_FROM_SCRATCH
```

The executable entry is the audited curve controller. It runs explicit
400-update serial segments (the first command contains
`--max_iterations 400`), verifies Actor, Critic, both normalizers, policy
standard deviation, optimizer, iteration field, and checkpoint SHA-256 at
every boundary, then evaluates seeds `17/42/2026` with `1024` environments for
`400` deterministic steps each. There is no hard total-update cap. It stops
only after two consecutive aggregate evaluations are at least 90%, differ by
at most one percentage point, and two consecutive boundaries show no more than
one percentage point absolute gain between the latest and preceding 100-update
training windows. Otherwise it performs a complete V2-state continuation.
The controller never loads a Physics-v1 policy or optimizer.

At this commit, PPO has not been started. The manifest and controller are
prepared for external audit only.

## Current scientific caveat

The six missing quantities now have explicit, bounded values sufficient for
simulation and ablation. Their provenance class remains
`ENGINEERING_HYPERPARAMETER`, `HARDWARE_SURROGATE_PRIOR`, or
`QUALIFIED_SIMULATION_SURROGATE`; none is silently promoted to a direct prior.
Target-cultivar, deployed pad, actuator, tactile, and perception measurements
still override these values before hardware transfer.
