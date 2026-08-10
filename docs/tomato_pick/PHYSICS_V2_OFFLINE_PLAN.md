# TomatoPick Physics-v2 offline improvement plan

Physics-v2 is the user's independent offline physics-improvement lineage. It
does not overwrite Clean-v5 or the team's Physics-v1 source baseline. Until a
stage explicitly changes and qualifies one physical family, Physics-v2 inherits
Physics-v1 exactly and formal training remains blocked.

## Non-negotiable boundaries

- Clean-v5 remains reproducible model evidence, not physical ground truth.
- Physics-v1 remains a frozen team baseline.
- Hardware-pending values cannot be relabeled as calibrated.
- A physics or reward semantic change requires an explicit profile revision,
  repeated isolated gates, and a new from-scratch PPO run.
- The current task controls the right A7/G20. Left-hand transfer remains blocked
  until axes, zero positions, limits, sensor order, and mount transforms are
  explicitly audited.

## Stages and acceptance

### 1. Version boundary and evidence registry

Outputs:

- independent profile, task, environment/config wrappers, PPO output identity;
- frozen Physics-v1 file hashes;
- parameter evidence and uncertainty registry;
- machine-readable step-1 gate.

Acceptance: parent hashes match; 66D/83D/11D and 120/20 Hz contracts are fixed;
wrappers contain no behavior overrides; formal training is fail-closed.

Known parent issue discovered during the compatibility audit: parent commit
`edc4197` contains a Physics-v1 derived USD whose actual SHA-256 is `2b5bc9...`,
while the Physics-v1 profile and handoff documents declare `6bde68...`. Physics-v2
freezes the actual checked-in file, leaves the parent untouched, and carries this
declared-versus-actual identity mismatch into the stage-2 asset audit.

### 2. Static robot asset audit

Outputs: machine-readable joint, mass, inertia, collision, mimic, side, and
frame report; reviewed issue list; no guessed hardware fit.

Acceptance: finite positive mass/inertia, expected 16+5 hand topology, passive
followers unpowered, joint axes/limits and right/left mapping accounted for,
collision coverage and exclusions explicit.

### 3. Fruit contact and damage proxy

Outputs: correlated geometry/mass prior, linear compliant-contact families,
soft/nominal/firm fixed cases, force/indentation/time damage telemetry.

Acceptance: mass follows volume and density; no energy gain; force grows with
indentation; damage is monotonic under damaging load and absent under safe load.

### 4. Stem modes and feasibility

Outputs: separately labeled learnability, literature-axial, and bend/twist-prior
profiles; conservative hand-wrench feasibility report.

Acceptance: one-substep spikes do not detach; sustained weak/median/strong loads
behave consistently; infeasible profiles are reported rather than silently
weakened.

### 5. Actuator and tactile uncertainty

Outputs: bounded delay, response, backlash, mimic, gain, bias, drift, saturation,
dropout, and stale-contact models, all marked hardware-pending.

Acceptance: deterministic seeds reproduce; limits are never exceeded; stale or
non-contact force cannot produce secure support or success.

### 6. Isolated physics gates

Outputs: free-fall, energy, incline, compression, rebound, hand free-motion,
self-collision, and stem-duration reports.

Acceptance: every invariant and scenario threshold passes before integrated PPO.

### 7. Scripted feasibility and anti-reward-hacking

Outputs: weak/median/strong scripted pick reports plus zero-action, random,
unsupported pull, impact, same-side contact, and drop counterfactuals.

Acceptance: scripted feasible cases complete detach-retain; invalid strategies
cannot receive full task success.

### 8. Numerical PPO and comparison

Outputs: 64-environment stability report, 256x4 PPO numerical report, and fixed
Clean-v5/Physics-v1/Physics-v2 counterfactual comparison.

Acceptance: finite states and losses, no PhysX overflow or energy growth, and an
explicit authorization decision before any formal-scale training.

## Step-1 gate command

```bash
python script/check_tomato_physics_v2_step1.py \
  --output logs/qualification/tomato_physics_v2/step1_scaffold_gate.json
python -m pytest -q tests/test_tomato_physics_v2.py
```

Neither command imports Isaac Sim, connects hardware, or starts training.

## Current offline result

Stages 1 through 7 are implemented and their pure/static gates pass.  The
machine-readable reports are written under
`logs/qualification/tomato_physics_v2/` (the directory is intentionally
ignored by Git because reports are regenerated evidence).

- The robot audit accounts for 71 links, 70 joints, both 16+5 hand
  articulations, all mass/inertia records, 59 enabled self-collision pairs and
  2426 filtered pairs.  It preserves the inherited Physics-v1 USD hash mismatch
  as a known parent issue.
- Eight independent sample seeds pass deterministic/finite checks.  Fixed
  soft/nominal/firm cases hold at 20 degrees, slide at 35 degrees, and have no
  analytic compression or rebound energy gain.
- Learnability, literature-axial and bend/twist stem modes are disjoint.  The
  literature envelope is not weakened to make it learnable.
- Weak/median/strong wrench-level twist scenarios detach and retain through the
  real load-duration, support-credit, break-credit and reward primitives.
  Unsupported, high-speed, same-side, stale-tactile and drop counterfactuals
  cannot produce task success.

The exact runtime is now installed at
`/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1`: Isaac Lab source tag
`v3.0.0-beta2.patch1`, Python 3.12.3, Isaac Sim 6.0.1.0, IsaacLab PhysX 1.1.3,
Torch 2.10.0+cu128 and RSL-RL 5.0.1.  CUDA PhysX on the local RTX 4060 loads
the formal Physics-v2 USD as a fixed-base 71-body, 56-independent-joint
articulation with ten authored mimic relations.

Stage-8 integration exposed and corrected two inherited compatibility defects:

- the frozen reset sampler emits `wxyz`, while Isaac Lab v3/Sim 6 math and
  tensor APIs use `xyzw`; Physics-v2 now adapts that boundary without changing
  the frozen Physics-v1 source;
- the inherited geometric fingertip IDs and contact-sensor prims were not
  aligned with the formal distal fingertip bodies.  Physics-v2 now uses the
  formal distal bodies and a simulation-derived low-force closure center.

After correction, 64-environment and 256-environment finite CUDA rollouts pass.
A 16-environment gentle reach-close qualification produces real fingertip
contact in 9 environments, reaches an enclosure score of 0.92 and a maximum
physical force of 3.27 N without pressure-limit termination.  The same deliberately
non-pulling trajectory also records unsupported detachments as drops, confirming
that contact alone is not mislabeled as a successful pick.

Before stage 8, run the read-only boundary report:

```bash
python script/check_tomato_physics_v2_training_preflight.py \
  --output logs/qualification/tomato_physics_v2/training_boundary_preflight.json
```

The exact runtime, articulation, contact trajectory, 64-environment and
256-environment non-training gates are complete.  The next operation is the
fail-closed 256-environment, four-update PPO numerical probe.  Formal-scale
training remains blocked regardless of that probe's result and requires a
separate decision.
