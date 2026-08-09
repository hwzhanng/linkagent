# TomatoPick Physics-v1 baseline

Physics-v1 is an independent, calibration-gated successor to Clean-v5.  It
preserves the successful policy interface, PPO architecture, initialization,
and bounded early exploration design, but it does not treat Clean-v5 physics
or its 96.18% evaluation as physically validated ground truth.

This document is the design and gate history. The source-only team handoff,
including exact commands and the explicit no-checkpoint boundary, is
`PHYSICS_V1_SOURCE_REPRODUCTION.md`.

## Frozen parent evidence

- Clean-v5 task: `LinkAgent-TomatoPick-Clean-v5`
- selected checkpoint: `model_1350.pt`
- checkpoint SHA-256:
  `c4ca9ac27aaa20806b630496362bcb9154d1e8076605f9d7026b58f719d124d0`
- source commit at Physics-v1 start: `9c4d71b`

The parent task, checkpoint, reward, environment, hand manifold, and robot USD
are not modified by Physics-v1.  The existing user change to the exported
`policy.pt` is also outside this work.

## Deliberate corrections

1. L20 has sixteen active joints and five mechanically passive followers.
   Physics-v1 removes position-drive authority from the followers and couples
   each driver/follower pair with one native compliant PhysX mimic joint. Its
   provisional natural frequency is selected by an explicit stability and
   tracking gate, not treated as measured cable/linkage stiffness; the user's
   motion capture must replace or confirm it.
2. Hand commands are limited at the 20 Hz policy boundary by the minimum of
   the vendor envelope and the active USD joint limit. The current follower
   speed ceiling remains the conservative asset value: raising it to a very
   large default, as generally recommended for mimic joints, produced excess
   constraint-driven speed in this specific lightweight hand. Hardware
   capture must replace or confirm both driver and follower envelopes.
3. Tomato contact is a rigid core with PhysX compliant contact, explicit
   average friction combination, low restitution, and contact/friction
   telemetry. Correlated friction and compliance samples are quantized into a
   bounded 4-by-4 material bank so thousands of environments do not create
   thousands of unrelated USD materials.
4. Fruit geometry, mass, compliance, and stem properties come from correlated
   profiles rather than independent uniform draws.
5. Stem failure is an axial/shear/bending interaction with load-duration
   damage.  A single 120 Hz impulse cannot detach the fruit.
6. Retention requires proximity, low relative speed, and load-bearing contact
   from the thumb plus at least one opposing finger. Proximity or two contacts
   on the same side alone are not a training success.
7. Contact force is accepted only when its reported contact point lies within
   8 mm of the sampled fruit surface. Stale reset forces are quarantined and
   logged rather than allowed to trigger success, reward, or excess-force
   termination.
8. The user-calibrated five-dimensional manifold is isolated from Clean-v5.
   The four finger coordinates cannot move the thumb, and the thumb coordinate
   cannot move the four fingers.  The reviewed grasp trajectory forms the
   finger cage first and closes the thumb over the outside only in the final
   quarter of the motion.
9. Physics-v1 enables articulation self collision with a derived right-hand
   matrix.  Cross-digit pairs and distal-finger/palm pairs collide; adjacent
   links, same-digit internal geometry, inactive left-hand geometry, and the
   coarse imported arm/torso geometry are filtered.  This prevents the thumb
   and fingers from passing through one another without turning every coarse
   convex mesh in the imported robot into an unstable self-contact pair.

## Formal-training readiness

Physics-v1 passed the from-scratch PPO gates under the explicitly provisional
literature-prior qualification, and a formal source-validation run was later
started from scratch. That run, its W&B history, and every Physics-v1
checkpoint are deliberately excluded from the source-only team handoff.
Hardware transfer remains blocked until the L20/A7 response, actual pad-fruit
contact, fruit compression/rebound, and cultivar-specific stem curves are
measured. Simulation readiness is not a claim of real-system calibration.
Any change to physics or reward semantics requires a new independent profile,
new hashes, repeated gates, and a new from-scratch PPO run; it must not resume
the omitted training checkpoint.

The 2,199-pair collision matrix is now baked once into the independent derived
USD
`linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda`. Runtime authoring is
zero pairs regardless of environment count; prestartup verifies its topology,
metadata, and matrix hash using one composed clone. The source robot USD is
unchanged.

## Current verification evidence

- Focused hand, viewer, reward, physics, and config tests are listed with the
  exact validation command in `PHYSICS_V1_SOURCE_REPRODUCTION.md`.
- One-environment final randomized gate: all checks green; maximum passive
  residual 0.01476 rad and maximum active hand velocity 1.71950 rad/s.  Log:
  `/tmp/tomato_physics_v1_final_1env.log`.
- Four-environment randomized gate: finite and green before the expanded
  passive-link stress gate.
- Sixty-four-environment randomized gates passed for seeds 17, 42, and 2026.
  Maximum passive residuals were respectively 0.33785, 0.25884, and 0.14385
  rad against the unchanged 0.35 rad gate; maximum active hand velocities were
  2.27805, 2.17380, and 2.26181 rad/s. Logs:
  `/tmp/tomato_physics_v1_armature002_mimic150_seed17.log`,
  `/tmp/tomato_physics_v1_armature002_mimic150_seed42.log`, and
  `/tmp/tomato_physics_v1_armature002_mimic150_seed2026.log`.
- The final provisional transmission uses 0.002 kg m2 active-joint armature,
  150 rad/s mimic natural frequency, damping ratio 1.2, and a 0.95 referred
  driver-speed factor.  These values suppress constraint-driven overspeed and
  leave the passive followers unpowered; they remain simulation priors until
  replaced by the user's recorded joint response.
- Self-collision matrix hash:
  `16653f7ed1d19e6b6bf0a5e25b74ad57c584c74eb548e8af18f3b2a473cd5755`.
  External fruit contact and hand self-collision are represented by a 12-body
  palm/distal-pad proxy with 59 distal-envelope/palm pairs.  The asset stores
  seven filters within that proxy and disables contact solving on 59
  out-of-scope rigid bodies while preserving their visual, inertial, and joint
  state.  This prevents visible distal penetration without the contact-
  complexity explosion caused by all 286 cross-digit segment pairs.
- Baked derived robot SHA-256:
  `6bde6855bcd0d6c2ab9e23c5588b516c684743cff980d7b37b5981834dc5aa35`.
  The 1-environment and 64-environment post-bake gates both passed with
  `runtime_authored_pair_count=0`; the seed-42 64-environment gate retained a
  0.10575 rad maximum passive residual.
- The isolated provisional physics report passes all gates.  At the nominal
  coefficient 0.55, the 20-degree incline puck moved only 4.7 micrometres in
  the onset window while the 35-degree puck moved 0.269 m.  A 2 cm low drop
  produced 1.10 mm maximum compliant penetration, no measurable rebound, and
  no energy gain.  All 4,096 sampled stems rejected a one-substep overload,
  broke after 5--9 sustained 120 Hz substeps, and recovered under subcritical
  load. Report:
  `logs/qualification/tomato_physics_v1/isolated_physics_gate.json`.
- The 256-environment, four-update RSL-RL preflight completed 32,768
  transitions. Value, surrogate, entropy, reward, and action-standard-
  deviation metrics were finite in every update; action std stayed 0.60 and
  no Traceback, NaN/Inf, PhysX overflow, or OOM occurred. Report:
  `logs/qualification/tomato_physics_v1/ppo_256x4_gate.json`.
- The frozen formal plan and every input hash are recorded in
  `logs/qualification/tomato_physics_v1/formal_training_manifest.json`.
  Printing or validating the plan is safe; execution additionally requires
  the exact confirmation token enforced by
  `script/train_tomato_physics_v1.py`.
- An isolated extreme-close counterfactual confirms that the matrix is active.
  With tomato collision disabled, self collision on leaves a 0.73396 rad final
  middle-finger target error and 0.44096 rad middle-follower residual because
  the commanded meshes are physically blocked.  Turning only self collision
  off reduces those values to 0.00202 and 0.01314 rad, respectively, allowing
  the same kinematic command to pass through.  The extreme contact case is a
  barrier proof, not a free-motion acceptance test; the normal randomized
  gates above are the acceptance tests. Logs:
  `/tmp/tomato_physics_v1_final_collisionstress_selfon.log` and
  `/tmp/tomato_physics_v1_final_collisionstress_selfoff.log`.

## Parameter evidence and limits

- The Linker Hand L20 V1.2 manual specifies 16 active degrees of freedom,
  21 joints (16 active plus 5 passive), linkage transmission, about 1.2 kg
  mass, 500 Hz communication, 0.9 s open-close time, per-joint range/speed,
  and 18 N thumb/20 N finger maximum fingertip force. Physics-v1 uses those
  values as command envelopes, but does not infer motor inertia, drive gains,
  backlash, follower compliance, or contact-pad friction from the brochure.
- The tomato contact paper reports a mean static coefficient of 0.305 on its
  rubber surface (0.284--0.344), and mean compression modulus 0.11 GPa,
  rupture force 51.27 N, and rupture deformation 9.52 mm. Those values are
  useful order checks, not a direct mapping to the user's pad or to a single
  PhysX spring/damper. The nominal 0.55 friction and 5000 N/m, 25 Ns/m contact
  are therefore explicitly marked calibration priors.
- The pedicel study reports mean pull-off forces 40.262, 44.781, and 72.003 N
  at mean abscission diameters 5.52, 6.27, and 7.75 mm, plus a 13.299--51.685 N
  three-point bending-force range. Physics-v1 samples the same physical order
  and correlates diameter with strength, but cultivar-specific axial,
  bending, rate, and duration data remain mandatory.
- NVIDIA's PhysX guidance distinguishes mimic natural frequency/damping ratio
  from drive stiffness/damping and recommends simple speed, load, penetration,
  and jitter scenes before integrated manipulation. Physics-v1 follows that
  gate order. The current 150 rad/s and 1.2 mimic values are numerical priors,
  not fitted hardware parameters.

Sources:

- Linker Hand L20 manual V1.2:
  https://a.storyblok.com/f/298593/x/50a0fc9e84/linker-hand-l20-v10-20260324.pdf
- NVIDIA joint and mimic tuning example:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/gripper_tuning_example.html
- Tomato friction and compression measurements:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6475739/
- Tomato pedicel tensile/bending characterization:
  https://www.mdpi.com/2073-4395/14/10/2274

## Hardware-capture handoff

The machine-readable protocol is under `calibration_protocol` in
`simulation/config/tomato_pick_physics_v1.yaml`. No SDK/CAN connection is
performed by this work. The user's offline recordings should contain:

1. L20 neutral/open/close, slow sweep, fast sweep, and reversal trials with
   command timestamps, all active targets, all 21 measured joint positions and
   velocities, and current/effort if available; five repetitions each.
2. A7 isolated small steps per joint with command/measured position, velocity,
   effort, and Cartesian pose; five repetitions per joint.
3. Actual pad/tomato incline-slip, 1/2/3/5 mm compression, and two low-height
   rebound tests, with fruit mass, axes, ripeness, wetness, and damage labels.
4. At least 20 matched axial and 20 bending stem tests with diameter, ripeness,
   loading direction/rate, and full force-displacement-time curves.

The first hardware fit changes only physical parameters. Reward, observation, action,
success definition, and the user's revised five-dimensional manifold remain
fixed while each physical family repeats its isolated gate. A red
hardware-calibration gate blocks deployment, but no longer blocks this
explicitly provisional simulation-only training run.

## Gate order

1. pure profile and load-duration tests;
2. one-environment import and finite-state smoke;
3. hand free-motion speed/linkage audit;
4. tomato drop, compression, and incline-slip provisional isolation gate;
5. axial, shear, bending, and duration provisional stem gate;
6. 64-environment randomized stability test;
7. merge the user-recorded manifold and repeat gates;
8. old-model counterfactual inspection only;
9. 256-environment four-update numerical preflight;
10. explicit authorization for a new from-scratch PPO run.

## Reset workspace v1

Physics-v1 no longer resets around one arm pose with only plus/minus three
degrees of independent joint noise. The reset now samples continuously inside
a seven-dimensional A7 joint ellipsoid, applies the actual USD forward
kinematics, and only then generates one fruit around that palm inside a bounded
local approach cone. The fruit keeps full yaw variation and up to 15 degrees
of stem tilt; the resulting pull axis remains present in the existing actor
observation.

The first reviewed point cloud contains 1,024 seed-42 samples. Relative to the
robot environment frame, palm positions span approximately x 0.313--0.426 m,
y -0.492---0.238 m, and z 1.765--2.005 m. Corresponding fruit positions span
x 0.096--0.651 m, y -0.553--0.066 m, and z 1.534--2.228 m. Every sample was
finite, initial contact force was zero, and minimum hand-fruit collision
clearance was 15.35 mm. A 64-environment, 64-step randomized action gate then
produced no success/failure/timeout reset, no out-of-bounds event, and no
numerical error. At that historical v1-reset checkpoint the passive-link gate
was still red; the final transmission configuration documented above now
passes it without changing the 0.35 rad threshold.

Artifacts:

- point cloud PNG:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_1024.png`
- point cloud rows:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_1024.csv`
- machine summary:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_1024.json`
- one-environment reset log:
  `/tmp/tomato_physics_v1_reset_workspace_gate_1x16.log`
- 64-environment dynamic reset log:
  `/tmp/tomato_physics_v1_reset_workspace_gate_64x64.log`

## Reset workspace v2

The v1 cloud was deliberately replaced after review found its arm coverage too
conservative. Reset workspace v2 expands the A7 joint-space ellipsoid to
`[42, 50, 58, 34, 48, 46, 52]` degrees in A1--A7 order. Every palm point is
still obtained from the actual USD forward kinematics with a ten-percent joint
limit margin. Samples outside that margin are rejected and redrawn instead of
being clipped onto a limit, so the distribution has no artificial boundary
mass. The fruit's local approach displacement remains 0.05--0.09 m; the larger
global fruit cloud therefore comes from real arm coverage rather than inflated
fruit offsets.

The final reviewed seed-42 cloud contains 2,048 paired samples. Palm positions
span x 0.109--0.483 m, y -0.640---0.042 m, and z 1.577--2.215 m. The robust
5th--95th percentile palm height is 1.681--2.120 m. Corresponding fruit centres
span x -0.079--0.756 m, y -0.694--0.237 m, and z 1.378--2.425 m; their robust
5th--95th percentile height is 1.559--2.221 m. All samples were finite, no
sample was clipped to a joint radius, initial fingertip contact force was zero,
and minimum hand-fruit collision clearance was 15.35 mm.

A 64-environment, 64-step randomized dynamic gate produced no termination,
timeout, initial contact, numerical failure, or workspace-specific failure.
At that historical v2-reset checkpoint the passive mimic tracking gate was
still red. The reset expansion did not introduce it, and the final
transmission configuration documented above now passes the same threshold.
This historical v2 statement is superseded by the readiness section above.

Artifacts:

- final point cloud PNG:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_broad_v2.png`
- point cloud rows:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_broad_v2.csv`
- machine summary:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_broad_v2.json`
- actual Kit overlay:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_kit_overlay_broad_v2.png`

## Reset workspace v3: X extension with closure margin

Workspace v3 retains the v2 base A7 ellipsoid and adds one measured correlated
joint mode, `[8.4, 17.9, 3.7, 22.0, 0.9, 2.6, -8.7]` degrees in A1--A7
order. Its coefficient is bounded asymmetrically to `[-1.0, 0.30]`. The
positive shell above 0.30 is excluded because a 2,048-sample hard gate found a
fixed-orientation closure singularity there. This mode
was obtained from the local USD forward-kinematics sensitivity matrix to extend
the palm primarily along robot-frame X while avoiding unnecessary Y/Z growth.
Stronger 24, 26, and 32 degree candidates were rejected because at least one
extreme reset could not simultaneously satisfy collision clearance and the
closed-grasp reach margin.

Fruit reset is now fail-closed. It is sampled 4.5--6.0 cm beyond the configured
closed-grasp centre and is redrawn after any collision-clearance push unless
all of the following are true:

- closure translation norm is at most 6.5 cm, leaving 5 mm for an outward
  collision-clearance correction above the 6.0 cm sampled approach maximum;
- at least 25 percent of each Cartesian policy translation range remains;
- initial hand-fruit clearance is at least 15 mm;
- a full iterative USD Jacobian IK closure preserves at least 3 percent of
  every physical A7 joint range and has at most 8 mm Cartesian error.

The final 2,048-sample seed-42 gate passed 2,048/2,048 closures. The maximum
closure IK position error was 1.82 mm, the minimum post-closure physical joint
margin was 3.95 percent, and the minimum remaining X/Y/Z policy translation
headroom was 13.54/5.22/13.17 cm. Initial contact force was zero and minimum
clearance was 15.30 mm. Relative to v2, full palm X span increased from 0.374 m
to 0.402 m (7.6 percent) and robust 5th--95th-percentile palm X span increased
from 0.200 m to 0.217 m (8.6 percent). Full palm Y/Z spans changed by only
about +0.45/+0.25 percent. Fruit centres occupy x -0.042--0.693 m,
y -0.734--0.106 m, and z 1.390--2.449 m; their narrower local approach envelope
is intentional reach margin rather than a loss of global arm randomization.

The 64-environment, 64-step dynamic gate also passed every reset-specific
check, including the new closure-distance and Cartesian-headroom assertions.
At that historical v3-reset checkpoint the passive mimic gate and revised hand
manifold were still pending. Both are now merged and pass their focused gates;
no PPO training was started at that point. The later gates and current status
are listed in the readiness section above.

Artifacts:

- final v3 point cloud PNG:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_xy_extended_v3.png`
- final v3 point cloud rows:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_xy_extended_v3.csv`
- final v3 machine summary:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_seed42_2048_xy_extended_v3.json`
- actual Kit overlay:
  `logs/diagnostics/tomato_physics_v1_reset_workspace/physics_v1_reset_workspace_kit_overlay_xy_extended_v3.png`
- dynamic gate log:
  `/tmp/tomato_physics_v1_xy_extended_v3_gate64x64.log`
