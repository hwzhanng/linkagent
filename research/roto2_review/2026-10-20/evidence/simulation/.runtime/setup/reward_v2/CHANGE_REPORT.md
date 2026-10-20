# Reward correction v2

User constraint: do not change core architecture.

## Confirmed failures

- With the old checkpoint, environment 0 over 10 seconds reported 2.426 turns by physics-substep angular velocity integration, while quaternion increments gave -0.00291 turns about world Z. This is not merely 20 Hz video aliasing. Raw six-component COM velocity and the derived angular velocity agreed exactly. The underlying simulator velocity/pose discrepancy is observed, not fully explained; the local Isaac Lab TGS configuration itself warns about noisy velocities. Physics settings were not changed.
- The old tactile code clipped any distal-link contact XY into the active rectangle and ignored depth/face. A link housing/backside contact was thus presented as sensor-pad contact.
- The old pressure-relief bonus could reward unload/reload cycles with the erroneous motion gate.

## Changes

- Reward and evaluation use unwrapped material-X heading from real object quaternions. Reward progress is growth of the episode's directed orientation high-water mark. Reverse motion, stalled progress and missing/lost pad support are penalized. Positive progress requires thumb and at least two other active pads and upright orientation.
- Pressure relief no longer gives positive reward; a small upper-tail pressure penalty remains. Existing pressure-Jacobian correction is unchanged; its motion measurement now uses pose change and its grip gate uses valid pad forces.
- Fingertip taxels now reject contacts outside the active rectangle and on the wrong side/depth. Shapes remain 5x12x6. Frames are fitted engineering priors from the vendor distal sensor-cap mesh, not hardware calibration. Surface force is still a filtered contact resultant and pressure remains a proxy.
- Evaluation independently reconstructs heading with scipy from saved quaternions, checks agreement with online accounting, and requires one net pose turn AND one supported new turn, >=65% active-pad support, >=95% upright, no physical failure and pressure below the existing limit.
- Video shows actual pose-derived rotation and colored active-pad corners; all markers are visual only.

## Unchanged

Network dimensions, actor input layout, four-frame stack, Encoder, GaussianPolicy, value network, PPO/GAE/optimizer implementations; upstream multimodal_rl source; PressureJacobianTrack algorithm; collision geometry, mass/inertia, actuation, solver settings. Reset grasp and domain curriculum were subsequently changed as documented below. New training starts from scratch; old results are preserved.

## Verification

CPU regression checks cover quaternion sign/wrapping, complete revolutions, static poses, side/back rejection, no-contact taxels, stall/release/reversal and repeated-arc reward exploits. Real Isaac Sim smoke: 8 environments x 20 control steps, finite values and unchanged 70/432 actor shapes. Old checkpoint under corrected sensing/reward is 0/64 successful; this is a diagnostic with a changed sensor contract, not a paired policy-performance comparison.

## Approach-guidance refinement (v3)

The early v2 checkpoint completed 64/64 full episodes without physical failure but had 0 secure-pad support and only 0.0188 average net turns. Survival had improved; the intended task had not. Therefore the next phase adds continuous reward-only geometry guidance before contact exists: active-pad distance to the ellipsoid surface and front-normal alignment toward the fruit. Contact quality and approach quality are each penalized relative to their ideal value (weight 0.15), so standing still cannot earn a positive return. Actual rotation still requires real active-pad contact. Failure penalty is increased to -60 so deliberately ending an episode cannot avoid the accumulated contact-deficit cost. This changes rewards only; there is no additional actor input, reset/physics change or optimizer modification.

## Contact-acquisition curriculum (v4)

After 95,998,320 cumulative corrected-reward transitions, independent v3 evaluation remains 0/64 successful, mean directed pose rotation 0.00440 turns, zero secure active-pad support. The model is NOT accepted. A contact-acquisition reward stage was therefore added as a training option, with unchanged policy inputs/network/PPO/reset/physics. It rewards actual pad coverage and full thumb+two-pad support, alongside real-rotation progress; rotation-stage stationary holding remains penalized. Approach geometry was changed from an exponential score with negligible guidance far from a pad to a linear distance cost. Contact-stage completion does not establish rotation success.

Independent grasp-search candidates were tested but NOT adopted: pose-centre geometry alone produced unstable or housing-supported grasps. One parallel diagnostic exhausted GPU memory and returned finite zero states; its result is explicitly invalidated. Training logs did not report the same error. Subsequent simulator diagnostics are serialized, logs checked for PhysX ABORT/allocator errors, and the physical probe now rejects collapsed all-zero joint state.

## Active-pad reset and low-noise curriculum (v5)

Real-physics CEM found a valid active-pad reset. Independent 64-environment zero-action checks give 96.72% secure support and 0 physical failures for nominal 64mm objects; gentle size/reset noise gives 62.99% support and 54/64 surviving. Full randomization remains unvalidated. These are reset validations, NOT policy results. A settled-state snapshot reset was separately tested (3.43% support), rejected, and its optional reset-target code removed.

Fresh seed44 rotation-stage 32M transitions with gentle randomization and default std/entropy failed: 0/64 success, 0.073% secure support, 62.5% failure. Fresh seed45 now uses nominal-size CONTACT curriculum, 16,384 environments, 32M transitions, initial action std 0.25 instead of 1 and entropy weight 0.005 instead of 0.05. These are CLI hyperparameter overrides; base YAML/network/PPO algorithm and pressure Jacobian implementation remain unchanged. A resumed run must explicitly retain entropy weight 0.005; restored checkpoint carries its learned std.
