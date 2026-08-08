# Tomato Pick Clean-v5: offline release and deployment pipeline

This document describes the fail-closed path from a PPO checkpoint to the
Agent runtime. None of the commands below opens CAN or a robot SDK unless a
hardware backend is explicitly selected and separately authorized.

## Frozen release

- Training run: `logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch`
- Hardened fixed evaluation: `logs/evaluations/tomato_pick_clean_v5_2026-08-08_fixed/release_manifest.json`
- Selected checkpoint: `model_1350.pt`
- Aggregate deterministic success: 96.1805% over 26,077 episodes and seeds 17, 42, 2026
- Worst-seed success: 96.0623%
- Frozen skill package: `artifacts/simulation/tomato_pick_clean_v5/model_1350_best`
- Runtime profile: `config/skills/tomato_pick_clean_v5_model1350_best.json`

`model_1600.pt` completed the same evaluation at 95.6390%, but is retained as
a candidate rather than promoted because it is not the aggregate best.

## Boundaries

The Agent depends on `RoboticsBundle`, not a mock implementation. The same
task/service composition can therefore select `mock`, `fake_gateway`, an
Isaac-owned bundle, `linkbot_g20_shadow`, or a future authorized hardware
bundle. Perception, grasp planning, motion safety, policy inference, and pick
verification are separate Protocol boundaries.

The independent verifier consumes tactile and explicit detach/drop evidence;
it never consumes PPO reward, policy logits, or a policy-reported success bit.

## Offline gates

Run the complete mock-to-production-adapter loop:

```bash
python -m unittest tests.test_fake_gateway_chain -v
```

Check the selected skill for simulation:

```bash
python script/check_tomato_deployment.py \
  --skill-profile config/skills/tomato_pick_clean_v5_model1350_best.json \
  --mode simulation --minimum-success-rate 0.90
```

The check validates every packaged SHA-256, frozen interface/layout versions,
the checkpoint hash, deterministic three-seed evaluation, the hardened
candidate-selection gate, and the requirement for an independent verifier.

## Hardware remains deliberately blocked

`linkbot_g20_shadow` is read-only and its lease manager rejects every command.
The hardware preflight will continue to fail until all of the following exist:

1. Real per-finger G20 force calibration and a released calibration ID.
2. Real-data detach, slip, drop, hold, and excess-force verifier thresholds.
3. Validated arm/hand topology and simultaneous-operation evidence.
4. A verified G20 stop/e-stop strategy and whole-system watchdog.
5. Explicit graded release of the selected skill for hardware.

The calibration utilities only normalize and fit externally recorded data;
they never open an SDK or CAN interface. Thresholds in
`config/verifiers/tomato_pick_real_pending.json` are placeholders and cannot
pass hardware preflight.
