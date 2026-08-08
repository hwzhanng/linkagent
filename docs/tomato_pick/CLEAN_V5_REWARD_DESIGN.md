# Tomato Pick Clean-v5 reward design

Status: **formal from-scratch training completed; fixed multi-seed release selected**.

Formal run:

- training directory: `/home/descfly/linkagent/logs/rsl_rl/tomato_pick_clean_v5/2026-08-08_02-27-24_tomato_pick_clean_v5_from_scratch`;
- log: `/tmp/tomato_clean_v5_formal_8192.log`;
- W&B run: `1vp9dp6x`;
- selected checkpoint: `model_1350.pt`;
- fixed deterministic evaluation: 96.1805% success over 26,077 episodes and
  seeds 17, 42, and 2026;
- worst-seed success: 96.0623%; 95% Wilson CI: 95.9410%–96.4065%;
- reproduction and continuation guide:
  `PPO_CLEAN_V5_MODEL1350_REPRODUCTION.md`.

## First principles

1. A random policy must be able to distinguish approach, contact, opposition,
   loading, detach, retention, and success.
2. Reversible motion cannot farm progress: reach, grasp, load, stable detach,
   and retention progress are paid only for a new per-episode maximum.
3. A single-finger collision receives no detach credit. Two fingers receive
   partial credit; thumb plus a support finger receives full credit.
4. Early learning must not see late failure penalties before successful
   trajectories exist. Late learning must not profit from immediate detach or
   repeated hold steps.
5. Partial attempts remain ordered. They are not all collapsed to the same
   terminal value, which previously taught PPO to avoid the fruit.

## One continuous objective

There is no phase machine and no scripted action. A single scalar `alpha`
smoothly interpolates the vulnerable back-end credit while leaving the bounded
front-end bridge intact.

Always active:

- reach best-progress: `5 * delta_best(reach)`;
- grasp best-progress: `10 * delta_best(second-strongest fingertip contact)`;
- opposed-contact best-progress: up to `5`;
- directional stem-load best-progress: `20 * delta_best(load * opposition)`.

Discovery endpoint (`alpha=0`):

- qualified break: up to `40` (`0` single finger, `0.4` any two fingers,
  `1.0` thumb plus support finger);
- best-ever retention progress: up to `40` (equivalent to `2` for each new
  longest hold step, but repeated or reset hold streaks earn nothing);
- success: `100`;
- no drop/excess/timeout penalty, plus a minimal `-0.01` non-terminal time
  cost so lazy timeout is strictly dominated by useful progress.

Aligned endpoint (`alpha=1`):

- stable detach after three retained policy steps: up to `40`;
- success: `200`;
- drop: `-40`; excess force / workspace failure: `-60`;
- pre-detach timeout: `-10`; post-detach timeout: `-30`;
- non-terminal time cost: `-0.1` per policy step.

Immediate break is multiplied by `1-alpha`; stable detach and failures are
multiplied by `alpha`. Bounded opposed/retention progress is used at both
endpoints, the time cost grows from `-0.01` to `-0.1`, and success grows
continuously from 100 to 200.

## Smooth schedule

| PPO update | alpha |
|---:|---:|
| 0-250 | 0.00 |
| 400 | 0.138 |
| 600 | 0.35 |
| 800 | 0.55 |
| 1000 | 0.75 |
| 1150 | 0.875 |
| 1300+ | 1.00 |

Interpolation uses smoothstep, so reward scale and its derivative do not jump
at a knot. The historical first success occurred near update 29; alignment is
therefore deliberately zero through update 250. A resumed run must set
`TOMATO_REWARD_ITERATION_OFFSET` to its checkpoint iteration.

## Physical and task predicates

- policy: 20 Hz; physics: 120 Hz;
- stem fracture: one overloaded physics substep, matching the recorded 64%
  discovery environment rather than silently making the sparse event harder;
- qualified detach credit requires any two fingertips at 0.60 N and gives full
  credit only to thumb plus a support finger;
- retained/success is the physical outcome: detached fruit within the hand
  distance bound, low relative speed, no drop or excess force. Load may move
  between fingertip sensors and the palm without resetting a valid hold;
- success: 20 consecutive retained policy steps.

## Audited behavior

Constructed discounted trajectories pass all ordering assertions:

| Trajectory | alpha=0 | alpha=1 |
|---|---:|---:|
| safe wait | -0.80 | -10.00 |
| single-contact drop | 12.00 | -28.00 |
| two-finger break then drop | 41.00 | -15.00 |
| thumb-opposed break then drop | 80.00 | 0.00 |
| stable hold 3 then drop | 85.84 | 45.87 |
| hold 19 then drop | 114.24 | 78.69 |
| hold 19 then excess | 114.24 | 61.92 |
| hold 20 and succeed | 197.66 | 276.63 |

Thus discovery strongly rewards the first successful chain. At the aligned
endpoint, collision and excess are worse, stability is ordered, and complete
success has a roughly 198-point margin over a 19-step drop.

The 8192-environment from-scratch front-end gate reached its first complete
success at update 32. At update 29 it had detach 63.89%, retained 0.11%, and
timeout 29.17%; the recorded 64% run first succeeded at update 29 with detach
65.83% and retained 0.03%. The three-update delay is therefore a sparse-event
variation, not avoidance or a broken behavior chain. All progress terms are
new-best deltas, so their observed growth cannot be produced by cycling.

## Gates completed

- 17 focused MDP/reward tests passed.
- Constructed trajectory reward audit passed every ordering assertion.
- 8192-environment from-scratch learning gate produced retained states from
  update 0 and its first success at update 32 without NaN, Inf, PhysX overflow,
  OOM, or reset failure.
- No SDK, CAN, or real robot was accessed.
