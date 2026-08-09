# Physics-v1 abscission-v1 credit-alignment root cause

The from-scratch 8192 x 50 gate was numerically healthy but failed the task
gate: iteration 49 reached 68.90% detach and 66.22% drop, with no success.

The first stochastic replay captured 9,834 exact 120 Hz fracture events. Only
30.70% had secure support, 41.53% had thumb support, and 18.95% met the
distance-and-relative-speed retention condition. Mean relative speed at
fracture was 0.18072 m/s versus the unchanged 0.10 m/s success limit.

The independent overlap replay captured another 3,036 fractures:

- 950 (31.29%) had secure support;
- 787 (25.92%) passed relative speed;
- 584 (19.24%) passed both kinematic limits;
- only 277 (9.12%) had both secure support and valid kinematics;
- 2,149 (70.78%) already passed distance, so relative speed and support—not
  center distance—are the dominant blockers.

The physics is now capable of detachment, but the inherited Clean-v5 soft
opposition/load and break credit rewards unsupported or high-speed fractures.
That credit is inconsistent with Physics-v1's unchanged physical success
predicate. The single repair is a Physics-v1-only secure-support credit gate:
continuous reach/grasp/opposition guidance remains unchanged, while stem-load
progress requires current measured secure support and one-shot break credit
requires secure support at the exact fracture substep. No physical threshold,
reset, observation, action, 5D manifold, collision model, PPO parameter, or
abscission envelope is changed.
