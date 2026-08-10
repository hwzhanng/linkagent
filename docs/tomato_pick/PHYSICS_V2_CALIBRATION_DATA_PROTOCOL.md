# Physics-v2 tomato calibration data protocol

This protocol prepares future physical measurements without inventing data and
without allowing a fit to modify the active simulation profile automatically.

Start from
`docs/tomato_pick/templates/tomato_physics_v2_calibration_dataset.template.json`.
An empty template is valid, but it produces no parameter proposal.

## Measurements per specimen

- Unique specimen ID and cultivar label.
- Ripeness index on the local documented 0–1 scale.
- Three ellipsoid semi-axes and mass.
- Dry or wet surface condition.
- Normal force at increasing indentation, including approximately 1, 2, 3,
  and 5 mm where safe.
- Incline slip angle using the deployed fingertip pad.
- Rebound height from at least two low drop heights.
- Axial stem separation force and bending separation moment in separate,
  explicitly labeled trials.

Loading speed, duration, operator, site, instrument IDs, and notes are part of
the evidence. Synthetic, estimated, and literature values must never be
labeled as measurements.

## CPU validation and fitting

```bash
python script/fit_tomato_physics_v2_calibration.py \
  --dataset /path/to/measured_dataset.json \
  --output /path/to/calibration_proposal.json \
  --require-review-ready
```

The fitter performs only transparent first-order calculations:

- density from ellipsoid volume and measured mass;
- zero-intercept linear contact stiffness per specimen;
- static friction as `tan(slip angle)`;
- restitution as `sqrt(rebound height / drop height)`;
- empirical p05/median/p95 stem loads and moments.

These fits match the current simplified Hookean model. They do not establish a
non-Hookean constitutive law or damage probability.

## Minimum evidence gates

- At least 10 specimens for geometry and contact stiffness.
- At least 10 dry and 10 wet slip trials.
- At least 20 rebound trials spanning at least two drop heights.
- At least 10 axial and 10 bending stem trials.

Passing these counts only makes the proposal ready for human review. It never
activates parameters, authorizes PPO training, or establishes cultivar-wide
ground truth. A reviewed profile revision must still pass CPU gates, exact
Isaac integration, and a new from-scratch PPO sequence.
