# Physics-v2 public priors

This folder freezes the first literature-derived crop/task prior package for the proposed Physics-v2 tomato-picking environment. It intentionally does **not** modify Physics-v1, its checkpoint, reward, action space, or robot USD.

## Scope

The package answers one question: **which quantities can be populated from public crop/mechanics evidence without using the user's robot for system identification, and how should those quantities enter simulation?**

It separates four evidence roles:

- `DIRECT_PRIOR`: public measurements can directly parameterize a crop/task distribution after unit conversion.
- `DERIVED_PRIOR`: a deterministic physical transform of a public measurement is used (for example three-point bending force -> center-load bending moment).
- `ENGINEERING_HYPERPARAMETER`: the public literature constrains the mechanism but does not identify a unique numerical value; the value must be treated as a model parameter and ablated, not as a biological measurement.
- `HARDWARE_CALIBRATION_REQUIRED`: pair-specific or robot-specific quantities must still be measured on the deployed A7/G20 system.

The package is **literature-prior only**. It is not a claim that the simulator is calibrated to the user's cultivar, fingertip material, actuator, or tactile sensor.

## Main conclusions

1. **Do not sample fruit mass independently of geometry.** Harvest-ready tomato dimensions and mass are strongly coupled. Physics-v2 should sample a correlated morphology profile and derive/check density from the sampled ellipsoid volume and mass. Public fully-ripe true-density measurements around 1000-1040 kg/m^3 provide a useful sanity gate, not a universal constant.

2. **Physics-v1's 4-8 N break range must not be described as pure axial abscission pull-off force.** A modern pedicel study measured mean axial pull-off forces of 40.262, 44.781, and 72.003 N for mean abscission-zone diameters 5.52, 6.27, and 7.75 mm. Physics-v2 should preserve this diameter dependence.

3. **Twist-assisted harvesting provides the bridge from those large axial forces to feasible dexterous harvesting.** In a separate 200-fruit manual-picking study, 77.5% of best-picking-state tomatoes required 360-540 deg rotation. In a 50-fruit test, after 360 deg rotation the instantaneous pull force averaged 11.66 N and reached 15.29 N maximum. Relative to the 6-7 mm pure-axial mean above, the post-360-deg mean is about 0.260 of intact axial capacity. This cross-study ratio is a medium-confidence *calibration target* for a torsion-damage model, not a universal material constant.

4. **Physics-v2 should explicitly model tension + bending + torsion, with shear kept as an uncertainty term until an abscission-zone shear dataset is available.** Published distal/proximal pedicel shearing tests use a cutting/shearing fixture and should not be silently reinterpreted as abscission-zone shear strength.

5. **Natural task geometry is not uniform.** A 378-bunch greenhouse survey found peduncle orientations front/back/left/right at 11.9/11.1/6.3/70.6%. This belongs in the world/task generator. The local PPO skill should remain expressed in its grasp frame and should not learn greenhouse-side bias directly.

6. **Human behavior supports a semi-enclosed, thumb-opposed grasp manifold.** A 200-fruit study found 85% semi-enclosed grasps; another 60-fruit instrumented-glove study found the distal thumb, index, and middle regions carried the largest pressures and fruit-stem separation lasted 0.26-1.56 s. These observations support retaining a low-dimensional hand manifold while adding enough differential-finger authority for retention/slip correction.

7. **Fruit tissue mechanics are useful as damage/OOD constraints, not as a direct PhysX contact spring.** Whole-fruit platen rupture, 2-mm puncture, tissue Young's moduli, and fracture stresses have different test geometries. Do not convert them directly into a single fingertip force limit or `compliant_contact_stiffness`.

8. **Finger-pad/tomato friction remains hardware calibration.** Published tomato-on-rubber static friction (~0.305 in one study) is a useful sensitivity reference but is not the coefficient of the deployed G20 pad against the target cultivar/wetness state.

## Files

- `physics_v2_public_priors.json`: machine-readable SI-unit source data, recommended samplers, and source/role labels.
- `SIMULATION_POLICY.md`: direct implementation policy for the Physics-v2 environment and qualification tests.
- `PUBLIC_PRIOR_EVIDENCE.md`: detailed evidence audit, conversions, caveats, and what each datum is allowed to mean.
- `SOURCE_MANIFEST.json`: source identities, DOI/URLs, access notes, and evidence grades.
- `PUBLIC_PRIOR_TABLE.csv`: flattened high-value parameters for quick review.

## Recommended usage

Load `physics_v2_public_priors.json` only in a **new Physics-v2 profile ID**. Do not resume a Physics-v1 checkpoint after changing crop physics or failure semantics.

The proposed first Physics-v2 implementation should preserve the existing policy interface until the action/observation ablations are explicitly started. The first change is the crop physics and validation target, not a simultaneous rewrite of PPO, reward, tactile, and hand action space.

## Explicitly not supplied by this package

The following remain real-hardware work and must not be replaced by literature values:

- A7 joint/Cartesian controller step response and latency;
- G20 command-to-joint mapping and latency on the deployed hand;
- G20 free-motion/load response, damping/friction/effort envelope;
- deployed G20 fingertip-pad x deployed tomato friction;
- raw tactile -> physical force calibration if Newton-valued tactile is used;
- robot link/hand mass, CoM and inertia if the vendor asset cannot be trusted;
- camera/extrinsic/vision error model for the actual perception stack.

A public prior may be used to initialize a search range for these quantities, but it cannot upgrade `hardware_transfer_ready` to true.
