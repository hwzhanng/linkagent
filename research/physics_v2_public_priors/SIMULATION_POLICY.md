# Physics-v2 simulation policy from public priors

This document turns the source values in `physics_v2_public_priors.json` into an implementable simulator policy. The intent is to prevent a common failure mode: copying biologically unrelated numbers into PhysX and calling the result calibrated.

## 1. Freeze Physics-v1 and create a new profile ID

Physics-v1/model_5150 remains the baseline. Any change to crop failure physics, fruit distribution, or the meaning of success creates a **new task/profile ID and requires from-scratch training**. Do not resume the v1 optimizer state after the MDP changes.

Suggested identity:

```text
LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1
```

## 2. Fruit sampler

### 2.1 Core harvest-ready distribution

Use S01 semi-ripe, early-firm and mid-late-firm profiles. Their axis/mass means and SDs are direct observations; the covariance matrix is not public. Therefore the correlation model itself must be declared an engineering assumption.

Recommended first implementation:

```text
profile ~ categorical([semi, early_firm, mid_late_firm], weights=[0.15, 0.55, 0.30])
z_size  ~ TruncNormal(0,1,-2,2)
eps_i   ~ TruncNormal(0,1,-2,2)
rho_corr = 0.85  # method hyperparameter, not a measured biological correlation
x_i = mean_i + sd_i * (rho_corr*z_size + sqrt(1-rho_corr^2)*eps_i)
mass = mean_mass + sd_mass * (rho_corr*z_size + sqrt(1-rho_corr^2)*eps_mass)
```

Why shared latent size? Sampling four independent uniforms for three dimensions + mass routinely creates implausible combinations (small fruit with large mass, thin fruit with a huge third axis). A shared latent preserves a physically plausible monotonic size relationship while being explicit that the exact covariance is unknown.

Reject/resample when any of the following holds:

```text
axis <= 0
ellipsoid aspect ratio outside [0.65, 1.20]
mass outside [0.05, 0.40] kg
implied density = mass / (4/3*pi*a*b*c) outside [800, 1200] kg/m^3
```

The density gate is deliberately broad. S02 gives fully-ripe 1000/1040 kg/m^3 for two cultivars; it should prevent absurd samples rather than force every cultivar into one density.

### 2.2 Large-fruit OOD

Do **not** mix the S04 Jiangshu No.1 152.7-378.2 g distribution into the core benchmark without labeling it. Use its measured lateral-diameter histogram as a robustness/OOD slice. Recommended curriculum:

```text
Phase A: 100% core public prior
Phase B after stable success: 80% core + 20% S04 large-tail OOD
Final evaluation: report core and large-tail separately
```

Weights are method choices, not observed greenhouse prevalence.

## 3. Crop geometry sampler

Sample abscission-zone diameter first:

```text
d_az ~ Uniform(5.0, 8.0) mm
```

This is maximum-entropy over the majority range reported by S05 because no continuous histogram was published.

For visible pedicel length, use S07 greenhouse varietal means as geometry randomization:

```text
L_pedicel ~ Uniform(29.2, 38.8) mm
L_from_AZ ~ Uniform(7.0, 10.4) mm
```

These are long-shelf-life varieties; they should affect visual/compliance geometry but must not silently redefine the S05 strength distribution.

## 4. Pure axial strength conditional on diameter

Interpolate between the S05 nodes:

```text
(d_mm, mean_N, sd_N)
(5.52, 40.262, 12.437)
(6.27, 44.781, 15.156)
(7.75, 72.003, 23.401)
```

For each sampled `d_az`, linearly interpolate mean and SD, then sample a positive truncated normal. Use e.g. `[max(5, mean-2.5*sd), mean+2.5*sd]` only as a numerical guard; log the amount of truncation.

**Do not keep Physics-v1's 4-8 N as the `Ft*` pure axial limit.** It can remain only as a historical effective-break baseline.

## 5. Bending threshold

S05 gives a 12-mm-span three-point bending experiment. For a center point load:

```text
M_break = F_break * span / 4
```

Derived means:

```text
5-6 mm AZ: 0.067152 N m  (SD 0.019167)
6-7 mm AZ: 0.112134 N m  (SD 0.014073)
7-8 mm AZ: 0.124821 N m  (SD 0.029559)
```

Alternatively use the source regression `F_N = -21.364 + 8.710*d_mm` in 5-8.5 mm and convert to moment. The grouped distribution is preferred because it retains uncertainty.

## 6. Torsion threshold and twist angle

S06 is old but provides scarce direct tomato torsion evidence. Convert kgf*cm to N*m using `0.0980665`.

Use broad qualification envelopes, not one fixed value:

```text
Mt* core sensitivity: 0.05-0.30 N m
Mt* OOD envelope:     0.02-0.40 N m
angular test rate:    0.5-1.5 rad/s
```

Representative suitable-maturity values from that study are ~0.167 N*m (H1370) and ~0.127 N*m (Chico). Twist-to-detachment ranges were 50-500 deg and 30-250 deg respectively, while the source's conservative harvesting conclusion was about 550 deg.

S04 supplies a newer behavior target: best-picking-state fruits predominantly required 360-540 deg rotation. Use **both**: S06 bounds the torsional material envelope; S04 defines the target harvesting behavior envelope.

## 7. Proposed multi-mode abscission model

Use a dimensionless interaction ratio:

```text
u = sqrt(
    (Ft/Ft*)^2 +
    (Fs/Fs*)^2 +
    (Mb/Mb*)^2 +
    (Mt/Mt*)^2
)
```

Where:

- `Ft*`: public diameter-conditioned axial limit from S05.
- `Mb*`: public diameter-conditioned bending moment from S05.
- `Mt*`: sampled torsional limit constrained by S06.
- `Fs*`: **not directly identified** for the abscission zone. The S05 distal/proximal shear values are cutting/shearing-fixture data. For v2 research, use a declared sensitivity prior such as `Fs*/Ft* ~ Triangular(0.60, 0.74, 0.85)` and ablate it.

This is already more defensible than labeling a distal cutting force as AZ shear strength.

## 8. Twist-induced integrity loss: remove the artificial 4-8 N shortcut

The most useful cross-study constraint is:

```text
S05 pure axial at d~6.27 mm: 44.781 N mean
S04 after 360 deg rotation: 11.66 N mean, 15.29 N max
```

Thus a medium-size nominal target for residual axial capacity after a representative 360-deg twist is:

```text
C_360_mean = 11.66 / 44.781 = 0.2604
C_360_max  = 15.29 / 44.781 = 0.3414
```

Across the three S05 diameter nodes, the mean residual ratios are 0.290, 0.260 and 0.162. These are **cross-cultivar, cross-experiment targets**; report them as such.

A numerically stable first mechanism is multiplicative integrity:

```text
C <- C * exp(-k_theta * abs(dtheta) * clamp(abs(Mt)/Mt*,0,1)^q)
capacity_now = C * capacity_intact
```

With `q=1`, a nominal 6.27-mm calibration that leaves `C=0.2604` after `2*pi` rad at full normalized torsion is:

```text
k_theta = -ln(0.2604)/(2*pi) = 0.2142 rad^-1
```

This is **not claimed as a biological constant**. It is a method parameter identified against two public qualification targets. Sweep it so the post-360-deg pull envelope covers the cross-study uncertainty.

This yields a key desired behavior:

```text
pure pull -> requires tens of newtons
controlled twist -> damages/weakens abscission layer
then moderate pull/bend -> completes separation
```

That is physically and scientifically much stronger than lowering the intact axial threshold to 4-8 N.

## 9. Subcritical duration/fatigue

S08 supports combined loading and reciprocating-twist fatigue, but it does not justify a unique fatigue exponent/time constant in this package. Keep the duration law explicitly as a method hyperparameter:

```text
if u > onset:
    D_load += dt/tau * ((u-onset)/(1-onset))^p
else:
    D_load recovers slowly or stays constant depending on ablation
```

Initial research defaults for a numerical preflight can be:

```text
onset = 0.55
p = 2.0
tau = 0.20 s
instantaneous_fail = 1.25
```

But every paper table/figure must label these as **model hyperparameters**, and an ablation/sensitivity study is required. The public-measurement contribution is the qualification envelope, not these constants.

## 10. Fruit damage model

Do not use a single hard `fingertip_force > X` derived from a paper.

S01 shows why:

- 2-mm probe exocarp rupture is only ~4.5-9.2 N depending on ripeness;
- 100-mm-platen whole-fruit rupture is ~85-141 N for harvest-ready stages and much larger for green fruit.

The geometry differs by orders of magnitude in contact area.

For Physics-v2 v1, use three levels:

1. **Safety/termination limit from the real hand** (hardware calibrated, not public).
2. **Public damage proxy** based on contact pressure/area or deformation, used as an auxiliary metric rather than a hard biological truth.
3. **OOD damage evaluation** using puncture/platen/tissue references to ensure learned policies do not exploit unrealistic penetrations.

If a later deformable-fruit model is added, S12/S13 provide tissue-level modulus/failure-stress/fracture-energy ranges. Do not feed those values directly to `RigidBodyMaterialCfg.compliant_contact_stiffness`.

## 11. Friction policy

Public tomato static friction against rubber (S03) has mean ~0.305. It is not the G20 fingertip coefficient.

Therefore:

```text
public package: use 0.28-0.34 only for sensitivity smoke tests
Physics-v2 transfer profile: block until actual G20-pad x tomato friction is measured
```

This prevents the literature package from overriding the hardware-calibration requirement.

## 12. Task/world orientation

At world-generation level use S09:

```text
front 0.119
back  0.111
left  0.063
right 0.706
```

Do **not** expose the categorical greenhouse side as a privileged shortcut to the local policy. The planner should create the grasp frame; the PPO should observe the pull/stem direction in that local frame.

If continuous region bounds from S09 Figure 3 are not reconstructed robustly, sample maximum-entropy within declared quadrants and label the within-quadrant distribution as an assumption.

## 13. Hand-manifold priors

Public human data supports, but does not uniquely define, a robot eigengrasp:

- S04: 85% semi-enclosed; thumb/index/ring are the main force digits in that experiment.
- S11: distal thumb/index/middle consistently have the largest pressures; thumb distal contacts first.
- S10: smaller fruit shifts toward pinch-like grasps, larger fruit toward palm/power grasps.

For PPO-v2:

```text
keep current 5D as baseline
add per-action saturation audit
if finger synergies saturate, test 7-8D with differential finger residuals
avoid jumping directly to 16D unless 8D fails
```

Suggested 8D extension:

```text
existing 5D
+ index-middle differential
+ ring-little differential
+ thumb proximal-distal differential
```

This is a paper ablation, not a public physical parameter.

## 14. Public qualification suite before PPO

Implement these tests **without a learned policy**:

### Q1: morphology

Sample >=100k fruit profiles. Report axis/mass/density/aspect histograms, rejection rate, and source-profile overlap.

Pass if no physically impossible samples appear and each source mean lies inside the generated support.

### Q2: pure axial abscission

For d=5.52/6.27/7.75 mm, scripted axial loading at ~1 mm/s should reproduce the corresponding mean/SD envelopes from S05.

### Q3: bending

Reproduce 12-mm-span center-load moments for the three diameter groups.

### Q4: torsion

At 0.5-1.5 rad/s, sampled torque/angle outcomes must stay inside the broad S06 envelope. Do not tune every sample to a single old cultivar.

### Q5: twist then pull

Script 360-deg twist then axial pull. For the 6-7 mm nominal slice, target residual pull around 11.66 N with a public observed maximum 15.29 N as a medium-confidence envelope. Report sensitivity to `k_theta`.

### Q6: strategy sanity

Compare peak grasp/attachment load for:

```text
pure pull
pure bend
pure twist
simultaneous twist+pull
staged twist-then-pull
```

The simulator should make multi-mode harvesting advantageous without making detachment free. If pure pull remains easiest at low force, the model has failed its purpose.

## 15. What should enter PPO only after these gates pass

Only after Q1-Q6 pass:

- correlated fruit/pedicel randomization;
- multi-mode attachment state;
- diameter-conditioned strength;
- twist integrity state;
- public-task orientation distribution;
- measured hardware randomization overlays (separate profile).

Then train from scratch and compare against frozen Physics-v1/model_5150.

The main scientific target is **smaller Sim->Real gap and a physically interpretable detach strategy**, not merely a higher simulator success percentage.
