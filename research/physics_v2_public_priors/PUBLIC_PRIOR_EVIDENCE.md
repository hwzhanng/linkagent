# Public-prior evidence audit

## Evidence rules

A number appears in the Physics-v2 machine profile only if its meaning and test geometry are recorded. Numbers from different fixtures are not treated as interchangeable.

The key distinction is between **crop material evidence** and **robot/contact-pair evidence**. Fruit size, mass, crop geometry and abscission mechanics can be initialized from public sources. A7/G20 dynamics and G20-pad/tomato contact cannot.

## 1. Fruit geometry, mass and maturity

### S01 - Syngenta Sibede, four ripening stages

Physical sample size was 14 fruit per stage.

| stage | major diameter mm | minor diameter mm | height mm | mass g |
|---|---:|---:|---:|---:|
| green | 70.73 +/- 6.18 | 66.63 +/- 6.78 | 57.86 +/- 4.74 | 154.116 +/- 41.599 |
| semi-ripe | 65.83 +/- 4.30 | 63.78 +/- 3.74 | 61.77 +/- 4.87 | 151.347 +/- 24.799 |
| early firm | 72.15 +/- 8.11 | 69.39 +/- 7.65 | 57.00 +/- 3.12 | 167.489 +/- 43.937 |
| mid-late firm | 71.70 +/- 6.69 | 68.54 +/- 5.63 | 62.07 +/- 20.44 | 156.586 +/- 44.144 |

The source reports suspiciously large maxima for height in the early/mid-late groups (112.43 and 132.61 mm). Physics-v2 therefore uses mean/SD plus an aspect-ratio gate instead of copying the raw maximum.

Mean ellipsoid volumes derived from the three mean axes are 135.8, 149.4 and 159.7 cm^3 for semi/early/mid-late. This is a geometry approximation, not a measured water-displacement volume.

### S02 - CV-613 and Naveen

Thirty fruits were used for dimensional measurement. Fully-ripe values:

| cultivar | geometric mean diameter mm | mass g | true volume cm^3 | true density kg/m^3 | firmness N/mm |
|---|---:|---:|---:|---:|---:|
| Naveen | 56.3 | 113.0 | 118 | 1000 | 1.45 |
| CV-613 | 52.1 | 114.8 | 110 | 1040 | 1.28 |

Semi-ripe density was 940 and 1030 kg/m^3. The study shows maturity/cultivar dependence, so a single universal density is not justified.

**Simulation conclusion:** sample geometry and mass together; derive implied ellipsoid density and reject extreme combinations. Use public density as a broad consistency gate.

### S04 - Jiangshu No.1, large-fruit tail

Two hundred fruit:

- lateral diameter 66.4-92.7 mm;
- longitudinal diameter 52.6-81.3 mm;
- lateral-height dimension 35.1-62.7 mm;
- mass 152.7-378.2 g.

Measured lateral-diameter proportions were 1%, 6.5%, 36%, 31.5%, 22%, and 3% in 65-70, 70-75, 75-80, 80-85, 85-90, 90-95 mm bins.

**Simulation conclusion:** preserve this as a named large-fruit robustness/OOD distribution rather than silently mixing it into the nominal cultivar prior.

## 2. Fruit damage and deformation

### S01 - 2 mm puncture

Mean exocarp rupture force (1 mm/s):

- green: 9.224 +/- 0.901 N;
- semi-ripe: 5.996 +/- 0.952 N;
- early firm: 5.408 +/- 0.688 N;
- mid-late firm: 4.521 +/- 0.654 N.

### S01 - whole-fruit 100 mm platen compression

Mean rupture force:

| stage | axial N | radial N |
|---|---:|---:|
| green | 708.68 | 534.97 |
| semi-ripe | 116.24 | 109.71 |
| early firm | 141.10 | 98.91 |
| mid-late firm | 113.60 | 85.33 |

This apparent conflict with 4-9 N puncture force is expected: probe/contact geometry and failure mode are completely different.

**Simulation conclusion:** neither table is a direct `maximum_fingertip_force_n`. Use them as damage/OOD constraints and validate actual fingertip contact separately.

### S12/S13 - tissue/fracture reference

S12 reports ranges across two tomato varieties:

- exocarp tensile failure stress 0.421-0.582 MPa;
- exocarp elastic modulus 4.601-9.59 MPa;
- mesocarp compression failure stress 0.122-0.229 MPa;
- mesocarp modulus 0.726-0.846 MPa;
- locular gel compression failure stress 0.012-0.016 MPa;
- locular gel modulus 0.048-0.124 MPa.

A separate validated XFEM study (S13, light-red JinshengF1) used exocarp E=20.35 MPa, Poisson ratio=0.4, failure stress=1.44 MPa and fracture energy=4.32 mJ/mm^2.

The spread itself is important evidence of cultivar/ripeness/model dependence.

**Simulation conclusion:** do not convert these Young's moduli into a single PhysX contact spring. Keep them for a future deformable-fruit model and damage sensitivity tests.

### S14 - impact/bruise OOD

Tomato bruise models were studied at 23, 71, 158 and 216 mJ impact energies, with risk depending on ripeness, impact location, impact duration, restitution and mass.

**Simulation conclusion:** use impact energy as an OOD damage metric; no one of those four values is a universal safe threshold.

## 3. Abscission-zone geometry and pure axial strength

### S05 - tensile test

Syngenta Spectrum, early firm, 24 pedicels across three groups, 1 mm/s loading.

| mean AZ diameter mm | pull-off N | tensile strength MPa |
|---:|---:|---:|
| 5.52 +/- 0.221 | 40.262 +/- 12.437 | 1.691 +/- 0.554 |
| 6.27 +/- 0.207 | 44.781 +/- 15.156 | 1.441 +/- 0.427 |
| 7.75 +/- 0.451 | 72.003 +/- 23.401 | 1.534 +/- 0.511 |

The majority AZ diameter was reported in the 5-8 mm range.

**Simulation conclusion:** sample diameter first and condition intact tensile capacity on diameter. A 4-8 N uniform break force is not a defensible pure-axial prior.

## 4. Bending

S05 performed a three-point test with a 12 mm support span and 1 mm/s loading.

| AZ group mm | break force N | derived M=F*L/4 N m |
|---|---:|---:|
| 5-6 | 22.384 +/- 6.389 | 0.067152 +/- 0.019167 |
| 6-7 | 37.378 +/- 4.691 | 0.112134 +/- 0.014073 |
| 7-8 | 41.607 +/- 9.853 | 0.124821 +/- 0.029559 |

The paper also reports `F_N = -21.364 + 8.710*d_mm`, R^2=0.707 for d in about 5-8.5 mm.

**Simulation conclusion:** use the grouped moment distributions as the bending failure prior; use the regression as a diagnostic/cross-check.

## 5. Pedicel cutting/shear: do not misuse

S05 measured roughly 62.054 +/- 15.727 N at a proximal pedicel section and 33.241 +/- 7.962 N at a distal section under its shearing fixture; it also showed that most distal/proximal diameters were about 3-5 mm.

This is highly relevant to a **cutting** end-effector. It is not the same experiment as applying lateral shear at the abscission zone during a dexterous twist/bend/pull maneuver.

**Simulation conclusion:** retain these values in the evidence database, but label `Fs*/Ft*` in the multi-mode abscission model as a method uncertainty until direct AZ shear evidence or real crop measurements are available.

## 6. Torsion

### S06 - direct torsion

H1370 and Chico tomatoes; test angular speed 0.5-1.5 rad/s.

Source maximum torsional-moment ranges:

- H1370: 0.5-4 kgf cm = 0.0490-0.3923 N m;
- Chico: 0.2-3 kgf cm = 0.0196-0.2942 N m.

Suitable-maturity representative values:

- H1370: 1.7 kgf cm = 0.1667 N m;
- Chico: 1.3 kgf cm = 0.1275 N m.

Twist-to-detachment ranges:

- H1370: 50-500 deg, suitable-maturity typical ~270 deg;
- Chico: 30-250 deg, suitable-maturity typical ~160 deg.

The paper's conservative conclusion states that approximately 550 deg rotary motion is necessary for harvesting tomatoes.

**Simulation conclusion:** use these broad values to constrain torsional capacity and rate sensitivity; do not overfit the old cultivars.

## 7. Twist-assisted detachment - the key Physics-v2 bridge

### S04 - manual strategy

For 200 manually picked tomatoes, rotation before separation:

| angle | count | fraction |
|---|---:|---:|
| <=180 deg | 12 | 6.0% |
| 180-360 deg | 33 | 16.5% |
| 360-540 deg | 155 | 77.5% |

For 50 best-picking-state tomatoes, after 360 deg rotation:

- instantaneous axial pull mean 11.66 N;
- maximum 15.29 N;
- pulling distance ~3-5 cm.

### Cross-study calculation

Divide 11.66 N by S05 pure-axial mean capacity:

| AZ diameter node | residual mean ratio after 360 deg | 15.29 N / intact mean |
|---|---:|---:|
| 5.52 mm | 0.290 | 0.380 |
| 6.27 mm | 0.260 | 0.341 |
| 7.75 mm | 0.162 | 0.212 |

This calculation crosses cultivar, ripeness and test protocol, so it is **not** a measured constitutive law. It is nevertheless far more defensible as a Physics-v2 qualification target than inventing a low intact tensile threshold.

For the 6.27-mm nominal case, a multiplicative integrity law `C=exp(-k_theta*theta)` fitted to `C(2*pi)=0.2604` gives `k_theta=0.2142 rad^-1` at full normalized torsional utilization. This value is therefore a *model parameter identified against public behavioral/mechanical endpoints*, not a plant constant.

## 8. Fatigue and combined loading

S08 concludes that:

- load transmitted to the abscission layer differs from motive input;
- combined loading is extremely important across plucking patterns;
- reciprocating twisting can involve fatigue break;
- longer/thicker peduncle/pedicel tends to make plucking more difficult.

No sufficiently matched public numeric fatigue exponent/time constant was found for the target abscission zone.

**Simulation conclusion:** fatigue must be a method term with explicit ablation. Tune it to Q-tests, do not label its constants as literature measurements.

## 9. Pedicel length

S07, 12 long-shelf-life tomato varieties:

- greenhouse pedicel length mean 3.32 cm; varietal mean range 2.92-3.88 cm;
- greenhouse distance from abscission layer mean 0.90 cm; range 0.70-1.04 cm;
- open-field corresponding means were 2.44 cm and 0.81 cm.

**Simulation conclusion:** useful geometry randomization, medium confidence for another cultivar. It should not redefine mechanical strength.

## 10. Greenhouse orientation/task distribution

S09 measured 378 bunches:

| peduncle orientation | count | probability |
|---|---:|---:|
| front | 45 | 11.9% |
| back | 42 | 11.1% |
| left | 24 | 6.3% |
| right | 267 | 70.6% |

The same work describes ripe-tomato working heights around 1150-1630 mm under its greenhouse setup and main stems 30-60 deg relative to ground.

**Simulation conclusion:** categorical orientation belongs in the global world/task distribution. The local PPO still receives a normalized grasp frame/pull direction so it does not memorize greenhouse side.

## 11. Human grasp priors

### S04 - 200 samples

- fingertip pinch: 10%;
- semi-enclosed: 85%;
- full coverage: 5%.

Among 20 pinch examples, 75% were <75 mm. All 10 full-coverage examples were >90 mm. The study identifies thumb/index/ring as the principal force digits in its semi-enclosed grasp analysis.

### S10 - size vs grasp type

Reported average major diameters associated with common grasp choices:

- power palm-thumb abduction: 79.8 +/- 12.9 mm;
- power pad-thumb abduction: 71.7 +/- 10.4 mm;
- pinch: 61.2 +/- 11.1 mm.

### S11 - instrumented glove

Sixty greenhouse tomatoes:

- complete pick duration 2.28-4.54 s;
- fruit-stem separation 0.26-1.56 s;
- release 0.34-2.02 s;
- distal thumb entered first;
- distal thumb, middle and index regions made the largest pressure contributions.

**Simulation conclusion:** the current low-dimensional hand approach is biologically sensible, but a size-conditioned/differential-finger extension should be tested if action saturation shows the 5D manifold is the bottleneck.

## 12. Friction public reference

S03 measured static friction of tomato against flat material surfaces:

- galvanized: mean 0.249 (0.221-0.286);
- aluminum: 0.212 (0.194-0.249);
- wood: 0.364 (0.338-0.404);
- rubber: 0.305 (0.284-0.344).

**Simulation conclusion:** this is not G20 fingertip-pad friction. The rubber interval can be used for a literature sensitivity smoke test; transfer remains blocked until the real pad/cultivar/wetness pair is measured.

## 13. Public-data gaps discovered by the audit

The research did **not** find a source strong enough to freeze the following as biological truth:

1. a continuous natural pedicel tilt distribution for the exact local tomato-picking task;
2. an abscission-zone shear failure distribution under multi-axis dexterous loading;
3. a numeric cyclic-fatigue law (onset, exponent, recovery) matched to the target cultivar;
4. effective PhysX compliant-contact stiffness/damping for a G20 pad and tomato;
5. G20-pad/tomato static/dynamic friction;
6. robot-specific motor/controller latency, damping, friction or torque transfer.

The correct response is to keep these gaps explicit, not fill them with unrelated literature numbers.

## 14. Direct paper hypothesis enabled by these priors

A defendable Physics-v2 question is:

> Can a dexterous robot learn lower-load, more stable tomato detachment by training on a diameter-conditioned, multi-mode abscission model in which torsion progressively reduces residual tensile/bending capacity, rather than training on an artificially low fixed break force?

The public qualification targets in this folder let that hypothesis be tested before any real-robot trial.
