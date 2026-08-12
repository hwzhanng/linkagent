# Physics-v2 engineering-prior supplement

This file fills the six public-data gaps with **bounded simulation priors**, not
new biological or hardware claims.  The machine-readable source of truth is
`ENGINEERING_PRIOR_SUPPLEMENT.json`.

| Gap | Simulation distribution | Why this is defensible now | Replacement evidence |
|---|---|---|---|
| Continuous pedicel pose | measured S09 category probabilities; category azimuth `Normal(0,18 deg)` truncated to `+/-45 deg`; polar tilt `Triangular(0,8,15 deg)` | smooths the measured categories without leaving the already-qualified 15-degree workspace | target greenhouse 3-D pedicel vectors |
| AZ shear | `Fs*/Ft* ~ Triangular(0.60,0.7423,0.90)` | the mode is the adjacent-section S05 ratio `33.241/44.781`; the broad support carries the fixture mismatch as uncertainty | direct lateral AZ tests |
| Cyclic fatigue | onset `Triangular(.45,.60,.75)`; exponent `Triangular(1.5,2,3)`; time `Triangular(.20,.55,1.20) s` | S08 supports the mechanism; the public human separation stage bounds the order of time | target-cultivar cyclic traces |
| Pad-fruit friction | effective static `Triangular(.30,.45,.70)`; dynamic/static `0.85` | contains the S03 rubber value `0.305` while retaining surface/pad uncertainty | deployed dry/wet incline-slip tests |
| Effective contact | stiffness `Triangular(3000,5385,12000) N/m`; coupled damping `10-60 Ns/m` | the mode is the S03 whole-fruit rupture secant `51.27/0.00952 = 5385 N/m`, used only as a rigid-contact surrogate | deployed 1/2/3/5 mm compression curves |
| A7/L20 dynamics | A7 URDF effort/velocity envelopes; L20 manual force/speed envelopes; accepted implicit gains; uniform 0-2 policy-step latency | enough to train a robustness prior while keeping the true system-identification gap explicit | timestamped A7/L20 step and reversal logs |

The current checked-in vendor hand is Linker Hand **L20** even though older task
naming says G20.  The supplement records this explicitly so a later hardware
fit cannot silently target the wrong hand.

No SDK, CAN, robot or tactile device is used to create these values.  Hardware
transfer remains blocked until the replacement gates above pass.
