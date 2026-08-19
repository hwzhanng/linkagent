# Physics-v2 first-success repair audit

Date: 2026-08-14

Status: code prepared for external audit; training remains stopped.

## Repairs 21-23: outcome discrimination after physical discoverability

Repairs 21 and 22 completed the same formal 8192 x 50 gate after replacing the
inherited 0.60 N-only opposition saturation with an equal blend of inherited
contact discovery and an actual mass/friction load-bearing target.  The
intervention improved grasp force and produced isolated secure/joint-ready
fractures, but neither run produced a retained transition or success.  Repair
22 ended with approximately 95.5% of detach outcomes becoming immediate drops.

The decisive reward audit found that the existing `DROP_FAILURE_WEIGHT=-40`
was multiplied by the general alignment alpha, which is exactly zero through
update 250.  Therefore every drop in the mandatory 50-update gate had zero
drop-failure cost even after physical grasp discoverability had improved.

Repair 23 makes one isolated correction: with the same audited training-only
environment switch, preserve zero drop penalty through update 10, then
smoothstep the existing drop penalty to its unchanged full weight by update
40.  It adds no positive term, changes no mechanics, and preserves the strict
termination, retention, hold, and success predicates.  The base/V1 default is
unchanged because its hook still returns the released alignment alpha.

The completed repair-23 gate rejected that correction: it reduced detach to
roughly 0.3-0.9% in the full-capacity tail, but every retained/hold/success
metric remained zero and tail joint-ready fractures fell to zero.  The change
therefore taught avoidance, not controlled retention, and is removed.

Repair 24 instead corrects the training mechanics shortcut exposed across
repairs 21-23.  Uniformly reducing tensile, shear, bending, and torsion made a
direct pull the easiest discovery strategy; twist-assisted breaks were exactly
zero and bend-assisted/moderate-pull breaks disappeared as capacity returned.
The new mode-aware curriculum keeps tensile and shear at 100% public capacity
from update zero, while bending and torsion alone start at 10% and smoothstep
to 100% by update 40.  Reward and strict outcome semantics remain unchanged.

The completed repair-24 gate validated the mode distinction but rejected its
schedule.  Updates 0-9 produced a mean 10.04% bend-assisted detach rate and
nonzero paper-candidate/retention-progress samples.  Once the ramp began at
update 10, mean detach fell to 2.10% in updates 20-29 and below 1% in the
full-physics tail before PPO learned a retained transition.  All 50 updates
were finite and the tail was fully public physics, but success and retained
were zero.  Repair 25 therefore changes only the curriculum timing: hold the
same mode-aware 10% bending/torsion scale through update 25, then smoothstep to
100% at update 40.  Tensile/shear remain 100% throughout, and reward, PPO,
termination, retention, success, and the gate budget remain unchanged.

Repair 25 was stopped after update 23 because its isolated timing change was
already causally harmful: detach rose to 28.98%, drop rose with it to 28.84%,
secure support at break was only 4.26%, and retained/success remained zero.
The extended global low bending capacity had made unsupported one-finger bends
the new shortcut.  Repair 26 keeps the same schedule but makes its temporary
bending/torsion assistance conditional on the existing strict thumb-plus-one-
other-finger support predicate.  Unsupported environments and all tensile/
shear modes stay at 100% public capacity.  At update 40 the conditional scale
is also 100%, so formal tail mechanics remain exactly public-prior mechanics.

The completed repair-26 gate was numerically healthy and kept drop/detach far
below repair 25, but it still produced no retained transition or success.  In
the assisted window, 93-99% of fractures had secure support while only 20-41%
also met the unchanged retained distance/speed predicates; as capacity ramped
back to public physics, joint-ready fractures collapsed to approximately zero.
Repair 27 therefore makes one further narrowing of the same training-only
mechanics curriculum: bending/torsion assistance requires both strict secure
support and current 120 Hz kinematic retention.  Every non-ready state remains
at full public capacity.  This changes no reward, PPO setting, physical prior,
retention predicate, hold duration, success threshold, or final tail physics.

Repair 27 was safely stopped after update 30 once its timing bottleneck was
decisive.  The joint gate worked exactly as designed: assisted-break and
joint-ready rates were equal throughout, joint-ready rose to 89.22% by update
25, and retention-progress reached its run maximum.  After the update-25 ramp
started, joint-ready fell to 9.38% and retention-progress reached zero by
update 30, before a complete hold was learned.  Repair 28 changes only the
timing of that now-validated joint-ready curriculum: hold 10% bending/torsion
through update 35, then smoothstep to exact public capacity at update 40.  The
mandatory update-40-to-49 public-physics tail remains unchanged.

Repair 28 completed all 50 updates and produced the first retained/success
transition at update 25.  Retention also recurred at updates 32, 33, and 35
while joint-ready fractures were approximately 93-98%.  The result still
failed the gate: as bending/torsion capacity returned to the public prior,
joint-ready, candidate, retention, and success collapsed to zero throughout
the update-40-to-49 public-physics tail.  The run was finite and error-free,
so this isolates transfer across the capacity schedule rather than numerical
instability or a broken retention predicate.

The reward audit then found a normalization incompatibility, not a missing
reward term.  The inherited best-ever stem-load score used the temporarily
scaled fracture capacity, so a 10%-capacity trajectory could exhaust its unit
load score before the public-capacity ramp.  Because best-ever progress cannot
pay again below that exhausted maximum, the learner lost the dense signal for
increasing physical load as the public threshold returned.  Repair 29 keeps
the same reward weights and topology but normalizes the instantaneous load
component against the fixed public capacities.  Assisted damage remains in
the existing sustained-damage quarter of the unit score, while the remaining
load-progress budget can now increase only as actual public-normalized load
increases.  Physical fracture still uses the unchanged training schedule and
the final ten updates remain exact public physics.

Repair 29 completed all 50 updates and validated the normalization correction
in isolation: public-normalized stem-load progress rose monotonically through
the capacity ramp and reached approximately 0.95 by update 48.  However, it
did not produce a complete hold or success, and joint-ready fractures were
almost absent in the public tail.  The learner had acquired larger public
loads but received that progress even when distance or relative speed had
already left the strict retained envelope.  Repair 30 therefore adds no term
or weight; it removes that incorrect credit.  During the audited assisted
training only, the existing public-normalized stem-load progress is paid when
the same current 120 Hz state satisfies strict secure support plus the
unchanged retained distance and relative-speed predicates.  Non-curriculum
evaluation keeps the released secure-support reward gate, and physical
fracture, retention, success, PPO, and the public tail are unchanged.

Repair 30 completed all 50 updates with finite metrics and no runtime error.
The stricter credit alignment worked during the low-capacity phase: break
joint-ready reached 97.70%, retained outcomes occurred at updates 20, 24, 25,
and 32, and public-normalized stem-load progress reached 0.22.  The remaining
failure was localized to curriculum pacing.  The five-update smoothstep from
update 35 to 40 reduced break joint-ready from 97.70% to zero and eliminated
retention, while public-normalized load progress continued to 0.6745 at update
49.  This is evidence against changing reward weights or final mechanics.
Repair 31 changes only the pacing: the identical 10%-to-100% smoothstep starts
at update 20 and still reaches exact public capacity at update 40, providing
20 adaptation updates while preserving the complete update-40-to-49 public
tail.

Repair 31 completed all 50 updates with finite metrics and no runtime error,
but produced no retained transition or success.  Extending the smoothstep did
remove the previous one-update cliff, yet joint-ready fractures declined
continuously as capacity rose: secure support, retained distance, and retained
speed each occurred intermittently but almost never intersected at fracture.
The exact-public tail ended with only 3.12% joint-ready fractures while
public-normalized load progress still reached 0.653.  This rejects another
schedule-only repair.

Repair 32 keeps the repair-31 schedule and unchanged strict boolean envelope.
Inside that envelope only, it modulates the existing public-normalized
load-progress credit by the geometric mean of center-distance and relative-
speed safety margins.  The modifier is exactly zero when either unchanged
retention boundary is reached, so it cannot provide weak credit outside the
strict predicate.  This is not the rejected continuous velocity-quality
substitution: it neither replaces break/retention reward nor creates a new
term.  It only allocates the same load-progress credit toward trajectories
that preserve both measured kinematic margins.  Physics capacities, schedule,
reward weights, PPO, retention, success, and the public tail are unchanged.

Repair 32 completed all 50 updates with finite metrics and no runtime error.
It produced one retained outcome at update 23, so the strict-envelope margin
was useful, but no success and no persistent tail retention.  Mean fracture
relative speed increased from 0.044 m/s over updates 0-19 to 0.694 m/s over
the exact-public updates 40-49, versus the unchanged 0.25 m/s retention limit.
The actor MDP audit then found a concrete observability defect: the feed-
forward actor received only the current perceived fruit-relative position and
could not observe fruit velocity or a previous position, although the critic
received exact fruit velocity and the gate judged actor behavior on fruit-to-
hand relative speed.  The actor therefore could not distinguish slow retained
loading from fast fruit escape in a single observation.

Repair 33 changes no reward, curriculum, mechanics, PPO setting, or outcome
predicate.  It extends only the from-scratch Physics-v2 actor from 66D to 69D
by appending the previous sample of the already inherited noisy perceived
fruit-center position relative to the palm.  The current/previous pair lets a
feed-forward actor infer relative motion from a deployable perception signal;
exact simulator velocity remains critic-only.  On the first post-reset sample,
previous equals current to prevent an artificial velocity impulse.  The
frozen Clean-v5/V1 66D actor, its exported policy, and all global interfaces
remain untouched.

## Superseded intermediate decision

The earlier decision in this position described the repair-41 torsion-only
curriculum. Repair 41 was subsequently stopped because its assisted torsion
fracture and break-credit rates stayed exactly zero, and repair 42 also
rejected cumulative axial-roll observation. That intermediate decision is no
longer the runnable candidate. The authoritative consolidated decision appears
after the repair-46 result below.

## Evidence from completed gates

| Repair | Completed updates | First success | Tail retained | Tail drop / detach | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 10 | 50 | none | 0 | 97.1642% | reject reward bridge |
| 11 | 50 | none | 0 | 98.7897% | reject; worse than repair 10 |
| 12 | 50 | none | 0 | 97.9398% | reject |
| 13 | 50 | none | 0 | 98.0676% | reject |
| 14 | 16 observed | none | 0 | incomplete | reject as unqualified partial run |
| 15 | 9 observed | none | 0 | incomplete | reject as unqualified partial run |
| 16 | 17 observed | none | 0 | incomplete | reject as unqualified partial run |
| 17 | 50 | none | 0 | 93.8824% | useful diagnosis, but still fails |

Repair 17 is the decisive negative result: restoring the successful V1-style
reward scale did not produce retention or success. That means the remaining
bottleneck cannot honestly be solved by stacking larger credit multipliers.

Repair 18 isolated physical discoverability. It was stopped after three
updates, so it is not a gate result and cannot authorize training. Its early
measurements were:

| Update | Capacity scale | Detach | Drop | Kinematic retained at break | Joint ready at break | Success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1509 | 19.44% | 19.33% | 6.96% | 0.12% | 0 |
| 1 | 0.1562 | 8.93% | 8.93% | 2.53% | 0 | 0 |
| 2 | 0.1667 | 6.30% | 6.30% | 2.34% | 0 | 0 |

This is enough to retain the curriculum as a single auditable candidate, but
not enough to claim that it passes the 50-update gate.

## Reverted changes

The following changes are removed because no completed gate demonstrated a
success benefit:

- 4x front-end progress scaling;
- 9x instantaneous load-ratio normalization;
- 14x stem-progress scaling;
- 40x controlled-break reward scaling;
- continuous velocity-quality reward substitution;
- weak/continuous opposition credit in place of secure support;
- post-break pre-hold readiness reward;
- V2 overrides of the inherited total reward, break reward, or retention
  reward;
- their helper functions, profile fields, log fields, and unit tests.

Historical failed reports remain evidence and are not rewritten.

## Repairs 37--43: release-speed and exact-substep credit diagnosis

Repair 37 added deployable fingertip-centroid-relative motion and repair 39
added the current deployable controller tracking error. Repair 39 materially
improved the exact-public tail over repair 37, reducing mean break speed from
`0.6231` to `0.3697 m/s` and raising joint-ready from `0.391%` to `0.937%`,
but still produced no retained outcome. Repair 40 tested one previous tracking
error sample. Its finite 50-update result reduced mean speed only to
`0.3245 m/s`, regressed joint-ready to `0.312%`, and again produced zero
retained outcomes. That redundant 6D sample is removed.

The repeated public-tail evidence showed twist-assisted fractures were
effectively zero while bend-assisted fractures persisted above the unchanged
`0.25 m/s` retention-speed boundary. Repair 41 tested torsion-only temporary
assistance. It was stopped safely after update 17 because assisted torsion
fractures and the grasped-break reward were exactly zero at every completed
update; all sparse detachments were unassisted public bends. The partial log
SHA-256 is
`45a3bb64009d202457507b98bfbd06dd171188ba5d2193f713857137f26325f9`.
This rejects torsion-only assistance rather than carrying a nonfunctional
patch forward.

The resulting audit found a distinct actor-state defect. The fixed pull-axis
direction is unchanged by roll about itself, and instantaneous tracking/command
signals cannot tell a feed-forward policy how much wrist twist has already
accumulated. Repair 42 restores the bending-plus-torsion curriculum and appends
one reset-safe, deployment-computable scalar: cumulative absolute actual
end-effector encoder rotation projected onto the fixed grasp axis, normalized
by one full revolution. It exposes neither fruit simulator orientation nor
stem state, sampled delay, reward, or acceptance thresholds. No reward term,
PPO setting, public-tail mechanic, or acceptance predicate changes.

Repair 42 completed all 50 finite updates but rejected that hypothesis. Its
exact-public tail contained no retained or successful outcome, only one
joint-ready-positive update, and mean fracture speed regressed to `0.8421
m/s`. The archived report SHA-256 is
`91d01c1e2c59739ae9c248dde3214602463e740cd8d6d387aaa34b2e864aa19a`;
the full-state `model_49.pt` SHA-256 is
`910f1b943efe00c6b239943827752523b33efb985a8b5cf920110d747913ec4c`.
The cumulative axial-rotation observation is removed rather than accumulated.

Repair 43 targets a concrete temporal-credit bug instead of adding a reward.
The environment records break support, distance, and speed at the exact 120 Hz
fracture substep, yet the inherited break reward also required grasp quality
and fruit-in-hand state measured only at the later 20 Hz policy boundary.
Movement during the remaining substeps of the same action could erase a valid
controlled-break credit. Repair 43 latches exact-substep grasp quality and
uses it with the existing strict substep gates. The `BREAK_GRASPED_WEIGHT`,
physics, PPO, termination, retention, success, and public-tail gate remain
unchanged.

Repair 43 passed 63/63 focused V2/V1/MDP contracts, the managed eligible suite
at 292 tests plus 103 subtests, Q1--Q6 with report SHA-256
`f823258b3520ba871b50897abb687ddef868f5ebc431f907c6e2ae6588783582`,
and the no-training 64x128 managed-Isaac smoke at 25/25 with report SHA-256
`176bf691cb63b2a13e2976b872a4c94117627eb8f208a90649f6a31795f6fa6a`
before the formal-scale gate.

## Repair 43 result and repair 44 hypothesis

Repair 43 finished 50/50 finite updates with a fully public update-40--49 tail
and no runtime error, but no terminal retained or successful episode. Its
archived report SHA-256 is
`2739548b7d6b1f8ece79f4bb58cd6890c313d47ba89b79aa4fd1bae835f31918`;
its complete `model_49.pt` SHA-256 is
`7240a632b6463312a6f1ba2cfa655d995802579d13181e7e561e3f5bf3d90c47`.

Nonzero exact break credit and retention-progress reward show that the repair
fixed real temporal credit and that transient retention occurred, while the
terminal-only retained logger hid it. The actor still receives no detachment
flag and a feed-forward network cannot retain a latent fracture phase across
the release-and-hold sequence. Repair 44 is a single causal model intervention:
the V2 actor becomes a one-layer GRU with 128 hidden units, followed by the
unchanged 512-256-128 ELU head. The 103D actor observation, 120D feed-forward
critic, reward, mechanics, PPO optimization, thresholds, and gate remain
unchanged. Episode-level ever-retained and maximum-hold latches are audit-only
and do not relax success.

Repair 44 preflight passed 65/65 focused contracts and 294 managed eligible
tests plus 103 subtests. Q1--Q6 report SHA-256:
`a9c57c4b6ae604c53e040a7aac89ef9ccb1c545abb746449c3cfa38e5447236e`.
The 64x128 no-training Isaac report remained 25/25 with SHA-256
`176bf691cb63b2a13e2976b872a4c94117627eb8f208a90649f6a31795f6fa6a`.
The new actor was also exercised by a 64-environment one-update PPO smoke:
the log instantiated `GRU(103, 128)`, kept the critic feed-forward at 120D,
completed finite optimization, emitted both episode-level retention metrics,
and wrote a complete recurrent checkpoint with SHA-256
`8450bfe72fd9cad186bd542158779401609878563d593095351597980b32c065`.

## Repair 44 result and repair 45 hypothesis

Repair 44 completed 50/50 finite updates with exact public capacities in the
tail and no runtime error. Report SHA-256:
`ce61df9c7e1f56ff888c06c2a750984c2ca5605d57bb763157d737dfb0e49d06`;
recurrent full-state checkpoint SHA-256:
`156463ee1a08dd422a23dd7fb05101b22f6075a12a5957d3cd9626a21a462cc5`.
The new audit confirmed transient retention (`0.14%` peak early; two positive
tail updates), but no terminal retention or success. The tail simultaneously
increased detachment and reached `0.524 m/s` mean fracture speed.

This identifies an objective-timing defect rather than another missing actor
history. The inherited reward blend has alpha zero through update 250, so the
entire 50-update V2 gate rewards controlled fracture but applies neither drop
penalties nor stable-detach credit. Repair 45 preserves ten discovery updates
and smoothsteps the existing endpoint blend to full outcome alignment at the
existing update-40 public-mechanics boundary. Reward terms and endpoint
weights, PPO, GRU, physics, predicates, and the gate are unchanged.

## Repair 45 result and repair 46 hypothesis

Repair 45 completed all 50 updates with finite metrics, exact public mechanics
and full outcome alignment in the update-40--49 tail. Archived report SHA-256:
`427819d16145dd968a7af2b27c0b69a6cb994351fa0a9c0adc78184d2839c546`;
full recurrent checkpoint SHA-256:
`0f9378d421b036aa9e648d360d8f60da5339bc2dd8e142d48ec7c16a97b758fc`.
It did not produce terminal retention or success.

The intervention separated the effects cleanly. Tail drop fell from repair
44's `1.519%` to `0.532%`, so early outcome penalties are useful. However,
the same alpha also multiplied the strict V2 `break_grasped` bridge by
`1-alpha`; at full alignment its value and exact joint-ready fracture both
became zero (repair 44 tail joint-ready mean: `1.041%`). The V2 event is not
the inherited exploitable fruit-in-hand break: repair 43 made it a one-shot
exact-substep conjunction of grasp quality, secure support, distance and
speed. Repair 46 keeps this already-aligned precursor at its unchanged
40-point weight while retaining repair 45's failure penalties and stable-
detach schedule. No new term, predicate relaxation, observation, physics or
PPO change is introduced, and the inherited Clean-v5/V1 behavior is preserved.

Repair 46 preflight passed 43 focused contracts, the complete managed suite at
305 tests plus 103 subtests, Q1--Q6 with report SHA-256
`e4467bc074dd74ce4cf878a3d51646e0069444cbc965554eab15ba1e7a17f334`,
and the 25/25 no-training 64x128 managed-Isaac gate with SHA-256
`5b2b404c11a8d4aa72f6e8d7a3c83a4d12b20d5a7e1b3f5a9405c0d1140c86d2`.
A 64-environment, 12-update fully aligned/full-public PPO smoke completed
finite optimization and wrote recurrent checkpoint SHA-256
`58f6f1b4cfbf6ccc72f53f5281e7e2d2409cd26cf147bc115a3a94a8976ba565`.

## Repair 46 result and repair 47 hypothesis

Repair 46 completed all 50 updates with finite metrics, exact public mechanics,
full outcome alignment, and no runtime error. Its archived report SHA-256 is
`e153f039e4904b7d513905438c77eaed7cdea5e1ee80f630da5c8b17784c5915`;
its complete recurrent `model_49.pt` SHA-256 is
`e62a5dee9657c63dbb109980294c8e8faa7115ea97c4454f08d46f5c6d402332`.
The checkpoint contains GRU weights `(384,103)` and `(384,128)`, 103D/120D
normalizers, 11D policy standard deviation, 21 Adam states, iteration 49, and
only finite tensors. It did not produce terminal retention or success. The
fully public tail had one joint-ready-positive update, zero ever-retained
updates, `0.507%` mean drop, `-27.728` mean reward, and `26.485` mean value
loss.

The strict bridge was therefore necessary but not the remaining sufficient
repair. Comparing the isolated interventions shows that terminal alignment is
useful: repair 44's tail drop was `1.519%`, versus `0.532%` in repair 45 and
`0.507%` in repair 46. But the same alpha also strengthened the dense time
cost from `-0.01` to `-0.10` per nonterminal step. At roughly 147 steps per
episode that term alone contributed about `-14.7`, while tail value loss rose
from repair 44's `4.666` to repair 45/46's `25.822/26.485`; transient retention
then vanished. Repair 47 separates this episode-length baseline from the
evidenced outcome channels: the strict break bridge, terminal failures,
stable-detach, success, physics, PPO, predicates, and acceptance are unchanged,
while V2 keeps the original bounded `-0.01` anti-stall cost throughout the
gate. Clean-v5 and Physics-v1 retain their original full blend.

## Consolidated next-gate candidate

Candidate ID: `physics_v2_retention_objective_v1`.

This is the only runnable candidate prepared for the next gate. It is not a
passed training result. It retains only interventions with a causal contract
or a completed-gate improvement:

- public Physics-v2 multi-mode mechanics and unchanged strict success,
  retention, hold, force, distance, and speed predicates;
- tensile/shear at full public capacity throughout and temporary
  bending/torsion assistance only inside the current strict support plus
  kinematic-retention envelope, with every mode at full public capacity for
  updates 40--49;
- public-capacity load normalization and strict retention-margin allocation,
  which produced the isolated repair-28/30/32 retained transitions without
  weakening acceptance;
- the deployable 103D repair-39 observation contract; its current tracking
  error reduced public-tail drop and break speed and increased joint-ready
  fracture relative to repair 37;
- exact-120 Hz fracture credit from repair 43, which removed a measured
  temporal-credit mismatch;
- the repair-44 128-unit GRU, which produced the first measurable transient
  public-tail retention chain without privileged detachment state;
- repair-45 terminal outcome alignment, which reduced public-tail drop from
  `1.519%` to approximately `0.5%`;
- repair-46's unchanged-weight, strict one-shot joint-ready fracture bridge;
- exactly one not-yet-gated correction: keep the dense anti-stall cost at
  `-0.01` while terminal outcomes align. This isolates the fivefold value-loss
  increase measured in repairs 45--46 without adding a term or changing any
  endpoint weight.

Rejected multipliers and permissive bridges from repairs 10--17, repair-36
damage reweighting, repair-38 episode phase, repair-40 tracking history,
repair-41 torsion-only assistance, and repair-42 cumulative axial rotation are
absent from the runnable code. No further repair may be stacked before this
candidate receives one clean `8192 x 32 x 50`, seed-42 gate.

## Gate policy

The audited profile default remains 8192 environments, 32 rollout steps,
seed 42, and 50 PPO updates. The runner now reads the gate budget from the
profile rather than embedding `50` in three separate launch paths.

The program never extends a failed gate automatically. If update 50 has not
passed, it may only recommend asking the user for a larger audited budget when
all of the following hold:

1. the process completed with finite metrics and no runtime error signature;
2. the final ten updates used full public-prior mechanics;
3. either an isolated retained/success outcome exists, or joint-ready
   fractures occur in at least three tail updates while the tail drop/detach
   ratio improves by at least five percentage points over the preceding ten
   updates.

Changing the budget requires a reviewed profile edit and explicit user
approval. A command-line override cannot silently enlarge it.

## Remaining validation before any new gate

- all CPU unit tests must pass;
- Q1-Q6 must pass against the current hashes;
- the no-training 64x128 managed-Isaac smoke test must pass at full public
  capacity;
- both prerequisite reports must contain current source fingerprints and the
  exact `physics_v2_retention_objective_v1` candidate ID;
- the controller must reject a missing, failed, or stale first-success report;
- no SDK, CAN, or real-hardware connection is allowed.
