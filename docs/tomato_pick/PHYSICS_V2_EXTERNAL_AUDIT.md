# Physics-v2 external-audit handoff

Generated: `2026-08-17T13:25:24+08:00`

Status: **public-prior qualification passed; the prior first-success gate
failed; formal PPO is not authorized or running; hardware is not connected**.

## Reproducible evidence

| Evidence | Result | SHA-256 |
|---|---:|---|
| CPU public-prior Q1-Q6 | pass; candidate and source hashes current | `274f287f1a39918059789b1c3269d526614348979a9d14af61aa6860148c953e` |
| Isaac vector 64 envs x 128 steps | 25/25 pass; no training | `32381b3f3819334a7ef9054eea8a5a24af0fa0a9227c0fc975556f2397a23e9c` |
| Previous 8192 x 32 x 50 gate (repair 46) | fail; zero success and zero tail retention | `e153f039e4904b7d513905438c77eaed7cdea5e1ee80f630da5c8b17784c5915` |
| Managed-environment test suite | 309 tests + 103 subtests pass | command below |

The committed evidence paths are:

- `logs/qualification/tomato_physics_v2/public_prior_cpu_gate.json`
- `logs/qualification/tomato_physics_v2/isaac_smoke_gate_64x128.json`
- `logs/qualification/tomato_physics_v2/first_success_8192x50_gate_repair46.json`

The first two reports identify the sole next candidate as
`physics_v2_retention_objective_v1` and fingerprint every shared runtime
input. The third report is historical repair-46 failure evidence, not a pass
for the frozen candidate.

## Scientific acceptance signal

The evaluator records final success separately from a break-time candidate.
`PhysicsV2/paper_strategy_success_rate` is true only when all of these hold:

1. torque-utilization-weighted effective twist of at least 360 degrees or
   bending utilization of at least 0.5;
2. axial break force in the nontrivial `1-15.29 N` range;
3. real fingertip support, including thumb opposition;
4. break-time distance and relative-speed retention;
5. the unchanged final secure-retention success predicate.

This signal is diagnostic only. It is not added to reward and therefore cannot
directly reward a scripted paper-looking motion.

## Training controller audit points

- New V2 policy only; no Physics-v1 checkpoint or optimizer may be loaded.
- `8192` environments, `32` rollout steps, seed `42`.
- Every segment passes explicit `--max_iterations 400`; there is no inherited
  `1800`-update terminal cap.
- Every continuation verifies Actor, Critic, actor/critic normalizers, policy
  standard deviation, optimizer, iteration field, and checkpoint SHA-256.
- Every boundary evaluates deterministic seeds `17`, `42`, and `2026`, each
  with `1024` environments for `400` steps, before continuation.
- Stop requires two consecutive aggregate success results at least 90%, within
  one percentage point, plus two consecutive 100-vs-100 curve plateaus.
- A lock forbids duplicate controllers; process checks forbid concurrent V2
  train/evaluate jobs; less than 8 GiB available memory forbids a new segment.
- The controller is resumable from its own auditable state/history, but refuses
  to adopt an untracked pre-existing initial run.

The controller has an exact-confirmation dry stop. This handoff does not invoke
its execute path.

## Reproduction commands

```bash
LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1 \
  PYTHONPATH="$PWD" \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  -m pytest -q

LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1 \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/qualify_tomato_physics_v2_public_priors.py

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/check_tomato_physics_v2.py --num-envs 64 --steps 128 \
  --output logs/qualification/tomato_physics_v2/isaac_smoke_gate_64x128.json \
  --viz none --device cuda:0

LINKAGENT_ISAACLAB_ROOT=/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1 \
  PYTHONPATH="$PWD" \
  /home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python \
  script/gate_tomato_physics_v2_first_success.py --preflight-only
```

## Non-claims

Passing these gates means the simulator is finite, internally consistent,
source-bounded where public evidence exists, explicit where engineering priors
fill gaps, and ready for a from-scratch training audit. It does not demonstrate
that PPO will discover the desired strategy, and it does not qualify real A7 or
L20/G20 transfer. The deployed pad/friction, compliance, actuator response,
target cultivar, perception error, and L20/G20 alias still require hardware
measurements before any real-robot claim.

## Repair-33 gate evidence and repair-34 scope

Repair 33 completed the mandatory from-scratch `8192 x 32 x 50`, seed-42 gate
without a runtime or numerical error. Its report SHA-256 is
`0c2c1a248aa729bd656d56b2c8a5bf92cf6b0e9ad7035aace2f89a18a051f022`;
`model_49.pt` SHA-256 is
`e6b7a3fac6e636d328fcc014c46ba6c8b283158ace1cd5965d4813217ba78cf4`.
The added current/previous perceived-position pair improved low-capacity
break-time control: joint-ready reached 81.25% at update 20. It did not survive
the public-capacity transfer: updates 40--49 had zero retained and success
outcomes, only one positive joint-ready iteration, and break relative speed as
high as 0.827 m/s.

The remaining defect is an actuator-state alias, not a reason to weaken the
gate or stack another reward. V2 samples a fixed 0--2-step action delay, while
the inherited feed-forward observation exposes only the previously applied
command. With a two-step delay, two newer issued commands can still be queued
but are absent from actor state. Repair 34 appends those two bounded 11D command
vectors, newest first. They are controller-owned deployable data; the sampled
delay remains hidden. Reward, PPO, public mechanics, capacity schedule,
termination, retention, and success predicates are unchanged.

Before launching the repair-34 gate, the complete repository suite passed
(`298 passed`, `103 subtests passed`), the CPU public-prior gate passed, and the
CUDA Isaac smoke gate passed all 25 checks with 64 environments for 128 steps.
The evidence SHA-256 values are
`00e858ff9d7d5087e6e21e41d097638512e6c7d6ac820028a4cfe69de8367163`
for `public_prior_cpu_gate.json` and
`08fced25984c3cac030bee37455322592491c772e550ae589c667a2d8f38061f`
for `isaac_smoke_gate_64x128.json`. The gate runner remains unchanged at
`773c8ca0de71b256044eaedfa228dec0431554703bd63a2b422e3bbf0902f37f`.

## Repair-34 result and repair-35 scope

Repair 34 also completed all 50 configured updates with finite metrics, no
runtime error signature, and exact public capacities throughout updates
40--49. It did not pass: retained and success were zero in every update. The
report SHA-256 is
`16f66e9ee294832c8a55b9b6fd19d6d86d0c4e055a2c2d8fe70d742c764be922`;
the complete `model_49.pt` SHA-256 is
`4ba59031ffb650f77da0b77aed5164ecdc31ccc19d1b48c03e7df41756f32982`.

Capacity-binned evidence rejects a missing-break-event explanation. At
bending/torsion capacity at most 0.15, mean joint-ready was 33.24% and the
inherited break-grasped reward was 0.1127. At capacity at least 0.90, both were
zero while stem-load progress rose to 0.1515 and mean break relative speed rose
to 0.381 m/s. Repair 35 therefore changes one training-only curriculum
abscissa: instead of spending updates 0--20 fixed at 10% capacity and using
only updates 20--40 for transfer, the same smoothstep spans updates 0--40.
The 10% and 100% endpoints, reward terms and weights, conditional assistance,
public mechanics, PPO, success/retention predicates, and exact-public tail are
unchanged.

## Repair-35 result and repair-36 scope

Repair 35 completed all 50 finite updates without a runtime error and kept
exact public capacities throughout updates 40--49. It produced one isolated
retained outcome at update 6 (`0.0008` retained rate) but no success and no
retained tail. At update 49 the inherited stem-load progress reward was still
`0.2047`, while twist assistance, break-grasped credit, and retention were all
zero; mean break relative speed was `0.4009 m/s` and mean stem integrity was
only `0.4313`. The report SHA-256 is
`10456316b8ca944f24fc6ca676b9caeb0b6b276155a5a9269eebaaccff835ecf`;
the complete `model_49.pt` SHA-256 is
`16acdc95462b429c5f09febc55f504e3882b3c88eca39abd65f31490825c43e4`.

This is evidence of a high-load impulse attractor, not justification for a new
reward term. Repair 36 preserves the existing bounded unit progress score and
its inherited reward weight, but reallocates its internal mixture from 75%
instantaneous public-capacity load plus 25% accumulated damage to 25% load plus
75% damage. The opposition gate, strict support/kinematic-retention envelope,
retention-margin modifier, curriculum, public mechanics, PPO, outcome
predicates, and 100% public update-40--49 tail are unchanged. The V1 helper
default remains 75/25; the new allocation is isolated to the V2 profile.

## Repair-36 result and repair-37 scope

Repair 36 completed all 50 finite updates, preserved exact public capacities
throughout updates 40--49, and had no runtime error. It produced no retained
or successful episode. The report SHA-256 is
`db2ccbdec9bb177a8293021ab4b357b87080ca229a87974fee164deb225bd825`;
the complete `model_49.pt` SHA-256 is
`1884dad0320e225c5357f646acec9c50c61b2deec6aebc5763306faaa9908f6c`.
Changing the bounded progress mixture from 75/25 instantaneous load/damage to
25/75 reduced exact-public tail stem-load reward from `0.1732` to `0.1021`,
but tail joint-ready remained zero. The change is therefore rejected rather
than accumulated into the next repair; repair 37 restores the V1 helper's
unchanged 75/25 default.

The failure isolates an actor-frame mismatch. In repair 36's public tail,
distance-ok averaged `16.46%`, secure support `4.32%`, speed-ok `3.00%`, and
kinematic-retained `1.02%`, yet their strict joint-ready intersection was zero.
The strict predicate measures fruit distance and speed relative to the
fingertip centroid, while the actor's motion history measures the fruit
relative to the palm. Finger articulation changes the first signal without
changing the second. Repair 37 appends current and previous perceived
fruit-center positions relative to the fingertip centroid in the palm frame.
It uses the existing noisy fruit perception and robot forward kinematics, so
it is deployable; it exposes neither exact simulator velocity nor any gate
threshold. Reward, public mechanics, curriculum, PPO, termination, retention,
success, and acceptance remain unchanged.

## Repair-37 result and repair-38 scope

Repair 37 completed all 50 finite updates with no runtime error and exact
public capacities throughout updates 40--49. It produced no retained or
successful episode, so it did not pass the first-success gate. Its report
SHA-256 is
`f068f5421c38c42658f572f26a96e1a7154f544d6876778a79abeebf0569863f`;
the complete `model_49.pt` SHA-256 is
`a3546e6cb99a3f0950e234a03652679c3181c3f9126b7f6c02b5525f17ada12e`.

The new deployable relative-motion signal is retained because it materially
improved the exact-public tail over repair 36: secure support rose from 4.32%
to 7.66%, kinematic-retained from 1.01% to 2.68%, joint-ready from 0% to
0.39%, and drop fell from 1.21% to 0.93%. The remaining alias is phase, not
another missing force reward: the feed-forward MLP cannot reliably distinguish
the required grasp, load, detach, and retain stages from instantaneous state
alone. Repair 38 adds one normalized elapsed-episode scalar. This clock is
controller-owned and deployment-computable; it exposes no stem-detached state,
success predicate, or sampled physics. Reward, public mechanics, curriculum,
PPO, termination, retention, success, and acceptance remain unchanged.

## Repair-38 result

Repair 38 completed all 50 finite updates with no runtime error and exact
public capacities throughout updates 40--49. It produced no retained or
successful episode. Its report SHA-256 is
`84722b15457b2cf42405da4952db99ada2d412e6cd34ff462a25e610c5699330`;
the complete `model_49.pt` SHA-256 is
`e0db0b603378e434317641ac09c61a34fd0d751911972306a4a96f7b3877a045`.

The episode clock reduced exact-public mean break speed from `0.6231` to
`0.3575 m/s` and raised distance-ok from `18.18%` to `23.93%`, but it regressed
secure support from `7.66%` to `5.55%`, kinematic-retained from `2.68%` to
`1.35%`, joint-ready from `0.39%` to `0.31%`, and raised drop from `0.93%` to
`1.41%`. Because the target retained intersection remained zero, the clock is
rejected and removed rather than accumulated into later repairs. Repair 39
starts from the evidenced 97D repair-37 observation baseline.

## Repair-39 scope

Repair 39 changes only deployable actor observability. A finite issued-command
queue does not reveal how far the arm actually lags its commanded pose while
loading a resisting stem, or the release of that lag after fracture. The actor
therefore receives a 6D command-to-actual end-effector pose error, expressed in
the local grasp basis, divided by the existing per-policy-action translation
and rotation scales, and clipped to `[-1, 1]`. Both inputs are available from
the controller target and encoder forward kinematics. Exact stem wrench,
damage, detachment, simulator velocity, sampled delay, and acceptance
thresholds remain hidden. Actor/critic dimensions become `103/120`; reward,
public mechanics, curriculum, PPO, termination, success, retention, and the
update-40--49 exact-public gate are unchanged.

The pre-training evidence for this change is: 21/21 focused CPU contracts,
300 tests plus 103 subtests in the managed suite, Q1--Q6 pass with report
SHA-256
`8ba8b1f36601d811f08ea4b3aec353b1b2702c8a2851daf5893a9d4cbd56ddab`,
and the 64-environment x 128-step Isaac smoke at 25/25 with report SHA-256
`e9ce398257b32a1313392fbdd5f69749cf2f866130abcb4828f9608d0dbb6b1b`.

## Repair-39 result and repair-40 scope

Repair 39 completed all 50 finite updates with exact public capacities through
updates 40--49 and no runtime error, but produced no retained or successful
episode. Its report SHA-256 is
`2caa537826fb6f9589a86eb03391ddc4788004f37de131b5cccc981747d5af7e`;
the full-state `model_49.pt` SHA-256 is
`79d3e7199ba22a60c19a36995538e6d054807a5f539a4c173526e07bfcc3cb5c`.
The checkpoint contains a 103D actor, 120D critic, both normalizers, 11D policy
standard deviation, 17 Adam states, and finite tensors at iteration 49.

The current tracking error is retained because it improved repair 37's
exact-public tail: drop fell from `0.931%` to `0.693%`, mean break distance
from `0.0361 m` to `0.0307 m`, mean break speed from `0.6231 m/s` to
`0.3697 m/s`, and joint-ready from `0.391%` to `0.937%`. A deterministic
seed-42 evaluation of repair 39 `model_49.pt` over 1024 environments x 400
steps observed 58 detaches, 55 bend-assisted breaks, 47 moderate-pull breaks,
only 4 speed-ok breaks, and zero secure-plus-speed or retained events. One
tracking-error sample exposes lag magnitude but not whether resisted load is
accumulating or releasing. Repair 40 appends only the reset-safe previous 6D
tracking-error sample, making that trend available to the feed-forward actor.
Actor/critic dimensions become `109/126`; reward, mechanics, curriculum, PPO,
termination, retention, success, and acceptance remain unchanged.

Repair 40 passed 21/21 focused contracts, the managed suite at 300 tests plus
103 subtests, CPU Q1--Q6 with report SHA-256
`29931b22b8bf5eefc2621e5d7bec6631f9290db1ecac432a355bc15a821a1d6f`,
and the 64-environment x 128-step Isaac smoke at 25/25 with report SHA-256
`b00d222c893f6d663a081d3c9d29a232850fe3d7aed55d4ce969066f40c46d7c`
before formal-scale training.

## Repair-40 result and repair-41 scope

Repair 40 completed all 50 finite updates, preserved exact public capacities
through updates 40--49, and emitted no runtime error. It produced no retained
or successful episode. The archived report SHA-256 is
`67b242ece042b0a39932198d6b67bff89e386a829a61520cfce9df7bac9aeaf6`;
the complete `model_49.pt` SHA-256 is
`eafc6f9529a80554513c835a5aacaa752853c6d5c7155667a3ef29077d818687`.
The checkpoint contains a 109D actor, 126D critic, both normalizers, 11D policy
standard deviation, 17 Adam states, and finite tensors at iteration 49.

The previous tracking-error sample is rejected rather than accumulated. It
reduced exact-public mean break speed only from `0.3697` to `0.3245 m/s`,
while joint-ready regressed from `0.937%` to `0.312%` and tail drop/detach
regressed. Repair 41 therefore returns to the evidenced 103D/120D repair-39
observation contract.

The next intervention targets the remaining physical shortcut rather than
adding reward or observation history. Across repairs 37, 39, and 40,
twist-assisted public-tail fractures were effectively zero, bend-assisted
fractures persisted, and mean release speed stayed above the unchanged
`0.25 m/s` retention limit. Repair 41 keeps tensile, shear, and bending at
100% public capacity from update zero. Only torsion receives the existing
temporary strict-envelope assistance, from 10% to 100% by update 40. This
forces the assisted discovery path toward low-translation twist instead of
cheap high-speed bending. Reward weights, PPO, public-tail mechanics,
retention, success, and acceptance predicates are unchanged.

## Repair-41 rejection and repair-42 scope

Repair 41 was stopped safely after update 17 because it had already isolated a
decisive failure mode: `torsion_curriculum_assisted_break_rate` and the grasped-
break reward were exactly zero at every completed update. All detachments were
sparse unassisted public-capacity bends. The partial log is preserved with
SHA-256
`45a3bb64009d202457507b98bfbd06dd171188ba5d2193f713857137f26325f9`.
The source intervention is rejected rather than trained blindly to update 50;
it removed the sparse bending discovery path but did not create a learnable
twist stage.

Repair 42 restores the previously audited bending-plus-torsion curriculum and
changes only deployable actor observability. The actor's pull-axis direction is
invariant to roll about that axis, while instantaneous angular velocity,
tracking error, and command history do not encode how much roll has already
accumulated. A new 1D reset-safe controller state integrates the absolute
successive actual end-effector encoder rotation projected onto the fixed grasp
axis, normalized by one physical revolution and clipped to `[0, 1]`. The
underlying controller accumulator is not clipped. This is not fruit simulator
orientation and does not use stem damage, detachment, sampled delay, reward, or
any acceptance threshold. Actor/critic dimensions become `104/121`; PPO,
public-tail mechanics, reward weights, retention, success, and the strict
50-update gate remain unchanged.

## Repair-42 result and repair-43 scope

Repair 42 completed 50 finite updates with exact public capacities throughout
updates 40--49 and no runtime error, but produced no retained or successful
episode. Its archived report SHA-256 is
`91d01c1e2c59739ae9c248dde3214602463e740cd8d6d387aaa34b2e864aa19a`;
the complete `model_49.pt` SHA-256 is
`910f1b943efe00c6b239943827752523b33efb985a8b5cf920110d747913ec4c`.
Its exact-public mean break speed regressed to `0.8421 m/s`, so the cumulative
axial-rotation observation is rejected and repair 43 returns to the evidenced
103D/120D repair-39 observation contract.

The report also exposed an independent temporal-credit defect. Physics-v2
already latched strict secure support and kinematic retention at the exact
120 Hz fracture substep, but inherited `break_grasped` credit still multiplied
those latches by grasp quality and fruit-in-hand state sampled at the later
20 Hz policy boundary. The remaining substeps of the already-issued action
could therefore turn a valid controlled break into zero reward. Repair 43
latches the exact-substep grasp quality and uses it with the existing secure,
distance, and speed latches. This changes neither reward weight nor predicate;
mechanics, PPO, termination, retention, success, and the public tail are
unchanged.

Before the formal repair-43 gate, the focused V2/V1/MDP contracts passed at
63/63 and the managed suite excluding the independently user-modified
Clean-v5 exported policy passed at 292 tests plus 103 subtests. Q1--Q6 passed
with report SHA-256
`f823258b3520ba871b50897abb687ddef868f5ebc431f907c6e2ae6588783582`;
the no-training 64-environment x 128-step managed-Isaac smoke passed all 25
checks with report SHA-256
`176bf691cb63b2a13e2976b872a4c94117627eb8f208a90649f6a31795f6fa6a`.

## Repair-43 result and repair-44 scope

Repair 43 completed all 50 finite updates, preserved exact public capacities
through updates 40--49, and emitted no runtime error. It produced no terminal
retained or successful episode. The archived report SHA-256 is
`2739548b7d6b1f8ece79f4bb58cd6890c313d47ba89b79aa4fd1bae835f31918`;
the complete `model_49.pt` SHA-256 is
`7240a632b6463312a6f1ba2cfa655d995802579d13181e7e561e3f5bf3d90c47`.
The checkpoint contains a 103D actor, 120D critic, both normalizers, 11D policy
standard deviation, 17 Adam states, and finite tensors at iteration 49.

Exact-substep credit was nonzero and retention-progress reward appeared even
though the terminal-only retained metric stayed zero. This identifies two
separate issues: valid transient retention was hidden by terminal sampling,
and the deployable actor must change phase after an unobserved fracture event.
Repair 44 therefore keeps the validated 103D observation values and replaces
only the V2 actor with a single-layer GRU with 128 hidden units. The critic
remains the inherited feed-forward MLP; PPO optimizer/losses, rewards, physics,
thresholds, action space, curriculum, and acceptance remain unchanged.
Episode-level `ever_retained_rate` and `mean_max_hold_steps` latches correct
the audit without relaxing the requirement for first success and repeated
tail success.

Repair 44 passed 65/65 focused contracts and the managed eligible suite at
294 tests plus 103 subtests. Q1--Q6 passed with report SHA-256
`a9c57c4b6ae604c53e040a7aac89ef9ccb1c545abb746449c3cfa38e5447236e`;
the no-training 64-environment x 128-step managed-Isaac smoke passed all 25
checks with report SHA-256
`176bf691cb63b2a13e2976b872a4c94117627eb8f208a90649f6a31795f6fa6a`.
A separate 64-environment x 1-update training smoke instantiated a GRU(103,
128) actor and feed-forward 120D critic, completed finite PPO optimization,
logged both new audit metrics, and wrote a complete recurrent checkpoint with
SHA-256
`8450bfe72fd9cad186bd542158779401609878563d593095351597980b32c065`.

## Repair-44 result and repair-45 scope

Repair 44 completed all 50 finite updates with a fully public update-40--49
tail and no runtime error. Its archived report SHA-256 is
`ce61df9c7e1f56ff888c06c2a750984c2ca5605d57bb763157d737dfb0e49d06`;
the complete recurrent `model_49.pt` SHA-256 is
`156463ee1a08dd422a23dd7fb05101b22f6075a12a5957d3cd9626a21a462cc5`.
The model contains a GRU(103,128) actor, 120D feed-forward critic, both
normalizers, 11D policy standard deviation, 21 Adam states, and finite tensors
at iteration 49.

The corrected audit showed real but short retention chains: early
`ever_retained_rate` reached `0.14%`, while the public tail retained events in
two updates but never reached terminal retention or success. In parallel,
tail detach rate rose to `1.549%` and mean fracture speed rose to `0.524 m/s`.
The inherited Clean-v5 objective explains this direction: throughout updates
0--49 its alignment alpha is exactly zero, so drop penalties and stable-detach
credit are disabled while each controlled break remains rewarded. That
schedule begins alignment only after update 250, beyond the mandatory V2 gate.

Repair 45 changes only this timing mismatch. It preserves discovery through
update 10, then smoothsteps the unchanged discovery/aligned objective endpoints
to full alignment at update 40, the existing full-public-mechanics boundary.
No term, weight, reward topology, physics value, PPO setting, threshold, or
acceptance predicate changes. GRU128 and the corrected retention audit are
retained because repair 44 supplied positive evidence for both.

## Repair-45 result and repair-46 scope

Repair 45 completed 50/50 finite updates with exact public mechanics and full
outcome alignment throughout updates 40--49. Its archived report SHA-256 is
`427819d16145dd968a7af2b27c0b69a6cb994351fa0a9c0adc78184d2839c546`;
the recurrent full-state `model_49.pt` SHA-256 is
`0f9378d421b036aa9e648d360d8f60da5339bc2dd8e142d48ec7c16a97b758fc`.
The checkpoint contains a GRU(103,128) actor, 120D feed-forward critic, both
normalizers, 11D policy standard deviation, 21 Adam states, and finite tensors
at iteration 49.

Outcome alignment reduced exact-public tail drop from repair 44's `1.519%` to
`0.532%`, but the shared blend also faded `break_grasped` to zero. Exact
joint-ready fracture fell from `1.041%` to zero and tail retention vanished.
That fade is appropriate for inherited coarse fruit-in-hand break credit, but
not for V2's replacement: V2 credit is one-shot and already requires exact
fracture-substep grasp quality, thumb-plus-other secure support, and unchanged
distance and speed retention. Repair 46 therefore keeps this same strict
40-point bridge available at every alignment value while preserving every
repair-45 failure penalty and stable-detach scale. It adds no reward term,
changes no individual weight, and leaves physics, PPO, predicates, observations,
GRU, public-tail requirements, and acceptance unchanged. Clean-v5 and
Physics-v1 retain their original break-credit fade.

Repair 46 preflight passed 43 focused reward/V2 contracts and the complete
managed suite at 305 tests plus 103 subtests. Q1--Q6 passed with report SHA-256
`e4467bc074dd74ce4cf878a3d51646e0069444cbc965554eab15ba1e7a17f334`;
the no-training 64x128 managed-Isaac smoke passed all 25 checks with SHA-256
`5b2b404c11a8d4aa72f6e8d7a3c83a4d12b20d5a7e1b3f5a9405c0d1140c86d2`.
A 64-environment, 12-update fully aligned/full-public PPO smoke remained finite
and wrote a complete recurrent checkpoint with SHA-256
`58f6f1b4cfbf6ccc72f53f5281e7e2d2409cd26cf147bc115a3a94a8976ba565`.

## Repair-46 result and repair-47 scope

Repair 46 completed all 50 finite updates with exact public capacities, full
outcome alignment, and no runtime error. Its archived report SHA-256 is
`e153f039e4904b7d513905438c77eaed7cdea5e1ee80f630da5c8b17784c5915`;
the complete recurrent `model_49.pt` SHA-256 is
`e62a5dee9657c63dbb109980294c8e8faa7115ea97c4454f08d46f5c6d402332`.
The checkpoint has GRU `(384,103)/(384,128)` weights, 103D/120D normalizers,
11D standard deviation, 21 Adam states, iteration 49, and finite tensors.

The exact-public tail still had no retained or successful outcome. Preserving
the strict break bridge recovered only one joint-ready-positive update; mean
reward was `-27.728` and mean value loss `26.485`. The comparison now isolates
the remaining objective-scale defect. Terminal alignment reduced tail drop
from repair 44's `1.519%` to repair 45/46's `0.532%/0.507%`, so those failure
terms are retained. Simultaneously changing the per-step cost from `-0.01` to
`-0.10` contributed about `-14.7` per episode and raised value loss more than
fivefold over repair 44 (`4.666`). Repair 47 therefore keeps the strict break,
terminal failure, stable-detach, and success channels unchanged while holding
only V2's dense anti-stall cost at its original bounded `-0.01` endpoint. It
adds no term, changes no endpoint weight, and leaves physics, PPO, observation,
predicates, the public tail, and acceptance unchanged. Clean-v5 and Physics-v1
retain their original time-cost blend.

## Frozen candidate for the next gate

`physics_v2_retention_objective_v1` is the sole prepared candidate. Repair 47
is one isolated, not-yet-gated correction on top of the evidence-backed
mechanics, deployable repair-39 observation contract, exact-substep fracture
credit, recurrent actor, and terminal outcome alignment. Rejected reward
multipliers, permissive retention bridges, episode phase, prior tracking-error
history, torsion-only assistance, and cumulative axial rotation are absent.

The CPU and Isaac prerequisite reports now fingerprint the shared runtime
inputs and candidate ID. The formal-scale gate refuses a missing or stale
fingerprint. The curve controller additionally requires the passing gate's
candidate ID and every runtime hash to match before any long segment can
start. A failed gate cannot auto-extend, and this handoff does not launch one.
