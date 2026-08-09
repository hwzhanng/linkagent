# Physics-v1 fracture feasibility root-cause audit

Status: **formal training blocked**

This audit is read-only with respect to the task MDP. It does not change the
reward, observations, actions, reset distribution, collision geometry, contact
parameters, fracture thresholds, or PPO configuration.

## Evidence

- The corrected stem-bridge lineage completed 100 updates with finite losses
  and exploration, but never reached retained or success.
- A stochastic replay of `model_99.pt` over 8,192 environments and 800 policy
  steps covered 6,553,600 states and 51,113 completed episodes. It produced no
  fracture, no retention, and no success.
- Replay maximum combined stem load was 0.73596 and maximum damage was 0.06044.
  Under secure fingertip support, mean combined load was only 0.08414 and mean
  damage was exactly zero.
- 20,000+ replay episodes terminated from excess fingertip force. At discovery
  alpha zero the reward penalty is absent, but excess force still terminates
  the episode and prevents continued damage accumulation.
- A 200,000-profile deterministic sampling audit of the frozen Physics-v1
  randomization found:

  - stem tensile force: 20.33--75.08 N, median 41.90 N;
  - pad static friction: 0.484--0.616, median 0.550;
  - with the 8 N per-fingertip terminal limit, the ideal five-tip tangential
    capacity is only 19.36--24.64 N;
  - only 0.48% of sampled stems are below that ideal five-tip pure-pull
    capacity; zero samples are feasible with two, three, or four fingertips.

The capacity calculation is deliberately optimistic: it assumes every active
fingertip simultaneously reaches 8 N normal force, all contacts contribute in
the useful direction, and there is no geometric or dynamic loss.

## Root cause

The current stem-strength prior and gentle-contact safety envelope do not share
a feasible operating region. The cited 20--75 N intact-stem tensile prior was
combined with a tomato-safe 8 N per-fingertip terminal limit and roughly 0.55
friction. PPO cannot learn a reliable policy for an MDP in which nearly every
randomized episode requires more tangential load than the hand is permitted to
transmit.

This explains the observed local behavior: bounded reach, grasp, opposition,
and subcritical load progress improve, while sustained damage, fracture,
retention, and success remain absent. It is a physics-feasibility problem, not
an optimizer failure and not evidence that the policy is frozen.

## Minimal physical correction

Do not change the reward again. Make one isolated physical correction: replace
the intact-stem tensile prior used as the detachment limit with a provisional
abscission-zone/twist-detachment envelope that is explicitly bounded by the
hand's safe transmissible wrench. Preserve duration damage, combined
tension/shear/bending, contact physics, reset, 5D manifold, action and
observation spaces, PPO, and the success predicate.

Before choosing numeric limits, require an offline feasibility gate:

1. For every randomized fruit/stem profile, compute a conservative hand wrench
   capacity using no more than three load-bearing contacts and a safety margin.
2. Require the randomized detachment envelope to overlap that conservative
   capacity with margin; reject profiles that do not.
3. Run scripted secure-grasp pulls/twists across the weak/median/strong profile
   quantiles. Each must accumulate damage and detach without crossing the 8 N
   per-fingertip terminal limit or producing an impulsive fly-away.
4. Re-run the 256-environment numerical gate and a fresh 8,192-environment
   first-success gate from zero. Formal training remains prohibited until the
   complete detach -> retained -> success chain appears.

If the 20--75 N literature prior must be retained unchanged, the alternative is
to raise the admissible contact-force envelope and add validated fruit
compression/damage mechanics. That is a larger physics redesign and is not the
recommended minimal correction.

## Source artifacts

- `first_success_8192x100_stem_bridge_extended_gate.json`
- `model99_stem_bridge_failure_audit_seed42_4096.json`
- `model99_break_substep_audit_seed42_8192x800.json`
- `simulation/config/tomato_pick_physics_v1.yaml`
