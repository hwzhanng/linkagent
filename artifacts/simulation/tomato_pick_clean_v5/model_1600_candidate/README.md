# Clean-v5 model 1600 candidate skill

This package freezes training iteration 1600 as a simulation-evaluation
candidate. It is not the final best checkpoint and is not approved for robot
hardware.

Runtime selection from the repository root:

```json
{
  "type": "torch_export",
  "manifest_path": "artifacts/simulation/tomato_pick_clean_v5/model_1600_candidate/skill_manifest.json",
  "device": "cpu",
  "allow_hardware": false
}
```

The repository profile is
`config/skills/tomato_pick_clean_v5_model1600_candidate.json`. TorchScript is
kept only as a compatibility artifact; `torch.export` is the preferred path
and both artifacts passed the same four-input equivalence gate.

The Agent receives only the five task-level tomato tools. The skill consumes
the frozen 66D observation at 20 Hz, emits the deterministic 11D normalized
action, and relies on `TomatoPickSkillRunner` for action supervision, control
leases, state freshness, independent verification, cancellation, and timeout.

Promotion remains blocked on deterministic fixed multi-seed selection, G20
force calibration, vendor-gateway shadow validation, and an explicit graded
hardware release.
