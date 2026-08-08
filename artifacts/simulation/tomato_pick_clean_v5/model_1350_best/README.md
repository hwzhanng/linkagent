# Clean-v5 model 1350 — fixed multi-seed best

This is the automatically selected deterministic simulation skill. Across
26,077 episodes using seeds 17, 42, and 2026 it achieved 96.18% success
(95% Wilson CI 95.94%–96.41%); the worst seed was 96.06%.

Use `config/skills/tomato_pick_clean_v5_model1350_best.json`. The preferred
runtime is `torch.export`; TorchScript remains a checked compatibility
artifact. Hardware use is deliberately blocked until tactile calibration,
real-data verifier thresholds, vendor shadow validation, and an explicit
graded release are complete.
