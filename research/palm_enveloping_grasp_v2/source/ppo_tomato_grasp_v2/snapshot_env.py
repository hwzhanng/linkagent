"""Terminal-transition snapshots for reliable per-episode evaluation.

Isaac Lab's DirectRLEnv resets completed environments inside ``step``.  A
metric collector that reads ``env.success`` or contact buffers after ``step``
therefore reads the newly reset episode.  This adapter captures the terminal
state immediately before the reset hook calls the v1 implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from simulation.ppo_tomato_grasp.physx_enveloping_env import EnvelopingPhysxTomatoGraspEnv


_SNAPSHOT_FIELDS = {
    "termination_code": ("last_termination_code", torch.long),
    "success": ("success", torch.bool),
    "strict_success": ("strict_success_latched", torch.bool),
    "dropped": ("dropped", torch.bool),
    "overloaded": ("overloaded", torch.bool),
    "damage": ("damage_score", torch.float32),
    "pick_started": ("pick_started", torch.bool),
    "stem_broken": ("stem_broke", torch.bool),
    "post_break_hold": ("stable_steps", torch.long),
    "contact_forces": ("contact_forces", torch.float32),
    "peak_force": ("peak_force", torch.float32),
    "palm_force": ("palm_force", torch.float32),
    "strict_hold_max": ("strict_hold_max", torch.long),
    "strict_coverage_max": ("strict_coverage_max", torch.float32),
    "strict_distal_participation": ("strict_distal_participation", torch.bool),
    "strict_first_palm": ("strict_first_palm", torch.long),
    "strict_first_three": ("strict_first_three", torch.long),
    "strict_contact_geometry_measured": ("strict_contact_geometry_measured", torch.bool),
    "last_drop_before_pick": ("last_drop_before_pick", torch.bool),
    "last_drop_after_pick": ("last_drop_after_pick", torch.bool),
    "last_drop_after_break": ("last_drop_after_break", torch.bool),
}


class TerminalSnapshotEnvelopingEnv(EnvelopingPhysxTomatoGraspEnv):
    """v1 physics with a reset-safe terminal snapshot buffer."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._v2_snapshot_enabled = False
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        device = self.device
        self.terminal_snapshot_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.terminal_snapshot: dict[str, torch.Tensor] = {}
        for key, (source, dtype) in _SNAPSHOT_FIELDS.items():
            source_value = getattr(self, source)
            self.terminal_snapshot[key] = torch.zeros_like(source_value, dtype=dtype)
        self._v2_snapshot_enabled = True

    def register_snapshot_fields(self, fields: dict[str, tuple[str, torch.dtype]]) -> None:
        """Register adapter-owned buffers after the parent constructor.

        v2 bookkeeping is allocated only after the Isaac Lab parent has
        created the scene, so it cannot be part of the static base snapshot
        table.  Registering it here keeps the terminal capture reset-safe
        without changing the frozen v1 environment.
        """
        for key, (source, dtype) in fields.items():
            if not hasattr(self, source):
                raise AttributeError(f"snapshot source {source!r} is not initialized")
            self.terminal_snapshot[key] = torch.zeros_like(getattr(self, source), dtype=dtype)
        self._registered_snapshot_fields = dict(fields)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            ids = self.robot._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._v2_snapshot_enabled and ids.numel() > 0:
            # Initial resets have a zero-length episode and must not become
            # fake terminal samples.  Done resets have nonzero episode length.
            eligible = self.episode_length_buf[ids] > 0
            if bool(eligible.any()):
                selected = ids[eligible]
                self.terminal_snapshot_valid[selected] = True
                fields = dict(_SNAPSHOT_FIELDS)
                fields.update(getattr(self, "_registered_snapshot_fields", {}))
                for key, (source, _dtype) in fields.items():
                    self.terminal_snapshot[key][selected] = getattr(self, source)[selected]
        super()._reset_idx(ids)

    def consume_terminal_snapshots(self, done: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return and clear snapshots for the done envs in this transition."""

        ids = torch.nonzero(done, as_tuple=False).flatten()
        if ids.numel() == 0:
            return {key: value[:0].clone() for key, value in self.terminal_snapshot.items()}
        if not bool(self.terminal_snapshot_valid[ids].all()):
            missing = ids[~self.terminal_snapshot_valid[ids]].detach().cpu().tolist()
            raise RuntimeError(f"missing terminal snapshot for done envs {missing}")
        result = {key: value[ids].clone() for key, value in self.terminal_snapshot.items()}
        self.terminal_snapshot_valid[ids] = False
        return result
