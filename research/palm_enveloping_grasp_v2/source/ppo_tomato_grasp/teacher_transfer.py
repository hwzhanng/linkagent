"""Read-only Clean-v5 teacher adapter for hand-only behavior cloning.

The released teacher owns a right arm plus a right G20 hand (66D -> 11D).
This project owns a fixed left hand (110D -> 10D).  Only the five semantic
hand synergies are transferable; Cartesian arm actions are deliberately
discarded and the five new tactile residuals start at zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTOR_OBSERVATION_BLOCKS,
)


TEACHER_OBSERVATION_DIM = 66
TEACHER_ACTION_DIM = 11
TEACHER_CHECKPOINT_SHA256 = (
    "c4ca9ac27aaa20806b630496362bcb9154d1e8076605f9d7026b58f719d124d0"
)


class CleanV5Teacher(nn.Module):
    """Exact deterministic actor topology stored in ``model_1350.pt``."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, TEACHER_OBSERVATION_DIM))
        self.register_buffer("std", torch.ones(1, TEACHER_OBSERVATION_DIM))
        self.mlp = nn.Sequential(
            nn.Linear(66, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 11),
        )

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | torch.device) -> "CleanV5Teacher":
        path = Path(checkpoint).expanduser().resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != TEACHER_CHECKPOINT_SHA256:
            raise ValueError(f"teacher checkpoint SHA-256 mismatch: {digest}")
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload.get("actor_state_dict")
        if not isinstance(state, dict) or int(payload.get("iter", -1)) != 1350:
            raise ValueError("teacher checkpoint contract or iteration mismatch")
        model = cls().to(device)
        model.mean.copy_(state["obs_normalizer._mean"])
        model.std.copy_(state["obs_normalizer._std"].clamp_min(1.0e-6))
        model.mlp.load_state_dict(
            {name.removeprefix("mlp."): value for name, value in state.items() if name.startswith("mlp.")}
        )
        model.eval()
        return model

    def forward(self, teacher_observation: torch.Tensor) -> torch.Tensor:
        if teacher_observation.shape[-1] != TEACHER_OBSERVATION_DIM:
            raise ValueError("teacher observation must end in 66 values")
        normalized = (teacher_observation - self.mean) / self.std
        return self.mlp(normalized)


def hand_actor_to_teacher_observation(actor: torch.Tensor) -> torch.Tensor:
    """Map deployable common fields from the 110D left-hand observation.

    Unavailable arm twist/pull-axis information is assigned its fixed-hand
    value (zero twist, palm +Z pull axis).  The teacher previous arm action is
    zero; its previous five synergy values come from the new actor's previous
    coarse action.  Orientation and extra tactile matrix moments have no
    Clean-v5 counterpart and are intentionally not smuggled into the teacher.
    """

    if actor.shape[-1] != 110 or not torch.isfinite(actor).all():
        raise ValueError("actor must be a finite tensor ending in 110 values")
    out = actor.new_zeros((*actor.shape[:-1], 66))
    out[..., 0:3] = actor[..., ACTOR_OBSERVATION_BLOCKS["fruit_center_in_palm"]] * (0.08 / 0.15)
    out[..., 3:6] = actor[..., ACTOR_OBSERVATION_BLOCKS["fruit_semi_axes"]]
    out[..., 6:9] = actor.new_tensor((0.0, 0.0, 1.0))
    out[..., 15:31] = actor[..., ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]]
    out[..., 31:47] = actor[..., ACTOR_OBSERVATION_BLOCKS["hand_joint_velocity"]]
    tactile = actor[..., ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]]
    out[..., 47:52] = tactile.reshape(*actor.shape[:-1], 5, 10)[..., 0]
    out[..., 52:55] = actor.new_tensor((0.0, 0.0, -1.0))
    previous = actor[..., ACTOR_OBSERVATION_BLOCKS["previous_action"]]
    out[..., 61:66] = previous[..., :5]
    return out.clamp(-1.0, 1.0)


def teacher_action_to_hand_action(teacher_action: torch.Tensor) -> torch.Tensor:
    """Discard teacher arm commands and transfer only semantic synergies."""

    if teacher_action.shape[-1] != 11 or not torch.isfinite(teacher_action).all():
        raise ValueError("teacher action must be a finite tensor ending in 11 values")
    result = teacher_action.new_zeros((*teacher_action.shape[:-1], 10))
    # Both source and destination use the same semantic 5D order.  Mirroring
    # is handled geometrically inside the left-hand manifold, not by changing
    # flexion/balance labels at this policy boundary.
    result[..., :5] = teacher_action[..., 6:11].clamp(-1.0, 1.0)
    return result


__all__ = (
    "CleanV5Teacher",
    "TEACHER_ACTION_DIM",
    "TEACHER_CHECKPOINT_SHA256",
    "TEACHER_OBSERVATION_DIM",
    "hand_actor_to_teacher_observation",
    "teacher_action_to_hand_action",
)
