"""A frozen feed-forward policy with a trainable recurrent action residual.

The feed-forward branch has the exact module names and shapes of the proven
minimal-v3 actor, so its checkpoint can be loaded without approximation.  The
LSTM branch is additive and its output projection is initialized to zero.
Consequently, the initial deterministic and stochastic policy distributions
are exactly those of the source actor while PPO can learn temporal corrections.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState, RNN
from rsl_rl.utils import unpad_trajectories


class ResidualLSTMModel(MLPModel):
    """Frozen MLP actor plus a zero-output LSTM residual branch."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        freeze_base: bool = True,
        freeze_obs_normalization: bool = True,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, "lstm")
        self.residual_head = nn.Linear(rnn_hidden_dim, self.distribution.input_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.freeze_base = freeze_base
        self.freeze_obs_normalization = freeze_obs_normalization
        if freeze_base:
            self.mlp.requires_grad_(False)
        if freeze_obs_normalization:
            self.obs_normalizer.requires_grad_(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        # Keep one shared, source-compatible normalization for both branches.
        latent = MLPModel.get_latent(self, obs)
        base_latent = unpad_trajectories(latent, masks) if masks is not None else latent
        base_output = self.mlp(base_latent)

        residual_latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        output = base_output + self.residual_head(residual_latent)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.freeze_obs_normalization:
            super().update_normalization(obs)

    def as_jit(self) -> nn.Module:
        return _TorchResidualLSTMModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxResidualLSTMModel(self, verbose)


class _TorchResidualLSTMModel(nn.Module):
    """TorchScript deployment wrapper with persistent recurrent state."""

    def __init__(self, model: ResidualLSTMModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.residual_head = copy.deepcopy(model.residual_head)
        self.deterministic_output = model.distribution.as_deterministic_output_module()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        base_output = self.mlp(x)
        residual_latent, (hidden, cell) = self.rnn(
            x.unsqueeze(0), (self.hidden_state, self.cell_state)
        )
        self.hidden_state[:] = hidden
        self.cell_state[:] = cell
        output = base_output + self.residual_head(residual_latent.squeeze(0))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0


class _OnnxResidualLSTMModel(nn.Module):
    """ONNX wrapper exposing recurrent state explicitly."""

    is_recurrent: bool = True

    def __init__(self, model: ResidualLSTMModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.residual_head = copy.deepcopy(model.residual_head)
        self.deterministic_output = model.distribution.as_deterministic_output_module()
        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.obs_normalizer(obs)
        base_output = self.mlp(x)
        residual_latent, (hidden, cell) = self.rnn(x.unsqueeze(0), (h_in, c_in))
        output = base_output + self.residual_head(residual_latent.squeeze(0))
        return self.deterministic_output(output), hidden, cell

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.input_size),
            torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["obs", "h_in", "c_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out", "c_out"]


__all__ = ["ResidualLSTMModel"]
