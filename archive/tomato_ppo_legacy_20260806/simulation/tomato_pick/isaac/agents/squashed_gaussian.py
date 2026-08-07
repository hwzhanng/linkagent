"""Bounded Gaussian exploration for the tomato PPO policy.

The deployable actor still emits all 22 action means independently.  During
training only, an optional rank-one covariance term lets the ten physical G20
flexion drivers explore coherent closing/opening motions in addition to their
independent noise.  This changes neither the deterministic export nor the
action interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from rsl_rl.modules.distribution import Distribution


class SquashedGaussianDistribution(Distribution):
    """Gaussian transformed by tanh with an exact, consistent log-probability."""

    _G20_CLOSURE_ACTION_INDICES = (6, 7, 8, 9, 10, 17, 18, 19, 20, 21)

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.35,
        min_std: float = 0.05,
        max_std: float = 0.80,
        hand_closure_correlation_std: float = 0.0,
        max_latent_mean: float = 4.0,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__(output_dim)
        if not 0.0 < min_std < init_std < max_std:
            raise ValueError("require 0 < min_std < init_std < max_std")
        if not 0.0 < epsilon < 1.0e-3:
            raise ValueError("epsilon must be in (0, 1e-3)")
        if not max_latent_mean > 0.0:
            raise ValueError("max_latent_mean must be > 0")
        if hand_closure_correlation_std < 0.0:
            raise ValueError("hand_closure_correlation_std must be >= 0")
        if hand_closure_correlation_std > 0.0 and output_dim != 22:
            raise ValueError(
                "correlated G20 exploration requires the frozen 22-D action layout"
            )
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.max_latent_mean = float(max_latent_mean)
        self.hand_closure_correlation_std = float(hand_closure_correlation_std)
        self.epsilon = float(epsilon)
        unit = (init_std - min_std) / (max_std - min_std)
        initial_logit = math.log(unit / (1.0 - unit))
        self.std_logit_param = nn.Parameter(
            torch.full((output_dim,), initial_logit)
        )
        closure_direction = torch.zeros(output_dim)
        if output_dim == 22:
            closure_direction[list(self._G20_CLOSURE_ACTION_INDICES)] = 1.0
        self.register_buffer(
            "_closure_direction",
            closure_direction,
            persistent=False,
        )
        self._distribution: Normal | MultivariateNormal | None = None
        self._latent_mean: torch.Tensor | None = None
        self._base_std: torch.Tensor | None = None
        self._last_latent_sample: torch.Tensor | None = None
        Normal.set_default_validate_args(False)
        MultivariateNormal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        # Bound the latent Gaussian mean before constructing the distribution.
        # PPO evaluates old rollout actions under the updated policy and then
        # exponentiates the log-probability difference.  An unbounded MLP mean
        # can make that ratio overflow even though the final tanh action is
        # finite, poisoning the actor and std parameter in one optimizer step.
        # Four latent units already map to |action| > 0.999, so this removes no
        # useful actuator authority while keeping the probability ratio finite.
        latent_mean = self.max_latent_mean * torch.tanh(
            mlp_output / self.max_latent_mean
        )
        base_std = (
            self.min_std
            + (self.max_std - self.min_std) * torch.sigmoid(self.std_logit_param)
        )
        self._latent_mean = latent_mean
        self._base_std = base_std.expand_as(mlp_output)
        if self.hand_closure_correlation_std > 0.0:
            scale_tril = self._correlated_scale_tril(base_std)
            self._distribution = MultivariateNormal(
                latent_mean,
                scale_tril=scale_tril,
            )
        else:
            self._distribution = Normal(latent_mean, self._base_std)
        self._last_latent_sample = None

    def sample(self) -> torch.Tensor:
        self._require_updated()
        self._last_latent_sample = self._distribution.rsample()  # type: ignore[union-attr]
        return torch.tanh(self._last_latent_sample)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        latent_mean = self.max_latent_mean * torch.tanh(
            mlp_output / self.max_latent_mean
        )
        return torch.tanh(latent_mean)

    def as_deterministic_output_module(self) -> nn.Module:
        return _BoundedDeterministicOutput(self.max_latent_mean)

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        self._require_updated()
        return torch.tanh(self._latent_mean)  # type: ignore[arg-type]

    @property
    def std(self) -> torch.Tensor:
        self._require_updated()
        return self._distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        """One-sample reparameterized estimate of transformed entropy."""

        self._require_updated()
        latent = self._last_latent_sample
        if latent is None:
            latent = self._distribution.rsample()  # type: ignore[union-attr]
        log_abs_det_jacobian = self._log_tanh_jacobian(latent)
        base_entropy = self._distribution.entropy()  # type: ignore[union-attr]
        if isinstance(self._distribution, Normal):
            base_entropy = base_entropy.sum(dim=-1)
        return base_entropy + log_abs_det_jacobian.sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        self._require_updated()
        # KL is invariant under the same bijective tanh transform.  Store the
        # latent Gaussian parameters, not the squashed action mean.
        # Store the diagonal/base standard deviation rather than a full
        # covariance per transition.  The fixed rank-one closure basis is part
        # of the distribution definition and can be reconstructed for KL.
        return (self._latent_mean, self._base_std)  # type: ignore[return-value]

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        self._require_updated()
        bounded = outputs.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        latent = torch.atanh(bounded)
        correction = self._log_tanh_jacobian(latent)
        base_log_prob = self._distribution.log_prob(latent)  # type: ignore[union-attr]
        if isinstance(self._distribution, Normal):
            base_log_prob = base_log_prob.sum(dim=-1)
        return base_log_prob - correction.sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_base_std = old_params
        new_mean, new_base_std = new_params
        if self.hand_closure_correlation_std > 0.0:
            old_distribution = MultivariateNormal(
                old_mean,
                scale_tril=self._correlated_scale_tril(old_base_std[0]),
            )
            new_distribution = MultivariateNormal(
                new_mean,
                scale_tril=self._correlated_scale_tril(new_base_std[0]),
            )
            return torch.distributions.kl_divergence(
                old_distribution,
                new_distribution,
            )
        per_dimension = torch.distributions.kl_divergence(
            Normal(old_mean, old_base_std), Normal(new_mean, new_base_std)
        )
        return per_dimension.sum(dim=-1)

    def _require_updated(self) -> None:
        if (
            self._distribution is None
            or self._latent_mean is None
            or self._base_std is None
        ):
            raise RuntimeError("distribution.update() must be called before use")

    def _correlated_scale_tril(self, base_std: torch.Tensor) -> torch.Tensor:
        """Return the Cholesky factor of diagonal plus rank-one covariance."""

        if base_std.ndim != 1 or base_std.shape[0] != self.output_dim:
            raise ValueError(
                f"base_std must have shape ({self.output_dim},), got {tuple(base_std.shape)}"
            )
        covariance = torch.diag(base_std.square())
        closure = self._closure_direction.to(dtype=base_std.dtype)
        covariance = covariance + (
            self.hand_closure_correlation_std**2
        ) * torch.outer(closure, closure)
        return torch.linalg.cholesky(covariance)

    @staticmethod
    def _log_tanh_jacobian(latent: torch.Tensor) -> torch.Tensor:
        """Compute log(1 - tanh(x)^2) without subtracting near-equal floats."""

        return 2.0 * (
            math.log(2.0) - latent - functional.softplus(-2.0 * latent)
        )


class _BoundedDeterministicOutput(nn.Module):
    """Export the exact bounded-mean action used by the training distribution."""

    def __init__(self, max_latent_mean: float) -> None:
        super().__init__()
        self.max_latent_mean = float(max_latent_mean)

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        latent_mean = self.max_latent_mean * torch.tanh(
            mlp_output / self.max_latent_mean
        )
        return torch.tanh(latent_mean)


__all__ = ["SquashedGaussianDistribution"]
