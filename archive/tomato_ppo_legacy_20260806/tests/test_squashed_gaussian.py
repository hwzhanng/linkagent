"""Numerical tests for the bounded tomato PPO action distribution."""

from __future__ import annotations

import unittest

import torch

from simulation.tomato_pick.isaac.agents.squashed_gaussian import (
    SquashedGaussianDistribution,
)


class SquashedGaussianDistributionTests(unittest.TestCase):
    def test_extreme_mlp_outputs_have_finite_actions_and_gradients(self) -> None:
        distribution = SquashedGaussianDistribution(7)
        mlp_output = torch.linspace(-1.0e6, 1.0e6, 700).reshape(100, 7)
        mlp_output.requires_grad_()
        distribution.update(mlp_output)
        actions = distribution.sample()
        loss = -(
            distribution.log_prob(actions).mean()
            + 0.01 * distribution.entropy.mean()
        )
        loss.backward()

        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions.abs() < 1.0).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mlp_output.grad).all())
        self.assertTrue(torch.isfinite(distribution.std_logit_param.grad).all())

    def test_ppo_probability_ratio_stays_finite_for_extreme_network_outputs(self) -> None:
        generator = torch.Generator().manual_seed(42)
        old = SquashedGaussianDistribution(7)
        new = SquashedGaussianDistribution(7)
        old_output = torch.randn((100_000, 7), generator=generator) * 1.0e4
        new_output = torch.randn((100_000, 7), generator=generator) * 1.0e4

        old.update(old_output)
        actions = old.sample()
        old_log_prob = old.log_prob(actions).detach()
        new.update(new_output)
        ratio = torch.exp(new.log_prob(actions) - old_log_prob)

        self.assertTrue(torch.isfinite(ratio).all())

    def test_export_module_matches_deterministic_output(self) -> None:
        distribution = SquashedGaussianDistribution(7)
        mlp_output = torch.linspace(-20.0, 20.0, 70).reshape(10, 7)
        expected = distribution.deterministic_output(mlp_output)
        actual = distribution.as_deterministic_output_module()(mlp_output)
        self.assertTrue(torch.equal(actual, expected))

    def test_every_action_dimension_changes_probability(self) -> None:
        old = SquashedGaussianDistribution(7)
        new = SquashedGaussianDistribution(7)
        old.update(torch.zeros((1, 7)))
        changed = torch.zeros((1, 7))
        for action_index in range(7):
            changed.zero_()
            changed[:, action_index] = 2.0
            new.update(changed)
            self.assertFalse(
                torch.equal(
                    old.log_prob(torch.zeros((1, 7))),
                    new.log_prob(torch.zeros((1, 7)),),
                )
            )
            self.assertGreater(new.kl_divergence(old.params, new.params).item(), 0.0)

    def test_correlated_hand_exploration_preserves_deterministic_output(self) -> None:
        independent = SquashedGaussianDistribution(22)
        correlated = SquashedGaussianDistribution(
            22,
            hand_closure_correlation_std=0.18,
        )
        mlp_output = torch.linspace(-2.0, 2.0, 220).reshape(10, 22)
        self.assertTrue(
            torch.equal(
                independent.deterministic_output(mlp_output),
                correlated.deterministic_output(mlp_output),
            )
        )

    def test_correlated_hand_samples_have_only_intended_shared_mode(self) -> None:
        torch.manual_seed(42)
        distribution = SquashedGaussianDistribution(
            22,
            init_std=0.25,
            hand_closure_correlation_std=0.18,
        )
        distribution.update(torch.zeros((50_000, 22)))
        actions = distribution.sample()
        centered = actions - actions.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
        variance = covariance.diag().clamp_min(1.0e-8)
        correlation = covariance / torch.sqrt(
            variance.unsqueeze(0) * variance.unsqueeze(1)
        )
        closure = distribution._G20_CLOSURE_ACTION_INDICES
        correlation = correlation.detach()
        self.assertGreater(float(correlation[closure[0], closure[1]]), 0.20)
        self.assertLess(abs(float(correlation[closure[0], 0])), 0.03)
        self.assertLess(abs(float(correlation[closure[0], 11])), 0.03)

    def test_correlated_log_prob_kl_and_gradients_are_finite(self) -> None:
        old = SquashedGaussianDistribution(
            22,
            hand_closure_correlation_std=0.18,
        )
        new = SquashedGaussianDistribution(
            22,
            hand_closure_correlation_std=0.18,
        )
        old_output = torch.zeros((128, 22), requires_grad=True)
        new_output = torch.full((128, 22), 0.1, requires_grad=True)
        old.update(old_output)
        actions = old.sample()
        old_log_prob = old.log_prob(actions).detach()
        new.update(new_output)
        new_log_prob = new.log_prob(actions)
        ratio = torch.exp(new_log_prob - old_log_prob)
        kl = new.kl_divergence(old.params, new.params)
        loss = -(new_log_prob.mean() + 0.001 * new.entropy.mean())
        loss.backward()

        self.assertTrue(torch.isfinite(ratio).all())
        self.assertTrue(torch.isfinite(kl).all())
        self.assertGreater(float(kl.mean().detach()), 0.0)
        self.assertTrue(torch.isfinite(new_output.grad).all())
        self.assertTrue(torch.isfinite(new.std_logit_param.grad).all())


if __name__ == "__main__":
    unittest.main()
