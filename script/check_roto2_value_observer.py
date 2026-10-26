"""Check that tracing and gradient hooks preserve updates bit for bit."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from simulation.roto2_pressure.value_gradient_diagnostic import ValueGradientDiagnostic


class Example:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 0.3]))
        self.optimizer = torch.optim.Adam([self.parameter], lr=0.01)
        self._value_clip = 0.2
        self._value_preprocessor = SimpleNamespace(running_mean=torch.zeros(1), running_variance=torch.ones(1))

    def _update(self):
        for epoch in range(2):
            for i in range(2):
                sampled_values = torch.zeros(4)
                sampled_returns = torch.tensor([-1., 1., 1., 0.])
                predicted_values = self.parameter * 1.
                if self._value_clip > 0:
                    predicted_values = sampled_values + torch.clamp(predicted_values - sampled_values,
                                                                    -self._value_clip, self._value_clip)
                loss = (predicted_values - sampled_returns).square().mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return False


def main():
    plain = Example(); observed = Example()
    before = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as temp:
        with ValueGradientDiagnostic(observed, temp) as observer:
            for _ in range(2):
                assert plain._update() == observed._update()
                assert torch.equal(plain.parameter, observed.parameter)
                assert torch.equal(plain.parameter.grad, observed.parameter.grad)
        assert len(observer.rows) == 8
        assert all(r['clipped_n'] == 3 and r['wrong_direction_clipped_n'] == 2
                   and r['clipped_zero_gradient_n'] == 3 for r in observer.rows)
        assert torch.equal(torch.get_rng_state(), before)
        for k,v in plain.optimizer.state[plain.parameter].items():
            assert torch.equal(v, observed.optimizer.state[observed.parameter][k])
    print(json.dumps(dict(passed=True, bitwise_parameters_gradients_adam_equal=True,
                         rng_unchanged=True, observer_minibatches=8,
                         known_clipping_counts_verified=True)))


if __name__ == '__main__':
    main()
