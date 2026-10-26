"""Evaluate the exact upstream loss fragment on synthetic inputs; no PPO update."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.external_roto2/multimodal_rl/multimodal_rl/rl/ppo.py'
OUT = ROOT / 'artifacts/simulation/rotation_v15/critic_information'


def main():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    fragment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for parent in ast.walk(node):
                body = getattr(parent, 'body', None)
                if not isinstance(body, list):
                    continue
                for i, stmt in enumerate(body[:-1]):
                    if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == 'self._value_clip > 0':
                        following = body[i + 1]
                        assert isinstance(following, ast.Assign)
                        assert ast.unparse(following.targets[0]) == 'value_loss'
                        fragment = ast.Module(body=[stmt, following], type_ignores=[])
    assert fragment is not None
    old = torch.zeros(4)
    prediction = torch.tensor([0.1, 0.5, -0.5, 0.5], requires_grad=True)
    target = torch.tensor([1., -1., 1., 1.])
    context = dict(torch=torch, F=F,
                   self=SimpleNamespace(_value_clip=0.2, _value_loss_scale=1.),
                   predicted_values=prediction, sampled_values=old, sampled_returns=target)
    exec(compile(fragment, str(SOURCE), 'exec'), context)
    gradient = torch.autograd.grad(context['value_loss'], prediction)[0]
    assert gradient[0] != 0 and torch.equal(gradient[1:], torch.zeros(3))
    reference = torch.autograd.grad(F.mse_loss(prediction, target), prediction)[0]
    result = dict(
        completed=True, policy_training=False, core_modified=False,
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        exact_extracted_fragment=ast.unparse(fragment),
        old_value=old.tolist(), prediction=prediction.tolist(), target=target.tolist(),
        clipped_loss_gradient=gradient.tolist(), unclipped_mse_gradient=reference.tolist(),
        conclusion='The value-loss branch has zero gradient outside the clipping interval, including predictions moving away from the return target.',
        limitation='Synthetic function test only. No measured incidence in real PPO updates and no causal attribution of training failure. Shared encoder may still receive policy gradients.',
        next_required_evidence='Measure per-minibatch clipping incidence and wrong-direction incidence on an actual bounded training batch before proposing any PPO-core repair.',
    )
    (OUT / 'VALUE_CLIP_GRADIENT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
