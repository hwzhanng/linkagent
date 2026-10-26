"""Read-only observation of actual upstream PPO loss tensors and their gradients.

No replacement losses, extra backwards, random draws or modified gradients.
The instance update wrapper enables a Python trace only inside PPO updates.
"""
import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch


class ValueGradientDiagnostic:
    def __init__(self, agent, output):
        self.agent = agent
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        assert not (self.output / 'minibatches.jsonl').exists(), 'Preserve prior diagnostics'
        self.original = agent._update
        self.code = self.original.__func__.__code__
        self.source = Path(inspect.getsourcefile(self.original))
        tree = ast.parse(self.source.read_text())
        lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'self._value_clip > 0']
        assert len(lines) == 1
        self.capture_line = lines[0]
        self.update = 0
        self.rows = []
        self.stream = None

    def _capture(self, local):
        prediction = local['predicted_values']
        old = local['sampled_values'].detach()
        target = local['sampled_returns'].detach()
        p = prediction.detach()
        clip = float(self.agent._value_clip)
        clipped = (p - old).abs() > clip
        wrong = clipped & ((p - old) * (target - old) < 0)
        projected = old + (p - old).clamp(-clip, clip)
        harmful = clipped & ((p - target).abs() > (projected - target).abs())
        squared_error = (p - target).square()
        values = torch.stack([
            clipped.sum(), wrong.sum(), harmful.sum(), squared_error.sum(),
            squared_error[clipped].sum(), squared_error[wrong].sum(),
            p.mean(), old.mean(), target.mean(),
            (p - old).abs().max(), (projected - target).square().mean(),
        ]).double().cpu().tolist()
        row = dict(update=self.update, epoch=int(local['epoch']), minibatch=int(local['i']),
                   n=p.numel(), clip=clip)
        row.update(zip(['clipped_n', 'wrong_direction_clipped_n', 'harmful_clipped_n',
                        'unclipped_squared_error_sum', 'clipped_squared_error_sum',
                        'wrong_direction_squared_error_sum', 'prediction_mean',
                        'old_value_mean', 'return_mean', 'max_prediction_shift',
                        'clipped_loss_mse'], values))
        # Hooks return None: the exact upstream gradient propagates unchanged.
        def observe_gradient(gradient):
            zero = gradient.detach() == 0
            counts = torch.stack([(zero & clipped).sum(),
                                  ((~zero) & clipped).sum(),
                                  ((~zero) & (~clipped)).sum()]).cpu().tolist()
            row.update(zip(['clipped_zero_gradient_n', 'clipped_nonzero_gradient_n',
                            'unclipped_nonzero_gradient_n'], counts))
            self.rows.append(row)
            self.stream.write(json.dumps(row) + '\n')
            return None
        prediction.register_hook(observe_gradient)
        if self.update in (1, 8, 32, 64) and local['i'] == 0 and local['epoch'] in (0, 1, 7):
            torch.save({'prediction': p.cpu(), 'old': old.cpu(), 'return': target.cpu(),
                        'clip': clip}, self.output / f'tensors_u{self.update}_e{local["epoch"]}.pt')

    def _global_trace(self, frame, event, arg):
        if frame.f_code is self.code:
            return self._local_trace
        return None

    def _local_trace(self, frame, event, arg):
        if event == 'line' and frame.f_lineno == self.capture_line:
            self._capture(frame.f_locals)
        return self._local_trace

    def __enter__(self):
        assert sys.gettrace() is None, 'Do not interfere with another debugger'
        self.stream = (self.output / 'minibatches.jsonl').open('w', buffering=1)
        def observed_update(*args, **kwargs):
            self.update += 1
            before = len(self.rows)
            if hasattr(self.agent, 'memory'):
                memory = self.agent.memory
                terminated = memory.get_tensor_by_name('terminated').detach()
                truncated = memory.get_tensor_by_name('truncated').detach()
                rewards = memory.get_tensor_by_name('rewards').detach()
                values = memory.get_tensor_by_name('values').detach()
                stats = torch.stack([terminated.sum(), truncated.sum(),
                                     rewards.mean(), values.mean(), values.std()]).cpu().tolist()
                rollout = dict(update=self.update, shape=list(rewards.shape))
                rollout.update(zip(['terminated', 'truncated', 'reward_mean',
                                    'old_value_mean_raw', 'old_value_std_raw'], stats))
                if 'gt' in memory.tensors:
                    gt=memory.get_tensor_by_name('gt').detach()
                    rollout['critic_reward_path_invalid_fraction']=float(gt.mean())
                with (self.output/'rollouts.jsonl').open('a') as stream:
                    stream.write(json.dumps(rollout)+'\n')
            sys.settrace(self._global_trace)
            try:
                result = self.original(*args, **kwargs)
            finally:
                sys.settrace(None)
            subset = self.rows[before:]
            assert subset and all(r['clipped_nonzero_gradient_n'] == 0 for r in subset)
            progress = dict(updates=self.update, minibatches=len(self.rows),
                            last_update_clipped_fraction=sum(r['clipped_n'] for r in subset)/sum(r['n'] for r in subset),
                            last_update_wrong_direction_fraction=sum(r['wrong_direction_clipped_n'] for r in subset)/sum(r['n'] for r in subset),
                            value_scaler_mean=self.agent._value_preprocessor.running_mean.cpu().tolist(),
                            value_scaler_variance=self.agent._value_preprocessor.running_variance.cpu().tolist())
            temp = self.output / 'PROGRESS.tmp'
            temp.write_text(json.dumps(progress, indent=2) + '\n')
            temp.replace(self.output / 'PROGRESS.json')
            return result
        self.agent._update = observed_update
        return self

    def __exit__(self, kind, value, traceback):
        sys.settrace(None)
        self.agent._update = self.original
        self.stream.close()
        result = dict(completed=kind is None, updates=self.update, minibatches=len(self.rows),
                      source=str(self.source), source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                      loss_or_gradient_modified=False,
                      diagnostic_policy_not_selected=True,
                      exception=None if kind is None else repr(value))
        (self.output / 'OBSERVER_COMPLETION.json').write_text(json.dumps(result, indent=2) + '\n')
        return False
