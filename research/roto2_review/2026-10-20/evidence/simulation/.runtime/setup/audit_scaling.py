"""Check finite saved state and exact restore of each scaling trial."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
folder = Path(__file__).resolve().parent
source = torch.load(folder/'scaling_base/checkpoints/last_agent.pt', map_location='cpu', weights_only=True)

def tensors(value, prefix=''):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f'{prefix}/{key}')
    elif isinstance(value, (list, tuple)):
        for key, child in enumerate(value):
            yield from tensors(child, f'{prefix}/{key}')

def exact(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a,b) for a,b in zip(left,right))
    return left == right

results = []
for trial in json.loads((folder/'scaling/results.json').read_text()):
    if trial.get('exit_code') != 0 or not trial.get('training_completed'):
        continue
    run = Path(trial['run_dir'])
    initial = torch.load(run/'checkpoints/initial_agent.pt', map_location='cpu', weights_only=True)
    final = torch.load(run/'checkpoints/final_agent.pt', map_location='cpu', weights_only=True)
    values = list(tensors(final))
    result = {
        'num_envs': trial['num_envs'], 'run_dir': str(run),
        'exact_restore_all_modules_and_optimizers': exact(source,initial),
        'tensor_count': len(values),
        'all_checkpoint_tensors_finite': all(bool(torch.isfinite(t).all()) for _,t in values),
        'optimizer_steps': {name: sorted({int(v['step']) for v in final[name]['state'].values()})
                            for name in ('encoder_optimiser','policy_optimiser','value_optimiser')},
        'parameter_change_l2': trial['parameter_change_l2'],
    }
    assert result['exact_restore_all_modules_and_optimizers'], result
    assert result['all_checkpoint_tensors_finite'], result
    results.append(result)
(folder/'scaling/checkpoint-audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
