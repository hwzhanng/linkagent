"""Validate real 16384-env resume and first periodic learned checkpoint."""
from datetime import datetime
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
folder=Path(__file__).resolve().parent
start=json.loads((folder/'scaling/formal-start.json').read_text())
launch=json.loads((folder/'scaling/formal-launch.json').read_text())
run=Path(start['run_dir'])
source=torch.load(launch['resume_checkpoint'],map_location='cpu',weights_only=True)
initial=torch.load(run/'checkpoints/initial_agent.pt',map_location='cpu',weights_only=True)
current=torch.load(run/'checkpoints/last_agent.pt',map_location='cpu',weights_only=True)

def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value,dict):
        for child in value.values(): yield from tensors(child)
    elif isinstance(value,(list,tuple)):
        for child in value: yield from tensors(child)

checks=json.loads((run/'wrapper_validation.json').read_text())
report={
    'checked_at':datetime.now().astimezone().isoformat(),
    'run_dir':str(run),'checkpoint':str(run/'checkpoints/last_agent.pt'),
    'exact_restore_all_modules_and_optimizers':exact(source,initial),
    'all_checkpoint_tensors_finite':all(bool(torch.isfinite(t).all()) for t in tensors(current)),
    'eval_reset_isolation_passed':checks['eval_reset_isolation_passed'],
    'parameter_change_l2':{key:sum(float((current[key][name]-value).square().sum())
                                 for name,value in initial[key].items()
                                 if value.is_floating_point() and not name.startswith('state_preprocessor.'))**.5
                           for key in ('encoder','policy','value')},
    'optimizer_steps':{name:sorted({int(v['step']) for v in current[name]['state'].values()})
                       for name in ('encoder_optimiser','policy_optimiser','value_optimiser')},
    'progress':json.loads((run/'progress.json').read_text()),
}
assert report['exact_restore_all_modules_and_optimizers']
assert report['all_checkpoint_tensors_finite']
assert report['eval_reset_isolation_passed']
assert all(v>0 for v in report['parameter_change_l2'].values())
assert all(min(steps)>7104 for steps in report['optimizer_steps'].values())
(folder/'scaling/formal-checkpoint-audit.json').write_text(json.dumps(report,indent=2)+'\n')
status=json.loads((folder/'status.json').read_text())
status.update(status='scaled_training_running_verified',updated_at=report['checked_at'])
status['resource_scaling']['formal_start_verification']=report
status['resource_scaling']['formal_measured_resources']=start['resources']
status['resource_scaling']['formal_measured_transitions_per_second']=start['observed_transitions_per_second']
status['independent_evaluation_at_scaling']=json.loads((folder/'scaling/evaluation-comparison.json').read_text())
(folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(report,indent=2))
