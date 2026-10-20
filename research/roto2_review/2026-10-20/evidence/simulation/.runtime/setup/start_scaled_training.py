"""Resume the selected benchmark branch in a persistent local user service."""
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
folder = root/'simulation/.runtime/setup'
selected = json.loads((folder/'scaling/selection.json').read_text())
evaluation = json.loads((folder/'scaling/evaluation-comparison.json').read_text())
assert all(v['simulation_physx_rollout_completed'] for v in evaluation.values())
n = selected['num_envs']
train_n = n - 16
prior = selected['cumulative_training_transitions']
# Keep complete 32-step PPO rollouts, rounding the original 100M target upward.
steps = math.ceil((99_999_840-prior)/(train_n*32))*32
additional = steps*train_n
unit = 'linkagent-roto2-ppo-scaled-20261019.service'
log = folder/'formal-training-scaled.log'
command = ['systemd-run','--user',f'--unit={unit}',
           '--property=Type=exec','--property=RemainAfterExit=yes',
           '--property=Restart=no','--property=KillSignal=SIGINT',
           '--property=TimeoutStopSec=45s',f'--property=WorkingDirectory={root}',
           f'--property=StandardOutput=append:{log}',f'--property=StandardError=append:{log}',
           '/usr/bin/bash',str(root/'simulation/.runtime/run_roto2.sh'),'train',
           '--num-envs',str(n),'--million-steps',str((additional+.01)/1e6),
           '--validate-contracts','--resume-checkpoint',selected['final_checkpoint']]
subprocess.run(command,check=True)
manifest = {'started_at':datetime.now().astimezone().isoformat(),
            'service':unit,'log':str(log),'num_envs':n,'train_envs':train_n,
            'eval_envs':16,'prior_training_transitions':prior,
            'additional_training_transitions':additional,'control_steps':steps,
            'total_training_transitions':prior+additional,
            'resume_checkpoint':selected['final_checkpoint'],
            'benchmark_transitions_per_second':selected['transitions_per_second'],
            'benchmark_resources':selected['resources'], 'command':command}
(folder/'scaling/formal-launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
status = json.loads((folder/'status.json').read_text())
status.update(updated_at=manifest['started_at'],status='scaled_training_initializing',
              formal_service=unit,formal_log=str(log),formal_run_dir=None,completion_gate=None)
status['planned_training'].update(num_envs=n,additional_million_steps=additional/1e6,
                                 prior_training_transitions=prior,total_training_transitions=prior+additional)
status['resource_scaling'] = manifest
(folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
