"""Evaluate the common starting policy and fastest completed scaling trial."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
folder = root/'simulation/.runtime/setup/scaling'
trials = json.loads((folder/'results.json').read_text())
selected = max((r for r in trials if r.get('training_completed') and r['exit_code'] == 0),
               key=lambda r:r['transitions_per_second'])
(folder/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
checkpoints = {
    'base': root/'simulation/.runtime/setup/scaling_base/checkpoints/last_agent.pt',
    'selected': Path(selected['final_checkpoint']),
}
results = {}
for label, checkpoint in checkpoints.items():
    output = folder/f'eval-{label}.json'
    with (folder/f'eval-{label}.log').open('w') as log:
        subprocess.run(['bash', str(root/'simulation/.runtime/run_roto2.sh'), 'evaluate',
                        '--evaluate-checkpoint', str(checkpoint), '--evaluation-output', str(output)],
                       cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    results[label] = json.loads(output.read_text())
    print(json.dumps({'label':label, **results[label]}),flush=True)
(folder/'evaluation-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
