"""Bounded four-batch diagnostic collection; never launches PPO."""
from pathlib import Path
import json,subprocess,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/simulation/rotation_v15/critic_information'
plan=json.loads((OUT/'COLLECTION_PLAN.json').read_text())
status={'status':'collecting','runs':[],'training_running':False}
def save():
    status['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
try:
    for row in plan['commands']:
        dest=Path(row['output']);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'COMPLETE.json').exists():
            status['runs'].append({**row,'status':'already_completed'});save();continue
        entry={**row,'status':'running'};status['runs'].append(entry);save()
        with (dest/'run.log').open('w') as log:
            proc=subprocess.Popen(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=proc.pid;save();rc=proc.wait()
        entry['returncode']=rc
        if rc or not (dest/'COMPLETE.json').exists():raise RuntimeError(f'Collection failed: {dest}')
        entry['status']='completed';save()
    hashes=json.loads((OUT/'FROZEN_CORE.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    status['status']='collection_completed';save()
except BaseException as e:
    status['status']='collection_failed';status['error']=repr(e);save();raise
