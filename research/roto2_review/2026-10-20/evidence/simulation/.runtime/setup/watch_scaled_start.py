"""Record formal startup and resources through the first periodic checkpoint."""
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import time

root=Path(__file__).resolve().parents[3]
folder=root/'simulation/.runtime/setup'
launch=json.loads((folder/'scaling/formal-launch.json').read_text())
since=datetime.fromisoformat(launch['started_at']).timestamp()
samples=[]
run=None
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    state=subprocess.check_output(['systemctl','--user','show',launch['service'],
                                   '--property=ActiveState','--value'],text=True).strip()
    if state!='active': raise RuntimeError(f'Training service state: {state}')
    if run is None:
        candidates=[p for p in (root/'logs/roto2_pressure_fruit/jacobian_seed42').glob('*/checkpoints/initial_agent.pt')
                    if p.stat().st_mtime>=since]
        if candidates:
            run=max(candidates,key=lambda p:p.stat().st_mtime).parent.parent
            (folder/'formal-run.txt').write_text(str(run)+'\n')
            status=json.loads((folder/'status.json').read_text())
            status.update(status='scaled_training_running',formal_run_dir=str(run),
                          completion_gate=str(run/'training_result.json'),
                          updated_at=datetime.now().astimezone().isoformat())
            (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps({'training_run':str(run)}),flush=True)
    if run:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                                     '--format=csv,noheader,nounits'],text=True).strip().split(',')
        samples.append(dict(time=time.time(),gpu_util=float(gpu[0]),gpu_memory_mib=float(gpu[1]),
                            gpu_power_w=float(gpu[2]),gpu_temperature_c=float(gpu[3])))
        if (run/'progress.json').exists():
            progress=json.loads((run/'progress.json').read_text())
            report={'run_dir':str(run),'progress':progress,'resources':{
                'sample_count':len(samples),
                'mean_gpu_util_percent':statistics.mean(s['gpu_util'] for s in samples),
                'mean_gpu_power_w':statistics.mean(s['gpu_power_w'] for s in samples),
                'peak_gpu_memory_mib':max(s['gpu_memory_mib'] for s in samples),
                'maximum_temperature_c':max(s['gpu_temperature_c'] for s in samples)},
                'observed_transitions_per_second':progress['training_transitions']/(time.time()-(run/'checkpoints/initial_agent.pt').stat().st_mtime)}
            (folder/'scaling/formal-start.json').write_text(json.dumps(report,indent=2)+'\n')
            (folder/'scaling/formal-resources.json').write_text(json.dumps(samples,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
            break
    time.sleep(2)
else:
    raise RuntimeError('No periodic checkpoint within 10 minutes; inspect service log')
