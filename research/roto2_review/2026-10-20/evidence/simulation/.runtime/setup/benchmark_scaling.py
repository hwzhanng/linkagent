"""Measure real PPO throughput and resource usage; no changes to PPO math."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
folder = root / 'simulation/.runtime/setup/scaling'
folder.mkdir(exist_ok=True)
checkpoint = root / 'simulation/.runtime/setup/scaling_base/checkpoints/last_agent.pt'
results_path = folder / 'results.json'
results = json.loads(results_path.read_text()) if results_path.exists() else []

for n in map(int, sys.argv[1:]):
    started = time.time()
    log_path = folder / f'{n}-envs.log'
    budget = ((n - 16) * 256 + .01) / 1e6
    command = ['bash', str(root/'simulation/.runtime/run_roto2.sh'), 'train',
               '--num-envs', str(n), '--million-steps', str(budget),
               '--resume-checkpoint', str(checkpoint)]
    samples = []
    resource_abort = False
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            mem = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            try:
                rss = next(int(line.split()[1]) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
            except (StopIteration,FileNotFoundError): rss=0
            sample={'time':time.time(), 'gpu_util':float(gpu[0]),'gpu_memory_mib':float(gpu[1]),'gpu_power_w':float(gpu[2]),'gpu_temperature_c':float(gpu[3]),'rss_mib':rss/1024,'available_ram_gib':mem['MemAvailable']/1024**2}
            samples.append(sample)
            if sample['available_ram_gib'] < 6 or sample['gpu_memory_mib'] > 31000:
                import signal
                os.killpg(process.pid,signal.SIGINT)
                resource_abort=True
                break
            time.sleep(1)
        try: code=process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGKILL); code=process.wait()
    result={'num_envs':n,'exit_code':code,'resource_abort':resource_abort,'total_wall_seconds':time.time()-started,'log':str(log_path)}
    matches=[]
    for p in (root/'logs/roto2_pressure_fruit/jacobian_seed42').glob('*/training_result.json'):
        if p.stat().st_mtime >= started: matches.append(p)
    if code==0 and matches:
        report=max(matches,key=lambda p:p.stat().st_mtime)
        training=json.loads(report.read_text())
        result.update(training)
        result['transitions_per_second']=training['training_transitions']/training['elapsed_seconds']
        result['run_dir']=str(report.parent)
        # Training duration excludes application startup and scene conversion.
        train_start = report.stat().st_mtime - training['elapsed_seconds']
        training_samples=[s for s in samples if train_start <= s['time'] <= report.stat().st_mtime]
    else: training_samples=samples
    if training_samples:
        result['resources']={
            'mean_gpu_util_percent':statistics.mean(s['gpu_util'] for s in training_samples),
            'mean_gpu_power_w':statistics.mean(s['gpu_power_w'] for s in training_samples),
            'peak_gpu_memory_mib':max(s['gpu_memory_mib'] for s in samples),
            'peak_rss_mib':max(s['rss_mib'] for s in samples),
            'minimum_available_ram_gib':min(s['available_ram_gib'] for s in samples),
            'max_temperature_c':max(s['gpu_temperature_c'] for s in samples)}
    (folder/f'{n}-resources.json').write_text(json.dumps(samples,indent=2)+'\n')
    results.append(result)
    results_path.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if resource_abort: break
