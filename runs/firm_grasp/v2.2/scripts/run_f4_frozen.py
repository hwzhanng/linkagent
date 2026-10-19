"""冻结候选后的 F4 同条件矩阵；逐个运行，任一进程错误即停止。"""
import json,subprocess,os
from pathlib import Path
N=Path(__file__).resolve().parent
freeze=json.loads((N/'F4_CANDIDATE_FREEZE.json').read_text())
if freeze['status']!='FROZEN_BEFORE_FINAL_MATRIX':raise RuntimeError('候选尚未冻结')
common=['--matrix','--support-release-power','4','--learning-rate','2e-6','--force-ratio','.1','--torque-ratio','.01']
jobs=[('F4_matrix_update000001',freeze['updated_checkpoint'],[]),('F4_matrix_initial',freeze['initial_checkpoint'],[]),('F4_matrix_legacy_v16',freeze['legacy_checkpoint'],['--legacy-thumb-baseline'])]
for name,checkpoint,extra in jobs:
 print('START',name,flush=True)
 subprocess.run(['python',str(N/'execute.py'),name,*common,'--actor-seed-checkpoint',checkpoint,*extra],env={**os.environ,'FIRM_SOURCE_NAME':'source-f3-v2_2','FIRM_ENTRY_MODULE':'matrix'},check=True)
 print('DONE',name,flush=True)
print('ALL_THREE_MATRICES_FINISHED',flush=True)
