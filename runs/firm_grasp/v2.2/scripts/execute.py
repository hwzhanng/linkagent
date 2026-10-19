"""新运行目录专用启动入口，所有输出均为唯一名称。"""
import sys,subprocess,os
from pathlib import Path
N=Path(__file__).resolve().parent
S=N/os.environ.get('FIRM_SOURCE_NAME','source-f3-v2_0')
EP='/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python'
common=['--seed','110','--support-ramp-policy-steps','20','--wrist-tilt-deg','41.68575721082819',
'--wrist-tilt-axis-world-xyz','.7374624811217811','.6651274395898447','.11727991320922924',
'--reset-fruit-reference-hand-mm','81.14439249','-15.65784216','193.53402406','--base-lift-m','.03',
'--reset-reference-offset-hand-mm','.2022','.8588','-.4707']
module=os.environ.get('FIRM_ENTRY_MODULE','run')
if module=='probe': common=common[2:]
name=sys.argv[1]
args=sys.argv[2:]
raise SystemExit(subprocess.call([EP,'-m','research.firm_grasp_v1.launch','--name',name,
'--module',module,'--num-envs','4','--',*common,*args],cwd=S,stdout=(N/'evidence'/f'{name}-launcher.log').open('x'),stderr=subprocess.STDOUT))
