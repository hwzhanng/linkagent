"""Isolated MuJoCo friction law check; not a fitted fingertip/tomato contact model."""
import json,pathlib,math,numpy as np,mujoco
R=pathlib.Path(__file__).resolve().parents[1];rows=[]
for mu in [.3,.55,.8]:
 critical=math.atan(mu)
 for delta in [-3,3]:
  angle=critical+math.radians(delta);g=(9.81*math.sin(angle),0,-9.81*math.cos(angle))
  xml=f'''<mujoco><option timestep="0.001" gravity="{g[0]} 0 {g[2]}" integrator="implicitfast" cone="elliptic"/><worldbody><geom type="plane" size="1 1 .1" friction="{mu} .005 .0001"/><body pos="0 0 .011"><freejoint/><geom type="box" size=".01 .01 .01" mass=".1" friction="{mu} .005 .0001"/></body></worldbody></mujoco>'''
  m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
  m.opt.gravity[:]=[0,0,-9.81]
  for _ in range(500):mujoco.mj_step(m,d)
  m.opt.gravity[:]=g;start=float(d.qpos[0])
  for _ in range(1000):mujoco.mj_step(m,d)
  displacement=float(d.qpos[0])-start;expected=delta>0;slipped=displacement>.01;assert slipped==expected,(mu,delta,displacement)
  rows.append({'mu':mu,'angle_deg':math.degrees(angle),'analytic_threshold_deg':math.degrees(critical),'displacement_m':displacement,'slipped':slipped,'expected':expected})
out={'scope':__doc__,'engine':mujoco.__version__,'tests':rows};(R/'results/contact_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
