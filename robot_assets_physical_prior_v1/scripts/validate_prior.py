"""Reproducible static and independent-engine qualification of an engineering prior.
No collision meshes in multibody probes; no Isaac/PhysX, policy, or hardware claim.
"""
import copy,json,math,pathlib,xml.etree.ElementTree as E
import numpy as np,mujoco
from inertia_math import urdf_bodies
R=pathlib.Path(__file__).resolve().parents[1];P=json.loads((R/'patch/simulation/config/linkbot_physical_prior_v1.json').read_text());src=R/'patch'/P['urdf'];r=E.parse(src).getroot();old=E.parse(R/'reference/source.urdf').getroot();b=urdf_bodies(src)
joints={j.get('name'):j for j in r.findall('joint')};ol={j.get('name'):j for j in old.findall('joint')};kinematics=[];mimic=[]
for n,j in joints.items():
 for key in ['origin','axis','parent','child']:
  a=j.find(key);o=ol[n].find(key);assert (a.attrib if a is not None else None)==(o.attrib if o is not None else None),(n,key)
 kinematics.append(n)
 if j.find('mimic') is not None:
  x=j.find('mimic');f=j.find('limit');d=joints[x.get('joint')].find('limit');a=float(x.get('multiplier'));off=float(x.get('offset'));v=[a*float(d.get(q))+off for q in ['lower','upper']];assert min(v)>=float(f.get('lower'))-1e-9 and max(v)<=float(f.get('upper'))+1e-9
  assert a*float(d.get('velocity'))<=float(f.get('velocity'))+1e-9
  assert float(f.get('effort'))==0;mimic.append(n)
assert len(mimic)==10
for side in ['left','right']:assert abs(sum(t['mass'] for n,t in b.items() if n.startswith(side+'_hand_'))-1.2)<1e-9
# Strip geometry explicitly. This is a mechanical skeleton probe, not contact validation.
for l in r.findall('link'):
 for key in ['visual','collision']:
  for e in l.findall(key):l.remove(e)
ext=E.SubElement(r,'mujoco');E.SubElement(ext,'compiler',{'fusestatic':'false','balanceinertia':'false','discardvisual':'true'})
skeleton=R/'results/skeleton.urdf';E.ElementTree(r).write(skeleton);m=mujoco.MjModel.from_xml_path(str(skeleton));xml=R/'results/model.xml';mujoco.mj_saveLastXML(str(xml),m);x=E.parse(xml).getroot();opt=x.find('option')
if opt is None:opt=E.SubElement(x,'option')
opt.set('timestep',str(1/240));opt.set('integrator','implicitfast');act=E.SubElement(x,'actuator')
active=[]
for j in x.findall('.//body/joint'):
 n=j.get('name');p=P['joints'][n];j.set('armature',str(p['armature_kg_m2']));j.set('damping',str(p['urdf_viscous_nm_s_rad']));j.set('frictionloss',str(p['urdf_coulomb_nm']))
 if p['role']!='passive':
  E.SubElement(act,'position',{'name':n+'_servo','joint':n,'kp':str(p['stiffness_nm_rad']),'kv':str(p['drive_damping_nm_s_rad']),'forcelimited':'true','forcerange':f"{-p['effort_limit_nm']} {p['effort_limit_nm']}"});active.append(n)
E.ElementTree(x).write(xml)
trials=[];rng=np.random.default_rng(260930)
# Nominal and deterministic combined stress scenarios; scales are NOT calibrated distributions.
scenarios=[('nominal',1.,1.,1.,1.),('heavy_weak',1.15,.6,.7,.5),('light_strong',.85,1.4,1.3,2.),('low_friction',1.,1.,.7,.5),('high_friction',1.,1.,1.3,2.)]
for label,mass,effort,damp,fric in scenarios:
 for seed in [11,37,83]:
  model=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(model);rng=np.random.default_rng(seed)
  names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)];ids={n:i for i,n in enumerate(names)};aid=np.array([ids[n] for n in active]);arms=np.array([i for i,n in enumerate(active) if P['joints'][n]['role']=='arm'])
  for i in range(1,model.nbody):
   if '_hand_' in mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i):model.body_mass[i]*=mass;model.body_inertia[i]*=mass
  model.actuator_forcerange[:]*=effort;model.actuator_biasprm[:,2]*=damp;model.dof_frictionloss[:]*=fric
  for n,j in P['joints'].items():d.qpos[ids[n]]=np.clip(0,j['lower_rad'],j['upper_rad'])
  # Representative arm park angles, small arm perturbation; arbitrary hand poses stay mimic-consistent.
  for n,a in [('L2_Joint',1.2),('R2_Joint',-1.2),('L4_Joint',-.7),('R4_Joint',.7)]:d.qpos[ids[n]]=a
  target=d.qpos[aid].copy();start=target.copy();end=start.copy()
  for k,n in enumerate(active):
   p=P['joints'][n];end[k]=np.clip(start[k]+(rng.uniform(-.15,.15) if p['role']=='arm' else .4),p['lower_rad']+.001,p['upper_rad']-.001)
  mujoco.mj_setConst(model,d);d.qpos[aid]=start
  peakv=0.;peak_joint=None;peak_time=None;peake=0.;peakm=0.;sat=0;max_limit_excursion=0;posterror=[]
  for step in range(1440):
   goal=start if step<240 or step>=960 else end
   v=np.array([P['joints'][n]['velocity_limit_rad_s'] for n in active]);target+=np.clip(goal-target,-v*model.opt.timestep,v*model.opt.timestep)
   mujoco.mj_forward(model,d);g=d.qfrc_bias.copy();d.ctrl[:]=target
   # Put gravity feedforward inside the same bounded actuator, never as unbounded external force.
   d.ctrl[arms]+=g[aid[arms]]/model.actuator_gainprm[arms,0]
   mujoco.mj_step(model,d);assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
   
   if float(abs(d.qvel).max())>peakv:peakv=float(abs(d.qvel).max());peak_joint=names[int(np.argmax(abs(d.qvel)))];peak_time=float(d.time)
   peake=max(peake,float(abs(d.actuator_force).max()));sat+=int(np.count_nonzero(abs(d.actuator_force)>=model.actuator_forcerange[:,1]*.999))
   max_limit_excursion=max(max_limit_excursion,float(np.max(np.maximum(model.jnt_range[:,0]-d.qpos,d.qpos-model.jnt_range[:,1]))))
   for n,j in P['joints'].items():
    if 'mimic' in j:
     z=j['mimic'];peakm=max(peakm,abs(float(d.qpos[ids[n]]-z['multiplier']*d.qpos[ids[z['joint']]]-z['offset'])))
   if step>=1320:posterror.append(float(np.max(abs(d.qpos[aid]-start))))
  warnings=d.warning.number.tolist();assert not any(warnings)
  trials.append({'scenario':label,'seed':seed,'seconds':float(d.time),'finite':True,'solver_warnings':warnings,'max_abs_velocity_rad_s':peakv,'peak_velocity_joint':peak_joint,'peak_velocity_time_s':peak_time,'max_abs_actuator_torque_nm':peake,'max_mimic_error_rad':peakm,'max_joint_limit_excursion_rad':max_limit_excursion,'settled_max_tracking_error_rad':float(max(posterror)),'saturated_joint_steps':sat,'joint_steps':1440*len(active)})
# Posture-space mass-matrix SPD and gravity support tests (no collision-feasibility claim).
model=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(model);names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)];ids={n:i for i,n in enumerate(names)};maxg={n:0. for n,p in P['joints'].items() if p['role']=='arm'};mineig=1.;rng=np.random.default_rng(260930)
for k in range(500):
 d.qpos[:]=rng.uniform(model.jnt_range[:,0],model.jnt_range[:,1]);d.qvel[:]=0
 for n,j in P['joints'].items():
  if 'mimic' in j:z=j['mimic'];d.qpos[ids[n]]=z['multiplier']*d.qpos[ids[z['joint']]]+z['offset']
 mujoco.mj_forward(model,d);M=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,d,M);mineig=min(mineig,float(np.linalg.eigvalsh(M)[0]));assert mineig>0
 for n in maxg:maxg[n]=max(maxg[n],abs(float(d.qfrc_bias[ids[n]])))
out={'engine':mujoco.__version__,'scope':'Geometry-free independent-engine mechanism and actuator probes, NOT Isaac/contact/grasp regression or hardware identification','static':{'bodies':len(b),'joints':len(joints),'dof':len(P['joints']),'unchanged_joint_transforms':len(kinematics),'consistent_mimics':len(mimic),'total_mass_kg':sum(t['mass'] for t in b.values())},'posture_samples':500,'minimum_mass_matrix_eigenvalue':mineig,'maximum_gravity_torque_nm':maxg,'trials':trials}
(R/'results/qualification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
