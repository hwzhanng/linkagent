"""Opt-in integration for the uncalibrated physical-prior candidate.
Only Physics-v1/v2 tasks are supported. Original task defaults stay unchanged.
"""
import copy,hashlib,json,os,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[2]
PROFILE=ROOT/'simulation/config/linkbot_physical_prior_v1.json'
def enabled():
 value=os.environ.get('LINKAGENT_PHYSICAL_PRIOR','0')
 if value not in ('0','1'):raise ValueError('LINKAGENT_PHYSICAL_PRIOR must be 0 or 1')
 if value=='1' and os.environ.get('LINKAGENT_INERTIA_FIX','0')=='1':raise ValueError('Use only one asset variant flag')
 return value=='1'
def load_profile():return json.loads(PROFILE.read_text())
def verify_files():
 manifest=json.loads(PROFILE.with_suffix('.integrity.json').read_text())
 for rel,digest in manifest['git_dependencies'].items():
  p=ROOT/rel
  if not p.is_file():raise FileNotFoundError(f'Physical-prior source dependency missing: {rel}')
  b=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  if actual!=digest:raise RuntimeError(f'Source dependency changed: {rel}; rebuild/revalidate candidate')
 for rel,digest in manifest['candidate_sha256'].items():
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Candidate checksum mismatch: {rel}')
 return load_profile()
def configure_robot(robot):
 from isaaclab.actuators import ImplicitActuatorCfg
 p=verify_files();robot=copy.deepcopy(robot);robot.spawn.usd_path=str(ROOT/p['usd']);robot.actuators={}
 # Three batched groups keep per-joint priors without 56 per-step Python actuator calls.
 for role in ('arm','hand_active','passive'):
  records={n:j for n,j in p['joints'].items() if j['role']==role}
  fields={'effort_limit_sim':'effort_limit_nm','velocity_limit_sim':'velocity_limit_rad_s','stiffness':'stiffness_nm_rad','damping':'drive_damping_nm_s_rad','armature':'armature_kg_m2','friction':'physx_joint_friction_coefficient'}
  kwargs={key:{'^'+re.escape(n)+'$':j[src] for n,j in records.items()} for key,src in fields.items()}
  robot.actuators[role]=ImplicitActuatorCfg(joint_names_expr=['^'+re.escape(n)+'$' for n in records],**kwargs)

 return robot

def validate_stage(root_prim_path):
 import numpy as np
 from pxr import Usd,UsdPhysics,Gf
 from isaaclab.sim.utils.stage import get_current_stage
 p=load_profile();stage=get_current_stage();root=stage.GetPrimAtPath(root_prim_path)
 bodies={x.GetName():x for x in Usd.PrimRange(root) if x.HasAPI(UsdPhysics.RigidBodyAPI)}
 if set(bodies)!=set(p['bodies']):raise RuntimeError('Physical-prior rigid-body topology mismatch')
 for name,t in p['bodies'].items():
  a=UsdPhysics.MassAPI(bodies[name]);m=float(a.GetMassAttr().Get());com=np.array(a.GetCenterOfMassAttr().Get());q=a.GetPrincipalAxesAttr().Get();d=np.array(a.GetDiagonalInertiaAttr().Get());R=np.column_stack([np.array(q.Transform(Gf.Vec3f(*v))) for v in np.eye(3)]);I=R@np.diag(d)@R.T;ref=np.array(t['tensor_kg_m2'])
  if abs(m-t['mass_kg'])>max(1e-7,t['mass_kg']*1e-6) or np.linalg.norm(com-t['com_m'])>1e-6 or np.linalg.norm(I-ref)>1e-6*np.linalg.norm(ref):raise RuntimeError(f'Physical-prior mass/inertia mismatch: {name}')
 return {'profile':p['profile_id'],'validated_bodies':len(bodies),'hardware_calibrated':False}
