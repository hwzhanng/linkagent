"""Build engineering-prior URDF and PhysX overlay; never overwrite source assets.
No claim of real-device calibration. Run from bundle: --source-repo checkout.
"""
import argparse,copy,json,math,os,pathlib,re,xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf,Sdf,Usd,UsdPhysics
from inertia_math import urdf_bodies, tensor
ROOT=pathlib.Path(__file__).resolve().parents[1]
ASSET='botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg'
UDIR='botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/urdf'
U=UDIR+'/linkbot_bimanual_sdk_physical_prior_v1.urdf'
USD=ASSET+'/linkbot_bimanual_sdk_physical_prior_v1.usda'
PROFILE='simulation/config/linkbot_physical_prior_v1.json'

def build(source,out):
 source=pathlib.Path(source).resolve();out=pathlib.Path(out).resolve();r=ET.parse(ROOT/'reference/source.urdf').getroot();old=copy.deepcopy(r)
 links={x.get('name'):x for x in r.findall('link')}; joints={x.get('name'):x for x in r.findall('joint')};records={}
 for n,l in links.items():
  if '_hand_' in n:
   i=l.find('inertial');m=i.find('mass');m.set('value',f'{float(m.get("value"))*1.2:.12g}')
   for k,v in i.find('inertia').attrib.items():i.find('inertia').set(k,f'{float(v)*1.2:.12g}')
 # The tiny distal CAD markers are coordinate-frame proxies, not 10g rigid pads.
 # Relocate their assumed mass to the adjacent real distal link; use a 1mg finite marker.
 bounds={z['path'].split('/meshes/')[-1]:z for z in json.loads((ROOT/'reference/verified_mesh_bounds.json').read_text()) if not z.get('missing')}
 for n,l in links.items():
  if not re.fullmatch(r'(left|right)_hand_(thumb_link5|(index|middle|ring|little)_link4)',n):continue
  fj=next(j for j in joints.values() if j.find('child').get('link')==n);assert fj.get('type')=='fixed';parent=links[fj.find('parent').get('link')];mi=l.find('inertial/mass');pi=parent.find('inertial');pm=pi.find('mass');mass=float(pm.get('value'));newmass=mass+float(mi.get('value'))-1e-6;pm.set('value',f'{newmass:.12g}')
  for k,v in pi.find('inertia').attrib.items():pi.find('inertia').set(k,f'{float(v)*newmass/mass:.12g}')
  key=l.find('visual/geometry/mesh').get('filename').split('/meshes/')[-1];box=np.array(bounds[key]['bounds_m']);dims=box[1]-box[0];center=box.mean(axis=0);inert=l.find('inertial');inert.find('origin').set('xyz',' '.join(str(v) for v in center));inert.find('origin').set('rpy','0 0 0');mi.set('value','0.000001');a=inert.find('inertia');diag=1e-6/12*(np.sum(dims*dims)-dims*dims)
  for k,v in zip(['ixx','iyy','izz'],diag):a.set(k,f'{v:.14g}')
  for k in ['ixy','ixz','iyz']:a.set(k,'0')
 for n,j in joints.items():
  if j.get('type')=='fixed':continue
  arm=bool(re.fullmatch('[LR][1-7]_Joint',n));lim=j.find('limit');dyn=j.find('dynamics')
  if dyn is None:dyn=ET.SubElement(j,'dynamics')
  passive=j.find('mimic') is not None
  if arm:
   idx=int(n[1])-1;eff=float(lim.get('effort'));vel=1.5
   coul=[.4,.4,.18,.18,.18,.09,.09][idx];b=[.25,.25,.12,.12,.12,.06,.06][idx]
   # Equivalent closed-loop gains, not measured hardware motor gains.
   K=[600,600,300,300,150,80,60][idx];D=[40,40,20,20,10,6,5][idx];rotor=.01;mu=.02
  else:
   finger=n.split('_hand_')[1].split('_joint')[0];idx=int(n[-1]);vel=1.0
   if passive:eff=0.;K=0.;D=0.;rotor=0.;mu=.002;coul=.0005;b=.001
   else:
    # Torque-scale priors: modest tip-force times typical CAD lever, not manufacturer ratings.
    eff=([.25,.25,.40,.30][idx] if finger=='thumb' else [.15,.60,.35][idx])
    K=4.0 if eff<=.15 else 6.0;D=.12;rotor=.0002;mu=.02;coul=.003;b=.005
  lim.set('effort',str(eff));lim.set('velocity',str(vel));dyn.set('damping',str(b));dyn.set('friction',str(coul))
  if passive:j.find('mimic').set('offset','0')
  records[n]={'role':'arm' if arm else ('passive' if passive else 'hand_active'),'effort_limit_nm':eff,'velocity_limit_rad_s':vel,'urdf_viscous_nm_s_rad':b,'urdf_coulomb_nm':coul,'stiffness_nm_rad':K,'drive_damping_nm_s_rad':D,'armature_kg_m2':rotor,'physx_joint_friction_coefficient':mu,'source':'arm torque ceilings retained from delivered SDK; other actuator values are explicit conservative engineering priors' if arm else 'engineering prior; limits are operating ceilings, not rated hardware maxima'}
 # Make the full driver interval feasible under zero-offset linear linkage.
 for n,j in joints.items():
  mim=j.find('mimic')
  if mim is None:continue
  a=float(mim.get('multiplier'));off=float(mim.get('offset'));f=j.find('limit');driver=joints[mim.get('joint')];d=driver.find('limit');lo,hi=sorted([(float(f.get('lower'))-off)/a,(float(f.get('upper'))-off)/a]);d.set('lower',str(max(float(d.get('lower')),lo)));d.set('upper',str(min(float(d.get('upper')),hi)));v=min(float(d.get('velocity')),.95*float(f.get('velocity'))/abs(a));d.set('velocity',str(v));records[driver.get('name')]['velocity_limit_rad_s']=v
  records[n]['mimic']={'joint':mim.get('joint'),'multiplier':a,'offset':off}
 for n,p in records.items():p['lower_rad']=float(joints[n].find('limit').get('lower'));p['upper_rad']=float(joints[n].find('limit').get('upper'))
 r.set('name','linkbot_bimanual_sdk_physical_prior_v1');p=out/U;p.parent.mkdir(parents=True,exist_ok=True);ET.indent(r,space='  ');ET.ElementTree(r).write(p,encoding='utf-8',xml_declaration=True)
 targets=urdf_bodies(p)
 original=source/ASSET/'linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda';stage=Usd.Stage.Open(str(original));dest=out/USD;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():dest.unlink()
 patch=Usd.Stage.CreateNew(str(dest));patch.SetMetadata('defaultPrim',stage.GetDefaultPrim().GetName());patch.SetMetadata('metersPerUnit',stage.GetMetadata('metersPerUnit'));patch.SetMetadata('kilogramsPerUnit',stage.GetMetadata('kilogramsPerUnit'));patch.SetMetadata('upAxis',stage.GetMetadata('upAxis'));patch.GetRootLayer().subLayerPaths=['linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda'];patch.GetRootLayer().customLayerData={'physicalPriorId':'linkbot_physical_prior_v1','calibrated':False,'scope':'URDF dynamic priors with inherited single-right-hand contact topology; not hardware certified'}
 bodies={x.GetName():x for x in stage.TraverseAll() if x.HasAPI(UsdPhysics.RigidBodyAPI)};jp={x.GetName():x for x in stage.TraverseAll() if x.IsA(UsdPhysics.RevoluteJoint)}
 assert set(bodies)==set(targets)
 for n,t in targets.items():
  prim=patch.OverridePrim(bodies[n].GetPath());a=UsdPhysics.MassAPI(prim);a.CreateMassAttr().Set(t['mass']);a.CreateCenterOfMassAttr().Set(Gf.Vec3f(*t['com']));a.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*t['diag']));a.CreatePrincipalAxesAttr().Set(t['q'])
 for n,p in records.items():
  prim=patch.OverridePrim(jp[n].GetPath());j=UsdPhysics.RevoluteJoint(prim);j.CreateLowerLimitAttr().Set(math.degrees(p['lower_rad']));j.CreateUpperLimitAttr().Set(math.degrees(p['upper_rad']));d=UsdPhysics.DriveAPI(prim,'angular');d.CreateTypeAttr().Set('force');d.CreateMaxForceAttr().Set(p['effort_limit_nm']);d.CreateStiffnessAttr().Set(p['stiffness_nm_rad']*math.pi/180);d.CreateDampingAttr().Set(p['drive_damping_nm_s_rad']*math.pi/180)
  for key,val in [('physxJoint:maxJointVelocity',math.degrees(p['velocity_limit_rad_s'])),('physxJoint:armature',p['armature_kg_m2']),('physxJoint:jointFriction',p['physx_joint_friction_coefficient'])]:prim.CreateAttribute(key,Sdf.ValueTypeNames.Float,custom=False).Set(val)
 # Both hands are mechanically passive. Copy the existing verified native right-hand mimic schema to left.
 for n,p in records.items():
  if p['role']!='passive' or not n.startswith('left_'):continue
  right=jp[n.replace('left_','right_',1)];left=patch.OverridePrim(jp[n].GetPath())
  for api in right.GetMetadata('apiSchemas').GetAppliedItems():
   if api.startswith('PhysxMimicJointAPI:'):left.AddAppliedSchema(api)
  for prop in right.GetProperties():
   if not prop.GetName().startswith('physxMimicJoint:'):continue
   if isinstance(prop,Usd.Attribute):left.CreateAttribute(prop.GetName(),prop.GetTypeName(),custom=False).Set(prop.Get())
   else:left.CreateRelationship(prop.GetName(),custom=False).SetTargets([Sdf.Path(str(v).replace('right_hand_','left_hand_')) for v in prop.GetTargets()])
 patch.GetRootLayer().Save()
 # Portable sibling reference; validate in a composed temporary layer rooted at actual source.
 test=Usd.Stage.CreateInMemory();test.GetRootLayer().subLayerPaths=[str(dest),str(original)];maxerr=0
 for n,t in targets.items():
  a=UsdPhysics.MassAPI(test.GetPrimAtPath(bodies[n].GetPath()));err=np.linalg.norm(tensor(a.GetDiagonalInertiaAttr().Get(),a.GetPrincipalAxesAttr().Get())-t['I'])/np.linalg.norm(t['I']);maxerr=max(maxerr,float(err));assert err<1e-6
 profile={'schema_version':1,'profile_id':'linkbot_physical_prior_v1','qualification':'public_spec_constrained_engineering_prior_not_calibrated','urdf':U,'usd':USD,'hand_mass_kg':1.2,'hand_mass_distribution':'preserve existing 55% palm,13% thumb,8% per finger; scale all hand masses and inertia tensors by 1.2, preserve CAD CoM and radii of gyration for structural links; redistribute 12g marker prior to adjacent distal links, leave 1mg box-shaped coordinate markers','mount_adapter_mass_kg':.03,'contact':{'static_friction':.55,'dynamic_friction':.45,'combine_mode':'average','stiffness_n_m':5000,'damping_n_s_m':25,'status':'retain inherited provisional contact model; unmeasured pad/cultivar; geometry/contact regression required'},'uncertainty_sweep':{'hand_mass_scale':[.85,1,1.15],'actuator_effort_scale':[.6,1,1.4],'drive_stiffness_scale':[.7,1,1.3],'drive_damping_scale':[.7,1,1.3],'armature_scale':[.5,1,2],'urdf_friction_scale':[.5,1,2],'contact_static_friction':[.3,.55,.8],'semantics':'engineering stress scenarios, not confidence intervals; do not apply independent mass/inertia scalings'},'joints':records,'sources':[{'url':'https://a.storyblok.com/f/298593/x/50a0fc9e84/linker-hand-l20-v10-20260324.pdf','revision':'V1.2 2026-03-24','use':'1.2kg complete hand; 16 active+5 passive; published speeds exceed chosen operating caps; 18N thumb/20N finger maxima are not joint torque constants'},{'url':'https://www.linkerbot.cn/details/LinkerHandL20/','use':'1.2kg cross-check; thumb15N differs from manual18N; use lower15N only as contextual force ceiling, not certified limiter'},{'url':'https://github.com/hwzhanng/linkagent/tree/ac71d6bb3a65ca6f0add5599f766caf856716bb3','use':'delivered SDK kinematics, arm inertia and 80/36/18Nm torque ceilings; legacy hand geometry/coordinate mapping retained'}],'full_isaac_contact_regression':False,'hardware_transfer_qualified':False}
 profile['bodies']={n:{'mass_kg':float(t['mass']),'com_m':t['com'].tolist(),'tensor_kg_m2':t['I'].tolist()} for n,t in targets.items()}
 pp=out/PROFILE;pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text(json.dumps(profile,indent=2)+'\n');(ROOT/'results/build.json').write_text(json.dumps({'body_count':len(targets),'joint_count':len(records),'total_mass_kg':sum(t['mass'] for t in targets.values()),'max_tensor_error':maxerr},indent=2));return profile
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source-repo',required=True);a.add_argument('--output-repo',default=str(ROOT/'patch'));args=a.parse_args();print(json.dumps(build(args.source_repo,args.output_repo),indent=2))
