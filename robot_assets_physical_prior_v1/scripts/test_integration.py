import copy,importlib.util,json,math,pathlib,sys,types,unittest,os
import numpy as np
from pxr import Usd,UsdPhysics
R=pathlib.Path(__file__).resolve().parents[1];P=json.loads((R/'patch/simulation/config/linkbot_physical_prior_v1.json').read_text())
spec=importlib.util.spec_from_file_location('prior',R/'patch/simulation/tomato_pick/physical_prior_asset.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class TestPrior(unittest.TestCase):
 def test_opt_in_default(self):
  old=dict(os.environ)
  try:
   os.environ.pop('LINKAGENT_PHYSICAL_PRIOR',None);self.assertFalse(h.enabled());os.environ['LINKAGENT_PHYSICAL_PRIOR']='invalid'
   with self.assertRaises(ValueError):h.enabled()
  finally:os.environ.clear();os.environ.update(old)
 def test_conflicting_flags(self):
  old=dict(os.environ)
  try:
   os.environ['LINKAGENT_PHYSICAL_PRIOR']='1';os.environ['LINKAGENT_INERTIA_FIX']='1'
   with self.assertRaises(ValueError):h.enabled()
  finally:os.environ.clear();os.environ.update(old)
 def test_exact_cfg_values_with_api_stub(self):
  old=sys.modules.get('isaaclab.actuators');stub=types.ModuleType('isaaclab.actuators');stub.ImplicitActuatorCfg=lambda **k:types.SimpleNamespace(**k);sys.modules['isaaclab.actuators']=stub;verify=h.verify_files;h.verify_files=lambda:P
  try:
   robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path='original'),actuators={});new=h.configure_robot(robot);self.assertEqual(robot.spawn.usd_path,'original');self.assertEqual(len(new.actuators),3)
   for n,p in P['joints'].items():
    a=new.actuators[p['role']];k='^'+n+'$';self.assertEqual(a.effort_limit_sim[k],p['effort_limit_nm']);self.assertEqual(a.velocity_limit_sim[k],p['velocity_limit_rad_s']);self.assertEqual(a.stiffness[k],p['stiffness_nm_rad']);self.assertEqual(a.damping[k],p['drive_damping_nm_s_rad']);self.assertEqual(a.armature[k],p['armature_kg_m2'])
   self.assertEqual(sum(v==0 for a in new.actuators.values() for v in a.effort_limit_sim.values()),10)
  finally:
   h.verify_files=verify
   if old is None:del sys.modules['isaaclab.actuators']
   else:sys.modules['isaaclab.actuators']=old
 def test_usd_joint_si_units_and_mimics(self):
  # Compose with source checkout so the portable relative layer can be resolved for this offline test.
  source=R/'reference/source_repo'/pathlib.Path(P['usd']).parent/'linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda';s=Usd.Stage.CreateInMemory();s.GetRootLayer().subLayerPaths=[str(R/'patch'/P['usd']),str(source)];joints={p.GetName():p for p in s.TraverseAll() if p.IsA(UsdPhysics.RevoluteJoint)}
  for n,p in P['joints'].items():
   j=joints[n];d=UsdPhysics.DriveAPI(j,'angular');self.assertAlmostEqual(d.GetMaxForceAttr().Get(),p['effort_limit_nm'],places=5);self.assertAlmostEqual(d.GetStiffnessAttr().Get()*180/math.pi,p['stiffness_nm_rad'],places=4);self.assertAlmostEqual(j.GetAttribute('physxJoint:maxJointVelocity').Get()*math.pi/180,p['velocity_limit_rad_s'],places=6)
   if p['role']=='passive':
    self.assertEqual(d.GetStiffnessAttr().Get(),0);self.assertTrue(any(x.startswith('PhysxMimicJointAPI:') for x in j.GetMetadata("apiSchemas").GetAppliedItems()),n)
    rels=[z for z in j.GetRelationships() if z.GetName().startswith('physxMimicJoint:')];self.assertTrue(rels,n)
    targets=[str(v) for z in rels for v in z.GetTargets()];self.assertTrue(any(p['mimic']['joint'] in v for v in targets),(n,targets))
 def test_explicit_stage_metadata(self):
  s=Usd.Stage.Open(str(R/'patch'/P['usd']))
  self.assertEqual(s.GetMetadata('defaultPrim'),'linkbot_bimanual_sdk')
  self.assertEqual(s.GetMetadata('metersPerUnit'),1.0)
  self.assertEqual(s.GetMetadata('kilogramsPerUnit'),1.0)
  self.assertEqual(s.GetMetadata('upAxis'),'Z')
 def test_all_links_physically_admissible(self):
  for n,b in P['bodies'].items():
   eig=np.linalg.eigvalsh(b['tensor_kg_m2']);self.assertGreater(b['mass_kg'],0);self.assertGreater(eig[0],0);self.assertLessEqual(eig[-1],eig[0]+eig[1]+1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
