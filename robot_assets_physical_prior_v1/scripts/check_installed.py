import argparse,json,pathlib,hashlib,importlib.util,numpy as np
from pxr import Usd,UsdPhysics,Gf
R=pathlib.Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--repo',required=True,type=pathlib.Path);args=a.parse_args();p=json.load(open(R/'patch/simulation/config/linkbot_physical_prior_v1.json'));s=Usd.Stage.Open(str(args.repo/p['usd']));assert s.GetDefaultPrim().IsValid();assert s.GetMetadata('metersPerUnit')==1.;assert s.GetMetadata('upAxis')=='Z';b={z.GetName():z for z in s.TraverseAll() if z.HasAPI(UsdPhysics.RigidBodyAPI)};assert len(b)==71
for n,t in p['bodies'].items():
 x=UsdPhysics.MassAPI(b[n]);assert abs(float(x.GetMassAttr().Get())-t['mass_kg'])<1e-6;q=x.GetPrincipalAxesAttr().Get();D=np.array(x.GetDiagonalInertiaAttr().Get());Q=np.column_stack([np.array(q.Transform(Gf.Vec3f(*v))) for v in np.eye(3)]);I=Q@np.diag(D)@Q.T;ref=np.array(t['tensor_kg_m2']);assert np.linalg.norm(I-ref)/np.linalg.norm(ref)<1e-6
manifest=json.load(open(R/'install_manifest.json'))
for rel,digest in manifest['patch_sha256'].items():assert hashlib.sha256((args.repo/rel).read_bytes()).hexdigest()==digest
out={'direct_sibling_composition':True,'bodies_verified':71,'installed_files_verified':len(manifest['patch_sha256']),'full_geometry_present':(args.repo/pathlib.Path(p['usd']).parent/'payloads/geometries.usd').is_file(),'default_prim':str(s.GetDefaultPrim().GetPath()),'units':'meters kg Z-up'};(R/'results/installed_composition.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
