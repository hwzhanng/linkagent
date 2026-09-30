import json,pathlib,xml.etree.ElementTree as E,numpy as np
from inertia_math import urdf_bodies
R=pathlib.Path(__file__).resolve().parents[1];p=json.loads((R/'patch/simulation/config/linkbot_physical_prior_v1.json').read_text());r=E.parse(R/'patch'/p['urdf']).getroot();b=urdf_bodies(R/'patch'/p['urdf']);bounds={x['path'].split('/meshes/')[-1]:x for x in json.loads((R/'reference/verified_mesh_bounds.json').read_text()) if not x.get('missing')};tested=[];missing=[]
for l in r.findall('link'):
 v=l.find('visual');m=v.find('geometry/mesh');key=m.get('filename').split('/meshes/')[-1];n=l.get('name')
 if key not in bounds:missing.append(n);continue
 assert v.find('origin').get('xyz')=='0 0 0' and v.find('origin').get('rpy')=='0 0 0'
 lo,hi=np.array(bounds[key]['bounds_m']);t=b[n];c=t['com'];cov=(np.trace(t['I'])/2*np.eye(3)-t['I'])/t['mass'];limit=(hi-c)*(c-lo);ok=bool((np.diag(cov)<=limit+1e-8).all() and (c>=lo-1e-5).all() and (c<=hi+1e-5).all());assert ok,n
 tested.append({'link':n,'mass_kg':t['mass'],'radius_of_gyration_m':np.sqrt(np.linalg.eigvalsh(t['I'])/t['mass']).tolist(),'com_inside_aabb':True,'axis_variance_bound_pass':ok})
out={'scope':'Necessary AABB support conditions only, not proof of detailed material/density distribution. Bounds from previous 67 SHA-verified source meshes; four large source meshes unavailable. Coordinate-marker priors use uniform boxes.','checked':len(tested),'not_checked':missing,'results':tested};(R/'results/geometry_validation.json').write_text(json.dumps(out,indent=2));print('Geometry necessary-condition pass',len(tested),'missing',missing)
