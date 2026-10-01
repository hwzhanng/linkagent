import pathlib,json,hashlib,csv
R=pathlib.Path(__file__).resolve().parents[1];patch=R/'patch';source=R/'reference/source_repo';T=json.loads((R/'reference/source_blob_hashes.json').read_text());deps=json.loads((R/'reference/source_usd_dependencies.json').read_text());base='botSDK/urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_training_mass1kg/';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();integritypath=patch/'simulation/config/linkbot_physical_prior_v1.integrity.json';files=[p for p in patch.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p!=integritypath]
integrity={'git_dependencies':{base+k:v for k,v in deps.items()},'candidate_sha256':{str(p.relative_to(patch)):sha(p) for p in files}};rel='botSDK/urdf/02_project_derived/ros2_packages/linkbot_bimanual_sdk_description/urdf/linkbot_bimanual_sdk_description.urdf';integrity['git_dependencies'][rel]=T[rel];integritypath.write_text(json.dumps(integrity,indent=2)+'\n');files.append(integritypath)
baseline=json.loads((R/'reference/accepted_baseline_sha256.json').read_text())
(R/'install_manifest.json').write_text(json.dumps({'base_commit':'ac71d6bb3a65ca6f0add5599f766caf856716bb3','baseline_sha256':baseline,'patch_sha256':{str(p.relative_to(patch)):sha(p) for p in files}},indent=2)+'\n')
p=json.loads((patch/'simulation/config/linkbot_physical_prior_v1.json').read_text())
with (R/'joint_parameters.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['joint','role','effort_limit_nm','velocity_limit_rad_s','lower_rad','upper_rad','urdf_viscous_nm_s_rad','urdf_coulomb_nm','stiffness_nm_rad','drive_damping_nm_s_rad','armature_kg_m2','physx_joint_friction_coefficient']);w.writeheader()
 for n,j in p['joints'].items():w.writerow({'joint':n,**{k:j[k] for k in w.fieldnames if k!='joint'}})
with (R/'body_parameters.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['link','mass_kg','com_x_m','com_y_m','com_z_m','ixx','ixy','ixz','iyy','iyz','izz'])
 for n,b in p['bodies'].items():I=b['tensor_kg_m2'];w.writerow([n,b['mass_kg'],*b['com_m'],I[0][0],I[0][1],I[0][2],I[1][1],I[1][2],I[2][2]])
print('Manifests and per-link/joint parameter tables updated')
