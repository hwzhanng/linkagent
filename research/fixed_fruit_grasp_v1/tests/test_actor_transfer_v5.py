"""奖励迁移只接受已审查版本，真实物理和保持身份不得改变；纯CPU测试。"""
import copy
import pytest
from research.fixed_fruit_grasp_v1.tests.test_actor_transfer_v4 import identities,validate
from research.fixed_fruit_grasp_v1.run_utils import sha256_json,contract_sha256,validate_checkpoint_compatibility

def reward_identities():
 m,c,h=identities();m['task_identity']['task_version']='fixed_fruit_grasp_v1.4.0';c['task_version']='fixed_fruit_grasp_v1.5.0'
 old=m['task_identity'];old['reward']={'contact':1,'hold':2,'bonus':10};c['reward']={'contact':.25,'hold':6,'bonus':30}
 m['progress']['total_updates_completed']=85
 files={'research/fixed_fruit_grasp_v1/'+name:'a'*64 for name in ['physical_hold.py','physical_contacts.py','collision_cooking.py']}
 m['source_and_asset_hashes']['training_semantics_source_files']=copy.deepcopy(files);h['training_semantics_source_files']=copy.deepcopy(files)
 m['task_identity_sha256']=sha256_json(old);m['contract_sha256']=contract_sha256(old)
 return m,c,h

def test_reward_only_transfer_preserves_actual_best_lineage():
 m,c,h=reward_identities();r=validate(m,c,h)
 assert r['source_total_updates_completed']==85 and r['new_semantics_updates_before']==0
 assert r['reward_profile_transfer_only'] and r['physical_criterion_unchanged']
 assert r['critic_optimizer_and_iteration_reset'] and r['actor_normalizer_transferred']
 assert not r['is_new_semantics_success_evidence']

def test_reward_transfer_without_new_reward_subtree_still_preserves_physics():
 m,c,h=reward_identities();c['reward']=copy.deepcopy(m['task_identity']['reward'])
 assert validate(m,c,h)['physical_criterion_unchanged']

@pytest.mark.parametrize('field',['physics','criteria','contact','stage','goal','unknown_new_field'])
def test_reward_transfer_checks_entire_physical_identity_including_new_fields(field):
 m,c,h=reward_identities();c[field]={'unreviewed':True}
 with pytest.raises(ValueError):validate(m,c,h)

@pytest.mark.parametrize('name',['physical_hold.py','physical_contacts.py','collision_cooking.py'])
@pytest.mark.parametrize('missing',[False,True])
def test_reward_transfer_rejects_modified_or_missing_actual_guard_code(name,missing):
 m,c,h=reward_identities();key='research/fixed_fruit_grasp_v1/'+name
 if missing:h['training_semantics_source_files'].pop(key)
 else:h['training_semantics_source_files'][key]='b'*64
 with pytest.raises(ValueError):validate(m,c,h)

@pytest.mark.parametrize('change',['force','speed','hold_substeps','release_step','fixture','normalizer_architecture','version_pair','source_digest','asset_bytes'])
def test_reward_transfer_rejects_unsafe_or_unreviewed_change(change):
 m,c,h=reward_identities()
 if change=='force':c['contact']['hard_high_n']=30
 if change=='speed':c['criteria']['hold_max_relative_speed_m_s']=.1
 if change=='hold_substeps':c['physics']['physical_hold_guard']={'required_substeps':20}
 if change=='release_step':c['stage']['support_release_step']=120
 if change=='fixture':c['physics']['support']='scripted_position'
 if change=='normalizer_architecture':m['agent_architecture_sha256']='wrong'
 if change=='version_pair':m['task_identity']['task_version']='fixed_fruit_grasp_v1.3.0';m['task_identity_sha256']=sha256_json(m['task_identity']);m['contract_sha256']=contract_sha256(m['task_identity'])
 if change=='source_digest':m['task_identity_sha256']='wrong'
 if change=='asset_bytes':h['referenced_asset_files']['/other/source-v4/assets/a.usd']='changed'
 with pytest.raises(ValueError):validate(m,c,h)

def test_reward_transfer_normal_resume_remains_strict():
 m,c,h=reward_identities()
 with pytest.raises(ValueError):validate_checkpoint_compatibility(m,c,'contact')
