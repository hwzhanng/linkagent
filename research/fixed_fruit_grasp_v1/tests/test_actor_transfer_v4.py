"""显式actor迁移的拒绝路径与物理身份保护，不启动Kit。"""
import copy
import pytest
from research.fixed_fruit_grasp_v1.run_utils import (
    CONTRACT_ID,contract_sha256,sha256_json,validate_explicit_actor_transfer,
    validate_checkpoint_compatibility,
)

def identities():
    old={'task_version':'fixed_fruit_grasp_v1.3.0'}
    for k in ('actuation','assets','asset_config','fruit','initial_pose','randomization','run','safety_guard'):
        old[k]={'value':1,'path':'/disk/source-v3/assets/a.usd'}
    old['physics']={k:1 for k in ('sim_dt_s','policy_dt_s','decimation','hand_collision_cooking','support','observation_dims')}
    old['criteria']={k:1 for k in ('hold_max_drift_m','hold_max_relative_speed_m_s','drop_height_tolerance_m','drop_max_drift_m','instability_max_fruit_speed_m_s','instability_max_joint_velocity_rad_s','joint_limit_margin_rad','penetration_limit_m','thumb_required')}
    old['criteria']['thumb_required']=False
    old['contact']={k:1 for k in ('force_threshold_n','hard_high_n','min_contact_fingers','real_self_collisions_enabled','palm_fruit_collision_filter_enabled','palm_only_counts_as_success','declared_self_collision_pairs','effective_self_collision_pairs','self_collision_penetration_limit_m','skipped_self_collision_pairs')}
    old['contact']['palm_only_counts_as_success']=False
    old['stage']={k:1 for k in ('stage','required_unsupported_hold_s','support_release_step','min_height_gain_m','requires_lift_program')}
    old['stage']['stage']='contact'
    old['asset_config']={'usd_path':'/disk/source-v3/assets/a.usd','asset_config_path':'/disk/source-v3/config/c.usd','asset_files':{'geom':'/disk/source-v3/assets/a.usd'},'fruit':old['fruit'],'actuation':old['actuation'],'physics':old['physics'],'contact':old['contact'],'criteria':old['criteria']}
    current=copy.deepcopy(old);current['task_version']='fixed_fruit_grasp_v1.4.0' 
    for k in ('actuation','assets','fruit','initial_pose','randomization','run','safety_guard'):current[k]['path']='/other/source-v4/assets/a.usd'
    m={'contract_id':CONTRACT_ID,'task_identity':old,'task_identity_sha256':sha256_json(old),'contract_sha256':contract_sha256(old),'stage':'contact','agent_architecture_sha256':'architecture','agent_config_comparison_sha256':'config','source_and_asset_hashes':{'referenced_asset_files':{'/disk/source-v3/assets/a.usd':'bytes'},'missing_referenced_paths':[]},'progress':{'total_updates_completed':55}}
    hashes={'referenced_asset_files':{'/other/source-v4/assets/a.usd':'bytes'},'missing_referenced_paths':[]}
    return m,current,hashes

def validate(m,c,h,actual='a'*64,expected='a'*64,arch='architecture'):
    return validate_explicit_actor_transfer(m,c,'contact',h,arch,'config',actual,expected)

def test_explicit_transfer_preserves_old_updates_and_resets_new_stage():
    m,c,h=identities();r=validate(m,c,h)
    assert r['source_total_updates_completed']==55
    assert r['new_semantics_updates_before']==0
    assert r['critic_optimizer_and_iteration_reset']
    assert r['actor_normalizer_transferred']
    assert not r['is_new_semantics_success_evidence']

def test_normal_resume_still_rejects_task_version_change():
    m,c,h=identities()
    with pytest.raises(ValueError):validate_checkpoint_compatibility(m,c,'contact')

@pytest.mark.parametrize('change',['pin','architecture','version','speed','force','asset','source_digest','unknown_updates','missing_asset'])
def test_transfer_rejects_unreviewed_or_physically_changed_input(change):
    m,c,h=identities();actual='a'*64;arch='architecture'
    if change=='pin':actual='b'*64
    if change=='architecture':arch='different'
    if change=='version':c['task_version']='fixed_fruit_grasp_v9'
    if change=='speed':c['criteria']['hold_max_relative_speed_m_s']=2
    if change=='force':c['contact']['hard_high_n']=2
    if change=='asset':h['referenced_asset_files']['/other/source-v4/assets/a.usd']='changed'
    if change=='source_digest':m['task_identity_sha256']='corrupted'
    if change=='unknown_updates':m['progress']['total_updates_completed']=True
    if change=='missing_asset':h['missing_referenced_paths']=['missing.usd']
    with pytest.raises(ValueError):validate(m,c,h,actual=actual,arch=arch)


def test_transfer_allows_declared_stricter_sampling_description_only():
    m,c,h=identities();c['physics']['physical_hold_guard']={'frequency_hz':120,'streak':120}
    c['contact']['penetration_guard_scope']='all declared sensors at physical substeps'
    assert validate(m,c,h)['new_semantics_updates_before']==0
