from pathlib import Path


def test_v3_proxy_filter_is_narrowly_scoped_to_support_insert():
    source = (Path(__file__).parents[1] / "collision_proxies.py").read_text()
    assert "FilteredPairsAPI" in source
    assert "palm_support_surface" in source
    assert "Tomato" not in source


def test_v3_cfg_replaces_v2_proxy_authoring_event():
    source = (Path(__file__).parents[1] / "env.py").read_text()
    assert "event.func = author_fingertip_collision_proxies" in source
    assert 'event.params["filter_palm_support"]' in source
    assert "ImplicitActuatorCfg" in source
    assert "set_dof_stiffnesses" in source


def test_scenario_uses_bounded_physical_joint_drive():
    import json

    scenario = json.loads(
        (Path(__file__).parents[1] / "scenarios" / "v3_large90_nominal.json").read_text()
    )
    assert scenario["physics"]["direct_position_servo_enabled"] is False
    assert scenario["physics"]["direct_servo_zero_velocity"] is True
    assert 0.02 <= scenario["physics"]["joint_drive_effort_limit_nm"] <= 1.0
    assert 0.05 <= scenario["physics"]["joint_drive_stiffness_nm_rad"] <= 5.0
