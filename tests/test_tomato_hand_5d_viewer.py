from __future__ import annotations

import xml.etree.ElementTree as ET

from module.skills.tomato_pick.hand_synergy import (
    PREGRASP_SYNERGY,
    g20_synergy_to_joint_targets,
)
from script.serve_tomato_hand_5d_viewer import (
    build_hand_only_urdf,
    map_synergy,
    viewer_config,
)


def test_hand_only_urdf_contains_no_arm_or_left_hand() -> None:
    root = ET.fromstring(build_hand_only_urdf())
    links = [item.attrib["name"] for item in root.findall("link")]
    joints = [item.attrib["name"] for item in root.findall("joint")]
    assert len(links) == 27
    assert len(joints) == 26
    assert all(name.startswith("right_hand_") for name in links)
    assert all(name.startswith("right_hand_") for name in joints)
    assert "right_hand_mount_joint" not in joints
    mesh_paths = [mesh.attrib["filename"] for mesh in root.iter("mesh")]
    assert mesh_paths
    assert all(path.startswith("package/meshes/right_hand/") for path in mesh_paths)


def test_viewer_initial_pose_is_the_current_production_5d_pregrasp() -> None:
    config = viewer_config()
    assert config["pregrasp_synergy"] == list(PREGRASP_SYNERGY)
    mapped = map_synergy(list(PREGRASP_SYNERGY))
    expected = g20_synergy_to_joint_targets(PREGRASP_SYNERGY)
    assert tuple(mapped["active_joint_targets"].values()) == expected
    assert len(mapped["active_joint_targets"]) == 16
    assert len(mapped["all_revolute_joint_targets"]) == 21


def test_all_five_dimensions_change_the_exact_production_mapping() -> None:
    baseline = map_synergy([0.0] * 5)["active_joint_targets"]
    for index in range(5):
        values = [0.0] * 5
        values[index] = 0.6
        assert map_synergy(values)["active_joint_targets"] != baseline


def test_synergy_api_rejects_out_of_range_values() -> None:
    try:
        map_synergy([0.0, 0.0, 0.0, 0.0, 1.01])
    except ValueError as error:
        assert "[-1, 1]" in str(error)
    else:
        raise AssertionError("out-of-range synergy was accepted")
