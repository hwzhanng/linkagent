from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from module.skills.tomato_pick.hand_synergy import HAND_LOWER_RAD, HAND_UPPER_RAD
from module.skills.tomato_pick.hand_synergy_calibrated_v2 import (
    CENTER_RAD,
    FINGER_ACTIVE_JOINT_INDICES,
    MAPPING_ID,
    NEGATIVE_ENDPOINTS_RAD,
    POSITIVE_ENDPOINTS_RAD,
    SOURCE_DRAFT_SHA256,
    THUMB_ACTIVE_JOINT_INDICES,
    g20_synergy_to_joint_targets_v2,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "simulation/config/tomato_hand_5d_user_calibrated_v2.json"
DRAFT = ROOT / "logs/hand_5d_viewer/latest_draft.json"


def test_frozen_calibration_matches_saved_draft_and_mapper() -> None:
    payload = json.loads(CALIBRATION.read_text())
    assert payload["mapping_id"] == MAPPING_ID
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == SOURCE_DRAFT_SHA256
    assert payload["source_draft_sha256"] == SOURCE_DRAFT_SHA256
    assert tuple(payload["anchors_rad"]["center"]) == CENTER_RAD
    assert payload["controlled_joint_indices_by_coordinate"] == [
        list(FINGER_ACTIVE_JOINT_INDICES),
        list(FINGER_ACTIVE_JOINT_INDICES),
        list(FINGER_ACTIVE_JOINT_INDICES),
        list(FINGER_ACTIVE_JOINT_INDICES),
        list(THUMB_ACTIVE_JOINT_INDICES),
    ]
    for index, (negative_name, positive_name) in enumerate(payload["endpoint_pairs"]):
        assert tuple(payload["anchors_rad"][negative_name]) == NEGATIVE_ENDPOINTS_RAD[index]
        assert tuple(payload["anchors_rad"][positive_name]) == POSITIVE_ENDPOINTS_RAD[index]


def test_center_and_all_ten_axis_endpoints_are_exact_on_controlled_joints() -> None:
    assert g20_synergy_to_joint_targets_v2((0.0,) * 5) == CENTER_RAD
    for coordinate in range(5):
        negative = [0.0] * 5
        positive = [0.0] * 5
        negative[coordinate] = -1.0
        positive[coordinate] = 1.0
        controlled = (
            THUMB_ACTIVE_JOINT_INDICES
            if coordinate == 4
            else FINGER_ACTIVE_JOINT_INDICES
        )
        fixed = tuple(index for index in range(16) if index not in controlled)
        negative_target = g20_synergy_to_joint_targets_v2(negative)
        positive_target = g20_synergy_to_joint_targets_v2(positive)
        assert np.allclose(
            np.asarray(negative_target)[list(controlled)],
            np.asarray(NEGATIVE_ENDPOINTS_RAD[coordinate])[list(controlled)],
            atol=1.0e-9,
        )
        assert np.allclose(
            np.asarray(positive_target)[list(controlled)],
            np.asarray(POSITIVE_ENDPOINTS_RAD[coordinate])[list(controlled)],
            atol=1.0e-9,
        )
        assert np.allclose(np.asarray(negative_target)[list(fixed)], np.asarray(CENTER_RAD)[list(fixed)])
        assert np.allclose(np.asarray(positive_target)[list(fixed)], np.asarray(CENTER_RAD)[list(fixed)])


def test_finger_coordinates_never_move_thumb_and_thumb_never_moves_fingers() -> None:
    center = np.asarray(CENTER_RAD)
    rng = np.random.default_rng(7)
    for _ in range(1024):
        finger_action = np.zeros(5)
        finger_action[:4] = rng.uniform(-1.0, 1.0, 4)
        finger_target = np.asarray(g20_synergy_to_joint_targets_v2(finger_action))
        assert np.allclose(
            finger_target[list(THUMB_ACTIVE_JOINT_INDICES)],
            center[list(THUMB_ACTIVE_JOINT_INDICES)],
        )
        thumb_action = np.zeros(5)
        thumb_action[4] = rng.uniform(-1.0, 1.0)
        thumb_target = np.asarray(g20_synergy_to_joint_targets_v2(thumb_action))
        assert np.allclose(
            thumb_target[list(FINGER_ACTIVE_JOINT_INDICES)],
            center[list(FINGER_ACTIVE_JOINT_INDICES)],
        )


def test_random_cube_is_finite_and_inside_physical_joint_limits() -> None:
    rng = np.random.default_rng(42)
    for action in rng.uniform(-1.0, 1.0, size=(32768, 5)):
        target = np.asarray(g20_synergy_to_joint_targets_v2(action))
        assert np.isfinite(target).all()
        assert (target >= np.asarray(HAND_LOWER_RAD) - 1.0e-12).all()
        assert (target <= np.asarray(HAND_UPPER_RAD) + 1.0e-12).all()


def test_local_five_axis_jacobian_has_full_rank() -> None:
    epsilon = 1.0e-5
    columns = []
    for coordinate in range(5):
        negative = np.zeros(5)
        positive = np.zeros(5)
        negative[coordinate] = -epsilon
        positive[coordinate] = epsilon
        columns.append(
            (
                np.asarray(g20_synergy_to_joint_targets_v2(positive))
                - np.asarray(g20_synergy_to_joint_targets_v2(negative))
            )
            / (2.0 * epsilon)
        )
    jacobian = np.stack(columns, axis=1)
    assert np.linalg.matrix_rank(jacobian, tol=1.0e-7) == 5


def test_mapping_is_continuous_at_each_piecewise_origin() -> None:
    epsilon = 1.0e-7
    center = np.asarray(CENTER_RAD)
    for coordinate in range(5):
        for sign in (-1.0, 1.0):
            action = np.zeros(5)
            action[coordinate] = sign * epsilon
            target = np.asarray(g20_synergy_to_joint_targets_v2(action))
            assert np.max(np.abs(target - center)) < 1.0e-6
