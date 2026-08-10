from __future__ import annotations

from copy import deepcopy

import pytest

from simulation.tomato_pick.data_protocol_v2 import (
    load_data_protocol_v2,
    validate_data_protocol_v2,
)


def test_data_protocol_has_disjoint_splits_and_independent_release_test() -> None:
    protocol = load_data_protocol_v2()
    assert protocol["splits"]["train"]["seeds"] == [43]
    assert protocol["splits"]["validation"]["seeds"] == [17]
    assert protocol["splits"]["test"]["seeds"] == [42, 2026]
    assert protocol["splits"]["train"]["checkpoint_selection_allowed"] is False
    assert protocol["splits"]["test"]["checkpoint_selection_allowed"] is True
    assert protocol["checkpoint_policy"]["training_metrics_may_release"] is False


def test_data_protocol_rejects_seed_overlap_and_training_release() -> None:
    protocol = load_data_protocol_v2()
    overlap = deepcopy(protocol)
    overlap["splits"]["test"]["seeds"] = [43, 2026]
    with pytest.raises(ValueError, match="seeds overlap"):
        validate_data_protocol_v2(overlap)
    unsafe = deepcopy(protocol)
    unsafe["checkpoint_policy"]["training_metrics_may_release"] = True
    with pytest.raises(ValueError, match="may not release"):
        validate_data_protocol_v2(unsafe)
