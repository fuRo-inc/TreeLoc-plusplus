#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "z_only_ground_preprocessor.py"
SPEC = importlib.util.spec_from_file_location("z_pre", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_temporal_cluster_rejects_outlier() -> None:
    values = [(1, -10.0), (2, -10.1), (3, -9.95), (4, -6.0)]
    cluster, runner_up = MODULE.select_1d_cluster(values, 0.25)
    assert len(cluster) == 3
    assert runner_up == 1


def test_update_changes_only_z_after_temporal_support() -> None:
    processor = MODULE.GroundZPreprocessor.__new__(MODULE.GroundZPreprocessor)
    processor.config = MODULE.GroundZConfig(
        temporal_window_queries=8,
        temporal_tolerance=0.25,
        temporal_min_support=3,
        max_update_z=0.15,
    )
    processor.candidates = MODULE.deque()
    values = {1: -10.00, 2: -10.05, 3: -9.95}
    processor._estimate = lambda query, state: {
        "query": query,
        "reference_index": 0,
        "anchor_distance": 0.0,
        "state_z_before": float(state[2, 3]),
        "state_z_candidate": values[query],
        "valid": True,
        "gate_reason": "accepted",
    }
    state = np.eye(4)
    state[:3, 3] = [12.0, -4.0, -11.0]
    original_xy_rotation = state.copy()
    for query in (1, 2, 3):
        state, result = processor.update(query, state)
    assert result["temporal_ready"]
    assert abs(state[2, 3] - (-10.85)) < 1e-9
    assert np.array_equal(state[:2, :], original_xy_rotation[:2, :])
    assert np.array_equal(state[:3, :3], original_xy_rotation[:3, :3])


if __name__ == "__main__":
    test_temporal_cluster_rejects_outlier()
    test_update_changes_only_z_after_temporal_support()
    print("z-only ground preprocessor temporal test: PASS")
