#!/usr/bin/env python3
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "analyze_z_only_ground_alignment.py"
SPEC = importlib.util.spec_from_file_location("z_only", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_recovers_z_without_changing_xy() -> None:
    rng = np.random.default_rng(7)
    xy = rng.uniform(-8.0, 8.0, size=(1200, 2))
    z = 0.08 * xy[:, 0] - 0.04 * xy[:, 1]
    reference = np.column_stack([xy, z])
    normals = np.tile(np.array([-0.08, 0.04, 1.0]), (len(xy), 1))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    z_error = 2.35
    current = reference + np.array([0.0, 0.0, z_error])
    current += rng.normal(0.0, 0.01, current.shape)
    outliers = rng.choice(len(current), 180, replace=False)
    current[outliers, 2] += rng.uniform(-1.5, 1.5, len(outliers))
    result = MODULE.estimate_z_only(current, reference, normals)
    assert result["ok"]
    assert abs(float(result["dz"]) + z_error) < 0.03
    assert float(result["after_median_abs"]) < 0.03


def test_cell_correspondences_are_one_vote_per_cell() -> None:
    rng = np.random.default_rng(11)
    xy = rng.uniform(-4.0, 4.0, size=(5000, 2))
    reference = np.column_stack([xy, 0.03 * xy[:, 0]])
    current = np.repeat(reference, 3, axis=0)
    current[:, 2] += 0.7
    cur, ref, normals = MODULE.build_cell_correspondences(
        current, reference, 0.75, 2.0, 5, 0.8
    )
    keys = np.floor(cur[:, :2] / 0.75).astype(int)
    assert len(keys) == len(np.unique(keys, axis=0))
    result = MODULE.estimate_z_only(cur, ref, normals)
    assert result["ok"]
    assert abs(float(result["dz"]) + 0.7) < 0.03


if __name__ == "__main__":
    test_recovers_z_without_changing_xy()
    test_cell_correspondences_are_one_vote_per_cell()
    print("z-only ground alignment synthetic test: PASS")
