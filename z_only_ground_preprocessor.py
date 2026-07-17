#!/usr/bin/env python3
"""Stateful, tree-independent Z-only ground preprocessing stage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

# Keep the module importable from tests/, another application, or an installed
# entry point without requiring the caller to set PYTHONPATH explicitly.
MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from tools import analyze_z_only_ground_alignment as ground


def read_window_trajectory(path: Path) -> np.ndarray:
    rows = []
    with path.open() as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                raise RuntimeError(f"Invalid trajectory row in {path}: {line}")
            rows.append(values)
    return np.asarray(rows, dtype=float)


@dataclass
class GroundZConfig:
    crop_radius: float = 15.0
    voxel: float = 0.20
    ground_cell: float = 0.75
    ground_quantile: float = 0.10
    ground_below: float = 0.10
    ground_above: float = 0.35
    min_cell_points: int = 3
    match_cell: float = 0.75
    normal_radius: float = 2.0
    min_normal_neighbors: int = 5
    min_abs_nz: float = 0.80
    huber_delta: float = 0.15
    trim_fraction: float = 0.80
    iterations: int = 10
    min_correspondences: int = 30
    max_abs_dz: float = 4.0
    max_after_median_abs: float = 0.20
    max_trimmed_rmse: float = 0.30
    max_rmse_ratio: float = 0.80
    minimum_dz_for_improvement_gate: float = 0.10
    max_anchor_distance: float = 5.0
    temporal_window_queries: int = 8
    temporal_tolerance: float = 0.25
    temporal_min_support: int = 3
    max_update_z: float = 0.15


def select_1d_cluster(
    candidates: list[tuple[int, float]], tolerance: float
) -> tuple[list[tuple[int, float]], int]:
    if not candidates:
        return [], 0
    neighborhoods = [
        [candidate for candidate in candidates if abs(candidate[1] - center[1]) <= tolerance]
        for center in candidates
    ]
    order = sorted(range(len(neighborhoods)), key=lambda i: (len(neighborhoods[i]), candidates[i][0]), reverse=True)
    best = neighborhoods[order[0]]
    runner_up = 0
    best_center = float(np.median([value for _, value in best]))
    for index in order[1:]:
        center = candidates[index][1]
        if abs(center - best_center) > tolerance:
            runner_up = max(runner_up, len(neighborhoods[index]))
    return best, runner_up


class GroundZPreprocessor:
    def __init__(
        self,
        reference_window_root: Path,
        current_window_root: Path,
        config: GroundZConfig | None = None,
    ) -> None:
        self.reference_root = Path(reference_window_root)
        self.current_root = Path(current_window_root)
        self.config = config or GroundZConfig()
        self.reference_trajectory = read_window_trajectory(self.reference_root / "trajectory.txt")
        self.current_trajectory = read_window_trajectory(self.current_root / "trajectory.txt")
        self.candidates: deque[tuple[int, float]] = deque()

    def _estimate(self, query: int, state: np.ndarray) -> dict:
        cfg = self.config
        center_odom = self.current_trajectory[query, 1:4]
        center_map = ground.transform_points(center_odom.reshape(1, 3), state)[0]
        distances = np.linalg.norm(self.reference_trajectory[:, 1:3] - center_map[:2], axis=1)
        reference_index = int(np.argmin(distances))
        anchor_distance = float(distances[reference_index])
        result = {
            "query": query,
            "reference_index": reference_index,
            "anchor_distance": anchor_distance,
            "state_z_before": float(state[2, 3]),
        }
        if anchor_distance > cfg.max_anchor_distance:
            return {**result, "valid": False, "gate_reason": "reference_anchor_too_far"}
        current = ground.read_cloud(ground.payload_path(self.current_root, query))
        reference = ground.read_cloud(ground.payload_path(self.reference_root, reference_index))
        current = ground.crop_xy(ground.transform_points(current, state), center_map[:2], cfg.crop_radius)
        reference = ground.crop_xy(reference, center_map[:2], cfg.crop_radius)
        current = ground.voxel_downsample(current, cfg.voxel)
        reference = ground.voxel_downsample(reference, cfg.voxel)
        current_ground = ground.low_envelope_ground(
            current, cfg.ground_cell, cfg.ground_quantile, cfg.ground_below,
            cfg.ground_above, cfg.min_cell_points,
        )
        reference_ground = ground.low_envelope_ground(
            reference, cfg.ground_cell, cfg.ground_quantile, cfg.ground_below,
            cfg.ground_above, cfg.min_cell_points,
        )
        cur, ref, normals = ground.build_cell_correspondences(
            current_ground, reference_ground, cfg.match_cell, cfg.normal_radius,
            cfg.min_normal_neighbors, cfg.min_abs_nz,
        )
        estimate = ground.estimate_z_only(
            cur, ref, normals, cfg.huber_delta, cfg.trim_fraction, cfg.iterations
        )
        result.update({
            "current_ground_points": len(current_ground),
            "reference_ground_points": len(reference_ground),
            "correspondences": len(cur),
            **estimate,
        })
        dz = float(estimate.get("dz", math.nan))
        result["state_z_candidate"] = float(state[2, 3]) + dz
        valid, reason = True, "accepted"
        if not bool(estimate.get("ok", False)):
            valid, reason = False, str(estimate.get("reason", "optimizer_failed"))
        elif len(cur) < cfg.min_correspondences:
            valid, reason = False, "too_few_correspondences"
        elif abs(dz) > cfg.max_abs_dz:
            valid, reason = False, "dz_too_large"
        elif float(estimate["trimmed_after_median_abs"]) > cfg.max_after_median_abs:
            valid, reason = False, "residual_too_large"
        elif float(estimate["trimmed_after_rmse"]) > cfg.max_trimmed_rmse:
            valid, reason = False, "trimmed_rmse_too_large"
        elif (
            float(estimate["trimmed_after_rmse"])
            >= cfg.max_rmse_ratio * float(estimate["trimmed_before_rmse"])
            and abs(dz) >= cfg.minimum_dz_for_improvement_gate
        ):
            valid, reason = False, "insufficient_improvement"
        result["valid"] = valid
        result["gate_reason"] = reason
        return result

    def update(self, query: int, state: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return a copy of state with at most Z changed, plus diagnostics."""
        try:
            result = self._estimate(query, state)
        except Exception as error:
            result = {
                "query": query,
                "reference_index": -1,
                "anchor_distance": math.nan,
                "state_z_before": float(state[2, 3]),
                "valid": False,
                "gate_reason": f"exception: {error}",
            }
        minimum_query = query - self.config.temporal_window_queries + 1
        while self.candidates and self.candidates[0][0] < minimum_query:
            self.candidates.popleft()
        if result["valid"]:
            self.candidates.append((query, float(result["state_z_candidate"])))
        cluster, runner_up = select_1d_cluster(
            list(self.candidates), self.config.temporal_tolerance
        )
        current_member = any(candidate_query == query for candidate_query, _ in cluster)
        ready = (
            len(cluster) >= self.config.temporal_min_support
            and len(cluster) > runner_up
            and current_member
        )
        updated = state.copy()
        target = math.nan
        step = 0.0
        if ready:
            target = float(np.median([value for _, value in cluster]))
            step = float(np.clip(
                target - float(state[2, 3]),
                -self.config.max_update_z,
                self.config.max_update_z,
            ))
            updated[2, 3] += step
        result.update({
            "temporal_support": len(cluster),
            "temporal_runner_up_support": runner_up,
            "current_is_temporal_member": current_member,
            "temporal_ready": ready,
            "state_z_target": target,
            "z_update": step,
            "state_z_after": float(updated[2, 3]),
        })
        return updated, result
