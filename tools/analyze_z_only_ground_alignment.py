#!/usr/bin/env python3
"""Diagnose a Z-only ground alignment for accepted TreeLoc++ SE(2) updates.

The replay's x, y and yaw are held fixed.  For every accepted query, the
current window is transformed to map coordinates with the replay state and a
single scalar dz is estimated from robust point-to-plane ground residuals.
No result is fed back into the replay; this script is intentionally an
offline diagnostic and gate-calibration tool.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def planar_transform(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    result[:3, 3] = [x, y, z]
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or len(points) == 0:
        return points.copy()
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def crop_xy(points: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    delta = points[:, :2] - center_xy
    return points[np.einsum("ij,ij->i", delta, delta) <= radius * radius]


def low_envelope_ground(
    points: np.ndarray,
    cell_size: float,
    quantile: float,
    below: float,
    above: float,
    min_cell_points: int,
) -> np.ndarray:
    """Select a terrain band relative to a robust low Z per XY cell."""
    if len(points) == 0:
        return points.copy()
    keys = np.floor(points[:, :2] / cell_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    keep = np.zeros(len(points), dtype=bool)
    for cell in range(inverse.max() + 1):
        indices = np.flatnonzero(inverse == cell)
        if len(indices) < min_cell_points:
            continue
        floor_z = float(np.quantile(points[indices, 2], quantile))
        dz = points[indices, 2] - floor_z
        keep[indices[(dz >= -below) & (dz <= above)]] = True
    return points[keep]


def estimate_ground_normals(
    ground: np.ndarray,
    radius: float,
    min_neighbors: int,
    minimum_abs_nz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local terrain normals with XY-neighborhood PCA."""
    if len(ground) == 0:
        return ground.copy(), np.empty((0, 3))
    tree = cKDTree(ground[:, :2])
    accepted_points, accepted_normals = [], []
    for point in ground:
        indices = tree.query_ball_point(point[:2], radius)
        if len(indices) < min_neighbors:
            continue
        neighborhood = ground[indices]
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / len(neighborhood)
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        if normal[2] < 0.0:
            normal = -normal
        if abs(normal[2]) < minimum_abs_nz:
            continue
        accepted_points.append(point)
        accepted_normals.append(normal)
    if not accepted_points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(accepted_points), np.asarray(accepted_normals)


def build_xy_correspondences(
    current: np.ndarray,
    reference: np.ndarray,
    reference_normals: np.ndarray,
    maximum_xy_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(current) == 0 or len(reference) == 0:
        empty = np.empty((0, 3))
        return empty, empty, empty, np.empty(0)
    distances, indices = cKDTree(reference[:, :2]).query(
        current[:, :2], k=1, distance_upper_bound=maximum_xy_distance
    )


def cell_representatives(points: np.ndarray, cell_size: float) -> dict[tuple[int, int], np.ndarray]:
    """Return one robust terrain sample per global XY cell."""
    if len(points) == 0:
        return {}
    keys = np.floor(points[:, :2] / cell_size).astype(np.int64)
    result: dict[tuple[int, int], np.ndarray] = {}
    for key in np.unique(keys, axis=0):
        mask = np.all(keys == key, axis=1)
        result[(int(key[0]), int(key[1]))] = np.median(points[mask], axis=0)
    return result


def build_cell_correspondences(
    current_ground: np.ndarray,
    reference_ground: np.ndarray,
    cell_size: float,
    normal_radius: float,
    min_normal_neighbors: int,
    minimum_abs_nz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match at most one current/reference terrain sample per global XY cell."""
    current_cells = cell_representatives(current_ground, cell_size)
    reference_cells = cell_representatives(reference_ground, cell_size)
    if not current_cells or not reference_cells:
        empty = np.empty((0, 3))
        return empty, empty, empty
    reference_samples = np.asarray(list(reference_cells.values()))
    normal_points, normals = estimate_ground_normals(
        reference_samples, normal_radius, min_normal_neighbors, minimum_abs_nz
    )
    normal_by_cell = {
        tuple(np.floor(point[:2] / cell_size).astype(np.int64)): (point, normal)
        for point, normal in zip(normal_points, normals)
    }
    shared = sorted(set(current_cells) & set(normal_by_cell))
    if not shared:
        empty = np.empty((0, 3))
        return empty, empty, empty
    return (
        np.asarray([current_cells[key] for key in shared]),
        np.asarray([normal_by_cell[key][0] for key in shared]),
        np.asarray([normal_by_cell[key][1] for key in shared]),
    )
    mask = np.isfinite(distances) & (indices < len(reference))
    return (
        current[mask],
        reference[indices[mask]],
        reference_normals[indices[mask]],
        distances[mask],
    )


def estimate_z_only(
    current: np.ndarray,
    reference: np.ndarray,
    normals: np.ndarray,
    huber_delta: float = 0.15,
    trim_fraction: float = 0.80,
    iterations: int = 10,
) -> dict[str, float | int | bool | str]:
    """Robust scalar point-to-plane least squares with fixed correspondences."""
    if not (len(current) == len(reference) == len(normals)) or len(current) == 0:
        return {"ok": False, "reason": "no_correspondences", "dz": math.nan}
    dz = 0.0
    active = np.ones(len(current), dtype=bool)
    for _ in range(iterations):
        residual = np.einsum(
            "ij,ij->i", normals, current + np.array([0.0, 0.0, dz]) - reference
        )
        count = max(3, int(math.ceil(trim_fraction * len(residual))))
        threshold = np.partition(np.abs(residual), count - 1)[count - 1]
        active = np.abs(residual) <= threshold
        r = residual[active]
        j = normals[active, 2]
        weights = np.ones(len(r))
        large = np.abs(r) > huber_delta
        weights[large] = huber_delta / np.abs(r[large])
        denominator = float(np.sum(weights * j * j))
        if denominator < 1e-9:
            return {"ok": False, "reason": "degenerate_normals", "dz": math.nan}
        step = -float(np.sum(weights * j * r)) / denominator
        dz += step
        if abs(step) < 1e-5:
            break
    before = np.einsum("ij,ij->i", normals, current - reference)
    after = np.einsum(
        "ij,ij->i", normals, current + np.array([0.0, 0.0, dz]) - reference
    )
    count = max(3, int(math.ceil(trim_fraction * len(after))))
    threshold = np.partition(np.abs(after), count - 1)[count - 1]
    active = np.abs(after) <= threshold
    return {
        "ok": True,
        "reason": "estimated",
        "dz": dz,
        "active": int(active.sum()),
        "before_rmse": float(np.sqrt(np.mean(before * before))),
        "after_rmse": float(np.sqrt(np.mean(after * after))),
        "before_median_abs": float(np.median(np.abs(before))),
        "after_median_abs": float(np.median(np.abs(after))),
        "trimmed_before_rmse": float(np.sqrt(np.mean(before[active] ** 2))),
        "trimmed_after_rmse": float(np.sqrt(np.mean(after[active] ** 2))),
        "trimmed_before_median_abs": float(np.median(np.abs(before[active]))),
        "trimmed_after_median_abs": float(np.median(np.abs(after[active]))),
        "trim_threshold": float(threshold),
    }


def read_cloud(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required to read PCD payloads") from error
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise RuntimeError(f"Empty or unreadable point cloud: {path}")
    return points


def payload_path(root: Path, index: int) -> Path:
    return root / "payloads" / f"payload_{index:06d}.pcd"


def analyze_row(row: pd.Series, args: argparse.Namespace) -> dict:
    query = int(row["query"])
    map_index = int(row["map_index"])
    state = planar_transform(
        float(row["state_x"]), float(row["state_y"]),
        float(row["state_z"]), float(row["state_yaw"]),
    )
    current = transform_points(read_cloud(payload_path(args.current_window_root, query)), state)
    reference = read_cloud(payload_path(args.reference_window_root, map_index))
    # The accepted LiDAR position is available directly in replay map coordinates.
    center_xy = row[["after_x", "after_y"]].to_numpy(dtype=float)
    current = crop_xy(current, center_xy, args.crop_radius)
    reference = crop_xy(reference, center_xy, args.crop_radius)
    current = voxel_downsample(current, args.voxel)
    reference = voxel_downsample(reference, args.voxel)
    current_ground = low_envelope_ground(
        current, args.ground_cell, args.ground_quantile,
        args.ground_below, args.ground_above, args.min_cell_points,
    )
    reference_ground = low_envelope_ground(
        reference, args.ground_cell, args.ground_quantile,
        args.ground_below, args.ground_above, args.min_cell_points,
    )
    cur, ref, normal = build_cell_correspondences(
        current_ground, reference_ground, args.match_cell,
        args.normal_radius, args.min_normal_neighbors, args.min_abs_nz,
    )
    result = estimate_z_only(
        cur, ref, normal, args.huber_delta, args.trim_fraction, args.iterations
    )
    output = {
        "query": query,
        "map_index": map_index,
        "state_z_before": float(row["state_z"]),
        "current_cropped_points": len(current),
        "reference_cropped_points": len(reference),
        "current_ground_points": len(current_ground),
        "reference_ground_points": len(reference_ground),
        "correspondences": len(cur),
        "median_xy_correspondence": (
            float(np.median(np.linalg.norm(cur[:, :2] - ref[:, :2], axis=1)))
            if len(cur) else math.nan
        ),
        **result,
    }
    dz = float(result.get("dz", math.nan))
    output["state_z_candidate"] = float(row["state_z"]) + dz
    valid, reason = True, "accepted"
    if not bool(result.get("ok", False)):
        valid, reason = False, str(result.get("reason", "optimizer_failed"))
    elif len(cur) < args.min_correspondences:
        valid, reason = False, "too_few_correspondences"
    elif abs(dz) > args.max_abs_dz:
        valid, reason = False, "dz_too_large"
    elif float(result["trimmed_after_median_abs"]) > args.max_after_median_abs:
        valid, reason = False, "residual_too_large"
    elif float(result["trimmed_after_rmse"]) > args.max_trimmed_rmse:
        valid, reason = False, "trimmed_rmse_too_large"
    elif (
        float(result["trimmed_after_rmse"])
        >= args.max_rmse_ratio * float(result["trimmed_before_rmse"])
        and abs(dz) >= args.minimum_dz_for_improvement_gate
    ):
        valid, reason = False, "insufficient_improvement"
    output["valid"] = valid
    output["gate_reason"] = reason
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-csv", type=Path, required=True)
    parser.add_argument("--reference-window-root", type=Path, required=True)
    parser.add_argument("--current-window-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-radius", type=float, default=15.0)
    parser.add_argument("--voxel", type=float, default=0.20)
    parser.add_argument("--ground-cell", type=float, default=0.75)
    parser.add_argument("--ground-quantile", type=float, default=0.10)
    parser.add_argument("--ground-below", type=float, default=0.10)
    parser.add_argument("--ground-above", type=float, default=0.35)
    parser.add_argument("--min-cell-points", type=int, default=3)
    parser.add_argument("--match-cell", type=float, default=0.75)
    parser.add_argument("--normal-radius", type=float, default=2.0)
    parser.add_argument("--min-normal-neighbors", type=int, default=5)
    parser.add_argument("--min-abs-nz", type=float, default=0.80)
    parser.add_argument("--max-xy-correspondence", type=float, default=0.45)
    parser.add_argument("--huber-delta", type=float, default=0.15)
    parser.add_argument("--trim-fraction", type=float, default=0.80)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-correspondences", type=int, default=30)
    parser.add_argument("--max-abs-dz", type=float, default=4.0)
    parser.add_argument("--max-after-median-abs", type=float, default=0.20)
    parser.add_argument("--max-trimmed-rmse", type=float, default=0.30)
    parser.add_argument("--max-rmse-ratio", type=float, default=0.80)
    parser.add_argument("--minimum-dz-for-improvement-gate", type=float, default=0.10)
    parser.add_argument("--query", type=int, action="append", default=[])
    return parser


def main() -> None:
    args = make_parser().parse_args()
    replay = pd.read_csv(args.replay_csv)
    required = {"query", "accepted", "map_index", "state_x", "state_y", "state_z",
                "state_yaw", "after_x", "after_y"}
    missing = sorted(required - set(replay.columns))
    if missing:
        raise RuntimeError(f"Replay CSV is missing columns: {missing}")
    selected = replay[replay["accepted"].astype(bool) & replay["map_index"].notna()].copy()
    if args.query:
        selected = selected[selected["query"].isin(args.query)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in selected.iterrows():
        try:
            result = analyze_row(row, args)
        except Exception as error:  # Preserve the remaining batch diagnostics.
            result = {
                "query": int(row["query"]), "map_index": int(row["map_index"]),
                "state_z_before": float(row["state_z"]), "ok": False, "valid": False,
                "reason": "exception", "gate_reason": f"exception: {error}",
            }
        rows.append(result)
        print(
            f"query={result['query']} map={result['map_index']} "
            f"dz={result.get('dz', math.nan):.3f} valid={int(result['valid'])} "
            f"reason={result['gate_reason']}"
        )
    output = pd.DataFrame(rows).sort_values("query")
    output_path = args.output_dir / "z_only_ground_alignment.csv"
    output.to_csv(output_path, index=False)
    valid = output[output.get("valid", False).astype(bool)] if len(output) else output
    print("\n=== summary ===")
    print(f"accepted replay rows: {len(selected)}")
    print(f"valid Z estimates: {len(valid)}")
    if len(valid):
        print(f"candidate Z median: {valid['state_z_candidate'].median():.6f}")
        print(f"candidate Z robust spread (MAD): {(valid['state_z_candidate'] - valid['state_z_candidate'].median()).abs().median():.6f}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
