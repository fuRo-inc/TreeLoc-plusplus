#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def read_trajectory(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def load_vertices(path):
    verts = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                verts.append((float(x), float(y), float(z)))
    return verts


def build_grid(points, cell=2.0):
    grid = {}
    for px, py, pz in points:
        key = (math.floor(px / cell), math.floor(py / cell))
        grid.setdefault(key, []).append((px, py, pz))
    return grid, cell


def build_terrain_index(points):
    if cKDTree is not None and points:
        xy = [(x, y) for x, y, _ in points]
        z = [pz for _, _, pz in points]
        return "kdtree", cKDTree(xy), z
    return "grid", build_grid(points)


def interp_z(index, x, y, k=6):
    if index[0] == "kdtree":
        _, tree, z_values = index
        k = min(k, len(z_values))
        dists, idxs = tree.query([x, y], k=k)
        if k == 1:
            dists = [dists]
            idxs = [idxs]
        num = 0.0
        den = 0.0
        for dist, idx in zip(dists, idxs):
            w = 1e9 if dist < 1e-6 else 1.0 / dist
            den += w
            num += w * z_values[idx]
        return num / den if den > 0.0 else 0.0

    _, grid_data = index
    grid, cell = grid_data
    cx = math.floor(x / cell)
    cy = math.floor(y / cell)
    candidates = []
    for radius in range(1, 16):
        candidates.clear()
        for gx in range(cx - radius, cx + radius + 1):
            for gy in range(cy - radius, cy + radius + 1):
                for px, py, pz in grid.get((gx, gy), []):
                    candidates.append((math.hypot(px - x, py - y), pz))
        if len(candidates) >= k:
            break
    best = sorted(candidates, key=lambda v: v[0])[:k]
    if not best:
        return 0.0
    if best[0][0] < 1e-6:
        return best[0][1]
    den = 0.0
    num = 0.0
    for d, z in best:
        w = 1e9 if d < 1e-6 else 1.0 / d
        den += w
        num += w * z
    return num / den


def local_to_global_xy(pose, x, y):
    _, tx, ty, _, qx, qy, qz, qw = pose
    yaw = quat_to_yaw(qx, qy, qz, qw)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return tx + c * x - s * y, ty + s * x + c * y


def copy_frame(src_csv, dst_csv, terrain_index, pose):
    pose_z = pose[3]
    with src_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        has_location_z = "location_z" in fieldnames
        if not has_location_z:
            if terrain_index is None:
                raise RuntimeError(f"{src_csv} has no location_z and no terrain source was provided")
            insert_at = fieldnames.index("location_y") + 1 if "location_y" in fieldnames else len(fieldnames)
            fieldnames.insert(insert_at, "location_z")
        rows = []
        for row in reader:
            if not has_location_z:
                x = float(row["location_x"])
                y = float(row["location_y"])
                gx, gy = local_to_global_xy(pose, x, y)
                row["location_z"] = f"{interp_z(terrain_index, gx, gy) - pose_z:.8f}"
            rows.append(row)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy_sequence(src, dst, count, terrain_path=None):
    traj = read_trajectory(src / "trajectory.txt")
    if count is None:
        count = len(traj)

    terrain_path = terrain_path or (src / "terrain.obj")
    vertices = load_vertices(terrain_path) if terrain_path and terrain_path.exists() else []
    terrain_index = build_terrain_index(vertices) if vertices else None

    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "trajectory.txt").open("w") as f:
        for row in traj[:count]:
            f.write(" ".join(f"{v:.12g}" for v in row) + "\n")

    for i in range(min(count, len(traj))):
        src_csv = src / f"TreeManagerState_{i}.csv"
        if src_csv.exists():
            copy_frame(src_csv, dst / src_csv.name, terrain_index, traj[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--terrain", type=Path, default=None)
    args = parser.parse_args()
    copy_sequence(args.source, args.output, args.frames, args.terrain)


if __name__ == "__main__":
    main()
