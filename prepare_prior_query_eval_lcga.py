#!/usr/bin/env python3

from pathlib import Path
import math
import os
import shutil
import sys

import numpy as np
import yaml


if len(sys.argv) != 2:
    raise SystemExit(
        f"usage: {sys.argv[0]} QUERY_INDEX"
    )

query_index = int(sys.argv[1])

search_radius_m = 50.0
minimum_map_trees = 3

map_dfi_root = Path(
    "/home/wataru/TreeLoc-plusplus/"
    "data/furo_cpp_relaxed_lcga"
)

query_dfi_root = Path(
    "/home/wataru/TreeLoc-plusplus/"
    "data/asaka_query_windowed_18_50_lcga"
)

query_trajectory_path = Path(
    "/home/wataru/sruppto_ws/src/sruppTo_for_user/"
    "control_set/draft/"
    "asaka_training_facility_forest_localization/"
    "maps/floor_0_0/"
    "treelocpp_dfi_input_windowed/"
    "trajectory.txt"
)

source_config = Path(
    "config/asaka_query48_prior50_inter_relaxed.yaml"
)

output_root = Path(
    f"data/asaka_query{query_index}_prior50_lcga_merged"
)

output_config = Path(
    f"config/asaka_query{query_index}_prior50_lcga_inter_relaxed.yaml"
)

state_path = Path(
    "results/sequential_replay/current_map_odom.yaml"
)


def normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)

    if norm <= 0.0:
        raise RuntimeError("invalid zero quaternion")

    return q / norm


def quat_to_rot(q):
    x, y, z, w = normalize(q)

    return np.array([
        [
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y - z*w),
            2.0 * (x*z + y*w),
        ],
        [
            2.0 * (x*y + z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z - x*w),
        ],
        [
            2.0 * (x*z - y*w),
            2.0 * (y*z + x*w),
            1.0 - 2.0 * (x*x + y*y),
        ],
    ], dtype=float)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = normalize(q1)
    x2, y2, z2, w2 = normalize(q2)

    return normalize(np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=float))


def read_trajectory(path):
    rows = []

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            values = [float(v) for v in line.split()]

            if len(values) != 8:
                raise RuntimeError(
                    f"invalid trajectory row in {path}: {line}"
                )

            rows.append(values)

    return rows


def count_trees(path):
    with path.open() as f:
        line_count = sum(
            1 for line in f
            if line.strip()
        )

    return max(0, line_count - 1)


for required_path in [
    map_dfi_root,
    query_dfi_root,
    query_trajectory_path,
    source_config,
    state_path,
]:
    if not required_path.exists():
        raise RuntimeError(
            f"required path not found: {required_path}"
        )


state = yaml.safe_load(
    state_path.read_text()
)

t_map_odom = np.array([
    float(state["position"]["x"]),
    float(state["position"]["y"]),
    float(state["position"]["z"]),
], dtype=float)

q_map_odom = np.array([
    float(state["orientation_xyzw"]["x"]),
    float(state["orientation_xyzw"]["y"]),
    float(state["orientation_xyzw"]["z"]),
    float(state["orientation_xyzw"]["w"]),
], dtype=float)


query_rows = read_trajectory(
    query_trajectory_path
)

if query_index < 0 or query_index >= len(query_rows):
    raise RuntimeError(
        f"query index {query_index} is outside trajectory "
        f"with {len(query_rows)} rows"
    )

query_row = query_rows[query_index]

stamp = query_row[0]
t_odom_lidar = np.asarray(
    query_row[1:4],
    dtype=float,
)
q_odom_lidar = np.asarray(
    query_row[4:8],
    dtype=float,
)

t_map_lidar_prior = (
    t_map_odom
    + quat_to_rot(q_map_odom) @ t_odom_lidar
)

q_map_lidar_prior = quat_multiply(
    q_map_odom,
    q_odom_lidar,
)


map_trajectory_path = (
    map_dfi_root / "trajectory.txt"
)

if not map_trajectory_path.exists():
    raise RuntimeError(
        f"Map trajectory not found: {map_trajectory_path}"
    )

map_rows = read_trajectory(
    map_trajectory_path
)

merged_query_index = len(map_rows)

source_query_csv = (
    query_dfi_root
    / f"TreeManagerState_{query_index}.csv"
)

if not source_query_csv.exists():
    raise RuntimeError(
        f"Query DFI not found: {source_query_csv}"
    )


selected_map_frames = []
skipped_missing_csv = 0
skipped_few_trees = 0

prior_x = float(t_map_lidar_prior[0])
prior_y = float(t_map_lidar_prior[1])

for map_index, map_row in enumerate(map_rows):
    map_x = float(map_row[1])
    map_y = float(map_row[2])

    distance = math.hypot(
        map_x - prior_x,
        map_y - prior_y,
    )

    if distance > search_radius_m:
        continue

    source_map_csv = (
        map_dfi_root
        / f"TreeManagerState_{map_index}.csv"
    )

    if not source_map_csv.exists():
        skipped_missing_csv += 1
        continue

    tree_count = count_trees(
        source_map_csv
    )

    if tree_count < minimum_map_trees:
        skipped_few_trees += 1
        continue

    selected_map_frames.append({
        "index": map_index,
        "distance": distance,
        "trees": tree_count,
        "source": source_map_csv,
    })


if output_root.exists():
    shutil.rmtree(output_root)

output_root.mkdir(
    parents=True,
    exist_ok=True,
)


for frame in selected_map_frames:
    destination = (
        output_root
        / f"TreeManagerState_{frame['index']}.csv"
    )

    os.symlink(
        frame["source"].resolve(),
        destination,
    )


destination_query_csv = (
    output_root
    / f"TreeManagerState_{merged_query_index}.csv"
)

shutil.copy2(
    source_query_csv,
    destination_query_csv,
)


trajectory_lines = [
    " ".join(
        f"{value:.9f}"
        for value in row
    )
    for row in map_rows
]

trajectory_lines.append(
    f"{stamp:.9f} "
    f"{t_map_lidar_prior[0]:.9f} "
    f"{t_map_lidar_prior[1]:.9f} "
    f"{t_map_lidar_prior[2]:.9f} "
    f"{q_map_lidar_prior[0]:.9f} "
    f"{q_map_lidar_prior[1]:.9f} "
    f"{q_map_lidar_prior[2]:.9f} "
    f"{q_map_lidar_prior[3]:.9f}"
)

(output_root / "trajectory.txt").write_text(
    "\n".join(trajectory_lines) + "\n"
)


cfg = yaml.safe_load(
    source_config.read_text()
)

cfg["dataset"]["root"] = str(
    output_root
)

output_config.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
    )
)


selected_map_frames.sort(
    key=lambda frame: frame["distance"]
)

print("query source index:", query_index)
print("merged query index:", merged_query_index)
print(
    "prior T_map_lidar position:",
    t_map_lidar_prior,
)
print(
    "prior quaternion xyzw:",
    q_map_lidar_prior,
)
print(
    "Map anchors inside radius:",
    sum(
        math.hypot(
            float(row[1]) - prior_x,
            float(row[2]) - prior_y,
        ) <= search_radius_m
        for row in map_rows
    ),
)
print(
    "selected valid Map frames:",
    len(selected_map_frames),
)
print(
    "skipped missing CSV:",
    skipped_missing_csv,
)
print(
    "skipped fewer than "
    f"{minimum_map_trees} trees:",
    skipped_few_trees,
)

print()
print("nearest selected Map frames:")
print("index distance_m trees")

for frame in selected_map_frames[:10]:
    print(
        f"{frame['index']:5d} "
        f"{frame['distance']:10.3f} "
        f"{frame['trees']:5d}"
    )

print()
print("Query trees:", count_trees(source_query_csv))
print("dataset:", output_root)
print("config:", output_config)
