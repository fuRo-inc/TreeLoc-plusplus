#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIAL_TRANSLATION = np.array([
    -429.06451416015625,
    822.0798950195312,
    -7.800044059753418,
])

INITIAL_QUATERNION_XYZW = np.array([
    -0.02772272742629447,
    0.04056485309201003,
    0.016247853804699283,
    0.9986600774660076,
])


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
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
    ])


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def planar_transform(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform[:3, 3] = [x, y, z]
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3))
    return (
        transform[:3, :3] @ points.T
    ).T + transform[:3, 3]


def read_trajectory(path: Path) -> list[np.ndarray]:
    poses = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise RuntimeError(f"invalid trajectory row: {raw_line}")
        x, y, z, qx, qy, qz, qw = map(float, fields[1:])
        transform = np.eye(4)
        transform[:3, :3] = quaternion_to_rotation(
            np.array([qx, qy, qz, qw])
        )
        transform[:3, 3] = [x, y, z]
        poses.append(transform)
    return poses


def read_tree_points(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return frame[[
        "location_x",
        "location_y",
        "location_z",
    ]].to_numpy(dtype=float)


def load_global_map_tree_points(
    map_root: Path,
    map_poses: list[np.ndarray],
    stride: int,
) -> np.ndarray:
    transformed_sets = []
    csv_paths = sorted(
        map_root.glob("TreeManagerState_*.csv"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for position, path in enumerate(csv_paths):
        if position % max(1, stride) != 0:
            continue
        index = int(path.stem.split("_")[-1])
        if index >= len(map_poses):
            continue
        points = read_tree_points(path)
        if len(points):
            transformed_sets.append(transform_points(map_poses[index], points))
    if not transformed_sets:
        return np.empty((0, 3))
    points = np.vstack(transformed_sets)
    # DFI windows contain repeated trees. Quantize only for overview rendering.
    keys = np.round(points[:, :2] / 0.25).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def parse_result(output: str) -> dict[str, str] | None:
    line = next(
        (
            line for line in output.splitlines()
            if line.startswith("localization_result ")
        ),
        None,
    )
    if line is None:
        return None
    return {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in line.split()[1:]
    }


def initial_planar_map_odom() -> np.ndarray:
    yaw = yaw_from_rotation(
        quaternion_to_rotation(INITIAL_QUATERNION_XYZW)
    )
    return planar_transform(
        INITIAL_TRANSLATION[0],
        INITIAL_TRANSLATION[1],
        INITIAL_TRANSLATION[2],
        yaw,
    )


def run_localizer(
    executable: Path,
    config: Path,
    query_root: Path,
    database_root: Path,
    prior_map_c: np.ndarray,
) -> tuple[dict[str, str] | None, str, int]:
    prior_yaw = yaw_from_rotation(prior_map_c[:3, :3])
    command = [
        str(executable),
        str(config),
        "--query_root", str(query_root),
        "--database_root", str(database_root),
        "--prior_x", str(prior_map_c[0, 3]),
        "--prior_y", str(prior_map_c[1, 3]),
        "--prior_z", str(prior_map_c[2, 3]),
        "--prior_yaw", str(prior_yaw),
        "--search_radius", "50",
        "--top_k", "1",
        "--match_distance", "1.5",
        "--refine_iterations", "5",
        "--dbh_soft_weight", "0.25",
        "--triangle_edge_tolerance", "0.5",
        "--triangle_max_hypotheses", "500",
        "--consensus_xy", "2.0",
        "--consensus_yaw", "0.15",
        "--min_consensus_support", "3",
        "--min_consensus_margin", "1",
        "--min_pairs", "6",
        "--min_query_coverage", "0.40",
        "--min_overlap", "0.0",
        "--max_mean_residual", "0.5",
        "--max_max_residual", "1.0",
        "--prior_gate_xy", "100",
        "--prior_gate_yaw", "3.2",
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return parse_result(completed.stdout), completed.stdout, completed.returncode


def points_from_rows(rows: list[dict], prefix: str) -> np.ndarray:
    return np.array([
        [row[f"{prefix}_x"], row[f"{prefix}_y"]]
        for row in rows
    ])


def plot_replay(
    rows: list[dict],
    map_poses: list[np.ndarray],
    global_map_trees: np.ndarray,
    map_tree_sets: list[np.ndarray],
    current_tree_sets: list[tuple[int, int, bool, np.ndarray]],
    initial_projection: np.ndarray,
    output: Path,
) -> None:
    queries = np.array([row["query"] for row in rows])
    accepted = np.array([row["accepted"] for row in rows], dtype=bool)
    before = points_from_rows(rows, "before")
    after = points_from_rows(rows, "after")

    all_path_points = np.vstack([before, after, initial_projection])
    center = np.median(all_path_points, axis=0)
    margin = 20.0
    map_xy = np.array([pose[:2, 3] for pose in map_poses])
    inside = (
        (np.abs(map_xy[:, 0] - center[0]) <= margin)
        & (np.abs(map_xy[:, 1] - center[1]) <= margin)
    )

    colors = plt.cm.viridis(
        np.linspace(0.05, 0.95, len(rows))
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(24, 14),
        constrained_layout=True,
    )

    global_axis = axes[0, 0]
    if len(global_map_trees):
        global_axis.scatter(
            global_map_trees[:, 0],
            global_map_trees[:, 1],
            s=5,
            color="0.45",
            alpha=0.22,
            label="Global Map trees (sampled)",
        )
    global_axis.plot(
        map_xy[:, 0],
        map_xy[:, 1],
        ".-",
        color="0.72",
        linewidth=0.7,
        markersize=2,
        label="Full Map LiDAR trajectory",
    )
    global_axis.plot(
        after[:, 0],
        after[:, 1],
        "-",
        color="tab:green",
        linewidth=2.2,
        label="Current replay interval",
    )
    global_axis.scatter(
        after[0, 0], after[0, 1],
        marker="o", color="tab:blue", s=90, label="Current start",
    )
    global_axis.scatter(
        after[-1, 0], after[-1, 1],
        marker="*", color="tab:red", s=150, label="Current end",
    )
    accepted_anchor_xy = np.array([
        [row["map_anchor_x"], row["map_anchor_y"]]
        for row in rows if row["accepted"]
    ])
    if len(accepted_anchor_xy):
        global_axis.scatter(
            accepted_anchor_xy[:, 0],
            accepted_anchor_xy[:, 1],
            marker="s",
            facecolors="none",
            edgecolors="tab:purple",
            s=55,
            label="Selected Map anchors",
        )
    route_min = np.min(all_path_points, axis=0) - 5.0
    route_max = np.max(all_path_points, axis=0) + 5.0
    global_axis.add_patch(plt.Rectangle(
        route_min,
        route_max[0] - route_min[0],
        route_max[1] - route_min[1],
        fill=False,
        edgecolor="tab:red",
        linewidth=1.5,
        linestyle="--",
        label="Local-detail extent",
    ))
    global_axis.set_title("Global Map overview")
    global_axis.set_xlabel("map X [m]")
    global_axis.set_ylabel("map Y [m]")
    global_axis.axis("equal")
    global_axis.grid(True)
    global_axis.legend(fontsize=8)

    overview = axes[0, 1]
    overview.plot(
        map_xy[inside, 0],
        map_xy[inside, 1],
        ".-",
        color="0.70",
        linewidth=1.0,
        markersize=3,
        label="Map LiDAR trajectory",
    )
    overview.plot(
        initial_projection[:, 0],
        initial_projection[:, 1],
        "o--",
        color="tab:blue",
        linewidth=1.5,
        label="Initial map<-odom projection",
    )
    overview.plot(
        after[:, 0],
        after[:, 1],
        "k-",
        linewidth=1.2,
        label="Sequential projected path",
    )
    for index, row in enumerate(rows):
        color = "tab:green" if row["accepted"] else "tab:red"
        overview.scatter(
            before[index, 0],
            before[index, 1],
            marker="o",
            facecolors="none",
            edgecolors="tab:orange",
            s=70,
            zorder=5,
        )
        overview.scatter(
            after[index, 0],
            after[index, 1],
            marker="*" if row["accepted"] else "x",
            color=color,
            s=120,
            zorder=6,
        )
        if index % 2 == 0 or not row["accepted"]:
            overview.annotate(
                f'Q{row["query"]}\n'
                f'{"A" if row["accepted"] else "R"}',
                after[index],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
        if row["accepted"]:
            overview.annotate(
                "",
                xy=after[index],
                xytext=before[index],
                arrowprops={
                    "arrowstyle": "->",
                    "color": "tab:orange",
                    "linewidth": 1.2,
                },
            )
            overview.scatter(
                row["map_anchor_x"],
                row["map_anchor_y"],
                marker="s",
                facecolors="none",
                edgecolors=colors[index],
                s=80,
            )
    overview.set_title(
        "Sequential map<-odom replay: LiDAR positions on map"
    )
    overview.set_xlabel("map X [m]")
    overview.set_ylabel("map Y [m]")
    overview.axis("equal")
    overview.grid(True)
    overview.legend(fontsize=8)

    trees_axis = axes[0, 2]
    for points in map_tree_sets:
        trees_axis.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            color="0.25",
            alpha=0.35,
        )
    for row_index, query, is_accepted, points in current_tree_sets:
        color = colors[row_index]
        trees_axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="x",
            s=28,
            color=color if is_accepted else "tab:red",
            alpha=0.65,
            label=f"Q{query}" if is_accepted else f"Q{query} rejected",
        )
    trees_axis.plot(
        after[:, 0],
        after[:, 1],
        "k.-",
        linewidth=1.0,
        label="Projected LiDAR",
    )
    for index, row in enumerate(rows):
        yaw = row["after_yaw"]
        trees_axis.arrow(
            row["after_x"],
            row["after_y"],
            1.2 * math.cos(yaw),
            1.2 * math.sin(yaw),
            color=colors[index],
            width=0.025,
            head_width=0.25,
            length_includes_head=True,
            alpha=0.8,
        )
    trees_axis.set_title(
        "Map trees (gray) and Current trees after each decision"
    )
    trees_axis.set_xlabel("map X [m]")
    trees_axis.set_ylabel("map Y [m]")
    trees_axis.axis("equal")
    trees_axis.grid(True)

    innovation_axis = axes[1, 0]
    parameter_planar = np.array([
        row["parameter_planar"]
        if row["accepted"] else np.nan
        for row in rows
    ])
    parameter_yaw_deg = np.array([
        math.degrees(row["parameter_yaw"])
        if row["accepted"] else np.nan
        for row in rows
    ])
    innovation_axis.bar(
        queries - 0.16,
        np.nan_to_num(parameter_planar),
        width=0.32,
        color=np.where(accepted, "tab:green", "tab:red"),
        alpha=0.75,
        label="parameter XY update [m]",
    )
    innovation_axis.set_xlabel("Query index")
    innovation_axis.set_ylabel("XY update [m]")
    innovation_axis.grid(True, axis="y")
    yaw_axis = innovation_axis.twinx()
    yaw_axis.plot(
        queries,
        parameter_yaw_deg,
        "o-",
        color="tab:purple",
        label="yaw update [deg]",
    )
    yaw_axis.set_ylabel("yaw update [deg]")
    innovation_axis.set_title(
        "Update applied at each payload (red = rejected/no update)"
    )
    lines1, labels1 = innovation_axis.get_legend_handles_labels()
    lines2, labels2 = yaw_axis.get_legend_handles_labels()
    innovation_axis.legend(lines1 + lines2, labels1 + labels2)

    state_axis = axes[1, 1]
    state_x = np.array([row["state_x"] for row in rows])
    state_y = np.array([row["state_y"] for row in rows])
    state_yaw = np.degrees(
        np.unwrap(np.array([row["state_yaw"] for row in rows]))
    )
    initial = initial_planar_map_odom()
    initial_yaw = math.degrees(yaw_from_rotation(initial[:3, :3]))
    state_axis.step(
        queries,
        state_x - initial[0, 3],
        where="post",
        label="map_odom x - initial [m]",
    )
    state_axis.step(
        queries,
        state_y - initial[1, 3],
        where="post",
        label="map_odom y - initial [m]",
    )
    state_axis.step(
        queries,
        state_yaw - initial_yaw,
        where="post",
        label="map_odom yaw - initial [deg]",
    )
    for query, is_accepted in zip(queries, accepted):
        if not is_accepted:
            state_axis.axvline(query, color="tab:red", alpha=0.18)
    state_axis.set_title("Held map<-odom state after each decision")
    state_axis.set_xlabel("Query index")
    state_axis.set_ylabel("Change from initial")
    state_axis.grid(True)
    state_axis.legend()

    quality_axis = axes[1, 2]
    supports = np.array([row["support"] for row in rows])
    pairs = np.array([row["pairs"] for row in rows])
    quality_axis.plot(
        queries,
        pairs,
        "o-",
        color="tab:blue",
        label="matched pairs",
    )
    quality_axis.plot(
        queries,
        supports,
        "s-",
        color="tab:purple",
        label="consensus support",
    )
    quality_axis.axhline(
        6, color="tab:blue", linestyle="--", alpha=0.45,
        label="pairs gate",
    )
    quality_axis.axhline(
        3, color="tab:purple", linestyle=":", alpha=0.55,
        label="support gate",
    )
    for query, is_accepted in zip(queries, accepted):
        if not is_accepted:
            quality_axis.axvline(query, color="tab:red", alpha=0.12)
    quality_axis.set_title("Localization evidence and rejected payloads")
    quality_axis.set_xlabel("Query index")
    quality_axis.set_ylabel("Count")
    quality_axis.grid(True)
    quality_axis.legend()

    figure.suptitle(
        f"TreeLoc++ sequential diagnostic replay, Query {queries[0]}-{queries[-1]}\n"
        "Open circle: before update, star: accepted after update, x: rejected",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=41)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument(
        "--tree-stride",
        type=int,
        default=3,
        help="Plot Current trees every N payloads to avoid overplotting.",
    )
    parser.add_argument(
        "--global-map-tree-stride",
        type=int,
        default=10,
        help="Use every Nth Map DFI when rendering the global overview.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    executable = args.repo / "build/treelocpp_localize_hypotheses"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_root = args.output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    map_poses = read_trajectory(args.map_root / "trajectory.txt")
    global_map_trees = load_global_map_tree_points(
        args.map_root,
        map_poses,
        args.global_map_tree_stride,
    )
    first_query_root = args.query_root / f"query_{args.start}"
    odom_poses = read_trajectory(first_query_root / "trajectory.txt")

    state = initial_planar_map_odom()
    initial_state = state.copy()
    rows = []
    map_tree_sets = []
    current_tree_sets = []
    used_map_indices = set()

    for query_index in range(args.start, args.end + 1):
        query_directory = args.query_root / f"query_{query_index}"
        odom_c = odom_poses[query_index]
        before_map_c = state @ odom_c

        result, raw_output, return_code = run_localizer(
            executable,
            args.config,
            query_directory,
            args.map_root,
            before_map_c,
        )
        (log_root / f"query_{query_index}.txt").write_text(
            raw_output,
            encoding="utf-8",
        )

        process_ok = return_code == 0 and result is not None
        accepted = process_ok and result["ok"] == "1"
        parameter_planar = math.nan
        parameter_yaw = math.nan
        map_index = -1

        if result is None:
            result = {
                "support": "0",
                "pairs": "0",
                "overlap": "nan",
                "mean_residual": "nan",
                "max_residual": "nan",
            }

        if accepted:
            candidate_x = float(result["map_odom_x"])
            candidate_y = float(result["map_odom_y"])
            candidate_yaw = float(result["map_odom_yaw"])
            parameter_planar = float(result["parameter_planar"])
            parameter_yaw = float(result["parameter_yaw"])
            map_index = int(result["map_idx"])

            # x, y, yawのみ更新。zは固定し、roll/pitchは0のまま。
            state = planar_transform(
                candidate_x,
                candidate_y,
                initial_state[2, 3],
                candidate_yaw,
            )

        after_map_c = state @ odom_c
        state_yaw = yaw_from_rotation(state[:3, :3])
        after_yaw = yaw_from_rotation(after_map_c[:3, :3])

        row = {
            "query": query_index,
            "accepted": accepted,
            "status": (
                "accepted" if accepted
                else "rejected" if process_ok
                else "unusable_query"
            ),
            "map_index": map_index,
            "support": int(result["support"]),
            "pairs": int(result["pairs"]),
            "overlap": float(result["overlap"]),
            "mean_residual": float(result["mean_residual"]),
            "max_residual": float(result["max_residual"]),
            "before_x": before_map_c[0, 3],
            "before_y": before_map_c[1, 3],
            "before_z": before_map_c[2, 3],
            "before_yaw": yaw_from_rotation(before_map_c[:3, :3]),
            "after_x": after_map_c[0, 3],
            "after_y": after_map_c[1, 3],
            "after_z": after_map_c[2, 3],
            "after_yaw": after_yaw,
            "state_x": state[0, 3],
            "state_y": state[1, 3],
            "state_z": state[2, 3],
            "state_yaw": state_yaw,
            "parameter_planar": parameter_planar,
            "parameter_yaw": parameter_yaw,
            "map_anchor_x": (
                map_poses[map_index][0, 3]
                if map_index >= 0 else math.nan
            ),
            "map_anchor_y": (
                map_poses[map_index][1, 3]
                if map_index >= 0 else math.nan
            ),
        }
        rows.append(row)

        current_csv = (
            query_directory / f"TreeManagerState_{query_index}.csv"
        )
        if current_csv.is_file():
            current_points = read_tree_points(current_csv)
            row_index = query_index - args.start
            plot_current_trees = (
                row_index % max(1, args.tree_stride) == 0
                or query_index == args.end
            )
            if plot_current_trees and len(current_points):
                current_tree_sets.append((
                    row_index,
                    query_index,
                    accepted,
                    transform_points(after_map_c, current_points),
                ))

        if map_index >= 0 and map_index not in used_map_indices:
            map_csv = args.map_root / f"TreeManagerState_{map_index}.csv"
            if map_csv.is_file():
                map_tree_sets.append(transform_points(
                    map_poses[map_index],
                    read_tree_points(map_csv),
                ))
                used_map_indices.add(map_index)

        print(
            f"Q{query_index}: "
            f"{row['status'].upper()} "
            f"map={map_index} support={row['support']} "
            f"pairs={row['pairs']} overlap={row['overlap']:.3f} "
            f"state=({row['state_x']:.6f}, {row['state_y']:.6f}, "
            f"{math.degrees(row['state_yaw']):.4f} deg)"
        )

    initial_projection = np.array([
        (initial_state @ odom_poses[index])[:2, 3]
        for index in range(args.start, args.end + 1)
    ])

    csv_path = args.output_dir / "sequential_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = args.output_dir / "sequential_replay.png"
    plot_replay(
        rows,
        map_poses,
        global_map_trees,
        map_tree_sets,
        current_tree_sets,
        initial_projection,
        png_path,
    )

    print()
    print("CSV:", csv_path)
    print("PNG:", png_path)


if __name__ == "__main__":
    main()
