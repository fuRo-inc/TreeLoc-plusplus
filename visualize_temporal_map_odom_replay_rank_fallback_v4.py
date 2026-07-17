#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import visualize_sequential_map_odom_replay_v3 as base
from z_only_ground_preprocessor import GroundZConfig, GroundZPreprocessor


def pose_xy_yaw(transform: np.ndarray) -> tuple[float, float, float]:
    return (
        float(transform[0, 3]),
        float(transform[1, 3]),
        base.yaw_from_rotation(transform[:3, :3]),
    )


def pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ax, ay, ayaw = pose_xy_yaw(a)
    bx, by, byaw = pose_xy_yaw(b)
    return math.hypot(ax - bx, ay - by), abs(base.wrap(ayaw - byaw))


def result_transform(result: dict[str, str], fixed_z: float) -> np.ndarray:
    return base.planar_transform(
        float(result["map_odom_x"]),
        float(result["map_odom_y"]),
        fixed_z,
        float(result["map_odom_yaw"]),
    )


def parse_rank_candidate(
    output: str,
    odom_c: np.ndarray,
    fixed_z: float,
) -> dict | None:
    """Return the printed rank-0 intrinsic hypothesis, even when consensus fails."""
    header = None
    for line in output.splitlines():
        if line.startswith("rank "):
            header = line.split()
            continue
        fields = line.split()
        if (
            header is None
            or len(fields) != len(header)
            or not fields
            or not fields[0].isdigit()
        ):
            continue
        row = dict(zip(header, fields))
        if row.get("intrinsic_ok") != "1":
            continue

        map_c = base.planar_transform(
            float(row["est_x"]),
            float(row["est_y"]),
            float(odom_c[2, 3] + fixed_z),
            float(row["est_yaw"]),
        )
        map_odom = map_c @ np.linalg.inv(odom_c)
        transform = base.planar_transform(
            float(map_odom[0, 3]),
            float(map_odom[1, 3]),
            fixed_z,
            base.yaw_from_rotation(map_odom[:3, :3]),
        )
        return {
            "transform": transform,
            "map_index": int(row["db_idx"]),
            "pairs": int(row["pairs"]),
            "overlap": float(row["overlap"]),
            "mean_residual": float(row["mean_residual"]),
            "max_residual": float(row["max_residual"]),
            "source": "rank_fallback",
        }
    return None


def projected_pose(candidate: dict, odom_reference: np.ndarray) -> np.ndarray:
    return candidate["transform"] @ odom_reference


def select_temporal_cluster(
    candidates: list[dict],
    odom_reference: np.ndarray,
    xy_tolerance: float,
    yaw_tolerance: float,
) -> dict:
    if not candidates:
        return {
            "members": [],
            "support": 0,
            "runner_up_support": 0,
            "ambiguous": False,
        }

    projected = [
        projected_pose(candidate, odom_reference)
        for candidate in candidates
    ]
    neighborhoods: list[list[int]] = []

    for center in projected:
        members = []
        for index, pose in enumerate(projected):
            xy, yaw = pose_distance(center, pose)
            if xy <= xy_tolerance and yaw <= yaw_tolerance:
                members.append(index)
        neighborhoods.append(members)

    def center_key(index: int) -> tuple:
        candidate = candidates[index]
        return (
            len(neighborhoods[index]),
            candidate["pairs"],
            candidate["overlap"],
            -candidate["mean_residual"],
            candidate["query"],
        )

    best_index = max(range(len(candidates)), key=center_key)
    best_members = neighborhoods[best_index]
    best_pose = projected[best_index]
    runner_up_support = 0

    for index, pose in enumerate(projected):
        xy, yaw = pose_distance(best_pose, pose)
        if xy <= xy_tolerance and yaw <= yaw_tolerance:
            continue
        runner_up_support = max(
            runner_up_support,
            len(neighborhoods[index]),
        )

    return {
        "members": best_members,
        "support": len(best_members),
        "runner_up_support": runner_up_support,
        "ambiguous": len(best_members) == runner_up_support,
    }


def aggregate_cluster(
    candidates: list[dict],
    member_indices: list[int],
    odom_reference: np.ndarray,
    fixed_z: float,
) -> np.ndarray:
    poses = [
        projected_pose(candidates[index], odom_reference)
        for index in member_indices
    ]
    xs = [pose[0, 3] for pose in poses]
    ys = [pose[1, 3] for pose in poses]
    yaws = [base.yaw_from_rotation(pose[:3, :3]) for pose in poses]
    map_c_yaw = math.atan2(
        sum(math.sin(yaw) for yaw in yaws),
        sum(math.cos(yaw) for yaw in yaws),
    )
    map_c = base.planar_transform(
        float(np.median(xs)),
        float(np.median(ys)),
        float(odom_reference[2, 3] + fixed_z),
        map_c_yaw,
    )
    map_odom = map_c @ np.linalg.inv(odom_reference)
    return base.planar_transform(
        float(map_odom[0, 3]),
        float(map_odom[1, 3]),
        fixed_z,
        base.yaw_from_rotation(map_odom[:3, :3]),
    )


def bounded_projected_update(
    previous: np.ndarray,
    target: np.ndarray,
    odom_c: np.ndarray,
    fixed_z: float,
    maximum_xy: float,
    maximum_yaw: float,
) -> np.ndarray:
    """Limit the correction at the current LiDAR pose, not at the odom origin."""
    previous_map_c = previous @ odom_c
    target_map_c = target @ odom_c

    dx = float(target_map_c[0, 3] - previous_map_c[0, 3])
    dy = float(target_map_c[1, 3] - previous_map_c[1, 3])
    distance = math.hypot(dx, dy)
    scale = min(1.0, maximum_xy / distance) if distance > 0.0 else 1.0

    previous_yaw = base.yaw_from_rotation(previous_map_c[:3, :3])
    target_yaw = base.yaw_from_rotation(target_map_c[:3, :3])
    yaw_step = max(
        -maximum_yaw,
        min(maximum_yaw, base.wrap(target_yaw - previous_yaw)),
    )

    updated_map_c = base.planar_transform(
        float(previous_map_c[0, 3] + scale * dx),
        float(previous_map_c[1, 3] + scale * dy),
        float(odom_c[2, 3] + fixed_z),
        base.wrap(previous_yaw + yaw_step),
    )
    updated = updated_map_c @ np.linalg.inv(odom_c)
    return base.planar_transform(
        float(updated[0, 3]),
        float(updated[1, 3]),
        fixed_z,
        base.yaw_from_rotation(updated[:3, :3]),
    )


def plot_temporal_gate(rows: list[dict], output: Path) -> None:
    queries = np.array([row["query"] for row in rows])
    intrinsic = np.array([row["intrinsic_ok"] for row in rows], dtype=bool)
    updated = np.array([row["accepted"] for row in rows], dtype=bool)
    temporal_support = np.array([row["temporal_support"] for row in rows])
    runner = np.array([row["temporal_runner_up_support"] for row in rows])
    innovations = np.array([
        row["candidate_to_state_xy"]
        if math.isfinite(row["candidate_to_state_xy"])
        else np.nan
        for row in rows
    ])
    yaw_innovations = np.array([
        math.degrees(row["candidate_to_state_yaw"])
        if math.isfinite(row["candidate_to_state_yaw"])
        else np.nan
        for row in rows
    ])

    figure, axes = plt.subplots(
        2, 1, figsize=(16, 9), constrained_layout=True
    )
    support_axis = axes[0]
    support_axis.plot(
        queries, temporal_support, "o-", label="dominant temporal support"
    )
    support_axis.plot(
        queries, runner, "s--", label="runner-up temporal support"
    )
    support_axis.axhline(3, color="tab:green", linestyle=":", label="lock gate")
    support_axis.scatter(
        queries[intrinsic], temporal_support[intrinsic],
        marker="o", facecolors="none", edgecolors="tab:orange",
        s=75, label="intrinsic candidate",
    )
    support_axis.scatter(
        queries[updated], temporal_support[updated],
        marker="*", color="tab:green", s=130, label="state update",
    )
    support_axis.set_title("Temporal candidate clustering")
    support_axis.set_ylabel("Distinct Query candidates")
    support_axis.grid(True)
    support_axis.legend(ncol=3)

    innovation_axis = axes[1]
    innovation_axis.plot(
        queries, innovations, "o-", color="tab:blue",
        label="candidate to held state XY [m]",
    )
    innovation_axis.axhline(
        2.0, color="tab:blue", linestyle=":", alpha=0.6,
        label="tracking XY gate",
    )
    yaw_axis = innovation_axis.twinx()
    yaw_axis.plot(
        queries, yaw_innovations, "s-", color="tab:purple",
        label="candidate to held state yaw [deg]",
    )
    yaw_axis.axhline(
        math.degrees(0.15), color="tab:purple", linestyle=":", alpha=0.6,
        label="tracking yaw gate",
    )
    innovation_axis.set_title("Candidate innovation after temporal lock")
    innovation_axis.set_xlabel("Query index")
    innovation_axis.set_ylabel("XY [m]")
    yaw_axis.set_ylabel("yaw [deg]")
    innovation_axis.grid(True)
    lines1, labels1 = innovation_axis.get_legend_handles_labels()
    lines2, labels2 = yaw_axis.get_legend_handles_labels()
    innovation_axis.legend(lines1 + lines2, labels1 + labels2, ncol=2)

    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=18)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--tree-stride", type=int, default=5)
    parser.add_argument("--global-map-tree-stride", type=int, default=5)
    parser.add_argument("--temporal-window", type=int, default=12)
    parser.add_argument("--temporal-xy", type=float, default=2.0)
    parser.add_argument("--temporal-yaw", type=float, default=0.15)
    parser.add_argument("--min-temporal-support", type=int, default=3)
    parser.add_argument("--tracking-xy", type=float, default=2.0)
    parser.add_argument("--tracking-yaw", type=float, default=0.15)
    parser.add_argument("--max-update-xy", type=float, default=0.3)
    parser.add_argument(
        "--max-update-yaw",
        type=float,
        default=math.radians(0.5),
        help="Maximum yaw correction per accepted query, in radians.",
    )
    parser.add_argument(
        "--initial-map-odom-x", type=float, required=True
    )
    parser.add_argument(
        "--initial-map-odom-y", type=float, required=True
    )
    parser.add_argument(
        "--initial-map-odom-z", type=float, required=True
    )
    parser.add_argument(
        "--initial-map-odom-yaw", type=float, required=True
    )
    parser.add_argument(
        "--reacquire-after-queries", type=int, default=60
    )
    parser.add_argument(
        "--reacquire-min-support", type=int, default=3
    )
    parser.add_argument(
        "--reacquire-max-xy", type=float, default=30.0
    )
    parser.add_argument(
        "--reacquire-max-yaw", type=float, default=0.15
    )
    parser.add_argument(
        "--z-reference-window-root", type=Path, required=True,
        help="Reference original-window dataset used by the pre-TreeLoc Z stage.",
    )
    parser.add_argument(
        "--z-current-window-root", type=Path, required=True,
        help="Current original-window dataset used by the pre-TreeLoc Z stage.",
    )
    parser.add_argument("--z-max-anchor-distance", type=float, default=5.0)
    parser.add_argument("--z-temporal-window", type=int, default=8)
    parser.add_argument("--z-temporal-tolerance", type=float, default=0.25)
    parser.add_argument("--z-min-temporal-support", type=int, default=3)
    parser.add_argument("--z-max-update", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    executable = args.repo / "build/treelocpp_localize_hypotheses"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_root = args.output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    map_poses = base.read_trajectory(args.map_root / "trajectory.txt")
    global_map_trees = base.load_global_map_tree_points(
        args.map_root, map_poses, args.global_map_tree_stride
    )
    first_query = args.query_root / f"query_{args.start}"
    odom_poses = base.read_trajectory(first_query / "trajectory.txt")
    state = base.planar_transform(
        args.initial_map_odom_x,
        args.initial_map_odom_y,
        args.initial_map_odom_z,
        args.initial_map_odom_yaw,
    )
    initial_state = state.copy()
    z_preprocessor = GroundZPreprocessor(
        args.z_reference_window_root,
        args.z_current_window_root,
        GroundZConfig(
            max_anchor_distance=args.z_max_anchor_distance,
            temporal_window_queries=args.z_temporal_window,
            temporal_tolerance=args.z_temporal_tolerance,
            temporal_min_support=args.z_min_temporal_support,
            max_update_z=args.z_max_update,
        ),
    )
    locked = False
    last_accepted_query = None
    candidates: list[dict] = []
    rows: list[dict] = []
    z_rows: list[dict] = []
    map_tree_sets: list[np.ndarray] = []
    current_tree_sets: list[tuple[int, int, bool, np.ndarray]] = []
    used_map_indices: set[int] = set()

    for query_index in range(args.start, args.end + 1):
        query_directory = args.query_root / f"query_{query_index}"
        odom_c = odom_poses[query_index]
        state, z_result = z_preprocessor.update(query_index, state)
        z_rows.append(dict(z_result))
        current_z = float(state[2, 3])
        before_map_c = state @ odom_c
        result, raw_output, return_code = base.run_localizer(
            executable, args.config, query_directory,
            args.map_root, before_map_c,
        )
        (log_root / f"query_{query_index}.txt").write_text(
            raw_output, encoding="utf-8"
        )

        process_ok = return_code == 0 and result is not None
        localizer_ok = process_ok and result["ok"] == "1"
        if result is None:
            result = {
                "support": "0", "pairs": "0", "overlap": "nan",
                "mean_residual": "nan", "max_residual": "nan",
                "runner_up_support": "0", "ambiguous": "0",
            }

        current_candidate_index = None
        candidate_to_state_xy = math.nan
        candidate_to_state_yaw = math.nan
        map_index = -1

        candidate = None
        if localizer_ok:
            candidate = {
                "transform": result_transform(result, current_z),
                "map_index": int(result["map_idx"]),
                "pairs": int(result["pairs"]),
                "overlap": float(result["overlap"]),
                "mean_residual": float(result["mean_residual"]),
                "max_residual": float(result["max_residual"]),
                "source": "same_query_consensus",
            }
        elif process_ok and locked:
            # Once a trustworthy temporal lock exists, a single intrinsically
            # valid anchor may cast one query-level vote.  It must still pass
            # the temporal membership and held-state tracking gates below.
            candidate = parse_rank_candidate(raw_output, odom_c, current_z)

        intrinsic_ok = candidate is not None
        if intrinsic_ok:
            map_index = int(candidate["map_index"])
            transform = candidate["transform"]
            candidate_to_state_xy, candidate_to_state_yaw = pose_distance(
                transform @ odom_c, state @ odom_c
            )
            candidates.append({
                "query": query_index,
                "transform": transform,
                **candidate,
            })
            current_candidate_index = len(candidates) - 1

        minimum_query = query_index - args.temporal_window + 1
        candidates = [
            candidate for candidate in candidates
            if candidate["query"] >= minimum_query
        ]
        if intrinsic_ok:
            current_candidate_index = next(
                index for index, candidate in enumerate(candidates)
                if candidate["query"] == query_index
            )

        cluster = select_temporal_cluster(
            candidates, odom_c, args.temporal_xy, args.temporal_yaw
        )
        current_is_member = (
            current_candidate_index is not None
            and current_candidate_index in cluster["members"]
        )
        temporal_ready = (
            cluster["support"] >= args.min_temporal_support
            and not cluster["ambiguous"]
            and current_is_member
        )
        tracking_ok = (
            not locked
            or (
                candidate_to_state_xy <= args.tracking_xy
                and candidate_to_state_yaw <= args.tracking_yaw
            )
        )
        queries_since_accept = (
            query_index - last_accepted_query
            if last_accepted_query is not None
            else -1
        )

        reacquire_ready = (
            locked
            and intrinsic_ok
            and localizer_ok
            and temporal_ready
            and not tracking_ok
            and queries_since_accept >= args.reacquire_after_queries
            and int(result["runner_up_support"]) == 0
            and int(result["ambiguous"]) == 0
            and cluster["support"] >= args.reacquire_min_support
            and cluster["runner_up_support"] == 0
            and candidate_to_state_xy <= args.reacquire_max_xy
            and candidate_to_state_yaw <= args.reacquire_max_yaw
        )

        tracking_ok = tracking_ok or reacquire_ready
        update_applied = (
            intrinsic_ok
            and temporal_ready
            and tracking_ok
        )

        parameter_planar = math.nan
        parameter_yaw = math.nan
        lidar_update_planar = math.nan
        if update_applied:
            previous = state.copy()
            previous_map_c = previous @ odom_c
            target_state = aggregate_cluster(
                candidates, cluster["members"], odom_c, current_z
            )
            state = (
                target_state
                if not locked or reacquire_ready
                else bounded_projected_update(
                    previous,
                    target_state,
                    odom_c,
                    current_z,
                    args.max_update_xy,
                    args.max_update_yaw,
                )
            )
            parameter_planar = math.hypot(
                state[0, 3] - previous[0, 3],
                state[1, 3] - previous[1, 3],
            )
            parameter_yaw = base.wrap(
                base.yaw_from_rotation(state[:3, :3])
                - base.yaw_from_rotation(previous[:3, :3])
            )
            updated_map_c = state @ odom_c
            lidar_update_planar = math.hypot(
                updated_map_c[0, 3] - previous_map_c[0, 3],
                updated_map_c[1, 3] - previous_map_c[1, 3],
            )
            locked = True
            last_accepted_query = query_index
            map_index = candidates[current_candidate_index]["map_index"]

        if not process_ok:
            status = "unusable_query"
        elif not intrinsic_ok:
            status = "intrinsic_rejected"
        elif not temporal_ready:
            status = "buffering" if not locked else "temporal_rejected"
        elif not tracking_ok:
            status = "tracking_rejected"
        elif update_applied:
            status = "locked" if sum(row["accepted"] for row in rows) == 0 else "updated"
        else:
            status = "held"

        if reacquire_ready and update_applied:
            status = "relocked"

        after_map_c = state @ odom_c
        after_yaw = base.yaw_from_rotation(after_map_c[:3, :3])
        row = {
            "query": query_index,
            "accepted": update_applied,
            "reacquired": reacquire_ready,
            "status": status,
            "localizer_ok": localizer_ok,
            "intrinsic_ok": intrinsic_ok,
            "candidate_source": (
                candidate["source"] if candidate is not None else "none"
            ),
            "map_index": map_index,
            "support": int(result["support"]),
            "same_query_runner_up_support": int(result.get("runner_up_support", "0")),
            "same_query_ambiguous": int(result.get("ambiguous", "0")),
            "temporal_support": cluster["support"],
            "temporal_runner_up_support": cluster["runner_up_support"],
            "temporal_ambiguous": cluster["ambiguous"],
            "current_is_temporal_member": current_is_member,
            "pairs": candidate["pairs"] if candidate is not None else 0,
            "overlap": candidate["overlap"] if candidate is not None else math.nan,
            "mean_residual": (
                candidate["mean_residual"] if candidate is not None else math.nan
            ),
            "max_residual": (
                candidate["max_residual"] if candidate is not None else math.nan
            ),
            "candidate_to_state_xy": candidate_to_state_xy,
            "candidate_to_state_yaw": candidate_to_state_yaw,
            "before_x": before_map_c[0, 3],
            "before_y": before_map_c[1, 3],
            "before_z": before_map_c[2, 3],
            "before_yaw": base.yaw_from_rotation(before_map_c[:3, :3]),
            "after_x": after_map_c[0, 3],
            "after_y": after_map_c[1, 3],
            "after_z": after_map_c[2, 3],
            "after_yaw": after_yaw,
            "state_x": state[0, 3],
            "state_y": state[1, 3],
            "state_z": state[2, 3],
            "state_yaw": base.yaw_from_rotation(state[:3, :3]),
            "z_reference_index": z_result.get("reference_index", -1),
            "z_anchor_distance": z_result.get("anchor_distance", math.nan),
            "z_measurement_valid": z_result.get("valid", False),
            "z_gate_reason": z_result.get("gate_reason", "unavailable"),
            "z_candidate": z_result.get("state_z_candidate", math.nan),
            "z_temporal_support": z_result.get("temporal_support", 0),
            "z_temporal_runner_up_support": z_result.get("temporal_runner_up_support", 0),
            "z_temporal_ready": z_result.get("temporal_ready", False),
            "z_target": z_result.get("state_z_target", math.nan),
            "z_update": z_result.get("z_update", 0.0),
            "z_correspondences": z_result.get("correspondences", 0),
            "z_trimmed_after_rmse": z_result.get("trimmed_after_rmse", math.nan),
            "z_trimmed_after_median_abs": z_result.get("trimmed_after_median_abs", math.nan),
            "parameter_planar": parameter_planar,
            "parameter_yaw": parameter_yaw,
            "lidar_update_planar": lidar_update_planar,
            "map_anchor_x": map_poses[map_index][0, 3] if map_index >= 0 else math.nan,
            "map_anchor_y": map_poses[map_index][1, 3] if map_index >= 0 else math.nan,
        }
        rows.append(row)

        current_csv = query_directory / f"TreeManagerState_{query_index}.csv"
        if current_csv.is_file():
            points = base.read_tree_points(current_csv)
            row_index = query_index - args.start
            if (
                row_index % max(1, args.tree_stride) == 0
                or query_index == args.end
            ) and len(points):
                current_tree_sets.append((
                    row_index, query_index, update_applied,
                    base.transform_points(after_map_c, points),
                ))

        if map_index >= 0 and map_index not in used_map_indices:
            map_csv = args.map_root / f"TreeManagerState_{map_index}.csv"
            if map_csv.is_file():
                map_tree_sets.append(base.transform_points(
                    map_poses[map_index], base.read_tree_points(map_csv)
                ))
                used_map_indices.add(map_index)

        print(
            f"Q{query_index}: {status.upper()} "
            f"intrinsic={int(intrinsic_ok)} "
            f"temporal={cluster['support']}/{cluster['runner_up_support']} "
            f"member={int(current_is_member)} "
            f"innovation=({candidate_to_state_xy:.3f} m, "
            f"{math.degrees(candidate_to_state_yaw):.2f} deg) "
            f"state=({state[0,3]:.6f}, {state[1,3]:.6f}, "
            f"z={state[2,3]:.6f}, "
            f"{math.degrees(base.yaw_from_rotation(state[:3,:3])):.4f} deg) "
            f"ground_z={z_result.get('gate_reason', 'unavailable')}/"
            f"{z_result.get('temporal_support', 0)} "
            f"dz={z_result.get('z_update', 0.0):.3f}"
        )

    initial_projection = np.array([
        (initial_state @ odom_poses[index])[:2, 3]
        for index in range(args.start, args.end + 1)
    ])
    csv_path = args.output_dir / "temporal_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    z_csv_path = args.output_dir / "ground_z_preprocessor.csv"
    z_fieldnames = list(dict.fromkeys(
        key for z_row in z_rows for key in z_row.keys()
    ))
    with z_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=z_fieldnames)
        writer.writeheader()
        writer.writerows(z_rows)

    base.plot_replay(
        rows, map_poses, global_map_trees, map_tree_sets,
        current_tree_sets, initial_projection,
        args.output_dir / "temporal_replay_map.png",
    )
    plot_temporal_gate(
        rows, args.output_dir / "temporal_gate_diagnostics.png"
    )
    print("\nCSV:", csv_path)
    print("Ground Z CSV:", z_csv_path)
    print("Map PNG:", args.output_dir / "temporal_replay_map.png")
    print("Gate PNG:", args.output_dir / "temporal_gate_diagnostics.png")


if __name__ == "__main__":
    main()
