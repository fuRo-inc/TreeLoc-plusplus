#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile


DATABASE_TREES = [
    (0.4, 1.2, 0.18),
    (2.8, -0.7, 0.26),
    (-1.5, 3.4, 0.34),
    (4.2, 2.1, 0.42),
    (-3.1, -1.8, 0.50),
    (1.1, 5.0, 0.58),
]

TREE_COLUMNS = [
    "axis_00", "axis_01", "axis_02",
    "axis_10", "axis_11", "axis_12",
    "axis_20", "axis_21", "axis_22",
    "location_x", "location_y", "location_z",
    "dbh", "dbh_valid", "dbh_approximation",
    "reconstructed", "number_clusters",
    "score", "payload_index", "tree_id",
]


def inverse_transform_xy(x, y, tx, ty, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx = x - tx
    dy = y - ty
    return c * dx + s * dy, -s * dx + c * dy


def write_trajectory(path, poses):
    lines = []
    for index, (x, y, z, yaw) in enumerate(poses):
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)
        lines.append(
            f"{index:.9f} {x:.12f} {y:.12f} {z:.12f} "
            f"0.000000000000 0.000000000000 {qz:.12f} {qw:.12f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trees(path, trees, payload_index, partial=False):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TREE_COLUMNS)
        writer.writeheader()
        for tree_id, (x, y, dbh) in enumerate(trees):
            writer.writerow({
                "axis_00": 1.0, "axis_01": 0.0, "axis_02": 0.0,
                "axis_10": 0.0, "axis_11": 1.0, "axis_12": 0.0,
                "axis_20": 0.0, "axis_21": 0.0, "axis_22": 1.0,
                "location_x": x, "location_y": y, "location_z": 0.0,
                # Partial candidates deliberately carry no usable diameter.
                # Their XY constellation must still be loadable/localizable.
                "dbh": "" if partial else dbh,
                "dbh_valid": 0 if partial else 1,
                "dbh_approximation": "" if partial else dbh,
                "reconstructed": 0 if partial else 1,
                "number_clusters": 1,
                "score": 1.0,
                "payload_index": payload_index,
                "tree_id": tree_id,
            })


def parse_result(output):
    line = next(
        line for line in output.splitlines()
        if line.startswith("localization_result ")
    )
    return {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in line.split()[1:]
    }


def run(
    executable,
    config,
    query_root,
    database_root,
    min_support,
    search_radius=4.0,
):
    command = [
        str(executable), str(config),
        "--query_root", str(query_root),
        "--database_root", str(database_root),
        "--prior_x", "3.0",
        "--prior_y", "-1.5",
        "--prior_z", "4.25",
        "--prior_yaw", str(math.radians(20.0)),
        "--search_radius", str(search_radius),
        "--top_k", "6",
        "--min_consensus_support", str(min_support),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return parse_result(completed.stdout)


def assert_close(actual, expected, tolerance=1.0e-5):
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"actual={actual}, expected={expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    map_c_x = 3.0
    map_c_y = -1.5
    map_c_yaw = math.radians(20.0)
    odom_c_x = 5.0
    odom_c_y = 2.0
    odom_c_yaw = math.radians(10.0)

    query_trees = [
        (*inverse_transform_xy(x, y, map_c_x, map_c_y, map_c_yaw), dbh)
        for x, y, dbh in DATABASE_TREES
    ]

    with tempfile.TemporaryDirectory(prefix="treelocpp_contract_") as temporary:
        root = Path(temporary)
        query_root = root / "query"
        database_root = root / "database"
        query_root.mkdir()
        database_root.mkdir()

        write_trajectory(
            query_root / "trajectory.txt",
            [(odom_c_x, odom_c_y, 2.0, odom_c_yaw)],
        )
        write_trees(
            query_root / "TreeManagerState_0.csv",
            query_trees,
            0,
            partial=True,
        )

        database_poses = (
            [(0.0, 0.0, 0.0, 0.0)] * 3
            + [(10.0, 0.0, 0.0, math.pi)] * 3
        )
        write_trajectory(
            database_root / "trajectory.txt",
            database_poses,
        )
        for index in range(6):
            write_trees(
                database_root / f"TreeManagerState_{index}.csv",
                DATABASE_TREES,
                index,
            )

        accepted = run(
            args.executable,
            args.config,
            query_root,
            database_root,
            3,
        )
        if accepted["ok"] != "1":
            raise AssertionError(accepted)
        if accepted["support"] != "3":
            raise AssertionError(accepted)

        expected_yaw = map_c_yaw - odom_c_yaw
        c = math.cos(expected_yaw)
        s = math.sin(expected_yaw)
        expected_x = map_c_x - (c * odom_c_x - s * odom_c_y)
        expected_y = map_c_y - (s * odom_c_x + c * odom_c_y)

        assert_close(float(accepted["map_c_x"]), map_c_x)
        assert_close(float(accepted["map_c_y"]), map_c_y)
        assert_close(float(accepted["map_c_z"]), 4.25)
        assert_close(float(accepted["map_c_roll"]), 0.0)
        assert_close(float(accepted["map_c_pitch"]), 0.0)
        assert_close(float(accepted["map_c_yaw"]), map_c_yaw)
        assert_close(float(accepted["odom_c_x"]), odom_c_x)
        assert_close(float(accepted["odom_c_y"]), odom_c_y)
        assert_close(float(accepted["odom_c_yaw"]), odom_c_yaw)
        assert_close(float(accepted["map_odom_x"]), expected_x)
        assert_close(float(accepted["map_odom_y"]), expected_y)
        assert_close(float(accepted["map_odom_z"]), 0.0)
        assert_close(float(accepted["map_odom_roll"]), 0.0)
        assert_close(float(accepted["map_odom_pitch"]), 0.0)
        assert_close(float(accepted["map_odom_yaw"]), expected_yaw)

        rejected = run(
            args.executable,
            args.config,
            query_root,
            database_root,
            4,
        )
        if rejected["ok"] != "0" or rejected["map_idx"] != "-1":
            raise AssertionError(rejected)
        for key in (
            "map_c_x", "map_c_y", "map_c_yaw",
            "map_odom_x", "map_odom_y", "map_odom_yaw",
            "difference_planar", "parameter_planar",
        ):
            if not math.isnan(float(rejected[key])):
                raise AssertionError(f"{key} must be nan: {rejected[key]}")

        ambiguous = run(
            args.executable,
            args.config,
            query_root,
            database_root,
            3,
            search_radius=20.0,
        )
        if (
            ambiguous["ok"] != "0"
            or ambiguous["map_idx"] != "-1"
            or ambiguous["support"] != "3"
            or ambiguous["runner_up_support"] != "3"
            or ambiguous["ambiguous"] != "1"
        ):
            raise AssertionError(ambiguous)
        for key in (
            "map_c_x", "map_c_y", "map_c_yaw",
            "map_odom_x", "map_odom_y", "map_odom_yaw",
        ):
            if not math.isnan(float(ambiguous[key])):
                raise AssertionError(f"{key} must be nan: {ambiguous[key]}")


if __name__ == "__main__":
    main()
