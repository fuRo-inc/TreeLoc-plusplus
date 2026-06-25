#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
from pathlib import Path


def rewrite_pose(line, dx, dy, yaw):
    vals = [float(x) for x in line.split()]
    vals[1] += dx
    vals[2] += dy
    half = yaw * 0.5
    vals[4], vals[5], vals[6], vals[7] = 0.0, 0.0, math.sin(half), math.cos(half)
    return " ".join(f"{v:.12g}" for v in vals)


def transform_csv(path, dx, dy, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    for row in rows:
        x = float(row["location_x"])
        y = float(row["location_y"])
        row["location_x"] = f"{c * x - s * y + dx:.8f}"
        row["location_y"] = f"{s * x + c * y + dy:.8f}"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dx", type=float, default=0.35)
    parser.add_argument("--dy", type=float, default=-0.2)
    parser.add_argument("--yaw-deg", type=float, default=2.0)
    args = parser.parse_args()
    if args.query.exists():
        shutil.rmtree(args.query)
    if args.database.exists():
        shutil.rmtree(args.database)
    shutil.copytree(args.source, args.query)
    shutil.copytree(args.source, args.database)
    yaw = math.radians(args.yaw_deg)
    lines = [rewrite_pose(line, args.dx, args.dy, yaw)
             for line in (args.query / "trajectory.txt").read_text().splitlines()
             if line.strip()]
    (args.query / "trajectory.txt").write_text("\n".join(lines) + "\n")
    for csv_path in args.query.glob("TreeManagerState_*.csv"):
        transform_csv(csv_path, args.dx, args.dy, yaw)


if __name__ == "__main__":
    main()
