<div align="center">

<h1>TreeLoc++: Multi-Session Forest Localization with Tree Geometry</h1>

Official repository for **TreeLoc++**.

<a href="https://github.com/minwoo0611/TreeLoc" target="_blank">TreeLoc</a> | TreeLoc++

</div>

### Recent Updates
- [2026/06/25] Initial TreeLoc++ branch with intra-session and inter-session C++ code.

### Contributions
- **TreeLoc++ extends TreeLoc to intra-session and inter-session forest localization** using TDH retrieval, pairwise tree-distance histograms, triangle verification, DBH-aware matching, and yaw-voting initialization.
- **TreeLoc++ uses tree-level CSV files with `location_z` read directly from CSV when available**, so runtime localization does not require `terrain.obj` or `trajectory_tree.txt`.
- **TreeLoc++ keeps parameters in YAML config files** for dataset paths, descriptor settings, neighbor augmentation, pairwise weighting, yaw voting, DBH gates, and vertical correction.

### Dataset

Full dataset will be opened after finalaccpetance.

This branch includes small public samples under `sample_data/`. The samples are generated from the Oxford V-02 and V-03 large-cluster traces and contain short sequences with `axis_*`, `location_x`, `location_y`, `location_z`, DBH, reconstruction, cluster, and score fields.

The runtime dataset format is:

```text
dataset_root/
├── trajectory.txt
├── TreeManagerState_0.csv
├── TreeManagerState_1.csv
└── ...
```

`trajectory.txt` uses:

```text
timestamp x y z qx qy qz qw
```

Each `TreeManagerState_<idx>.csv` must include:

- `axis_00` ... `axis_22`
- `location_x`, `location_y`, `location_z`
- `dbh` or `dbh_approximation`

Optional columns used when available:

- `reconstructed`
- `number_clusters`
- `score` or `scores`
- `alignment_z`

`tools/prepare_sample_data.py` copies TreeLoc-style CSV rows without `trajectory_tree.txt` normalization. If the source CSV already has `location_z`, the value is preserved. If `location_z` is missing, pass `--terrain` to fill tree heights from `terrain.obj` with 6-NN inverse-distance interpolation.

### Prerequisites

TreeLoc++ is a C++17 CMake project.

- CMake >= 3.16
- C++17 compiler
- Eigen3

Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev
```

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

This builds:

- `treelocpp_intra`
- `treelocpp_inter`

### Usage

Intra-session:

```bash
./build/treelocpp_intra config/default.yaml
```

Inter-session:

```bash
./build/treelocpp_inter config/inter.yaml
```

Regenerate the sample from a full local TreeLoc++ sequence:

```bash
python3 tools/prepare_sample_data.py \
  --source /path/to/oxford/V-02_large_cluster \
  --terrain /path/to/oxford/terrain.obj \
  --output sample_data/oxford_v02 \
  --frames 60
```

The 60-frame samples are intended for build and runtime smoke tests. Intra-session metrics should be measured on a full sequence because a short slice may not contain positive pairs after temporal filtering.

Full intra-session evaluation should use the full sequence:

```bash
python3 tools/prepare_sample_data.py \
  --source /path/to/oxford/V-02_large_cluster \
  --terrain /path/to/oxford/terrain.obj \
  --output data/oxford_v02_intra

./build/treelocpp_intra config/full_v02.yaml
```

Full inter-session evaluation uses Oxford V-03 as query and Oxford V-02 as map:

```bash
python3 tools/prepare_sample_data.py \
  --source /path/to/oxford/V-02_large_cluster \
  --terrain /path/to/oxford/terrain.obj \
  --output data/oxford_v02

python3 tools/prepare_sample_data.py \
  --source /path/to/oxford/V-03_large_cluster \
  --terrain /path/to/oxford/terrain.obj \
  --output data/oxford_v03

./build/treelocpp_inter config/inter_v02_v03.yaml
```

`config/inter_v02_v03.yaml` uses local TreeLoc-style axis alignment, V-region test polygons, and a 5 m ground-truth radius.

### Configuration

Main parameters are in:

- `config/default.yaml` for intra-session evaluation
- `config/inter.yaml` for inter-session evaluation
- `config/full_v02.yaml` for full Oxford V-02 intra-session evaluation
- `config/full_v03.yaml` for full Oxford V-03 intra-session evaluation
- `config/inter_v02_v03.yaml` for full Oxford V-03-to-V-02 inter-session evaluation

The defaults preserve the TreeLoc++ experiment settings used in the previous single-file implementation: TDH + pairwise retrieval, triangle hash reranking, DBH-aware triangle matching, yaw voting, neighbor augmentation, t-aware overlap, and z/roll/pitch correction from precomputed tree heights.

### Acknowledgement

TreeLoc++ builds on TreeLoc and tree-level representations extracted with RealtimeTrees from forest LiDAR data.
