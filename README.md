# Z-only ground alignment diagnostic

The package contains an offline diagnostic and a stateful preprocessing stage.
The preprocessing stage is tree-independent: at each query it uses the current
map-to-odom XY/yaw prior to select the nearest reference window, updates only Z,
and then passes that state to TreeLoc++. TreeLoc++ continues to estimate only
XY/yaw.

The reference payload is selected from the accepted row's `map_index`. Ground
points are extracted independently in each payload, so an initial vertical
offset does not remove the true terrain before optimization. Matching uses one
robust terrain sample per shared global XY cell. This prevents vegetation
outliers and dense current payloads from creating thousands of many-to-one
nearest-neighbor votes. Gates use the same trimmed residual population as the
robust optimizer rather than returning rejected outliers to the full RMSE.

## Install and test

```bash
tar -xf ~/Downloads/TreeLoc-plusplus-z-only-ground-alignment.tar \
  -C ~/TreeLoc-plusplus

cd ~/TreeLoc-plusplus
python3 tests/test_z_only_ground_alignment.py
```

## Run the 42 accepted-query diagnostic

```bash
SE2_RESULT=\
/home/wataru/TreeLoc-plusplus/results/asaka_forest_4_localization_new_partial_se2_reacquire_v1

REFERENCE_WINDOWS=\
/home/wataru/realtime_trees_cpp_ref/data/asaka_forest_4_map_windowed_a2_r20_v005_v1

CURRENT_WINDOWS=\
/home/wataru/realtime_trees_cpp_ref/data/asaka_forest_localization_new_current_windowed_a2_r20_v005_v1

python3 tools/analyze_z_only_ground_alignment.py \
  --replay-csv "$SE2_RESULT/temporal_replay.csv" \
  --reference-window-root "$REFERENCE_WINDOWS" \
  --current-window-root "$CURRENT_WINDOWS" \
  --output-dir "$SE2_RESULT/z_only_ground_v1"
```

The result is `z_only_ground_alignment.csv`. `state_z_candidate` is
`state_z_before + dz`; only rows with `valid=1` should be considered. Before
online integration, inspect the valid count, the median/MAD of candidate Z,
and discontinuities over query index.

## Pre-TreeLoc integration

`z_only_ground_preprocessor.py` exposes `GroundZPreprocessor.update(query,
state)`. The returned transform is a copy with only element `[2, 3]` changed.
It does not read trees or TreeLoc++ decisions. The v4 replay calls this method
before `run_localizer()` for every query and writes independent diagnostics to
`ground_z_preprocessor.csv`.

Compared with the v3 replay command, use
`visualize_temporal_map_odom_replay_rank_fallback_v4.py` and add:

```bash
  --z-reference-window-root "$REFERENCE_WINDOWS" \
  --z-current-window-root "$CURRENT_WINDOWS" \
  --z-temporal-window 8 \
  --z-temporal-tolerance 0.25 \
  --z-min-temporal-support 3 \
  --z-max-update 0.15
```

The Z stage holds its state on missing files, bad residuals, insufficient
temporal support, or exceptions; those failures do not abort TreeLoc++.
