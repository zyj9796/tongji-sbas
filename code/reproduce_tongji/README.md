# Tongji Clean Equal-Height InSAR Reproduction Code

This directory contains the executable scripts used for the current active
Tongji building-height reproduction. The active workflow uses the optimized
clean equal-height building vector and keeps only strict InSAR outputs for
buildings that pass the current quality gates.

The shapefile `height` field is a diagnostic prior only. It is not used to
fill, fit, or rescale the active InSAR-only product.

## Runtime

Python packages:

```bash
env UV_CACHE_DIR=tmp/uv-cache uv venv .venv
env UV_CACHE_DIR=tmp/uv-cache uv pip install --python .venv/bin/python -r code/reproduce_tongji/requirements.txt
```

System/GAMMA dependencies:

- GDAL command-line tools
- GAMMA tools for baseline selection, differential interferograms, geocoding,
  and phase simulation

## Active Inputs

- Config:
  `configs/tongji_reproduction.json`
- Optimized clean building vector:
  `data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.geojson`
- Clean SAR roof projection:
  `work/projection/20200708_clean_equal_height_roof_projection_sar.geojson`
- Clean roof mask:
  `work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy`
- Audited redshift roof mask:
  `work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy`
- Audited clean-ID split island label:
  `work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy`
- Audited clean-ID split island table:
  `work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv`
- Clean FID/UID map:
  `results/tables/clean_equal_height_fid_uid_map.csv`
- GAMMA differential interferograms:
  `work/gamma_sbas/intf_triangular_dsm/`
- Pair list:
  `work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv`
- Radar-coordinate DSM/HGT:
  `work/gamma_sbas/dem/20200708_dsm_rdc.hgt`
- Amplitude-dispersion raster:
  `work/mli/amplitude_dispersion_crop_bmp.npy`

## Rebuild The Clean Vector Workflow

Build or refresh the optimized equal-height building vector:

```bash
.venv/bin/python code/reproduce_tongji/clean_equal_height_building_vectors.py
```

Build the clean SAR roof projection:

```bash
.venv/bin/python code/reproduce_tongji/build_clean_equal_height_roof_projection.py
```

Build the full-area clean roof mask, then split islands by `clean_id`:

```bash
.venv/bin/python code/reproduce_tongji/make_building_islands.py \
  --projection-geojson work/projection/20200708_clean_equal_height_roof_projection_sar.geojson \
  --surface roof \
  --uid-mask work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy \
  --island-label work/masks/island_label_clean_equal_height_roof_only_full_area_128.npy \
  --islands-csv work/masks/islands_clean_equal_height_roof_only_full_area_128.csv \
  --uid-preview work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.png \
  --island-preview work/masks/island_label_clean_equal_height_roof_only_full_area_128.png \
  --summary results/metadata/island_extraction_clean_equal_height_roof_only_full_area_128_summary.json

.venv/bin/python code/reproduce_tongji/make_clean_id_split_islands.py
```

Apply the audited projection-bias correction only to red QC buildings whose
InSAR-internal audit improves reliability without losing strict solutions:

```bash
.venv/bin/python code/reproduce_tongji/optimize_red_building_mask_shifts.py \
  --accept-clean-ids-file work/projection/cleanid_redshift_audited_accept_clean_ids.csv \
  --out-mask work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy \
  --out-csv work/projection/cleanid_split_red_building_mask_shift_metrics_audited.csv \
  --summary results/metadata/cleanid_split_red_building_mask_shift_audited_summary.json

.venv/bin/python code/reproduce_tongji/make_clean_id_split_islands.py \
  --uid-mask work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy \
  --out-island-label work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy \
  --out-islands-csv work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv \
  --preview work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.png \
  --summary results/metadata/island_extraction_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split_summary.json
```

Run strict roof-only paper-like LGR inversion on the audited mask:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-tongji \
.venv/bin/python code/reproduce_tongji/estimate_pixel_lgr_building_heights.py \
  --pairs-csv work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv \
  --intf-root work/gamma_sbas/intf_triangular_dsm \
  --island-label work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy \
  --fid-mask work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy \
  --islands-csv work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv \
  --fid-uid-map results/tables/clean_equal_height_fid_uid_map.csv \
  --reference-height-rdc work/gamma_sbas/dem/20200708_dsm_rdc.hgt \
  --ground-dem-m 4.0 \
  --residual-sign 1.0 \
  --min-pairs 12 \
  --min-coherence 0.75 \
  --amplitude-dispersion-npy work/mli/amplitude_dispersion_crop_bmp.npy \
  --max-amplitude-dispersion 0.4 \
  --max-pixel-rmse-rad 1.25 \
  --min-bperp-span-m 120 \
  --min-pixels 20 \
  --unwrap-method paper \
  --output-islands work/height/island_pixel_lgr_heights_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv \
  --output-points work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv \
  --summary results/metadata/pixel_lgr_building_heights_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_summary.json \
  --figure "" \
  --figure-svg ""
```

Aggregate to clean building vectors and build the final top-down Grubbs product:

```bash
.venv/bin/python code/reproduce_tongji/aggregate_clean_equal_height_insar_heights.py \
  --points work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv \
  --height-statistic grubbs_top \
  --out-geojson results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topstats_insar_only.geojson \
  --out-csv results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topstats_insar_only.csv \
  --summary results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topstats_summary.json \
  --height-map-png "" \
  --height-map-svg results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg \
  --diagnostic-png "" \
  --diagnostic-svg "" \
  --omit-height-field

.venv/bin/python code/reproduce_tongji/build_likely_top_height_from_tests.py \
  --input results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topstats_insar_only.geojson \
  --selection-mode top_down_grubbs \
  --out-geojson results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson \
  --out-csv results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv \
  --out-png "" \
  --out-svg "" \
  --summary results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_summary.json \
  --omit-height-field \
  --chinese-labels
```

## Active Outputs

- Height points:
  `work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv`
- Island heights:
  `work/height/island_pixel_lgr_heights_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv`
- Building GeoJSON:
  `results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson`
- Building CSV:
  `results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv`
- Summary:
  `results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_summary.json`
- Method and optimization plan:
  `docs/current_literature_route_and_optimization_plan.md`

## Roof-core stable-ground optimization (2026-08-23)

The optimized branch uses the GAMMA-corrected roof polygons directly, removes one-pixel roof edges when enough core pixels remain, deletes cross-building roof conflicts, and selects a stable-ground phase datum outside the buffered building support. Stable ground is defined only by the 4 m reference surface, amplitude dispersion, and coherence; vector `height` is never read for masking, inversion, ambiguity selection, QC, or filling.

The solver adds `--unwrap-method temporal_multistart` and `--stable-reference-mask`. It removes the coherence-weighted circular ground phase center for every interferogram, searches multiple DEM-residual ambiguity seeds, applies the existing coherence/DA/pair/RMSE/baseline gates, rejects physically impossible height branches, and reports a quality-weighted roof-pixel median. Unsolved buildings remain `NaN`.

Primary optimized outputs:

- `results/geodata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.geojson`
- `results/tables/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.csv`
- `results/metadata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_summary.json`
- `picall/12_屋顶核心区SBAS高度图.svg`
- `picall/14_严格有解建筑三维重建.svg`
- `picall/17_屋顶核心区SBAS建筑高度分布.svg`

Strict optimized result: 260 solved and 768 unsolved buildings; median/P05/P95 height = 23.56/4.48/49.28 m. Negative and over-120 m accepted heights are both zero. The previous active branch solved 239 buildings.

A damped (`alpha=0.6`) projection iteration was tested with the initial time-cross-validated SAR registration frozen. It reduced solved coverage from 260 to 228, lost 38 initial solutions, and only 11 common buildings met both `|delta H| <= 1 m` and roof-centroid shift `<= 0.5 pixel`. The iteration therefore failed the predeclared acceptance rule and was not adopted. See `work/roof_sbas_optimized/iteration_convergence_summary.json`. This prevents an unstable feedback loop from replacing the strict initial solution.

## Figures

- Audited redshift final figures:
  `results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg`

## Current Status

- Clean buildings: 1028
- Clean roof projection success: 1026
- Clean-ID split islands: 662
- Multi-clean-ID islands after split: 0
- Audited projection shifts accepted: 16 / 81 red QC buildings
- LGR islands processed: 635
- Islands with height: 241
- Strict InSAR solved buildings: 239
- Buildings without strict InSAR solution: 789
- Top-down Grubbs median height: 22.55 m
- Top-down Grubbs P05/P95: 3.70 / 48.99 m
- Reliability counts: high 168, medium 42, review 29, no solution 789
- Median coherence: 0.852
- Median amplitude dispersion: 0.336
- Median LGR RMSE: 0.737 rad
- Median valid pairs: 17
- Shapefile `height` field use: not read for comparison or quality control

The status block above describes the earlier Python/LGR branch and is retained for reproducibility; it is no longer the active GAMMA-native experiment.

## GAMMA-native SBAS continuation (2026-08-23)

`run_gamma_native_ipta_sbas.py` forms and filters the GAMMA differential interferograms, builds the IPTA stack, runs `multi_def_pt`, and then runs the actual GAMMA SBAS solver `mb_pt`. It supports equal interferogram weights or a three-column GAMMA `sigma` file estimated by robust MAD on stable-ground residuals. The vector `height` field is merged only after the solution is frozen.

`run_gamma_building_mcf_sbas.py` evaluates building-wise phase-reference and SBAS variants. Despite its historical filename, its accepted current path is not direct MCF: direct `mcf_pt`, local-from-scratch `multi_def_pt`, local residual MCF, and ZJC graph unwrap followed by `mb_pt` were all tested and rejected because they collapsed absolute roof-ground height. The current path retains the global baseline-time integer ambiguity from GAMMA `multi_def_pt`, then runs GAMMA `mb_pt(gamma=100)` independently for each building with nearby stable-ground points.

Current roof-core engineering candidate:

- CSV: `results/tables/tongji_building_height_gamma_roofcore_buildingmb_gamma100_insar_only.csv`
- GeoJSON: `results/geodata/tongji_building_height_gamma_roofcore_buildingmb_gamma100_insar_only.geojson`
- Summary: `results/metadata/tongji_building_height_gamma_roofcore_buildingmb_gamma100_summary.json`
- 318 accepted buildings; median/P05/P95 = 14.13/1.88/35.94 m; median phase residual = 0.268 rad.

Paper-strict roof-core candidate (`mean coherence >= 0.75`, 1.5×IQR then median):

- CSV: `results/tables/tongji_building_height_gamma_roofcore_coh075_gamma100_median_insar_only.csv`
- GeoJSON: `results/geodata/tongji_building_height_gamma_roofcore_coh075_gamma100_median_insar_only.geojson`
- Summary: `results/metadata/tongji_building_height_gamma_roofcore_coh075_gamma100_median_summary.json`
- 231 accepted buildings; median/P05/P95 = 8.83/0.86/30.86 m; median phase residual = 0.252 rad.

Neither candidate uses prior heights for inversion, QC, calibration, aggregation, or filling. Neither is yet accepted as a final accuracy product, and the existing Figure 17 has not been replaced.

Additional diagnostic programs:

- `correct_gamma_punw_closure.py`: integer `2π` correction on redundant network chords; bridges remain unchanged.
- `simulate_gamma_height_sensitivity.py`: GAMMA orbital phase sensitivity in rad/m.
- `run_zjc_dense_island_gamma_sbas.py`: full layover-island local unwrap followed by GAMMA `mb_pt`, including an explicit original-MATLAB rectangle/zero-background compatibility mode.
- `select_stable_roof_top_points.py`: early/late and long/short baseline repeatability audit for isolated rooftop candidates.

The highest-stable-point output is diagnostic only and was rejected because a consistent integer alias can survive every subset test. None of these experiments authorizes replacing Figure 17; missing/rejected buildings remain null and are never filled from vector `height`.

## Adaptive-window roof projection rerun (2026-08-24)

The roof-only adaptive local-window projection has now been propagated through the complete 48-pair GAMMA workflow. The rebuilt mask contains 553 in-scene buildings, 575 roof islands, 87,446 roof-core pixels, 55,504 reliable roof points, and 21,332 stable-ground reference pixels. The IPTA/SBAS rerun uses `geometry_wrapped_init` followed by per-building `mb_pt(gamma=100)` and local-ground subtraction.

Current strict hybrid output:

- CSV: `results/tables/tongji_building_height_adaptive_window_network48_gamma100_strict.csv`
- GeoJSON: `results/geodata/tongji_building_height_adaptive_window_network48_gamma100_strict.geojson`
- Summary: `results/metadata/tongji_building_height_adaptive_window_network48_gamma100_strict_summary.json`
- Height map: `picall/17_屋顶核心区SBAS建筑高度分布.svg`

Of 503 candidates, 381 receive a GAMMA numerical solution and 361 pass the frozen strict gates. Median/P05/P95 building height is 16.05/5.15/47.74 m and median phase sigma is 0.191 rad. All 667 unsolved buildings remain null; `filled_from_prior` is zero throughout. Vector height is used only for R-D roof initialization, adaptive-window size, and integer-cycle initialization, so this remains a hybrid/non-independent product rather than a pure-InSAR accuracy validation.
