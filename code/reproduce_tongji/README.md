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
