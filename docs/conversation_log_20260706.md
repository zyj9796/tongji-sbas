# Conversation Log 2026-07-06

## User Requests

1. Continue previous work.
2. Explain the follow-up optimization plan.
3. Summarize completed work.
4. Clean and merge files, remove unused outputs, and redefine current work and next optimization ideas according to the thesis literature.
5. Manage the project with Git.
6. Continue optimizing figure output so only one SVG is produced:
   `results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg`.
7. Save the conversation.

## Current Literature-Aligned Route

The active method follows the thesis route:

- Use building footprints as a prior constraint before InSAR unwrapping.
- Project clean equal-height building footprints into SAR coordinates.
- Build roof-only SAR masks.
- Split islands by `clean_id` to avoid mixed-building islands.
- Use audited SAR-amplitude/edge projection correction only where InSAR-internal metrics improve.
- Run paper-like island-local unwrapping.
- Run SBAS/LGR DEM-residual inversion.
- Convert height as `DSM_RDC + residual - 4 m`.
- Select final roof-top height using top-down Grubbs testing, not p95/p90 fallback.
- Do not use the building vector `height` field for fitting, filtering, calibration, filling, selection, or QC.

## Final Active Product

Branch:

- `cleanid_redshift_audited`

Final outputs:

- GeoJSON: `results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson`
- CSV: `results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv`
- Summary: `results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_summary.json`
- Only retained figure: `results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg`

Current statistics:

- Total buildings: 1028
- Strict InSAR solved: 239
- No strict solution: 789
- Reliability: high 168, medium 42, review 29, no_solution 789
- Median top height: 22.55 m
- P05/P95 top height: 3.70 / 48.99 m

## Cleanup Performed

Removed or ignored:

- Temporary caches and `__pycache__`
- Old PDF render caches
- DEM0/no-HGT method-test outputs
- APS/deformation side branches
- p90/p95/likely_top/p95-floor historical outputs
- Full redshift branch that lost 3 strict solutions
- Large unused `.mat`, `.tif`, `.zip`, `.png`, `.bmp` files from the copied original-code area
- Generated figures except the requested 203 SVG

Retained:

- Core code
- Config
- Clean vector inputs
- Lightweight final tables/GeoJSON/metadata
- Current LGR height points and island-height CSVs
- Necessary DSM differential interferograms under `work/gamma_sbas/intf_triangular_dsm`
- Current main documentation

Main documentation:

- `docs/current_literature_route_and_optimization_plan.md`
- `code/reproduce_tongji/README.md`
- `agent.md`

## Git State

Repository initialized on branch `main`.

Commits:

- `2f9c9ae Initial curated InSAR height workflow`
- `f0f7eed Limit figure output to topstats SVG`

The working tree was clean after the last optimization.

## Code Changes From Figure Output Optimization

Scripts updated to support empty figure paths:

- `code/reproduce_tongji/estimate_pixel_lgr_building_heights.py`
- `code/reproduce_tongji/aggregate_clean_equal_height_insar_heights.py`
- `code/reproduce_tongji/build_likely_top_height_from_tests.py`

Behavior:

- Empty output path means do not write that figure.
- README commands now generate only:
  `results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg`

## Next Optimization Plan

1. Audit the remaining 29 review buildings.
2. Classify the 789 no-solution buildings by failure cause.
3. Run small, controlled threshold sensitivity tests only after failure classification.
4. Extend audited projection correction only to no-solution buildings with strong SAR evidence.
5. Build LOD1 output from final `height_insar_m`, leaving no-solution buildings empty or separately flagged.
