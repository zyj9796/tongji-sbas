# Agent Current State

Last updated: 2026-07-06

The workspace has been cleaned to the current literature-aligned InSAR building-height route.

Main documentation:

- `docs/current_literature_route_and_optimization_plan.md`
- `code/reproduce_tongji/README.md`

Current active branch:

- `cleanid_redshift_audited`
- clean equal-height building vector
- roof-only SAR mask
- clean-ID split islands
- audited SAR-amplitude/edge projection correction
- paper-like island-local unwrapping
- SBAS/LGR DEM-residual inversion
- top-down Grubbs final roof-top height selection

Final product:

- GeoJSON: `results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson`
- CSV: `results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv`
- Summary: `results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_summary.json`

Current statistics:

- Buildings total: 1028
- Strict InSAR solved: 239
- No strict solution: 789
- Reliability: high 168, medium 42, review 29, no_solution 789
- Median top height: 22.55 m
- P05/P95 top height: 3.70 / 48.99 m

Important rule:

- The building vector `height` field is not used for fitting, filtering, calibration, selection, filling, or QC.
