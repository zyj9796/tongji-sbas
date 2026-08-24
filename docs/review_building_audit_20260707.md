# Review Building Audit 2026-07-07

## Scope

This audit continues priority 1 of `current_literature_route_and_optimization_plan.md`: classify the remaining 29 review buildings using only InSAR-internal metrics and SAR amplitude/edge evidence. The building-vector `height` field is not used for fitting, filtering, calibration, selection, filling, or QC.

Inputs:

- `results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv`
- `work/projection/cleanid_split_red_building_mask_shift_metrics_audited.csv`
- `results/diagnostics/cleanid_redshift_audited_red_buildings/diagnostic_manifest.csv`

Outputs:

- `docs/review_building_audit_20260707.csv`
- `docs/review_building_audit_20260707.svg`
- `docs/review_building_audit_20260707_summary.json`

## Findings

All 29 review buildings are in `max_reject_use_top_down_grubbs_many_removed`: the highest roof-height hypothesis is rejected and top-down Grubbs removes more than two upper-tail pixels before retaining a top value.

Audit class counts:

| audit_class | buildings |
|---|---:|
| `correctable_candidate` | 3 |
| `keep_review` | 26 |

Primary reason counts:

| primary_reason | buildings |
|---|---:|
| `multi_scatterer_or_upper_tail_mixture` | 10 |
| `lgr_model_rmse_unstable` | 6 |
| `near_threshold_quality` | 5 |
| `top_grubbs_tail_instability` | 4 |
| `local_projection_bias_candidate` | 3 |
| `post_shift_upper_tail_mixture` | 1 |

Key groups:

- Correctable projection-bias candidates: 344, 576, 600
- Candidate no-solution downgrades under stricter internal support rules: none
- Buildings whose audited shift was already accepted but still remain review: 555, 824, 843

## Interpretation

The dominant issue is not global low coherence or DA failure. Review buildings have median coherence and amplitude-dispersion values comparable to high/medium buildings, but their upper-tail height distributions are unstable. This points mainly to mixed roof/facade scatterers, local bright-ridge leakage, or unresolved projection/local-mask mismatch.

Projection expansion should be narrow. Only buildings flagged as `correctable_candidate` should enter another local shift experiment first, and acceptance should require improved strict InSAR reliability, not amplitude score alone.

The `downgrade_candidate` class is conservative: it marks review buildings with weak internal support in several dimensions. None of the 29 buildings met that stricter downgrade rule in this run, so no review building should be converted to `no_solution` automatically.

## Recommended Next Step

Before threshold sensitivity tests, continue priority 2 by decomposing the 789 no-solution buildings by exact failure cause: no roof mask, too few mask pixels, coherence, DA, valid-pair count, Bperp span, RMSE, or unwrapping/LGR failure.
