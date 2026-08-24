# No-Solution Failure Audit 2026-07-07

## Scope

This continues priority 2 of `current_literature_route_and_optimization_plan.md`: decompose the 789 buildings with `no_solution` using the current saved mask, island, LGR, point outputs, and the 2026-07-07 LGR gate diagnostic rerun. The building-vector `height` field is not used.

Outputs:

- `docs/no_solution_failure_audit_20260707.csv`
- `docs/no_solution_failure_audit_20260707.svg`
- `docs/no_solution_failure_audit_20260707_summary.json`
- `docs/lgr_failure_gate_diagnostics_buildings_20260707.csv`
- `docs/lgr_failure_gate_diagnostics_islands_20260707.csv`
- `docs/lgr_failure_gate_diagnostics_20260707_summary.json`

## Refined Counts

| class | buildings |
|---|---:|
| `no_sar_roof_island` | 383 |
| `lgr_valid_pairs_lt_min_after_unwrap` | 355 |
| `amplitude_dispersion_pixels_too_few` | 27 |
| `lgr_rmse_gt_max` | 24 |

## Interpretation

The largest group has no clean SAR roof island in the audited mask: 383 buildings. These should not enter threshold sensitivity first; they need a projection/mask and SAR bright-ridge evidence screen.

The dominant LGR-stage failure is `lgr_valid_pairs_lt_min_after_unwrap`: 355 buildings have too few pixels reaching `min_pairs>=12` after coherence screening and island-local unwrapping. This points to temporal/coherence support, unwrap coverage, and valid-pair threshold as the first LGR-side sensitivity target.

A smaller LGR-stage group is `lgr_rmse_gt_max`: 24 buildings have enough valid-pair/Bperp support but fall below 20 retained pixels after `RMSE<=1.25 rad`. These are the proper candidates for a controlled `RMSE 1.25 -> 1.50` sensitivity test.

Only 27 buildings fail at the DA screen with fewer than 20 pixels after `DA<=0.40`. These are the proper candidates for a controlled `DA 0.40 -> 0.45` test.

No current no-solution building is primarily classified as Bperp-span failure in this diagnostic run.

## Recommended Optimization Order

1. Projection/SAR-evidence screen for the 383 `no_sar_roof_island` buildings.
2. Controlled min-pairs sensitivity for the 355 `lgr_valid_pairs_lt_min_after_unwrap` buildings, starting with `min_pairs 12 -> 10` and reporting added/review/lost buildings.
3. Controlled RMSE sensitivity for the 24 `lgr_rmse_gt_max` buildings, using `RMSE 1.25 -> 1.50` only after checking spatial clustering and residual behavior.
4. Controlled DA sensitivity for the 27 `amplitude_dispersion_pixels_too_few` buildings, using `DA 0.40 -> 0.45` only with SAR evidence checks.
