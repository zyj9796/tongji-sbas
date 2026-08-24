# No-Solution Gate Sensitivity Pretest 2026-07-07

## Scope

This diagnostic pretest evaluates whether small gate changes can make current no-solution buildings reach at least 20 retained LGR pixels. It does not rebuild the active height product and does not use the building-vector `height` field.

Variants:

- `min_pairs10`: target `lgr_valid_pairs_lt_min_after_unwrap`, change `min_pairs 12 -> 10`.
- `rmse150`: target `lgr_rmse_gt_max`, change `RMSE 1.25 -> 1.50 rad`.
- `da045`: target `amplitude_dispersion_pixels_too_few`, change `DA 0.40 -> 0.45`.

## Results

| variant | target buildings | reach >=20 LGR pixels | median final pixels | max final pixels |
|---|---:|---:|---:|---:|
| `da045` | 27 | 0 | 0.0 | 0 |
| `min_pairs10` | 355 | 26 | 0.0 | 39 |
| `rmse150` | 24 | 12 | 20.5 | 46 |


Candidate clean IDs:

- `min_pairs10`: 103, 834, 116, 144, 1003, 121, 626, 205, 448, 138, 209, 694, 177, 856, 939, 986, 92, 136, 141, 220, 323, 807, 165, 880, 200, 455
- `rmse150`: 766, 625, 940, 821, 180, 238, 203, 171, 248, 623, 888, 149
- `da045`: none

## Interpretation

`min_pairs10` has limited but real yield: 26 of 355 valid-pairs/unwrap failures reach the 20-pixel diagnostic threshold. A full product rebuild with `min_pairs=10` should therefore be small and strictly compared against the current baseline for added buildings, lost buildings, review increment, median RMSE, and spatial clustering.

`rmse150` has the highest proportional yield: 12 of 24 RMSE failures reach the 20-pixel threshold. This should be tested after or alongside `min_pairs10`, but accepted buildings need careful residual review because this change directly weakens the model-fit gate.

`da045` has zero yield in this pretest. Relaxing DA alone should not be prioritized for the current 27 DA-limited no-solution buildings.

The 383 `no_sar_roof_island` buildings remain a separate projection/SAR-evidence problem rather than a threshold-sensitivity problem.
