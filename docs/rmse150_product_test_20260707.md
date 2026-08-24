# RMSE=1.50 Product Test 2026-07-07

## Scope

This is a docs-only product-level test of `RMSE 1.25 -> 1.50 rad` with all other active gates unchanged: coherence `>=0.75`, DA `<=0.40`, min pairs `>=12`, Bperp span `>=120 m`, audited mask, paper-like island unwrapping, and top-down Grubbs selection. It does not replace the active result under `results/`.

## Results

Change counts:

| class | buildings |
|---|---:|
| `added_solution` | 12 |
| `reliability_changed` | 25 |
| `retained_solution` | 214 |
| `still_no_solution` | 777 |

Variant reliability counts:

| class | buildings |
|---|---:|
| `high` | 179 |
| `medium` | 45 |
| `no_solution` | 777 |
| `review` | 27 |

Added-solution reliability counts:

| class | buildings |
|---|---:|
| `high` | 8 |
| `medium` | 3 |
| `review` | 1 |

Added clean IDs:

`171, 238, 248, 623, 766, 821, 888, 940, 149, 203, 625, 180`

Retained-building height delta, RMSE=1.50 minus current:

- median: 0.000 m
- P05/P95: 0.000 / 15.780 m

## Interpretation

`RMSE=1.50` increases solved buildings from 239 to 251 with no lost solutions. It adds 12 buildings, fewer than the `min_pairs=10` branch, and lowers final review count to 27. However, the variant shifts the height distribution upward and weakens the model-fit gate, so it should not replace the current product without residual and spatial audits.
