# min_pairs=10 Product Test 2026-07-07

## Scope

This is a docs-only product-level test of `min_pairs 12 -> 10` with all other active gates unchanged: coherence `>=0.75`, DA `<=0.40`, RMSE `<=1.25 rad`, Bperp span `>=120 m`, audited mask, paper-like island unwrapping, and top-down Grubbs selection. It does not replace the active result under `results/`.

## Results

Change counts:

| class | buildings |
|---|---:|
| `added_solution` | 37 |
| `reliability_changed` | 33 |
| `retained_solution` | 206 |
| `still_no_solution` | 752 |

Variant reliability counts:

| class | buildings |
|---|---:|
| `high` | 182 |
| `medium` | 57 |
| `no_solution` | 752 |
| `review` | 37 |

Added-solution reliability counts:

| class | buildings |
|---|---:|
| `high` | 27 |
| `medium` | 9 |
| `review` | 1 |

Added clean IDs:

`92, 103, 116, 121, 136, 138, 144, 149, 165, 177, 180, 200, 205, 220, 248, 346, 448, 625, 626, 766, 807, 834, 844, 856, 880, 940, 1003, 141, 209, 323, 455, 623, 694, 820, 939, 986, 203`

Retained-building height delta, min_pairs10 minus current:

- median: 0.000 m
- P05/P95: 0.000 / 8.897 m

## Interpretation

`min_pairs=10` increases solved buildings from 239 to 276 with no lost solutions. It adds 37 buildings, but also increases final review count relative to the current product. This is useful as a candidate branch, not an automatic replacement.

The current strict product should remain the main result unless the added buildings pass the same review audit: internal LGR residual checks, top-tail stability, spatial clustering, and SAR/mask consistency. The next controlled test should run `RMSE 1.25 -> 1.50` separately because its pretest yield is high but it weakens the model-fit gate.
