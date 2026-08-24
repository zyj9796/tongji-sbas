# Threshold Variant Decision 2026-07-07

## Scope

This compares the two completed docs-only product-level threshold tests against the current strict product. The building-vector `height` field is not used.

## Added Buildings

| group | count | clean IDs |
|---|---:|---|
| `min_pairs10` added | 37 | `92, 103, 116, 121, 136, 138, 141, 144, 149, 165, 177, 180, 200, 203, 205, 209, 220, 248, 323, 346, 448, 455, 623, 625, 626, 694, 766, 807, 820, 834, 844, 856, 880, 939, 940, 986, 1003` |
| `rmse150` added | 12 | `149, 171, 180, 203, 238, 248, 623, 625, 766, 821, 888, 940` |
| overlap | 8 | `149, 180, 203, 248, 623, 625, 766, 940` |
| min_pairs10 only | 29 | `92, 103, 116, 121, 136, 138, 141, 144, 165, 177, 200, 205, 209, 220, 323, 346, 448, 455, 626, 694, 807, 820, 834, 844, 856, 880, 939, 986, 1003` |
| rmse150 only | 4 | `171, 238, 821, 888` |

## Large Retained Height Changes

Buildings already solved in the current product but changing by more than 8 m under a threshold variant are listed in `docs/threshold_variant_large_height_change_audit_20260707.csv`.

| variant | large-change retained buildings |
|---|---:|
| `minpairs10` | 16 |
| `rmse150` | 21 |


## Decision

Keep the current strict product as the main result.

Use `min_pairs10` as the higher-yield candidate branch: it adds more buildings and does not lose current solutions, but it raises review count and needs targeted audit of the 37 added buildings plus retained buildings with large height changes.

Use `rmse150` only as a cautious residual-audit branch: it adds fewer buildings and has lower review count, but it weakens the model-fit gate and shows larger upper-tail height changes among retained buildings.

Do not prioritize `DA 0.40 -> 0.45`; the gate pretest produced zero buildings reaching 20 retained LGR pixels.
