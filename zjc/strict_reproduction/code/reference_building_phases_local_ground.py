#!/usr/bin/env python3
"""Reference each building pixel to nearby closure-stable ground scatterers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def wrap(values: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--ground-pair-root", type=Path, required=True)
    parser.add_argument("--building-pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--closure-rms-max", type=float, default=0.5)
    parser.add_argument("--nearest-ground-count", type=int, default=32)
    parser.add_argument("--ground-consistency-count", type=int, default=16)
    args = parser.parse_args()

    pair_paths = sorted(args.ground_pair_root.glob("*.npz"))
    names = [path.stem for path in pair_paths]
    pairs = [tuple(name.split("_")) for name in names]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    phase = np.stack([np.load(path)["wrapped_phase_rad"] for path in pair_paths]).astype(np.float64)
    dates = sorted({date for pair in pairs for date in pair})
    triangles = []
    for ia, first in enumerate(dates):
        for ib in range(ia + 1, len(dates)):
            second = dates[ib]
            for ic in range(ib + 1, len(dates)):
                third = dates[ic]
                if (first, second) in pair_index and (second, third) in pair_index and (first, third) in pair_index:
                    triangles.append(
                        (pair_index[(first, second)], pair_index[(second, third)], pair_index[(first, third)])
                    )
    closure = np.stack([wrap(phase[a] + phase[b] - phase[c]) for a, b, c in triangles])
    closure_rms = np.sqrt(np.mean(np.square(closure), axis=0))
    stable = closure_rms <= args.closure_rms_max
    if stable.sum() < 50:
        raise RuntimeError("too few closure-stable ground reference points")

    ground = np.load(args.ground_points)
    ground_xy_m = np.column_stack((ground["col"] * 1.249135, ground["row"] * 1.790613))[stable]
    stable_phase = phase[:, stable]
    stable_closure = closure_rms[stable]
    first_building = np.load(args.building_pair_root / f"{names[0]}.npz")
    building_xy_m = np.column_stack(
        (first_building["col"] * 1.249135, first_building["row"] * 1.790613)
    )
    tree = cKDTree(ground_xy_m)
    distance, neighbour = tree.query(building_xy_m, k=args.nearest_ground_count)
    base_weight = 1.0 / np.maximum(distance, 1.0)
    base_weight /= np.maximum(stable_closure[neighbour], 0.02)
    # Estimate pair-specific reliability of every ground point from its local
    # circular agreement.  The first neighbour is the point itself and is
    # excluded so that a noisy sample cannot certify itself.
    ground_distance, ground_neighbour = tree.query(
        ground_xy_m, k=args.ground_consistency_count + 1
    )
    ground_distance = ground_distance[:, 1:]
    ground_neighbour = ground_neighbour[:, 1:]
    ground_base_weight = 1.0 / np.maximum(ground_distance, 1.0)
    ground_base_weight /= np.maximum(stable_closure[ground_neighbour], 0.02)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pair_results = []
    resultant_values = []
    raw_resultant_values = []
    ground_reliability_values = []
    for index, name in enumerate(names):
        building = np.load(args.building_pair_root / f"{name}.npz")
        local_ground_phasor = np.sum(
            ground_base_weight * np.exp(1j * stable_phase[index][ground_neighbour]),
            axis=1,
        )
        ground_reliability = np.abs(local_ground_phasor) / np.sum(ground_base_weight, axis=1)
        neighbour_phase = stable_phase[index][neighbour]
        # Spatial/closure weights are augmented by independently estimated
        # pair-specific local agreement.  A short circular-Huber iteration
        # limits isolated phase outliers without inventing an absolute phase.
        weights = base_weight * np.maximum(ground_reliability[neighbour], 0.02) ** 2
        raw_phasor = np.sum(weights * np.exp(1j * neighbour_phase), axis=1)
        raw_total_weight = np.sum(weights, axis=1)
        raw_resultant = np.abs(raw_phasor) / raw_total_weight
        local_reference = np.angle(raw_phasor)
        robust_weights = weights
        for _ in range(3):
            residual = np.abs(wrap(neighbour_phase - local_reference[:, None]))
            robust_weights = weights * np.minimum(1.0, 1.0 / np.maximum(residual, 1.0e-6))
            phasor = np.sum(robust_weights * np.exp(1j * neighbour_phase), axis=1)
            local_reference = np.angle(phasor)
        total_weight = np.sum(robust_weights, axis=1)
        resultant = np.abs(phasor) / total_weight
        referenced = wrap(
            building["filtered_wrapped_phase_rad"].astype(np.float64) - local_reference
        )
        np.savez_compressed(
            args.output_root / f"{name}.npz",
            row=building["row"],
            col=building["col"],
            local_ground_reference_phase_rad=local_reference.astype(np.float32),
            local_ground_reference_raw_resultant=raw_resultant.astype(np.float32),
            local_ground_reference_resultant=resultant.astype(np.float32),
            ground_referenced_filtered_wrapped_phase_rad=referenced.astype(np.float32),
        )
        resultant_values.append(resultant)
        raw_resultant_values.append(raw_resultant)
        ground_reliability_values.append(ground_reliability)
        pair_results.append(
            {
                "pair_name": name,
                "local_reference_resultant_median": float(np.median(resultant)),
                "local_reference_resultant_p05": float(np.quantile(resultant, 0.05)),
                "local_reference_raw_resultant_median": float(np.median(raw_resultant)),
                "ground_local_reliability_median": float(np.median(ground_reliability)),
            }
        )

    all_resultant = np.stack(resultant_values)
    all_raw_resultant = np.stack(raw_resultant_values)
    all_ground_reliability = np.stack(ground_reliability_values)
    summary = {
        "method": "closure-stable local ground circular reference",
        "triangle_closure_count": len(triangles),
        "closure_rms_threshold_rad": args.closure_rms_max,
        "all_ground_point_count": int(len(closure_rms)),
        "closure_stable_ground_point_count": int(stable.sum()),
        "nearest_ground_count": args.nearest_ground_count,
        "ground_consistency_count": args.ground_consistency_count,
        "nearest_ground_distance_m_quantiles": {
            str(q): float(np.quantile(distance, q)) for q in (0.05, 0.5, 0.95, 1.0)
        },
        "local_reference_resultant_quantiles": {
            str(q): float(np.quantile(all_resultant, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "local_reference_raw_resultant_quantiles": {
            str(q): float(np.quantile(all_raw_resultant, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "ground_local_reliability_quantiles": {
            str(q): float(np.quantile(all_ground_reliability, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "pair_results": pair_results,
        "policy": "ground phases define only local phase zero; ground points do not enter building-height aggregation",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "pair_results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
