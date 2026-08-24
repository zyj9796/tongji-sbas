#!/usr/bin/env python3
"""Build a temporally closure-consistent local ground phase reference.

Pair phases are first synchronized onto acquisition-date phase nodes at every
closure-stable ground point.  Date phases, rather than independent pair phases,
are spatially interpolated to buildings.  Pair references formed by date-phase
differences therefore preserve all temporal triangle closures by construction.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


RANGE_PIXEL_M = 1.249135
AZIMUTH_PIXEL_M = 1.790613


def wrap(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def qdict(value: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(value, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--ground-pair-root", type=Path, required=True)
    parser.add_argument("--building-pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--closure-rms-max", type=float, default=0.5)
    parser.add_argument("--temporal-fit-rms-max", type=float, default=0.7)
    parser.add_argument("--nearest-ground-count", type=int, default=24)
    parser.add_argument("--synchronization-iterations", type=int, default=40)
    args = parser.parse_args()

    pair_paths = sorted(args.ground_pair_root.glob("*.npz"))
    names = [path.stem for path in pair_paths]
    pairs_text = [tuple(name.split("_")) for name in names]
    dates = sorted({date for pair in pairs_text for date in pair})
    date_index = {date: index for index, date in enumerate(dates)}
    pairs = [(date_index[a], date_index[b]) for a, b in pairs_text]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    phase = np.stack([np.load(path)["wrapped_phase_rad"] for path in pair_paths]).astype(np.float64)

    triangles: list[tuple[int, int, int]] = []
    for a in range(len(dates)):
        for b in range(a + 1, len(dates)):
            for c in range(b + 1, len(dates)):
                if (a, b) in pair_index and (b, c) in pair_index and (a, c) in pair_index:
                    triangles.append((pair_index[(a, b)], pair_index[(b, c)], pair_index[(a, c)]))
    closure = np.stack([wrap(phase[ab] + phase[bc] - phase[ac]) for ab, bc, ac in triangles])
    closure_rms = np.sqrt(np.mean(closure**2, axis=0))
    closure_stable = closure_rms <= args.closure_rms_max

    # Initialize date phases by a breadth-first spanning tree rooted at the
    # reference acquisition.  Observation convention is phi_ab=theta_a-theta_b.
    date_phase = np.full((len(dates), phase.shape[1]), np.nan, dtype=np.float64)
    date_phase[0] = 0.0
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in dates]
    for edge, (a, b) in enumerate(pairs):
        adjacency[a].append((b, edge, -1))
        adjacency[b].append((a, edge, +1))
    visited = {0}
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for other, edge, sign in adjacency[node]:
            if other in visited:
                continue
            date_phase[other] = wrap(date_phase[node] + sign * phase[edge])
            visited.add(other)
            queue.append(other)
    if len(visited) != len(dates):
        raise RuntimeError("interferogram network is disconnected")

    # Circular coordinate descent gives the equal-weight angular least-squares
    # solution while keeping the first date as a common zero gauge.
    for _ in range(args.synchronization_iterations):
        for node in range(1, len(dates)):
            phasor = np.zeros(phase.shape[1], dtype=np.complex128)
            for other, edge, sign in adjacency[node]:
                if sign == -1:  # node is primary: theta_node=phi+theta_other
                    estimate = phase[edge] + date_phase[other]
                else:  # node is secondary: theta_node=theta_other-phi
                    estimate = date_phase[other] - phase[edge]
                phasor += np.exp(1j * estimate)
            date_phase[node] = np.angle(phasor)

    residual = np.stack(
        [wrap(phase[edge] - (date_phase[a] - date_phase[b])) for edge, (a, b) in enumerate(pairs)]
    )
    temporal_fit_rms = np.sqrt(np.mean(residual**2, axis=0))
    stable = closure_stable & (temporal_fit_rms <= args.temporal_fit_rms_max)
    if stable.sum() < 100:
        raise RuntimeError("too few temporally synchronized stable ground points")

    ground = np.load(args.ground_points)
    ground_xy = np.column_stack((ground["col"] * RANGE_PIXEL_M, ground["row"] * AZIMUTH_PIXEL_M))[stable]
    synchronized = date_phase[:, stable]
    stable_closure = closure_rms[stable]
    stable_fit = temporal_fit_rms[stable]
    first_building = np.load(args.building_pair_root / f"{names[0]}.npz")
    building_xy = np.column_stack(
        (first_building["col"] * RANGE_PIXEL_M, first_building["row"] * AZIMUTH_PIXEL_M)
    )
    tree = cKDTree(ground_xy)
    distance, neighbour = tree.query(building_xy, k=args.nearest_ground_count)
    weights = 1.0 / np.maximum(distance, 1.0)
    weights /= np.maximum(stable_closure[neighbour], 0.02)
    weights /= np.maximum(stable_fit[neighbour], 0.05)

    local_date_phase = np.zeros((len(dates), len(building_xy)), dtype=np.float64)
    local_date_resultant = np.ones_like(local_date_phase)
    for date in range(1, len(dates)):
        neighbour_phase = synchronized[date][neighbour]
        phasor = np.sum(weights * np.exp(1j * neighbour_phase), axis=1)
        local_date_phase[date] = np.angle(phasor)
        local_date_resultant[date] = np.abs(phasor) / np.sum(weights, axis=1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    pair_resultants = []
    for edge, (a, b) in enumerate(pairs):
        building = np.load(args.building_pair_root / f"{names[edge]}.npz")
        local_reference = wrap(local_date_phase[a] - local_date_phase[b])
        referenced = wrap(
            building["filtered_wrapped_phase_rad"].astype(np.float64) - local_reference
        )
        pair_resultant = np.minimum(local_date_resultant[a], local_date_resultant[b])
        np.savez_compressed(
            args.output_root / f"{names[edge]}.npz",
            row=building["row"],
            col=building["col"],
            local_ground_reference_phase_rad=local_reference.astype(np.float32),
            local_ground_reference_resultant=pair_resultant.astype(np.float32),
            ground_referenced_filtered_wrapped_phase_rad=referenced.astype(np.float32),
        )
        pair_resultants.append(pair_resultant)

    pair_resultants_array = np.stack(pair_resultants)
    summary = {
        "method": "ground date-phase angular synchronization followed by local circular interpolation",
        "date_count": len(dates),
        "pair_count": len(pairs),
        "temporal_triangle_count": len(triangles),
        "all_ground_point_count": int(phase.shape[1]),
        "closure_stable_ground_point_count": int(closure_stable.sum()),
        "temporally_synchronized_ground_point_count": int(stable.sum()),
        "closure_rms_max_rad": args.closure_rms_max,
        "temporal_fit_rms_max_rad": args.temporal_fit_rms_max,
        "temporal_fit_rms_rad_quantiles_all": qdict(temporal_fit_rms),
        "nearest_ground_count": args.nearest_ground_count,
        "nearest_ground_distance_m_quantiles": qdict(distance),
        "local_date_reference_resultant_quantiles": qdict(local_date_resultant[1:]),
        "pair_reference_resultant_quantiles": qdict(pair_resultants_array),
        "policy": "ground defines phase zero only; no ground or Floor value enters building-height aggregation",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
