#!/usr/bin/env python3
"""Reproduce 3x3 opening and pixel-space DBSCAN building-island extraction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def dbscan_pixels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Exact Euclidean DBSCAN for integer raster coordinates without sklearn."""
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, eps, return_length=True)
    core_mask = counts >= min_samples
    core_points = points[core_mask]
    if not len(core_points):
        return np.full(len(points), -1, dtype=np.int32)

    core_tree = cKDTree(core_points)
    pairs = core_tree.query_pairs(eps, output_type="ndarray")
    diagonal = np.column_stack((np.arange(len(core_points)), np.arange(len(core_points))))
    edges = np.vstack((pairs, pairs[:, ::-1], diagonal)) if len(pairs) else diagonal
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(core_points), len(core_points)),
    ).tocsr()
    _, core_labels = connected_components(graph, directed=False)

    # Deterministic labels ordered by the top-left core point.
    minima = []
    for label in np.unique(core_labels):
        component = core_points[core_labels == label]
        minima.append((int(component[:, 0].min()), int(component[:, 1].min()), int(label)))
    relabel = {old: new for new, (_, _, old) in enumerate(sorted(minima), start=1)}
    core_labels = np.array([relabel[int(value)] for value in core_labels], dtype=np.int32)

    labels = np.full(len(points), -1, dtype=np.int32)
    labels[core_mask] = core_labels
    border_indices = np.flatnonzero(~core_mask)
    if len(border_indices):
        distance, nearest = core_tree.query(points[border_indices], k=1, distance_upper_bound=eps)
        valid = np.isfinite(distance)
        labels[border_indices[valid]] = core_labels[nearest[valid]]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid-rdc", type=Path, required=True)
    parser.add_argument("--floor-rdc", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=10000)
    parser.add_argument("--lines", type=int, default=7000)
    parser.add_argument("--floor-min-exclusive", type=float, default=30.0)
    parser.add_argument("--eps", type=float, default=2.5)
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()

    shape = (args.lines, args.width)
    uid = np.memmap(args.uid_rdc, dtype=">f4", mode="r", shape=shape)
    floor = np.memmap(args.floor_rdc, dtype=">f4", mode="r", shape=shape)
    candidate = (uid > 0) & (floor > args.floor_min_exclusive)
    opened = ndimage.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
    points = np.argwhere(opened)
    labels = dbscan_pixels(points, args.eps, args.min_samples)

    args.output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_root / "paper_highrise_island_points.npz",
        row=points[:, 0].astype(np.int32),
        col=points[:, 1].astype(np.int32),
        label=labels,
        building_uid=uid[points[:, 0], points[:, 1]].astype(np.int32),
        floor=floor[points[:, 0], points[:, 1]].astype(np.int16),
    )

    rows = []
    for label in sorted(value for value in np.unique(labels) if value > 0):
        select = labels == label
        island_points = points[select]
        island_uid = uid[island_points[:, 0], island_points[:, 1]].astype(np.int64)
        island_floor = floor[island_points[:, 0], island_points[:, 1]]
        uid_values, uid_counts = np.unique(island_uid, return_counts=True)
        dominant_uid = int(uid_values[np.argmax(uid_counts)])
        rows.append(
            {
                "island_id": int(label),
                "pixel_count": int(select.sum()),
                "row_min": int(island_points[:, 0].min()),
                "row_max": int(island_points[:, 0].max()),
                "col_min": int(island_points[:, 1].min()),
                "col_max": int(island_points[:, 1].max()),
                "building_uid_count": int(len(uid_values)),
                "dominant_building_uid": dominant_uid,
                "floor_median": float(np.median(island_floor)),
                "floor_max": int(island_floor.max()),
            }
        )
    with (args.output_root / "paper_highrise_islands.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    noise_count = int(np.sum(labels < 0))
    summary = {
        "input_all_building_pixels": int(np.sum(uid > 0)),
        "floor_rule": f"Floor > {args.floor_min_exclusive:g}",
        "candidate_pixels": int(candidate.sum()),
        "opened_pixels": int(opened.sum()),
        "opening_structure": [3, 3],
        "dbscan_eps_px": args.eps,
        "dbscan_min_samples": args.min_samples,
        "island_count": len(rows),
        "clustered_pixels": int(np.sum(labels > 0)),
        "noise_pixels": noise_count,
        "policy": "Floor is used only to recover the original-code high-rise target set; no Floor value enters InSAR height inversion.",
    }
    (args.output_root / "paper_highrise_islands_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
