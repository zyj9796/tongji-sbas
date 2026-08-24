#!/usr/bin/env python3
"""Unwrap stable-ground phases, fit robust ramps, and reference building phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay


TWOPI = 2.0 * np.pi


def wrap(values: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * values))


def delaunay_edges(coords: np.ndarray) -> np.ndarray:
    triangles = Delaunay(coords).simplices
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def mst_unwrap(phase: np.ndarray, coords: np.ndarray, quality: np.ndarray) -> np.ndarray:
    edges = delaunay_edges(coords)
    distance = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    edge_quality = np.sqrt(quality[edges[:, 0]] * quality[edges[:, 1]])
    cost = distance / np.maximum(edge_quality, 1e-6)
    graph = coo_matrix(
        (np.r_[cost, cost], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(phase), len(phase)),
    ).tocsr()
    tree = minimum_spanning_tree(graph)
    tree = (tree + tree.T).tocsr()
    root = int(np.argmax(quality))
    result = np.full(len(phase), np.nan, dtype=np.float64)
    result[root] = float(phase[root])
    stack = [root]
    while stack:
        current = stack.pop()
        for neighbour in tree.indices[tree.indptr[current] : tree.indptr[current + 1]]:
            if np.isfinite(result[neighbour]):
                continue
            result[neighbour] = result[current] + float(wrap(phase[neighbour] - phase[current]))
            stack.append(int(neighbour))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("ground-reference MST is disconnected")
    return result


def robust_plane(coords: np.ndarray, unwrapped: np.ndarray, base_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (coords[:, 0] - 2500.0) / 2500.0
    y = (coords[:, 1] - 583.0) / 583.0
    design = np.column_stack((np.ones(len(x)), x, y))
    weight = base_weight.astype(np.float64).copy()
    beta = np.zeros(3)
    for _ in range(20):
        root_w = np.sqrt(np.maximum(weight, 1e-12))
        beta_new = np.linalg.lstsq(design * root_w[:, None], unwrapped * root_w, rcond=None)[0]
        residual = unwrapped - design @ beta_new
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        u = residual / (4.685 * scale)
        bisquare = np.square(1.0 - np.square(u))
        bisquare[np.abs(u) >= 1.0] = 0.0
        weight_new = base_weight * bisquare
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta, weight = beta_new, weight_new
            break
        beta, weight = beta_new, weight_new
    return beta, weight


def evaluate_plane(beta: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    x = (col / 2.0 - 2500.0) / 2500.0
    y = (row / 6.0 - 583.0) / 583.0
    return beta[0] + beta[1] * x + beta[2] * y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--ground-pair-root", type=Path, required=True)
    parser.add_argument("--building-pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    ground = np.load(args.ground_points)
    ground_coords = np.column_stack((ground["mli_col"], ground["mli_row"])).astype(np.float64)
    quality = 1.0 / np.maximum(ground["amplitude_dispersion"].astype(np.float64), 0.02)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, ground_path in enumerate(sorted(args.ground_pair_root.glob("*.npz")), start=1):
        name = ground_path.stem
        print(f"[{index}/48] {name}", flush=True)
        ground_pair = np.load(ground_path)
        building_pair = np.load(args.building_pair_root / f"{name}.npz")
        observed = ground_pair["wrapped_phase_rad"].astype(np.float64)
        unwrapped = mst_unwrap(observed, ground_coords, quality)
        beta, final_weight = robust_plane(ground_coords, unwrapped, quality)
        ground_prediction = (
            beta[0]
            + beta[1] * ((ground_coords[:, 0] - 2500.0) / 2500.0)
            + beta[2] * ((ground_coords[:, 1] - 583.0) / 583.0)
        )
        ground_residual = wrap(observed - ground_prediction)
        building_prediction = evaluate_plane(beta, building_pair["col"], building_pair["row"])
        referenced = wrap(building_pair["filtered_wrapped_phase_rad"].astype(np.float64) - building_prediction)
        output = args.output_root / f"{name}.npz"
        np.savez_compressed(
            output,
            row=building_pair["row"],
            col=building_pair["col"],
            ground_ramp_coefficients=beta,
            ground_referenced_filtered_wrapped_phase_rad=referenced.astype(np.float32),
        )
        item = {
            "pair_name": name,
            "ground_point_count": int(len(observed)),
            "robust_inlier_count": int((final_weight > 0).sum()),
            "plane_coefficients_rad": beta.tolist(),
            "ground_circular_residual_median_rad": float(np.median(ground_residual)),
            "ground_circular_residual_nmad_rad": float(1.4826 * np.median(np.abs(ground_residual - np.median(ground_residual)))),
            "ground_circular_residual_rmse_rad": float(np.sqrt(np.mean(np.square(ground_residual)))),
        }
        results.append(item)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pair_count": len(results),
        "median_ground_residual_nmad_rad": float(np.median([item["ground_circular_residual_nmad_rad"] for item in results])),
        "median_ground_residual_rmse_rad": float(np.median([item["ground_circular_residual_rmse_rad"] for item in results])),
        "minimum_robust_inlier_count": int(min(item["robust_inlier_count"] for item in results)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
