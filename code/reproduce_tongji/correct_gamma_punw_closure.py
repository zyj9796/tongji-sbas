#!/usr/bin/env python3
"""Correct redundant-edge integer ambiguities using GAMMA phase closure.

A maximum-coherence spanning tree is kept unchanged. For each non-tree edge,
the integer number of 2-pi cycles required to agree with its tree path is added
point by point. This enforces network closure without changing bridge edges or
using vector building heights.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--point-metadata", default="work/gamma_native_ipta_sbas/ipta/point_metadata.csv")
    p.add_argument("--punw", default="work/gamma_native_ipta_sbas/ipta/punw")
    p.add_argument("--pmask", default="work/gamma_native_ipta_sbas/ipta/pmask")
    p.add_argument("--pcc", default="work/gamma_building_globalunw_gamma100_full/pcc")
    p.add_argument("--output-punw", default="work/gamma_native_ipta_sbas/ipta/punw_closure_corrected")
    p.add_argument("--output-cycles", default="work/gamma_native_ipta_sbas/ipta/punw_closure_correction_cycles.npy")
    p.add_argument("--summary", default="results/metadata/gamma_punw_closure_correction_summary.json")
    return p.parse_args()


def canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs_csv, dtype={"master": str, "slave": str}).reset_index(drop=True)
    metadata = pd.read_csv(args.point_metadata).sort_values("point_index").reset_index(drop=True)
    npoints, npairs = len(metadata), len(pairs)
    punw = np.memmap(args.punw, dtype=">f4", mode="r", shape=(npairs, npoints))
    pcc = np.memmap(args.pcc, dtype=">f4", mode="r", shape=(npairs, npoints))
    pmask = np.fromfile(args.pmask, dtype=np.uint8)[:npoints] > 0
    ground = metadata["point_class"].eq("ground").to_numpy() & pmask
    reference_rows = metadata.index[metadata["is_phase_reference"].astype(str).str.lower().eq("true")].to_numpy()
    if len(reference_rows) != 1:
        raise ValueError("exactly one phase reference point is required")
    reference = int(reference_rows[0])

    edge_to_record: dict[tuple[str, str], int] = {}
    graph = nx.Graph()
    pair_weights: list[float] = []
    for record, row in enumerate(pairs.itertuples(index=False)):
        edge = canonical(str(row.master), str(row.slave))
        edge_to_record[edge] = record
        values = np.asarray(pcc[record, ground], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        weight = float(np.nanmedian(values)) if len(values) else 0.0
        pair_weights.append(weight)
        graph.add_edge(edge[0], edge[1], weight=weight, record=record)
    if not nx.is_connected(graph):
        raise ValueError("pair graph must be connected")
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    tree_edges = {canonical(a, b) for a, b in tree.edges}
    chord_edges = [edge for edge in edge_to_record if edge not in tree_edges]

    relative = np.empty((npairs, npoints), dtype=np.float64)
    for record in range(npairs):
        relative[record] = np.asarray(punw[record], dtype=np.float64) - float(punw[record, reference])
    corrected_relative = relative.copy()
    correction_cycles = np.zeros((npairs, npoints), dtype=np.int16)

    def phase_along(a: str, b: str) -> np.ndarray:
        edge = canonical(a, b)
        values = corrected_relative[edge_to_record[edge]]
        return values if edge == (a, b) else -values

    for u, v in chord_edges:
        path = nx.shortest_path(tree, u, v)
        path_phase = np.zeros(npoints, dtype=np.float64)
        for a, b in zip(path[:-1], path[1:]):
            path_phase += phase_along(a, b)
        record = edge_to_record[(u, v)]
        cycles = np.rint((path_phase - corrected_relative[record]) / (2.0 * np.pi)).astype(np.int16)
        cycles[~pmask] = 0
        correction_cycles[record] = cycles
        corrected_relative[record] += 2.0 * np.pi * cycles

    output = Path(args.output_punw)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_stack = np.memmap(output, dtype=">f4", mode="w+", shape=(npairs, npoints))
    for record in range(npairs):
        out_stack[record] = (corrected_relative[record] + float(punw[record, reference])).astype(np.float32)
    out_stack.flush()
    Path(args.output_cycles).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_cycles, correction_cycles)

    triangles = [
        nodes for nodes in itertools.combinations(sorted(graph.nodes), 3)
        if graph.has_edge(nodes[0], nodes[1]) and graph.has_edge(nodes[1], nodes[2]) and graph.has_edge(nodes[0], nodes[2])
    ]
    bad_after = np.zeros(npoints, dtype=np.int16)
    for a, b, c in triangles:
        closure = phase_along(a, b) + phase_along(b, c) - phase_along(a, c)
        bad_after += np.rint(closure / (2.0 * np.pi)).astype(np.int16) != 0

    changed = np.any(correction_cycles != 0, axis=0) & pmask
    summary = {
        "method": "maximum-stable-ground-coherence spanning tree fixed; integer 2pi corrections applied only to redundant chord edges",
        "prior_height_used": False,
        "reference_point": reference,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "tree_edges": len(tree_edges),
        "chord_edges": len(chord_edges),
        "bridges_unchanged": [list(edge) for edge in nx.bridges(graph)],
        "triangles": len(triangles),
        "accepted_points": int(pmask.sum()),
        "points_with_any_cycle_correction": int(changed.sum()),
        "fraction_changed": float(changed.sum() / pmask.sum()),
        "nonzero_cycle_corrections": int(np.count_nonzero(correction_cycles[:, pmask])),
        "max_abs_cycle_correction": int(np.max(np.abs(correction_cycles[:, pmask]))),
        "points_with_nonzero_triangle_integer_after": int(np.sum((bad_after > 0) & pmask)),
        "pair_weights_stable_ground_median_coherence": pair_weights,
        "tree_edge_records": [edge_to_record[edge] + 1 for edge in sorted(tree_edges)],
        "chord_edge_records": [edge_to_record[edge] + 1 for edge in chord_edges],
        "output_punw": str(output),
        "output_cycles": args.output_cycles,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
