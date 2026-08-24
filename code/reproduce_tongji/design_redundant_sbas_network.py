#!/usr/bin/env python3
"""Design a redundant SBAS network while preserving the frozen 37-pair seed.

The optimization is independent of building-height priors.  Candidate edges are
ranked first by graph robustness (edge connectivity, bridge removal, minimum
degree), then by an InSAR-only coherence proxy and temporal/baseline diversity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def graph_metrics(graph: nx.Graph) -> dict[str, object]:
    connected = nx.is_connected(graph)
    degrees = np.asarray([value for _, value in graph.degree()], dtype=float)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected": connected,
        "components": nx.number_connected_components(graph),
        "edge_connectivity": nx.edge_connectivity(graph) if connected else 0,
        "bridges": len(list(nx.bridges(graph))) if connected else None,
        "degree_min": int(degrees.min()) if len(degrees) else 0,
        "degree_median": float(np.median(degrees)) if len(degrees) else 0.0,
        "degree_max": int(degrees.max()) if len(degrees) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--quality", default="work/baselines/gamma_native_full71_pair_quality.csv")
    parser.add_argument("--target-pairs", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    seed = pd.read_csv(args.seed, dtype={"master": str, "slave": str})
    candidates = pd.read_csv(args.candidates, dtype={"master": str, "slave": str})
    nodes = sorted(set(seed["master"]) | set(seed["slave"]))
    candidates = candidates[
        candidates["master"].isin(nodes) & candidates["slave"].isin(nodes)
    ].drop_duplicates(["master", "slave"]).copy()
    candidates["abs_bperp_m"] = candidates["bperp_m"].abs()

    quality_path = Path(args.quality)
    if quality_path.exists():
        quality = pd.read_csv(quality_path, dtype={"master": str, "slave": str})
        candidates = candidates.merge(
            quality[["master", "slave", "mean_cc"]],
            on=["master", "slave"],
            how="left",
        )
    else:
        candidates["mean_cc"] = np.nan
    # Conservative quality proxy for pairs not yet formed. It is used only as
    # a tie-break after graph robustness, never as a height-dependent score.
    proxy = 0.56 - 0.0011 * candidates["dt_days"] - 0.00018 * candidates["abs_bperp_m"]
    candidates["quality_score"] = candidates["mean_cc"].fillna(proxy).clip(0.0, 1.0)

    selected = {(str(r.master), str(r.slave)): "frozen_seed" for r in seed.itertuples(index=False)}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(selected)
    seed_metrics = graph_metrics(graph)

    while len(selected) < min(args.target_pairs, len(candidates)):
        best_key: tuple[object, ...] | None = None
        best_pair: tuple[str, str] | None = None
        for row in candidates.itertuples(index=False):
            pair = (str(row.master), str(row.slave))
            if pair in selected:
                continue
            trial = graph.copy()
            trial.add_edge(*pair)
            metrics = graph_metrics(trial)
            endpoint_degree = min(graph.degree(pair[0]), graph.degree(pair[1]))
            # Lexicographic objective: topology dominates signal-quality proxy.
            key = (
                int(metrics["edge_connectivity"]),
                -int(metrics["bridges"] or 0),
                int(metrics["degree_min"]),
                -endpoint_degree,
                float(row.quality_score),
                -abs(float(row.abs_bperp_m) - 180.0),
                -float(row.dt_days),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = pair
        if best_pair is None:
            break
        selected[best_pair] = "redundancy_addition"
        graph.add_edge(*best_pair)

    output = candidates[
        [(str(r.master), str(r.slave)) in selected for r in candidates.itertuples(index=False)]
    ].copy()
    output["selection_class"] = [
        selected[(str(r.master), str(r.slave))] for r in output.itertuples(index=False)
    ]
    output = output.sort_values(["master", "slave"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    summary = {
        "policy": "height-prior-independent graph robustness, then coherence/time/baseline tie-break",
        "candidate_pairs": int(len(candidates)),
        "seed_pairs": int(len(seed)),
        "target_pairs": int(args.target_pairs),
        "selected_pairs": int(len(output)),
        "seed_graph": seed_metrics,
        "selected_graph": graph_metrics(graph),
        "new_pairs": int((output["selection_class"] == "redundancy_addition").sum()),
        "output": str(output_path),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
