#!/usr/bin/env python3
"""Prune a redundant SBAS network using coherence under graph constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


def metrics(graph: nx.Graph) -> dict[str, object]:
    degree = dict(graph.degree())
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected": nx.is_connected(graph),
        "edge_connectivity": nx.edge_connectivity(graph) if nx.is_connected(graph) else 0,
        "bridges": len(list(nx.bridges(graph))) if nx.is_connected(graph) else None,
        "degree_min": min(degree.values()),
        "degree_max": max(degree.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--target-pairs", type=int, default=48)
    parser.add_argument("--min-edge-connectivity", type=int, default=2)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    network = pd.read_csv(args.network, dtype={"master": str, "slave": str})
    quality = pd.read_csv(args.quality, dtype={"master": str, "slave": str})
    table = network.drop(columns=["mean_cc"], errors="ignore").merge(
        quality, on=["master", "slave"], how="left"
    )
    graph = nx.Graph()
    graph.add_edges_from(zip(table["master"], table["slave"]))
    initial = metrics(graph)
    removed: list[dict[str, object]] = []

    # Reconsider the worst observations first. Long temporal intervals break
    # ties because they are more exposed to temporal decorrelation.
    order = table.sort_values(
        ["mean_cc", "fraction_cc_ge_055", "dt_days"],
        ascending=[True, True, False],
    )
    for row in order.itertuples(index=False):
        if graph.number_of_edges() <= args.target_pairs:
            break
        edge = (str(row.master), str(row.slave))
        graph.remove_edge(*edge)
        trial = metrics(graph)
        if (
            bool(trial["connected"])
            and int(trial["edge_connectivity"]) >= args.min_edge_connectivity
            and int(trial["degree_min"]) >= args.min_degree
        ):
            removed.append(
                {
                    "master": edge[0],
                    "slave": edge[1],
                    "mean_cc": float(row.mean_cc),
                    "dt_days": int(row.dt_days),
                }
            )
        else:
            graph.add_edge(*edge)

    keep = {(str(a), str(b)) for a, b in graph.edges()}
    keep |= {(b, a) for a, b in keep}
    output = table[
        [(str(r.master), str(r.slave)) in keep for r in table.itertuples(index=False)]
    ].copy().sort_values(["master", "slave"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    payload = {
        "policy": "remove lowest-coherence pairs while preserving graph constraints; no height used",
        "target_pairs": args.target_pairs,
        "selected_pairs": int(len(output)),
        "initial_graph": initial,
        "selected_graph": metrics(graph),
        "removed_pairs": removed,
        "output": str(path),
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
