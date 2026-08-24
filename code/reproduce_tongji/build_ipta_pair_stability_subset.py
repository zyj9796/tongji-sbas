#!/usr/bin/env python3
"""Build a frozen, graph-robust IPTA pair subset for stability validation.

Pairs are removed from lowest to highest network quality only when the retained
acquisition graph stays at least two-edge-connected and every date keeps degree
two or greater.  No building height or inversion result is read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ipta", required=True)
    p.add_argument("--pairs-csv", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--target-pairs", type=int, default=36)
    p.add_argument("--summary", required=True)
    return p.parse_args()


def graph_metrics(pairs: pd.DataFrame) -> dict[str, int | bool]:
    graph = nx.Graph()
    graph.add_edges_from(pairs[["master", "slave"]].itertuples(index=False, name=None))
    connected = nx.is_connected(graph)
    return {
        "connected": connected,
        "edge_connectivity": nx.edge_connectivity(graph) if connected else 0,
        "bridges": len(list(nx.bridges(graph))) if connected else -1,
        "minimum_degree": min(dict(graph.degree()).values()) if graph.number_of_nodes() else 0,
    }


def subset_records(source: Path, destination: Path, indices: list[int], record_bytes: int) -> None:
    raw = source.read_bytes()
    if len(raw) % record_bytes:
        raise ValueError(f"{source} is not an integer number of {record_bytes}-byte records")
    records = len(raw) // record_bytes
    if max(indices) >= records:
        raise ValueError(f"{source} has {records} records but subset requests {max(indices)}")
    destination.write_bytes(b"".join(raw[i * record_bytes : (i + 1) * record_bytes] for i in indices))


def main() -> None:
    args = parse_args()
    source = Path(args.source_ipta)
    output_root = Path(args.output_root)
    output_ipta = output_root / "ipta"
    output_ipta.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv, dtype={"master": str, "slave": str})
    pairs = pairs.drop_duplicates(["master", "slave"]).reset_index(drop=True)
    keep = set(range(len(pairs)))
    quality_columns = [column for column in ["quality_score", "mean_cc"] if column in pairs]
    if not quality_columns:
        raise ValueError("pairs CSV requires quality_score or mean_cc")
    for index in pairs.sort_values(quality_columns, kind="stable").index:
        if len(keep) <= args.target_pairs:
            break
        trial_indices = sorted(keep - {int(index)})
        metrics = graph_metrics(pairs.iloc[trial_indices])
        if metrics["connected"] and metrics["edge_connectivity"] >= 2 and metrics["minimum_degree"] >= 2:
            keep.remove(int(index))
    keep_indices = sorted(keep)
    selected = pairs.iloc[keep_indices].copy().reset_index(drop=True)
    selected["original_pair_record"] = np.asarray(keep_indices, dtype=int) + 1
    metrics = graph_metrics(selected)
    if len(selected) != args.target_pairs:
        raise RuntimeError(f"could retain only {len(selected)} pairs, target was {args.target_pairs}")

    selected_csv = output_root / "pairs_stability_subset.csv"
    selected.to_csv(selected_csv, index=False)

    original_itab = [line.split() for line in (source / "pairs.itab").read_text().splitlines() if line.strip()]
    if len(original_itab) != len(pairs):
        raise ValueError("pairs.itab record count differs from pairs CSV")
    (output_ipta / "pairs.itab").write_text(
        "\n".join(
            f"{original_itab[index][0]} {original_itab[index][1]} {new_record} 1"
            for new_record, index in enumerate(keep_indices, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = pd.read_csv(source / "point_metadata.csv")
    npoints = len(metadata)
    metadata.to_csv(output_ipta / "point_metadata.csv", index=False)
    for name in ["tongji.plist", "pmask", "pSLC_par"]:
        shutil.copy2(source / name, output_ipta / name)
    subset_records(source / "pdiff", output_ipta / "pdiff", keep_indices, npoints * 8)
    subset_records(source / "punw", output_ipta / "punw", keep_indices, npoints * 4)
    pbase_record_bytes = (source / "pbase").stat().st_size // len(pairs)
    subset_records(source / "pbase", output_ipta / "pbase", keep_indices, pbase_record_bytes)
    sensitivity = np.load(source / "phase_height_sensitivity_rad_per_m.npy", mmap_mode="r")
    np.save(output_ipta / "phase_height_sensitivity_rad_per_m.npy", np.asarray(sensitivity[keep_indices]))

    summary = {
        "method": "drop lowest-quality pairs while preserving edge-connectivity >=2 and minimum degree >=2",
        "source_pairs": len(pairs),
        "retained_pairs": len(selected),
        "dropped_original_records": [int(index + 1) for index in sorted(set(range(len(pairs))) - keep)],
        "uses_building_height": False,
        "graph": metrics,
        "artifacts": {"source_dir": str(output_root), "pairs_csv": str(selected_csv)},
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
