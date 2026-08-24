#!/usr/bin/env python3
"""Aggregate temporally stable rooftop PS only when they form a height cluster.

This is a strict follow-up to ``select_stable_roof_top_points.py``. Singleton
solutions are forbidden. Stable candidates are grouped in one-dimensional
height space; the largest cluster is selected and its median is reported.
Vector prior heights are merged only after the rule has been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def connected_clusters(group: pd.DataFrame, tolerance: float) -> list[pd.DataFrame]:
    ordered = group.sort_values("mb_height_m").reset_index(drop=True)
    if ordered.empty:
        return []
    split = np.r_[True, np.diff(ordered["mb_height_m"].to_numpy(float)) > tolerance]
    labels = np.cumsum(split)
    return [part.copy() for _, part in ordered.groupby(labels)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--point-audit",
        default="work/gamma_building_roofcore_closurecorrected_gamma100_full/stable_rooftop_point_audit.csv",
    )
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--cluster-tolerance", type=float, default=5.0)
    p.add_argument("--minimum-cluster-points", type=int, default=2)
    p.add_argument("--maximum-cluster-iqr", type=float, default=8.0)
    p.add_argument("--cluster-selection", choices=["largest", "highest"], default="largest")
    p.add_argument("--output", default="results/tables/tongji_building_height_stable_rooftop_cluster_insar_only.csv")
    p.add_argument("--geojson", default="results/geodata/tongji_building_height_stable_rooftop_cluster_insar_only.geojson")
    p.add_argument("--summary", default="results/metadata/tongji_building_height_stable_rooftop_cluster_summary.json")
    args = p.parse_args()

    audit = pd.read_csv(args.point_audit)
    stable = audit[audit["stable"].eq(True)].copy()
    rows: list[dict[str, object]] = []
    rejected_singleton = 0
    rejected_iqr = 0
    for clean_id, group in stable.groupby("clean_id"):
        clusters = connected_clusters(group, args.cluster_tolerance)
        eligible = [part for part in clusters if len(part) >= args.minimum_cluster_points]
        if not eligible:
            rejected_singleton += 1
            continue
        if args.cluster_selection == "highest":
            winner = max(
                eligible,
                key=lambda part: (
                    float(part["mb_height_m"].median()),
                    len(part),
                    -float(part["fit_rms_rad"].median()),
                ),
            )
        else:
            # Largest cluster first; then lower phase RMS and lower subset spread.
            winner = min(
                eligible,
                key=lambda part: (
                    -len(part),
                    float(part["fit_rms_rad"].median()),
                    float(part["subset_height_spread_m"].median()),
                ),
            )
        q1, q3 = np.percentile(winner["mb_height_m"], [25, 75])
        iqr = float(q3 - q1)
        if iqr > args.maximum_cluster_iqr:
            rejected_iqr += 1
            continue
        height = float(winner["mb_height_m"].median())
        rows.append(
            {
                "clean_id": int(clean_id),
                "insar_height_m": height,
                "stable_cluster_points": int(len(winner)),
                "stable_candidates_total": int(len(group)),
                "cluster_min_m": float(winner["mb_height_m"].min()),
                "cluster_max_m": float(winner["mb_height_m"].max()),
                "cluster_iqr_m": iqr,
                "median_phase_sigma_rad": float(winner["phase_sigma_rad"].median()),
                "median_fit_rms_rad": float(winner["fit_rms_rad"].median()),
                "median_subset_spread_m": float(winner["subset_height_spread_m"].median()),
                "solution_source": "GAMMA_mb_pt_repeated_rooftop_PS_cluster",
                "filled_from_prior": False,
            }
        )

    result = pd.DataFrame(rows)
    buildings = gpd.read_file(args.buildings)
    prior = buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"})
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result["insar_height_m"] - result["prior_height_m"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    geo = buildings.merge(result, on="clean_id", how="left")
    geojson = Path(args.geojson)
    geojson.parent.mkdir(parents=True, exist_ok=True)
    geo.to_file(geojson, driver="GeoJSON")

    compared = result[["insar_height_m", "prior_height_m"]].dropna()
    summary = {
        "method": f"{args.cluster_selection} repeated stable rooftop PS height cluster; singleton forbidden",
        "prior_used_in_selection_or_filling": False,
        "thresholds": {
            "cluster_tolerance_m": args.cluster_tolerance,
            "minimum_cluster_points": args.minimum_cluster_points,
            "maximum_cluster_iqr_m": args.maximum_cluster_iqr,
            "cluster_selection": args.cluster_selection,
        },
        "stable_candidate_buildings": int(stable["clean_id"].nunique()),
        "buildings_solved": int(len(result)),
        "rejected_without_repeated_cluster": int(rejected_singleton),
        "rejected_cluster_iqr": int(rejected_iqr),
        "height_median_m": float(result["insar_height_m"].median()) if len(result) else None,
        "height_p05_p95_m": [
            float(result["insar_height_m"].quantile(0.05)),
            float(result["insar_height_m"].quantile(0.95)),
        ] if len(result) else None,
        "phase_sigma_median_rad": float(result["median_phase_sigma_rad"].median()) if len(result) else None,
        "subset_spread_median_m": float(result["median_subset_spread_m"].median()) if len(result) else None,
        "post_selection_prior_comparison": {
            "count": int(len(compared)),
            "mae_m": float((compared["insar_height_m"] - compared["prior_height_m"]).abs().mean()) if len(compared) else None,
            "correlation": float(compared.corr().iloc[0, 1]) if len(compared) > 1 else None,
        },
        "artifacts": {"result_csv": str(output), "result_geojson": str(geojson)},
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
