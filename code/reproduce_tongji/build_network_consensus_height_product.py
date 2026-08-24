#!/usr/bin/env python3
"""Build a prior-unfilled consensus from two frozen GAMMA SBAS networks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network48", required=True)
    p.add_argument("--network36", required=True)
    p.add_argument("--full-vector", required=True)
    p.add_argument("--max-difference", type=float, default=4.0)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-geojson", required=True)
    p.add_argument("--summary", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = pd.read_csv(args.network48).rename(
        columns={
            "insar_height_m": "height_network48_m",
            "median_phase_sigma_rad": "phase_sigma_network48_rad",
            "roof_height_iqr_m": "roof_iqr_network48_m",
        }
    )
    b = pd.read_csv(args.network36).rename(
        columns={
            "insar_height_m": "height_network36_m",
            "median_phase_sigma_rad": "phase_sigma_network36_rad",
            "roof_height_iqr_m": "roof_iqr_network36_m",
        }
    )
    columns_a = [
        "clean_id", "height_network48_m", "phase_sigma_network48_rad", "roof_iqr_network48_m",
        "support_points", "ground_points", "unwrap_valid_support_points",
        "gamma_sbas_output_points", "points_after_iqr", "median_mean_coherence",
    ]
    columns_b = ["clean_id", "height_network36_m", "phase_sigma_network36_rad", "roof_iqr_network36_m"]
    common = a[[column for column in columns_a if column in a]].merge(
        b[[column for column in columns_b if column in b]], on="clean_id", how="inner", validate="one_to_one"
    )
    common["network_height_difference_m"] = (
        common["height_network36_m"] - common["height_network48_m"]
    )
    common["absolute_network_height_difference_m"] = common["network_height_difference_m"].abs()
    accepted = common[
        common["absolute_network_height_difference_m"].le(float(args.max_difference))
    ].copy()
    accepted["insar_height_m"] = accepted[["height_network48_m", "height_network36_m"]].median(axis=1)
    accepted["median_phase_sigma_rad"] = accepted[
        ["phase_sigma_network48_rad", "phase_sigma_network36_rad"]
    ].median(axis=1)
    accepted["roof_height_iqr_m"] = accepted[
        ["roof_iqr_network48_m", "roof_iqr_network36_m"]
    ].median(axis=1)
    accepted["solution_source"] = "GAMMA_equal_network48_36_consensus"
    accepted["filled_from_prior"] = np.int8(0)
    accepted["insar_only"] = np.int8(0)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output_csv, index=False)

    full = gpd.read_file(args.full_vector)
    result_columns = [
        "insar_height_m", "support_points", "ground_points", "unwrap_valid_support_points",
        "gamma_sbas_output_points", "points_after_iqr", "roof_height_iqr_m",
        "median_phase_sigma_rad", "median_mean_coherence", "solution_source",
        "filled_from_prior", "insar_only", "prior_height_m", "difference_to_prior_m",
        "height_network48_m", "height_network36_m", "network_height_difference_m",
        "absolute_network_height_difference_m",
    ]
    for column in result_columns:
        if column in {"solution_source"}:
            full[column] = None
        else:
            full[column] = np.nan
    merge_columns = [column for column in accepted.columns if column in result_columns or column == "clean_id"]
    full = full.drop(columns=[column for column in merge_columns if column != "clean_id" and column in full], errors="ignore")
    full = full.merge(accepted[merge_columns], on="clean_id", how="left", validate="one_to_one")
    full["filled_from_prior"] = full["filled_from_prior"].fillna(0).astype(np.int8)
    full["insar_only"] = full["insar_only"].fillna(0).astype(np.int8)
    # Prior is attached only after the consensus has been frozen; it is never
    # used for acceptance, aggregation, or filling.
    if "height" in full:
        solved = full["insar_height_m"].notna()
        full.loc[solved, "prior_height_m"] = pd.to_numeric(full.loc[solved, "height"], errors="coerce")
        full.loc[solved, "difference_to_prior_m"] = (
            full.loc[solved, "insar_height_m"] - full.loc[solved, "prior_height_m"]
        )
    output_geojson = Path(args.output_geojson)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    full.to_file(output_geojson, driver="GeoJSON")

    values = accepted["insar_height_m"]
    summary = {
        "method": "median of frozen equal-weight GAMMA 48-pair and graph-robust 36-pair solutions",
        "stability_gate_m": float(args.max_difference),
        "gate_origin": "rounded frozen-pilot |median difference| + 3 x robust MAD = 3.83 m",
        "uses_prior_for_gate_or_height": False,
        "prior_filled_buildings": 0,
        "network48_solved": int(len(a)),
        "network36_solved": int(len(b)),
        "common_solved": int(len(common)),
        "accepted_buildings": int(len(accepted)),
        "unsolved_buildings": int(len(full) - len(accepted)),
        "height_median_m": float(values.median()),
        "height_p05_p95_m": [float(values.quantile(0.05)), float(values.quantile(0.95))],
        "height_maximum_m": float(values.max()),
        "network_difference_median_m": float(common["network_height_difference_m"].median()),
        "network_absolute_difference_p95_m": float(common["absolute_network_height_difference_m"].quantile(0.95)),
        "artifacts": {"csv": str(output_csv), "geojson": str(output_geojson)},
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
