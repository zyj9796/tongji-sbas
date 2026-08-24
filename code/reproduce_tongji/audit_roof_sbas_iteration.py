#!/usr/bin/env python3
"""Audit a roof-height/GAMMA iteration without consulting vector height."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def roof_centroids(path: str) -> pd.DataFrame:
    gdf = gpd.read_file(path).set_crs(None, allow_override=True)
    roofs = gdf[gdf["surface"] == "roof"].copy()
    roofs["roof_col"] = roofs.geometry.centroid.x
    roofs["roof_row"] = roofs.geometry.centroid.y
    return roofs[["clean_id", "roof_col", "roof_row"]]


def height_table(path: str, output_name: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    candidates = [column for column in ("insar_height_m", "height_insar_m") if column in table]
    if not candidates:
        raise ValueError(f"No GAMMA/SBAS height column found in {path}")
    return table[["clean_id", candidates[0]]].rename(columns={candidates[0]: output_name})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-heights", required=True)
    p.add_argument("--updated-heights", required=True)
    p.add_argument("--initial-projection", required=True)
    p.add_argument("--updated-projection", required=True)
    p.add_argument("--height-tolerance-m", type=float, default=1.0)
    p.add_argument("--pixel-tolerance", type=float, default=0.5)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    initial = height_table(args.initial_heights, "height_initial_m")
    updated = height_table(args.updated_heights, "height_updated_m")
    table = initial.merge(updated, on="clean_id", how="outer")
    c0 = roof_centroids(args.initial_projection).rename(columns={"roof_col": "roof_col_initial", "roof_row": "roof_row_initial"})
    c1 = roof_centroids(args.updated_projection).rename(columns={"roof_col": "roof_col_updated", "roof_row": "roof_row_updated"})
    table = table.merge(c0, on="clean_id", how="left").merge(c1, on="clean_id", how="left")
    table["height_abs_change_m"] = (table["height_updated_m"] - table["height_initial_m"]).abs()
    table["roof_centroid_shift_pixels"] = np.hypot(
        table["roof_col_updated"] - table["roof_col_initial"],
        table["roof_row_updated"] - table["roof_row_initial"],
    )
    table["solved_initial"] = table["height_initial_m"].notna()
    table["solved_updated"] = table["height_updated_m"].notna()
    table["converged"] = (
        table["solved_initial"]
        & table["solved_updated"]
        & (table["height_abs_change_m"] <= args.height_tolerance_m)
        & (table["roof_centroid_shift_pixels"] <= args.pixel_tolerance)
    )
    initial_count = int(table["solved_initial"].sum())
    updated_count = int(table["solved_updated"].sum())
    common = table[table["solved_initial"] & table["solved_updated"]]
    lost = int((table["solved_initial"] & ~table["solved_updated"]).sum())
    convergence_fraction = float(table["converged"].sum() / max(len(common), 1))
    adopt = bool(lost == 0 and convergence_fraction >= 0.80)
    table["iteration_adopted"] = adopt
    table["height_attribute_used_for_acceptance"] = False
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    summary = {
        "method": "Compare independent SBAS height solutions and GAMMA roof-centroid movement; vector height is not read",
        "height_attribute_used_for_acceptance": False,
        "initial_solved_buildings": initial_count,
        "updated_solved_buildings": updated_count,
        "common_solved_buildings": int(len(common)),
        "lost_after_update": lost,
        "new_after_update": int((~table["solved_initial"] & table["solved_updated"]).sum()),
        "height_change_median_m": float(common["height_abs_change_m"].median()),
        "height_change_p95_m": float(common["height_abs_change_m"].quantile(0.95)),
        "roof_centroid_shift_median_pixels": float(common["roof_centroid_shift_pixels"].median()),
        "roof_centroid_shift_p95_pixels": float(common["roof_centroid_shift_pixels"].quantile(0.95)),
        "converged_buildings": int(table["converged"].sum()),
        "convergence_fraction_of_common": convergence_fraction,
        "acceptance_rule": "no initially solved building may be lost and at least 80% of common solutions must satisfy both tolerances",
        "height_tolerance_m": args.height_tolerance_m,
        "roof_centroid_tolerance_pixels": args.pixel_tolerance,
        "adopt_iteration": adopt,
        "decision": "retain_initial_strict_solution" if not adopt else "adopt_updated_solution",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
