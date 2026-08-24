#!/usr/bin/env python3
"""Resolve stable rooftop PS ambiguities with GAMMA R-D displacement geometry.

For every temporally stable GAMMA SBAS point, this script translates the
projected building bottom through a 0--180 m height grid using the GAMMA-derived
per-metre range/azimuth displacement. A point is accepted only if its SBAS
height is compatible with a height at which the translated footprint covers
that SAR pixel. The vector height is never copied into the output, but it was
used upstream to establish the initial bottom/roof projection pair; therefore
this product is explicitly HYBRID and is not an independent InSAR-only result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import affinity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--point-audit",
        default="work/gamma_building_roofcore_closurecorrected_gamma100_full/stable_rooftop_point_audit.csv",
    )
    p.add_argument("--point-metadata", default="work/gamma_native_ipta_sbas/ipta/point_metadata.csv")
    p.add_argument("--projection", default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--height-step", type=float, default=0.5)
    p.add_argument("--geometry-tolerance", type=float, default=5.0)
    p.add_argument("--pixel-buffer", type=float, default=0.51)
    p.add_argument("--aggregation", choices=["median", "minimum_mismatch"], default="median")
    p.add_argument("--output", default="results/tables/tongji_building_height_gamma_rd_geometry_constrained.csv")
    p.add_argument("--geojson", default="results/geodata/tongji_building_height_gamma_rd_geometry_constrained.geojson")
    p.add_argument("--summary", default="results/metadata/tongji_building_height_gamma_rd_geometry_constrained_summary.json")
    args = p.parse_args()

    audit = pd.read_csv(args.point_audit)
    stable = audit[audit["stable"].eq(True)].copy()
    metadata = pd.read_csv(args.point_metadata).set_index("point_index")
    projection = gpd.read_file(args.projection)
    height_grid = np.arange(0.0, 180.0 + 0.5 * args.height_step, args.height_step)
    models: dict[int, tuple[object, float, float]] = {}
    for clean_id, group in projection.groupby("clean_id"):
        bottom = group[group["surface"].eq("bottom")]
        roof = group[group["surface"].eq("roof")]
        if bottom.empty or roof.empty:
            continue
        geometry_height = float(group["height_prior_m"].iloc[0])
        if not np.isfinite(geometry_height) or geometry_height <= 0:
            continue
        bottom_geom = bottom.geometry.iloc[0]
        roof_geom = roof.geometry.iloc[0]
        models[int(clean_id)] = (
            bottom_geom,
            float((roof_geom.centroid.x - bottom_geom.centroid.x) / geometry_height),
            float((roof_geom.centroid.y - bottom_geom.centroid.y) / geometry_height),
        )

    point_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for clean_id, group in stable.groupby("clean_id"):
        if int(clean_id) not in models:
            continue
        group = group.reset_index(drop=True)
        bottom, dx_per_m, dy_per_m = models[int(clean_id)]
        group_meta = metadata.loc[group["point_index"].to_numpy(int)]
        group_x = group_meta["range_pixel"].to_numpy(float)
        group_y = group_meta["azimuth_pixel"].to_numpy(float)
        coverage = np.zeros((len(height_grid), len(group)), dtype=bool)
        for hi, height in enumerate(height_grid):
            geom = affinity.translate(
                bottom, xoff=dx_per_m * height, yoff=dy_per_m * height
            ).buffer(args.pixel_buffer)
            coverage[hi] = np.asarray(shapely.contains_xy(geom, group_x, group_y), dtype=bool)
        compatible: list[dict[str, object]] = []
        for point_offset, row in enumerate(group.itertuples(index=False)):
            valid_heights = height_grid[coverage[:, point_offset]]
            if len(valid_heights):
                mismatch = float(np.min(np.abs(valid_heights - float(row.mb_height_m))))
                geom_min, geom_max = float(valid_heights.min()), float(valid_heights.max())
                nearest = float(valid_heights[np.argmin(np.abs(valid_heights - float(row.mb_height_m)))])
            else:
                mismatch, geom_min, geom_max, nearest = np.inf, np.nan, np.nan, np.nan
            accepted = bool(mismatch <= args.geometry_tolerance)
            record = {
                "clean_id": int(clean_id),
                "point_index": int(row.point_index),
                "sbas_height_m": float(row.mb_height_m),
                "geometry_height_min_m": geom_min,
                "geometry_height_max_m": geom_max,
                "nearest_geometry_height_m": nearest,
                "geometry_mismatch_m": mismatch,
                "phase_sigma_rad": float(row.phase_sigma_rad),
                "subset_height_spread_m": float(row.subset_height_spread_m),
                "accepted_geometry_consistent": accepted,
            }
            point_rows.append(record)
            if accepted:
                compatible.append(record)
        if compatible:
            values = np.asarray([r["sbas_height_m"] for r in compatible], dtype=float)
            if args.aggregation == "minimum_mismatch":
                selected = min(
                    compatible,
                    key=lambda r: (
                        r["geometry_mismatch_m"],
                        r["phase_sigma_rad"],
                        r["subset_height_spread_m"],
                    ),
                )
                building_height = float(selected["sbas_height_m"])
                selected_point_index = int(selected["point_index"])
            else:
                building_height = float(np.median(values))
                selected_point_index = None
            # A singleton remains explicitly marked and must not be treated as
            # high confidence even when it is geometrically consistent.
            result_rows.append(
                {
                    "clean_id": int(clean_id),
                    "height_m": building_height,
                    "selected_point_index": selected_point_index,
                    "geometry_consistent_points": int(len(values)),
                    "singleton_solution": bool(len(values) == 1),
                    "height_min_m": float(values.min()),
                    "height_max_m": float(values.max()),
                    "median_geometry_mismatch_m": float(np.median([r["geometry_mismatch_m"] for r in compatible])),
                    "median_phase_sigma_rad": float(np.median([r["phase_sigma_rad"] for r in compatible])),
                    "median_subset_spread_m": float(np.median([r["subset_height_spread_m"] for r in compatible])),
                    "solution_source": "HYBRID_GAMMA_mb_pt_plus_RD_height_position_consistency",
                    "filled_from_prior": False,
                    "insar_only": False,
                }
            )

    point_audit = Path(args.output).with_name(Path(args.output).stem + "_point_audit.csv")
    point_audit.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(point_audit, index=False)
    result = pd.DataFrame(result_rows)
    buildings = gpd.read_file(args.buildings)
    prior = buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"})
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result["height_m"] - result["prior_height_m"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    geo = buildings.merge(result, on="clean_id", how="left")
    geojson = Path(args.geojson); geojson.parent.mkdir(parents=True, exist_ok=True); geo.to_file(geojson, driver="GeoJSON")
    compared = result[["height_m", "prior_height_m"]].dropna()
    repeated = result[result["singleton_solution"].eq(False)]
    summary = {
        "method": f"temporally stable GAMMA SBAS rooftop PS gated by R-D height-position consistency; {args.aggregation} aggregation",
        "product_class": "hybrid geometry-constrained diagnostic; not InSAR-only",
        "prior_policy": "height values are never copied or filled; prior-derived initial roof projection participates in geometry and prevents independent validation against that same prior",
        "thresholds": {"height_step_m": args.height_step, "geometry_tolerance_m": args.geometry_tolerance, "pixel_buffer": args.pixel_buffer, "aggregation": args.aggregation},
        "buildings_with_any_solution": int(len(result)),
        "singleton_buildings": int(result["singleton_solution"].sum()) if len(result) else 0,
        "repeated_point_buildings": int(len(repeated)),
        "height_median_m": float(result["height_m"].median()) if len(result) else None,
        "height_p05_p95_m": [float(result["height_m"].quantile(.05)), float(result["height_m"].quantile(.95))] if len(result) else None,
        "post_solution_prior_comparison_not_independent": {
            "count": int(len(compared)),
            "mae_m": float((compared["height_m"] - compared["prior_height_m"]).abs().mean()) if len(compared) else None,
            "correlation": float(compared.corr().iloc[0, 1]) if len(compared) > 1 else None,
        },
        "artifacts": {"result_csv": str(output), "result_geojson": str(geojson), "point_audit": str(point_audit)},
    }
    summary_path = Path(args.summary); summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
