#!/usr/bin/env python3
"""Add a dual-network, spatially connected upper-roof plateau to a frozen consensus.

The paper's IQR + median estimate remains the default.  A building is raised only
when both independently rerun interferogram networks contain the same spatially
connected, internally flat upper plateau.  No vector/prior height is read until
the InSAR-only selection and updated heights have been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", default="results/tables/tongji_building_height_network_consensus_gamma100.csv")
    parser.add_argument("--network48-shards", default="work/gamma_paperquality_globalground_full/building_shards")
    parser.add_argument("--network36-shards", default="work/gamma_paperquality_stability36_full/building_shards")
    parser.add_argument("--point-metadata", default="work/gamma_native_ipta_adaptive_window_paper_quality_network48/ipta/point_metadata.csv")
    parser.add_argument("--full-vector", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    parser.add_argument("--spatial-radius-px", type=float, default=3.1)
    parser.add_argument("--height-link-m", type=float, default=3.0)
    parser.add_argument("--min-plateau-points", type=int, default=4)
    parser.add_argument("--min-plateau-fraction", type=float, default=0.20)
    parser.add_argument("--max-plateau-iqr-m", type=float, default=3.0)
    parser.add_argument("--min-uplift-m", type=float, default=1.5)
    parser.add_argument("--max-uplift-m", type=float, default=12.0)
    parser.add_argument("--max-network-difference-m", type=float, default=3.0)
    parser.add_argument("--min-point-jaccard", type=float, default=0.50)
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_guarded_upper_roof_consensus_gamma100.csv")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_guarded_upper_roof_consensus_gamma100.geojson")
    parser.add_argument("--summary", default="results/metadata/tongji_building_height_guarded_upper_roof_consensus_gamma100_summary.json")
    return parser.parse_args()


def iqr_keep(values: np.ndarray) -> np.ndarray:
    q1, q3 = np.nanpercentile(values, [25, 75])
    spread = float(q3 - q1)
    return (values >= q1 - 1.5 * spread) & (values <= q3 + 1.5 * spread)


def upper_plateau(
    shard: Path,
    coordinates: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if not shard.exists():
        return None
    payload = np.load(shard)
    point_ids = payload["point_index"].astype(np.int64)
    heights = payload["height_m"].astype(float)
    sigmas = payload["phase_sigma_rad"].astype(float)
    valid = np.isfinite(heights) & np.isfinite(sigmas)
    point_ids, heights, sigmas = point_ids[valid], heights[valid], sigmas[valid]
    if len(heights) < 2 * args.min_plateau_points:
        return None
    keep = iqr_keep(heights)
    point_ids, heights, sigmas = point_ids[keep], heights[keep], sigmas[keep]
    if len(heights) < 2 * args.min_plateau_points:
        return None

    xy = coordinates.loc[point_ids, ["range_pixel", "azimuth_pixel"]].to_numpy(float)
    adjacency: list[list[int]] = [[] for _ in range(len(heights))]
    for left, right in cKDTree(xy).query_pairs(args.spatial_radius_px):
        if abs(float(heights[left] - heights[right])) <= args.height_link_m:
            adjacency[left].append(right)
            adjacency[right].append(left)

    components: list[np.ndarray] = []
    visited: set[int] = set()
    for start in range(len(heights)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(np.asarray(component, dtype=np.int64))

    full_median = float(np.nanmedian(heights))
    candidates: list[dict[str, object]] = []
    for component in components:
        component_height = heights[component]
        count = int(len(component))
        fraction = float(count / len(heights))
        q1, q3 = np.nanpercentile(component_height, [25, 75])
        spread = float(q3 - q1)
        median = float(np.nanmedian(component_height))
        uplift = float(median - full_median)
        if (
            count >= args.min_plateau_points
            and fraction >= args.min_plateau_fraction
            and spread <= args.max_plateau_iqr_m
            and args.min_uplift_m <= uplift <= args.max_uplift_m
        ):
            candidates.append(
                {
                    "height_m": median,
                    "point_count": count,
                    "point_fraction": fraction,
                    "iqr_m": spread,
                    "phase_sigma_rad": float(np.nanmedian(sigmas[component])),
                    "point_ids": set(point_ids[component].tolist()),
                    "uplift_m": uplift,
                }
            )
    return max(candidates, key=lambda item: float(item["height_m"])) if candidates else None


def main() -> None:
    args = parse_args()
    frozen = pd.read_csv(args.consensus)
    coordinates = pd.read_csv(args.point_metadata).set_index("point_index", drop=False)
    root48, root36 = Path(args.network48_shards), Path(args.network36_shards)

    rows: list[dict[str, object]] = []
    for record in frozen.to_dict("records"):
        clean_id = int(record["clean_id"])
        candidate48 = upper_plateau(root48 / f"{clean_id}_points.npz", coordinates, args)
        candidate36 = upper_plateau(root36 / f"{clean_id}_points.npz", coordinates, args)
        use_plateau = False
        point_jaccard = np.nan
        plateau_difference = np.nan
        if candidate48 is not None and candidate36 is not None:
            ids48 = candidate48["point_ids"]
            ids36 = candidate36["point_ids"]
            point_jaccard = len(ids48 & ids36) / max(1, len(ids48 | ids36))
            plateau_difference = abs(float(candidate48["height_m"]) - float(candidate36["height_m"]))
            use_plateau = (
                point_jaccard >= args.min_point_jaccard
                and plateau_difference <= args.max_network_difference_m
            )

        updated = dict(record)
        updated["paper_median_height_m"] = float(record["insar_height_m"])
        updated["upper_plateau_applied"] = np.int8(use_plateau)
        updated["upper_plateau_point_jaccard"] = point_jaccard
        updated["upper_plateau_network_difference_m"] = plateau_difference
        updated["aggregation_extension"] = "paper_IQR_median"
        if use_plateau:
            height48 = float(candidate48["height_m"])
            height36 = float(candidate36["height_m"])
            updated["height_network48_m"] = height48
            updated["height_network36_m"] = height36
            updated["network_height_difference_m"] = height36 - height48
            updated["absolute_network_height_difference_m"] = abs(height36 - height48)
            updated["insar_height_m"] = float(np.median([height48, height36]))
            updated["upper_plateau_uplift_m"] = updated["insar_height_m"] - updated["paper_median_height_m"]
            updated["roof_height_iqr_m"] = float(np.median([candidate48["iqr_m"], candidate36["iqr_m"]]))
            updated["median_phase_sigma_rad"] = float(
                np.median([candidate48["phase_sigma_rad"], candidate36["phase_sigma_rad"]])
            )
            updated["points_after_iqr"] = int(min(candidate48["point_count"], candidate36["point_count"]))
            updated["aggregation_extension"] = "dual_network_spatial_upper_plateau"
        else:
            updated["upper_plateau_uplift_m"] = 0.0
        rows.append(updated)

    result = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    # The InSAR-only selection is now frozen.  Load the vector prior only for
    # non-independent, post-hoc diagnostics and never for selecting a plateau.
    full = gpd.read_file(args.full_vector)
    result_columns = [column for column in result.columns if column != "clean_id"]
    full = full.drop(columns=[column for column in result_columns if column in full], errors="ignore")
    full = full.merge(result, on="clean_id", how="left", validate="one_to_one")
    solved = full["insar_height_m"].notna()
    full["prior_height_m"] = np.nan
    full["difference_to_prior_m"] = np.nan
    if "height" in full:
        full.loc[solved, "prior_height_m"] = pd.to_numeric(full.loc[solved, "height"], errors="coerce")
        full.loc[solved, "difference_to_prior_m"] = (
            full.loc[solved, "insar_height_m"] - full.loc[solved, "prior_height_m"]
        )
    output_geojson = Path(args.output_geojson)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    full.to_file(output_geojson, driver="GeoJSON")

    changed = result[result["upper_plateau_applied"].eq(1)].copy()
    values = result["insar_height_m"].astype(float)
    summary = {
        "method": "paper IQR+median with dual-network spatially connected upper-roof plateau gate",
        "prior_used_for_selection_or_height": False,
        "prior_filled_buildings": 0,
        "accepted_buildings": int(len(result)),
        "upper_plateau_buildings": int(len(changed)),
        "upper_plateau_clean_ids": changed["clean_id"].astype(int).tolist(),
        "uplift_median_p95_max_m": (
            [
                float(changed["upper_plateau_uplift_m"].median()),
                float(changed["upper_plateau_uplift_m"].quantile(0.95)),
                float(changed["upper_plateau_uplift_m"].max()),
            ]
            if len(changed)
            else [None, None, None]
        ),
        "height_median_p05_p95_max_m": [
            float(values.median()),
            float(values.quantile(0.05)),
            float(values.quantile(0.95)),
            float(values.max()),
        ],
        "frozen_gates": {
            "spatial_radius_px": args.spatial_radius_px,
            "height_link_m": args.height_link_m,
            "min_plateau_points": args.min_plateau_points,
            "min_plateau_fraction": args.min_plateau_fraction,
            "max_plateau_iqr_m": args.max_plateau_iqr_m,
            "uplift_range_m": [args.min_uplift_m, args.max_uplift_m],
            "max_network_difference_m": args.max_network_difference_m,
            "min_point_jaccard": args.min_point_jaccard,
        },
        "artifacts": {"csv": str(output_csv), "geojson": str(output_geojson)},
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
