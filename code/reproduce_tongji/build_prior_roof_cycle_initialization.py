#!/usr/bin/env python3
"""Build an explicit prior-roof phase-cycle initialization table.

The table is not a height result.  A point is retained only when it lies in the
eroded, conflict-free roof core generated from the GAMMA R-D projection.  The
vector height initializes the integer phase cycle for GAMMA ``mb_pt``; missing
buildings are omitted and are never filled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="work/gamma_native_ipta_sbas/ipta/point_metadata.csv")
    p.add_argument("--roof-owner", default="work/roof_sbas_optimized/roof_core_clean_id_mask.npy")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--projection-heights", default="", help="Optional projection GeoJSON; use roof height_prior_m actually used for that iteration")
    p.add_argument("--output", default="results/tables/tongji_prior_roof_cycle_initialization_points.csv")
    p.add_argument("--summary", default="results/metadata/tongji_prior_roof_cycle_initialization_summary.json")
    args = p.parse_args()

    metadata = pd.read_csv(args.metadata).sort_values("point_index").reset_index(drop=True)
    owner = np.load(args.roof_owner).astype(np.int64)
    az = metadata["azimuth_pixel"].to_numpy(np.int64)
    rg = metadata["range_pixel"].to_numpy(np.int64)
    if az.max() >= owner.shape[0] or rg.max() >= owner.shape[1]:
        raise ValueError("Roof-owner raster does not cover the IPTA point coordinates")
    owner_at_point = owner[az, rg]
    keep = metadata["point_class"].eq("roof").to_numpy() & (owner_at_point > 0)
    points = metadata.loc[keep, ["point_index", "mean_coherence", "amplitude_dispersion"]].copy()
    points["clean_id"] = owner_at_point[keep]

    if args.projection_heights:
        projected = gpd.read_file(args.projection_heights)
        projected = projected[projected["surface"].eq("roof")]
        height = projected[["clean_id", "height_prior_m"]].drop_duplicates("clean_id").set_index("clean_id")["height_prior_m"]
        initialization_source = "iteration_projection_height_GAMMA_RD_projected_eroded_roof_core"
    else:
        buildings = gpd.read_file(args.buildings)
        height = buildings[["clean_id", "height"]].drop_duplicates("clean_id").set_index("clean_id")["height"]
        initialization_source = "clean_equal_height_GAMMA_RD_projected_eroded_roof_core"
    points["searched_height_m"] = points["clean_id"].map(height).astype(float)
    points = points[np.isfinite(points["searched_height_m"]) & points["searched_height_m"].gt(0)].copy()
    points["nuisance_rate_m_per_year"] = 0.0
    points["selected_for_building_cluster"] = True
    points["initialization_only"] = True
    points["filled_from_prior"] = False
    points["initialization_source"] = initialization_source

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(output, index=False)
    counts = points.groupby("clean_id").size()
    summary = {
        "product_class": "phase_cycle_initialization_not_height_result",
        "policy": "prior height initializes GAMMA phase cycles only; no missing result is filled",
        "roof_owner": args.roof_owner,
        "projection_heights": args.projection_heights or None,
        "points": int(len(points)),
        "buildings": int(points["clean_id"].nunique()),
        "buildings_with_at_least_two_points": int((counts >= 2).sum()),
        "buildings_with_at_least_four_points": int((counts >= 4).sum()),
        "output": str(output),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
