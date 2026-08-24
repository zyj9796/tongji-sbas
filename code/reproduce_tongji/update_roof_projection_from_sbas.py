#!/usr/bin/env python3
"""Update roof-only SAR geometry with a previous GAMMA SBAS height solution.

The R-D displacement per metre is recovered from the original GAMMA bottom/roof
projection.  Valid SBAS heights move the already registered roof along that
vector.  Buildings without a previous SBAS solution retain their initialization
geometry solely so they may be attempted again; no height result is filled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registered-roofs", required=True)
    p.add_argument("--rd-projection", required=True)
    p.add_argument("--heights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    roofs = gpd.read_file(args.registered_roofs).set_crs(None, allow_override=True)
    roofs = roofs[roofs["surface"].eq("roof")].copy()
    rd = gpd.read_file(args.rd_projection).set_crs(None, allow_override=True)
    heights = pd.read_csv(args.heights)
    height_lookup = heights.dropna(subset=["insar_height_m"]).set_index("clean_id")["insar_height_m"].to_dict()

    models: dict[int, tuple[float, float, float]] = {}
    for clean_id, group in rd.groupby("clean_id"):
        bottom = group[group["surface"].eq("bottom")]
        roof = group[group["surface"].eq("roof")]
        if bottom.empty or roof.empty:
            continue
        initial_height = float(group["height_prior_m"].iloc[0])
        if not np.isfinite(initial_height) or initial_height <= 0:
            continue
        dx = float((roof.geometry.iloc[0].centroid.x - bottom.geometry.iloc[0].centroid.x) / initial_height)
        dy = float((roof.geometry.iloc[0].centroid.y - bottom.geometry.iloc[0].centroid.y) / initial_height)
        models[int(clean_id)] = (initial_height, dx, dy)

    rows = []
    shifts = []
    updated = 0
    for row in roofs.itertuples(index=False):
        clean_id = int(row.clean_id)
        props = row._asdict()
        geom = props.pop("geometry")
        model = models.get(clean_id)
        solved_height = height_lookup.get(clean_id)
        if model is not None and solved_height is not None and np.isfinite(float(solved_height)):
            initial_height, dx, dy = model
            delta_height = float(solved_height) - initial_height
            geom = affinity.translate(geom, xoff=dx * delta_height, yoff=dy * delta_height)
            props["height_prior_m"] = float(solved_height)
            props["iteration_height_source"] = "previous_GAMMA_SBAS_solution"
            props["previous_sbas_solution"] = 1
            updated += 1
            shifts.append(float(np.hypot(dx * delta_height, dy * delta_height)))
        else:
            props["iteration_height_source"] = "initial_geometry_unsolved_not_height_result"
            props["previous_sbas_solution"] = 0
        props["height_role"] = "iteration_geometry_initialization_only_not_fill"
        props["geometry"] = geom
        rows.append(props)

    output = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(output_path, driver="GeoJSON")
    payload = {
        "method": "previous GAMMA SBAS height updates registered roof along GAMMA R-D displacement-per-metre vector",
        "registered_roofs": args.registered_roofs,
        "rd_projection": args.rd_projection,
        "height_solution": args.heights,
        "buildings": int(output["clean_id"].nunique()),
        "updated_from_previous_sbas": int(updated),
        "retained_initial_geometry_without_solution": int(len(output) - updated),
        "median_roof_shift_pixels": float(np.median(shifts)) if shifts else None,
        "p95_roof_shift_pixels": float(np.percentile(shifts, 95)) if shifts else None,
        "prior_fill_used": False,
        "output": str(output_path),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
