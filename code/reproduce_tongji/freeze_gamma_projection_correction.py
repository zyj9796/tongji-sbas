#!/usr/bin/env python3
"""Reapply a fixed validated SAR correction to an updated GAMMA projection.

This separates a roof-height geometry update from a new image-registration fit,
so iteration convergence is not confounded by changing global/local shifts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import affinity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection", required=True)
    p.add_argument("--current-metrics", required=True)
    p.add_argument("--reference-metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()

    projection = gpd.read_file(args.projection).set_crs(None, allow_override=True)
    current = pd.read_csv(args.current_metrics).set_index("clean_id")
    reference = pd.read_csv(args.reference_metrics).set_index("clean_id")
    rows = []
    for row in projection.itertuples(index=False):
        clean_id = int(row.clean_id)
        cur = current.loc[clean_id]
        ref = reference.loc[clean_id]
        current_row = int(cur.gamma_global_row_shift) + int(cur.local_row_shift)
        current_col = int(cur.gamma_global_col_shift) + int(cur.local_col_shift)
        reference_row = int(ref.gamma_global_row_shift) + int(ref.local_row_shift)
        reference_col = int(ref.gamma_global_col_shift) + int(ref.local_col_shift)
        props = row._asdict()
        props["geometry"] = affinity.translate(row.geometry, xoff=reference_col - current_col, yoff=reference_row - current_row)
        props["global_row_shift"] = int(ref.gamma_global_row_shift)
        props["global_col_shift"] = int(ref.gamma_global_col_shift)
        props["fixed_local_row_shift"] = int(ref.local_row_shift)
        props["fixed_local_col_shift"] = int(ref.local_col_shift)
        props["correction_iteration_policy"] = "fixed_from_initial_temporal_cross_validation"
        rows.append(props)
    output = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_file(args.output, driver="GeoJSON")
    payload = {
        "method": "Updated GAMMA geometry with global/local SAR corrections frozen to the initial time-cross-validated solution",
        "height_attribute_used_for_correction": False,
        "features": int(len(output)),
        "buildings": int(output.clean_id.nunique()),
        "all_geometries_valid": bool(output.geometry.is_valid.all()),
        "projection": args.projection,
        "reference_metrics": args.reference_metrics,
        "output": args.output,
    }
    Path(args.summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
