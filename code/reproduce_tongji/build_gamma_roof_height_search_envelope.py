#!/usr/bin/env python3
"""Build a prior-height-independent SAR roof-position search envelope.

The existing GAMMA R-D bottom/roof projection is used only to recover the local
SAR displacement vector per metre.  Every building is then swept over the same
0..Hmax height interval.  The original vector height is not used as a search
bound, score, inversion value, or missing-value fill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely import affinity
from shapely.geometry import mapping
from shapely.ops import unary_union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--maximum-height-m", type=float, default=180.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    source = gpd.read_file(args.projection)
    features: list[dict[str, object]] = []
    skipped: list[int] = []
    for clean_id, group in source.groupby("clean_id"):
        bottom_rows = group[group["surface"].eq("bottom")]
        roof_rows = group[group["surface"].eq("roof")]
        if bottom_rows.empty or roof_rows.empty:
            skipped.append(int(clean_id))
            continue
        bottom = bottom_rows.geometry.iloc[0]
        initial_roof = roof_rows.geometry.iloc[0]
        initial_height = float(group["height_prior_m"].iloc[0])
        if initial_height <= 0 or bottom.is_empty or initial_roof.is_empty:
            skipped.append(int(clean_id))
            continue
        dx_per_m = (initial_roof.centroid.x - bottom.centroid.x) / initial_height
        dy_per_m = (initial_roof.centroid.y - bottom.centroid.y) / initial_height
        top = affinity.translate(
            bottom,
            xoff=dx_per_m * args.maximum_height_m,
            yoff=dy_per_m * args.maximum_height_m,
        )
        envelope = unary_union([bottom, top]).convex_hull.buffer(0)
        common = {
            "clean_id": int(clean_id),
            "uid": int(clean_id),
            "height_prior_m": float(args.maximum_height_m),
            "search_height_min_m": 0.0,
            "search_height_max_m": float(args.maximum_height_m),
            "range_displacement_per_m": float(dx_per_m),
            "azimuth_displacement_per_m": float(dy_per_m),
            "height_role": "common_search_envelope_only_not_prior_or_fill",
        }
        for surface, geometry in (("bottom", bottom), ("roof", top), ("layover", envelope)):
            features.append(
                {
                    "type": "Feature",
                    "properties": {**common, "surface": surface},
                    "geometry": mapping(geometry),
                }
            )

    payload = {
        "type": "FeatureCollection",
        "name": "tongji_gamma_roof_height_search_envelope",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "features": features,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "method": "GAMMA R-D displacement-per-m extrapolation with common 0..Hmax search interval",
        "source_projection": args.projection,
        "source_vector_height_used_as_search_bound_or_value": False,
        "maximum_height_m": args.maximum_height_m,
        "buildings": len(features) // 3,
        "skipped_clean_ids": skipped,
        "output": str(output),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
