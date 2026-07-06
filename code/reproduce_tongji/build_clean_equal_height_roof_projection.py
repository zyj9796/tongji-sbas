#!/usr/bin/env python3
"""Build roof SAR projection for equal-height-cleaned building vectors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.geojson")
    parser.add_argument("--fid-uid-map", default="results/tables/touying_fid_uid_map.csv")
    parser.add_argument("--source-projection", default="tmp/touying_roof_workflow/results/blue_aligned/20200708_full_area_projection_sar_col_row_brightness_optimized.geojson")
    parser.add_argument("--output-geojson", default="work/projection/20200708_clean_equal_height_roof_projection_sar.geojson")
    parser.add_argument("--output-fid-map", default="results/tables/clean_equal_height_fid_uid_map.csv")
    parser.add_argument("--summary", default="results/metadata/clean_equal_height_roof_projection_summary.json")
    args = parser.parse_args()

    clean = gpd.read_file(args.clean_buildings)
    fid_uid = pd.read_csv(args.fid_uid_map)
    uid_to_fid = dict(zip(fid_uid["uid"].astype(int), fid_uid["touying_fid"].astype(int)))

    src_data = json.loads(Path(args.source_projection).read_text(encoding="utf-8"))
    roof_by_fid = {}
    for feat in src_data.get("features", []):
        props = feat.get("properties", {})
        if props.get("surface") != "roof":
            continue
        fid = int(float(props.get("fid", props.get("source_fid", -1))))
        if fid > 0:
            roof_by_fid[fid] = shape(feat["geometry"])

    features = []
    map_rows = []
    missing_source_uids = []
    missing_roof_fids = []
    for row in clean.itertuples(index=False):
        clean_id = int(row.clean_id)
        source_uids = [int(x) for x in str(row.source_uids).split(";") if x.strip()]
        source_fids = []
        geoms = []
        for uid in source_uids:
            fid = uid_to_fid.get(uid)
            if fid is None:
                missing_source_uids.append(uid)
                continue
            source_fids.append(fid)
            geom = roof_by_fid.get(fid)
            if geom is None:
                missing_roof_fids.append(fid)
                continue
            geoms.append(geom)
        if not geoms:
            continue
        merged = unary_union(geoms)
        props = {
            "fid": clean_id,
            "uid": clean_id,
            "clean_id": clean_id,
            "surface": "roof",
            "height_m": float(getattr(row, "height")),
            "floor": int(getattr(row, "Floor")) if pd.notna(getattr(row, "Floor")) else None,
            "source_uids": ";".join(map(str, source_uids)),
            "source_touying_fids": ";".join(map(str, sorted(set(source_fids)))),
        }
        features.append({"type": "Feature", "properties": props, "geometry": mapping(merged)})
        map_rows.append(
            {
                "uid": clean_id,
                "touying_fid": clean_id,
                "height_prior_m": float(getattr(row, "height")),
                "floor_prior": int(getattr(row, "Floor")) if pd.notna(getattr(row, "Floor")) else None,
                "source_uids": ";".join(map(str, source_uids)),
                "source_touying_fids": ";".join(map(str, sorted(set(source_fids)))),
            }
        )

    out = {
        "type": "FeatureCollection",
        "name": "tongji_clean_equal_height_roof_projection_sar",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "features": features,
    }
    Path(args.output_geojson).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_geojson).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_fid_map).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(map_rows).to_csv(args.output_fid_map, index=False)

    summary = {
        "clean_buildings": args.clean_buildings,
        "source_projection": args.source_projection,
        "clean_polygons": int(len(clean)),
        "projected_clean_roofs": int(len(features)),
        "missing_source_uid_count": int(len(set(missing_source_uids))),
        "missing_roof_fid_count": int(len(set(missing_roof_fids))),
        "outputs": {
            "projection_geojson": args.output_geojson,
            "fid_uid_map": args.output_fid_map,
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
