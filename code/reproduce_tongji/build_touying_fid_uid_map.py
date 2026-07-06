#!/usr/bin/env python3
"""Build the mapping between Touying source FIDs and local building UIDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_projection_fids(path: Path) -> set[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fids = set()
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("surface") != "bottom":
            continue
        if props.get("fid") in (None, ""):
            continue
        fids.add(int(float(props["fid"])))
    return fids


def load_raster_fids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    rows = csv.DictReader(path.open("r", encoding="utf-8", newline=""))
    fids = set()
    for row in rows:
        value = row.get("primary_uid")
        if value not in (None, ""):
            fids.add(int(float(value)))
    return fids


def build_mapping(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    full = gpd.read_file(args.full_shp).to_crs("EPSG:4326")
    extent = gpd.read_file(args.extent_geojson).to_crs("EPSG:4326")
    local = gpd.read_file(args.local_buildings).to_crs("EPSG:4326")
    projection_fids = load_projection_fids(Path(args.projection_geojson))
    raster_fids = load_raster_fids(Path(args.islands_csv))

    inter = (
        gpd.sjoin(full.reset_index().rename(columns={"index": "touying_fid"}), extent[["geometry"]], predicate="intersects", how="inner")
        .sort_values("touying_fid")
        .reset_index(drop=True)
    )
    if len(inter) != len(local):
        raise RuntimeError(f"Local subset length mismatch: local={len(local)} intersects={len(inter)}")

    out = local.copy()
    out["touying_fid"] = inter["touying_fid"].astype(int).to_numpy()
    if "uid" not in out.columns:
        out["uid"] = range(1, len(out) + 1)
    out["in_touying_blue_projection"] = out["touying_fid"].isin(projection_fids)
    out["in_touying_blue_raster"] = out["touying_fid"].isin(raster_fids)

    rows = []
    for _, row in out.drop(columns="geometry").iterrows():
        rows.append(
            {
                "uid": int(row["uid"]),
                "touying_fid": int(row["touying_fid"]),
                "height_prior_m": float(row.get("height_prior_m", row.get("height", 0.0))),
                "floor_prior": float(row.get("floor_prior", row.get("Floor", 0.0))),
                "in_touying_blue_projection": bool(row["in_touying_blue_projection"]),
                "in_touying_blue_raster": bool(row["in_touying_blue_raster"]),
            }
        )
    table = pd.DataFrame(rows)
    local_fids = set(table["touying_fid"].astype(int))
    summary = {
        "local_buildings": int(len(table)),
        "touying_projection_features": int(len(projection_fids)),
        "touying_projection_fids_in_local": int(len(local_fids & projection_fids)),
        "touying_projection_fids_outside_local": int(len(projection_fids - local_fids)),
        "local_buildings_missing_projection": int(len(local_fids - projection_fids)),
        "touying_raster_primary_fids": int(len(raster_fids)),
        "touying_raster_primary_fids_in_local": int(len(local_fids & raster_fids)),
        "local_buildings_represented_as_primary_raster_fid": int(table["in_touying_blue_raster"].sum()),
        "note": "touying_fid is the original full tongji_clip.shp feature index used by touying_roof_workflow.",
    }
    return out, table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-shp", default="tmp/reference_inputs/shp/tongji_clip.shp")
    parser.add_argument("--extent-geojson", default="tmp/reference_inputs/rslc_extent/tongji_rslc_extent_wgs84.geojson")
    parser.add_argument("--local-buildings", default="results/geodata/tongji_buildings_prepared.geojson")
    parser.add_argument("--projection-geojson", default="work/projection/20200708_blue_aligned_bottom_touying.geojson")
    parser.add_argument("--islands-csv", default="work/masks/islands_touying_blue_bottom.csv")
    parser.add_argument("--output-csv", default="results/tables/touying_fid_uid_map.csv")
    parser.add_argument("--output-geojson", default="results/geodata/touying_fid_uid_buildings.geojson")
    parser.add_argument("--summary", default="results/metadata/touying_fid_uid_map_summary.json")
    args = parser.parse_args()

    gdf, table, summary = build_mapping(args)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_geojson).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    gdf.to_file(args.output_geojson, driver="GeoJSON")
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
