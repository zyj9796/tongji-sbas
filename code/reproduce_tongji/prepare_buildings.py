#!/usr/bin/env python3
"""Prepare Tongji building vectors with stable IDs and metric attributes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import pandas as pd


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(config: dict, output_gpkg: Path, output_geojson: Path, output_csv: Path, summary_path: Path) -> dict:
    shp = Path(config["paths"]["buildings_shp"])
    prior_field = config["height_aggregation"]["prior_height_field"]
    floor_field = config["height_aggregation"]["floor_field"]

    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf.insert(0, "uid", range(1, len(gdf) + 1))
    gdf["original_id"] = gdf["Id"] if "Id" in gdf.columns else None
    gdf["height_prior_m"] = pd.to_numeric(gdf.get(prior_field), errors="coerce")
    gdf["floor_prior"] = pd.to_numeric(gdf.get(floor_field), errors="coerce")
    gdf["height_insar_m"] = pd.NA
    gdf["height_final_m"] = gdf["height_prior_m"]
    gdf["height_source"] = "prior"
    gdf["qc_flag"] = "not_processed"
    gdf["n_points"] = 0
    gdf["n_islands"] = 0

    metric = gdf.to_crs("EPSG:32651")
    gdf["area_m2"] = metric.geometry.area
    centroids = metric.geometry.centroid.to_crs("EPSG:4326")
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(output_gpkg, layer="buildings", driver="GPKG")
    gdf.to_file(output_geojson, driver="GeoJSON")
    csv_cols = [c for c in gdf.columns if c != "geometry"]
    gdf[csv_cols].to_csv(output_csv, index=False)

    original_id_unique = int(gdf["original_id"].nunique(dropna=False)) if "original_id" in gdf else 0
    summary = {
        "input": str(shp),
        "output_gpkg": str(output_gpkg),
        "output_geojson": str(output_geojson),
        "output_csv": str(output_csv),
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "metric_crs_for_area": "EPSG:32651",
        "uid_min": int(gdf["uid"].min()) if len(gdf) else None,
        "uid_max": int(gdf["uid"].max()) if len(gdf) else None,
        "original_id_unique_count": original_id_unique,
        "height_prior_nonnull": int(gdf["height_prior_m"].notna().sum()),
        "floor_prior_nonnull": int(gdf["floor_prior"].notna().sum()),
        "area_m2_min": float(gdf["area_m2"].min()) if len(gdf) else None,
        "area_m2_median": float(gdf["area_m2"].median()) if len(gdf) else None,
        "area_m2_max": float(gdf["area_m2"].max()) if len(gdf) else None,
        "note": "Use uid for all SAR-coordinate projection, mask, island, and height aggregation joins.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--output-gpkg", default="results/geodata/tongji_buildings_prepared.gpkg")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_buildings_prepared.geojson")
    parser.add_argument("--output-csv", default="results/tables/tongji_buildings_prepared.csv")
    parser.add_argument("--summary", default="results/metadata/buildings_prepared_summary.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    summary = prepare(
        config,
        Path(args.output_gpkg),
        Path(args.output_geojson),
        Path(args.output_csv),
        Path(args.summary),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
