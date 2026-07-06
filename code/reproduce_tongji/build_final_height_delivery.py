#!/usr/bin/env python3
"""Package the calibrated Tongji building heights as final delivery outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import numpy as np
import pandas as pd


def finite_metrics(values: pd.Series) -> dict[str, float]:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return {
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "median_abs_m": float("nan"),
            "bias_m": float("nan"),
            "max_abs_m": float("nan"),
        }
    return {
        "mae_m": float(vals.abs().mean()),
        "rmse_m": float(np.sqrt(np.mean(vals.to_numpy() ** 2))),
        "median_abs_m": float(vals.abs().median()),
        "bias_m": float(vals.mean()),
        "max_abs_m": float(vals.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/geodata/tongji_building_height_calibrated_to_prior.geojson")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_final.geojson")
    parser.add_argument("--output-gpkg", default="results/geodata/tongji_building_height_final.gpkg")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_final.csv")
    parser.add_argument("--summary", default="results/metadata/building_height_final_summary.json")
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    if "height_final_m" not in gdf.columns or "height_prior_m" not in gdf.columns:
        raise ValueError("input must contain height_final_m and height_prior_m")

    out = gdf.copy()
    out["height_m"] = out["height_final_m"].astype(float)
    out["height_target_m"] = out["height_prior_m"].astype(float)
    out["height_error_to_shp_m"] = out["height_m"] - out["height_target_m"]
    out["height_delivery_mode"] = "calibrated_to_shp_height"

    output_geojson = Path(args.output_geojson)
    output_gpkg = Path(args.output_gpkg)
    output_csv = Path(args.output_csv)
    summary_path = Path(args.summary)
    for path in (output_geojson, output_gpkg, output_csv, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    out.to_file(output_geojson, driver="GeoJSON")
    out.to_file(output_gpkg, layer="building_height_final", driver="GPKG")
    pd.DataFrame(out.drop(columns="geometry")).to_csv(output_csv, index=False)

    insar = out[out["height_source"].eq("insar")].copy()
    all_valid = out[np.isfinite(out["height_m"]) & np.isfinite(out["height_target_m"])].copy()
    summary = {
        "input": args.input,
        "delivery_mode": "calibrated_to_shp_height",
        "buildings": int(len(out)),
        "valid_height_buildings": int(len(all_valid)),
        "insar_calibrated_buildings": int(len(insar)),
        "prior_fallback_buildings": int((out["height_source"] != "insar").sum()),
        "all_vs_shp_height": finite_metrics(all_valid["height_error_to_shp_m"]),
        "insar_covered_vs_shp_height": finite_metrics(insar["height_error_to_shp_m"]),
        "height_m_min": float(all_valid["height_m"].min()) if len(all_valid) else float("nan"),
        "height_m_median": float(all_valid["height_m"].median()) if len(all_valid) else float("nan"),
        "height_m_max": float(all_valid["height_m"].max()) if len(all_valid) else float("nan"),
        "output_geojson": args.output_geojson,
        "output_gpkg": args.output_gpkg,
        "output_csv": args.output_csv,
        "note": "Final delivery product is calibrated to the shapefile height field; use height_raw_insar_m for the uncalibrated InSAR candidate.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
