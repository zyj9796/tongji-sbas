#!/usr/bin/env python3
"""Build a full-area per-building height estimate.

Trusted InSAR source-range heights are used where available. All remaining
buildings receive a DSM footprint estimate using DSM p95 minus the constant
bare-earth DEM assumption.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated.*", category=DeprecationWarning)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds


def valid_values(ds: rasterio.io.DatasetReader, geom, pad_m: float = 0.0) -> np.ndarray:
    target = geom.buffer(pad_m) if pad_m else geom
    if target.is_empty:
        return np.array([], dtype=np.float32)
    minx, miny, maxx, maxy = target.bounds
    try:
        window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform).round_offsets().round_lengths()
        arr = ds.read(1, window=window, boundless=True, fill_value=ds.nodata).astype(np.float32)
    except Exception:
        return np.array([], dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    transform = ds.window_transform(window)
    mask = geometry_mask([target], out_shape=arr.shape, transform=transform, invert=True)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if ds.nodata is not None:
        vals = vals[vals != float(ds.nodata)]
    return vals


def pct(vals: np.ndarray, q: float) -> float:
    return float(np.percentile(vals, q)) if vals.size else float("nan")


def add_dsm_stats(gdf: gpd.GeoDataFrame, dsm_path: Path, ground_dem_m: float, ring_outer_m: float, ring_inner_m: float) -> gpd.GeoDataFrame:
    with rasterio.open(dsm_path) as ds:
        projected = gdf.to_crs(ds.crs)
        rows = []
        for geom in projected.geometry:
            footprint = valid_values(ds, geom)
            ring_geom = geom.buffer(ring_outer_m).difference(geom.buffer(ring_inner_m))
            ring = valid_values(ds, ring_geom)
            fp05 = pct(footprint, 5)
            fp50 = pct(footprint, 50)
            fp95 = pct(footprint, 95)
            ring10 = pct(ring, 10)
            height_const_ground = fp95 - ground_dem_m if np.isfinite(fp95) else np.nan
            height_ring_ground = fp95 - ring10 if np.isfinite(fp95) and np.isfinite(ring10) else np.nan
            rows.append(
                {
                    "dsm_footprint_pixels": int(footprint.size),
                    "dsm_ring_pixels": int(ring.size),
                    "dsm_footprint_p05_m": fp05,
                    "dsm_footprint_median_m": fp50,
                    "dsm_footprint_p95_m": fp95,
                    "dsm_ring_ground_p10_m": ring10,
                    "height_dsm_p95_minus_4m_m": max(0.0, height_const_ground) if np.isfinite(height_const_ground) else np.nan,
                    "height_dsm_p95_minus_ring_m": max(0.0, height_ring_ground) if np.isfinite(height_ring_ground) else np.nan,
                    "dsm_relief_p95_p05_m": fp95 - fp05 if np.isfinite(fp95) and np.isfinite(fp05) else np.nan,
                }
            )
    out = gdf.copy()
    stats = pd.DataFrame(rows)
    for col in stats.columns:
        out[col] = stats[col].to_numpy()
    return out


def plot_outputs(gdf: gpd.GeoDataFrame, out_map: Path, out_diag: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=260)
    gdf.plot(ax=ax, column="height_est_m", cmap="viridis", legend=True, linewidth=0.05, edgecolor="#333333")
    ax.set_axis_off()
    ax.set_title("Full-area building height estimate")
    fig.tight_layout()
    out_map.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_map)
    plt.close(fig)

    insar = gdf[gdf["height_est_source"] == "insar_source_range_trusted"].copy()
    fallback = gdf[gdf["height_est_source"] == "dsm_footprint_p95_minus_4m"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=230)
    axes[0, 0].hist(gdf["height_est_m"].dropna(), bins=55, color="#4477aa", edgecolor="white")
    axes[0, 0].set_title("All estimated heights")
    axes[0, 0].set_xlabel("m")
    axes[0, 1].scatter(gdf["height_prior_m"], gdf["height_est_m"], s=8, alpha=0.45, color="#117733")
    lim = max(float(gdf["height_prior_m"].max()), float(gdf["height_est_m"].max()), 1.0)
    axes[0, 1].plot([0, lim], [0, lim], color="0.25", lw=0.8, ls="--")
    axes[0, 1].set_title("Estimate vs rough shp height prior")
    axes[0, 1].set_xlabel("height prior, m")
    axes[0, 1].set_ylabel("height estimate, m")
    counts = gdf["height_est_source"].value_counts()
    axes[1, 0].bar(counts.index, counts.values, color=["#117733" if "insar" in k else "#88ccee" for k in counts.index])
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_title("Height source counts")
    axes[1, 1].hist(fallback["height_est_m"].dropna(), bins=50, alpha=0.75, label="DSM fallback", color="#88ccee", edgecolor="white")
    if not insar.empty:
        axes[1, 1].hist(insar["height_est_m"].dropna(), bins=25, alpha=0.75, label="trusted InSAR", color="#117733", edgecolor="white")
    axes[1, 1].set_title("Height distributions by source")
    axes[1, 1].set_xlabel("m")
    axes[1, 1].legend()
    fig.tight_layout()
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_diag)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_buildings_normalized.geojson")
    parser.add_argument("--trusted", default="results/tables/tongji_building_height_paper_roof_100_source_range_trusted.csv")
    parser.add_argument("--dsm", default="data/dsm/tongji_real_dsm_1m_rslc_extent.tif")
    parser.add_argument("--ground-dem-m", type=float, default=4.0)
    parser.add_argument("--ring-outer-m", type=float, default=8.0)
    parser.add_argument("--ring-inner-m", type=float, default=1.0)
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_full_area_estimate.geojson")
    parser.add_argument("--output-gpkg", default="results/geodata/tongji_building_height_full_area_estimate.gpkg")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_full_area_estimate.csv")
    parser.add_argument("--summary", default="results/metadata/tongji_building_height_full_area_estimate_summary.json")
    parser.add_argument("--map", default="results/pic_all/53_full_area_building_height_estimate_map.png")
    parser.add_argument("--diagnostic-figure", default="results/pic_all/54_full_area_building_height_estimate_diagnostics.png")
    args = parser.parse_args()

    buildings = gpd.read_file(args.buildings)
    trusted = pd.read_csv(args.trusted)
    trusted_cols = ["uid", "height_insar_m", "confidence_class", "median_coh", "source_pixel_count_used", "island_id"]
    trusted = trusted[[col for col in trusted_cols if col in trusted.columns]].rename(
        columns={
            "height_insar_m": "height_insar_trusted_m",
            "confidence_class": "insar_confidence_class",
            "median_coh": "insar_median_coh",
            "source_pixel_count_used": "insar_pixel_count_used",
            "island_id": "insar_island_id",
        }
    )
    out = buildings.merge(trusted, on="uid", how="left")
    out = add_dsm_stats(out, Path(args.dsm), args.ground_dem_m, args.ring_outer_m, args.ring_inner_m)
    has_trusted = out["height_insar_trusted_m"].notna()
    has_dsm = out["height_dsm_p95_minus_4m_m"].notna()
    out["height_est_m"] = np.select(
        [has_trusted, has_dsm],
        [out["height_insar_trusted_m"], out["height_dsm_p95_minus_4m_m"]],
        default=out["height_prior_m"],
    )
    out["height_est_source"] = np.select(
        [has_trusted, has_dsm],
        ["insar_source_range_trusted", "dsm_footprint_p95_minus_4m"],
        default="prior_no_dsm_fallback",
    )
    out["height_est_minus_prior_m"] = out["height_est_m"] - out["height_prior_m"]
    out["height_est_abs_diff_prior_m"] = out["height_est_minus_prior_m"].abs()
    out["height_est_note"] = np.where(
        has_trusted,
        "Trusted roof-mask paper-unwrapped LGR source-range candidate",
        "DSM footprint p95 minus constant 4 m bare-earth DEM fallback",
    )
    out.loc[out["height_est_source"] == "prior_no_dsm_fallback", "height_est_note"] = "No valid DSM footprint pixels; rough shapefile height used only as last-resort coverage fallback"

    out_geojson = Path(args.output_geojson)
    out_gpkg = Path(args.output_gpkg)
    out_csv = Path(args.output_csv)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_geojson, driver="GeoJSON")
    out.to_file(out_gpkg, driver="GPKG")
    pd.DataFrame(out.drop(columns="geometry")).to_csv(out_csv, index=False)
    plot_outputs(out, Path(args.map), Path(args.diagnostic_figure))

    summary = {
        "buildings": int(len(out)),
        "height_estimated_buildings": int(out["height_est_m"].notna().sum()),
        "ground_dem_m": args.ground_dem_m,
        "source_counts": {str(k): int(v) for k, v in out["height_est_source"].value_counts().sort_index().items()},
        "height_est_median_m": float(out["height_est_m"].median()),
        "height_est_p05_m": float(out["height_est_m"].quantile(0.05)),
        "height_est_p95_m": float(out["height_est_m"].quantile(0.95)),
        "diagnostic_mae_to_height_prior_m": float(out["height_est_abs_diff_prior_m"].mean()),
        "diagnostic_bias_to_height_prior_m": float(out["height_est_minus_prior_m"].mean()),
        "diagnostic_median_abs_diff_to_height_prior_m": float(out["height_est_abs_diff_prior_m"].median()),
        "outputs": {
            "geojson": args.output_geojson,
            "gpkg": args.output_gpkg,
            "csv": args.output_csv,
            "map": args.map,
            "diagnostic_figure": args.diagnostic_figure,
        },
        "note": "Full-area estimate. Trusted InSAR is used where QC passed; otherwise DSM p95 minus 4 m is used. Shapefile height is diagnostic only, not used to fit the estimate.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
