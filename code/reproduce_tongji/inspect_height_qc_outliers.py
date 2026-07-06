#!/usr/bin/env python3
"""Create focused inspection products for non-OK building-height QC cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_CLASSES = ("prior_mismatch", "low_coherence", "invalid_negative_height")


def classify_action(row: pd.Series) -> str:
    if row["height_qc_class"] == "low_coherence":
        return "manual_phase_or_dsm_check"
    if row["height_qc_class"] == "invalid_negative_height":
        return "inspect_residual_sign_reference_or_unwrap"
    if row["height_qc_class"] == "prior_mismatch":
        if pd.notna(row["height_insar_m"]) and pd.notna(row["height_prior_m"]):
            if float(row["height_insar_m"]) < float(row["height_prior_m"]):
                return "inspect_layover_shadow_or_missing_roof_pixels"
            return "inspect_prior_height_or_island_assignment"
        return "manual_height_check"
    return "no_action"


def build_outlier_table(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf[gdf["height_qc_class"].isin(TARGET_CLASSES)].copy()
    if out.empty:
        return out
    out["height_signed_diff_m"] = out["height_insar_m"] - out["height_prior_m"]
    out["qc_severity"] = np.where(
        out["height_qc_class"] == "low_coherence",
        (0.4 - out["median_coh"]).clip(lower=0),
        out["height_abs_diff_m"].fillna(0),
    )
    out["recommended_review"] = out.apply(classify_action, axis=1)

    centroids = out.to_crs("EPSG:3857").geometry.centroid.to_crs(out.crs)
    out["centroid_lon"] = centroids.x
    out["centroid_lat"] = centroids.y

    sort_cols = ["height_qc_class", "qc_severity", "height_abs_diff_m"]
    return out.sort_values(sort_cols, ascending=[True, False, False])


def plot_outliers(gdf: gpd.GeoDataFrame, outliers: gpd.GeoDataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=220)

    base = gdf.to_crs("EPSG:4326")
    out_wgs = outliers.to_crs("EPSG:4326")
    base.plot(ax=axes[0, 0], color="#eeeeee", linewidth=0.03, edgecolor="#aaaaaa")
    colors = {"prior_mismatch": "#cc6677", "low_coherence": "#ddaa33", "invalid_negative_height": "#882255"}
    for cls, color in colors.items():
        sub = out_wgs[out_wgs["height_qc_class"] == cls]
        if not sub.empty:
            sub.plot(ax=axes[0, 0], color=color, linewidth=0.05, edgecolor="#333333", label=cls)
    axes[0, 0].set_title("Buildings requiring QC inspection")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    axes[0, 0].legend(frameon=False, fontsize=8)

    mismatch = outliers[outliers["height_qc_class"] == "prior_mismatch"]
    axes[0, 1].axhline(0, color="#333333", linewidth=0.8)
    if not mismatch.empty:
        axes[0, 1].scatter(
            mismatch["height_prior_m"],
            mismatch["height_signed_diff_m"],
            s=16,
            alpha=0.7,
            color="#cc6677",
            edgecolors="none",
        )
    axes[0, 1].set_title("Prior mismatch: InSAR minus prior")
    axes[0, 1].set_xlabel("Prior height m")
    axes[0, 1].set_ylabel("Signed difference m")

    low = outliers[outliers["height_qc_class"] == "low_coherence"]
    if not low.empty:
        axes[1, 0].hist(low["median_coh"].dropna(), bins=20, color="#ddaa33", edgecolor="white")
    axes[1, 0].axvline(0.4, color="#333333", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("Low-coherence buildings")
    axes[1, 0].set_xlabel("Median coherence")
    axes[1, 0].set_ylabel("Count")

    label_map = {
        "inspect_layover_shadow_or_missing_roof_pixels": "Layover/shadow\nor missing roof pixels",
        "manual_phase_or_dsm_check": "Manual phase\nor DSM check",
        "inspect_prior_height_or_island_assignment": "Prior height\nor island assignment",
        "inspect_residual_sign_reference_or_unwrap": "Residual sign,\nreference, or unwrap",
    }
    counts = outliers["recommended_review"].value_counts()
    labels = [label_map.get(idx, str(idx)) for idx in counts.index]
    axes[1, 1].barh(labels, counts.values, color="#4477aa")
    axes[1, 1].set_title("Recommended review buckets")
    axes[1, 1].set_xlabel("Buildings")

    for ax in axes.ravel():
        ax.grid(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-geojson", default="results/geodata/tongji_building_height_qc.geojson")
    parser.add_argument("--out-csv", default="results/tables/tongji_building_height_qc_outliers.csv")
    parser.add_argument("--out-geojson", default="results/geodata/tongji_building_height_qc_outliers.geojson")
    parser.add_argument("--summary", default="results/metadata/building_height_qc_outliers_summary.json")
    parser.add_argument("--figure", default="results/pic_all/25_building_height_qc_outliers.png")
    args = parser.parse_args()

    gdf = gpd.read_file(args.qc_geojson)
    outliers = build_outlier_table(gdf)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(outliers.drop(columns="geometry"))
    table.to_csv(args.out_csv, index=False)
    outliers.to_file(args.out_geojson, driver="GeoJSON")
    plot_outliers(gdf, outliers, Path(args.figure))

    summary = {
        "input_qc_geojson": args.qc_geojson,
        "outlier_buildings": int(len(outliers)),
        "qc_counts": {str(k): int(v) for k, v in outliers["height_qc_class"].value_counts().sort_index().items()},
        "recommended_review_counts": {str(k): int(v) for k, v in outliers["recommended_review"].value_counts().sort_index().items()},
        "max_abs_height_diff_m": float(outliers["height_abs_diff_m"].max()) if not outliers.empty else None,
        "min_low_coherence": float(outliers.loc[outliers["height_qc_class"] == "low_coherence", "median_coh"].min())
        if (outliers["height_qc_class"] == "low_coherence").any()
        else None,
        "out_csv": args.out_csv,
        "out_geojson": args.out_geojson,
        "figure": args.figure,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
