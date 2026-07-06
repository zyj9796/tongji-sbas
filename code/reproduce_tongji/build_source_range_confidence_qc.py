#!/usr/bin/env python3
"""Confidence QC for the roof-mask source-range height candidate."""

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


def confidence_class(row: pd.Series) -> str:
    if row["height_source"] != "insar":
        return "prior_only"
    h = float(row["height_insar_m"])
    coh = row.get("median_coh", np.nan)
    err = row.get("err_to_prior_m", np.nan)
    abs_err = abs(err) if pd.notna(err) else np.nan
    prior = row.get("height_prior_m", np.nan)
    if pd.isna(h) or h <= 0:
        return "reject_nonpositive"
    if pd.notna(coh) and coh < 0.45:
        return "review_low_coherence"
    if pd.notna(abs_err) and pd.notna(prior):
        if abs_err >= 25.0:
            return "review_large_gap"
        if abs_err >= 15.0 and abs_err / max(float(prior), 1.0) >= 0.6:
            return "review_prior_gap"
    if pd.notna(coh) and coh >= 0.6:
        return "high_confidence"
    return "medium_confidence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_building_height_paper_roof_100_source_range_qc.geojson")
    parser.add_argument("--points", default="work/height/height_points_paper_roof_100_source_range.csv")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_paper_roof_100_source_range_confidence_qc.csv")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_paper_roof_100_source_range_confidence_qc.geojson")
    parser.add_argument("--summary", default="results/metadata/tongji_building_height_paper_roof_100_source_range_confidence_qc_summary.json")
    parser.add_argument("--figure", default="results/pic_all/51_paper_roof_100_source_range_confidence_qc.png")
    args = parser.parse_args()

    gdf = gpd.read_file(args.buildings)
    pts = pd.read_csv(args.points)
    pts["height_source_range_m"] = pd.to_numeric(pts["height_source_range_m"], errors="coerce")
    pts_extra = pts[
        [
            "uid",
            "touying_fid",
            "island_id",
            "island_uid_count",
            "pixel_count_used",
            "building_height_p05_m",
            "building_height_p95_m",
            "dem_error_source_range_m",
        ]
    ].copy()
    pts_extra = pts_extra.rename(
        columns={
            "pixel_count_used": "source_pixel_count_used",
            "building_height_p05_m": "source_height_p05_m",
            "building_height_p95_m": "source_height_p95_m",
        }
    )
    gdf = gdf.merge(pts_extra, on="uid", how="left")
    gdf["err_to_prior_m"] = np.where(
        gdf["height_source"] == "insar",
        gdf["height_insar_m"] - gdf["height_prior_m"],
        np.nan,
    )
    gdf["abs_err_to_prior_m"] = gdf["err_to_prior_m"].abs()
    gdf["confidence_class"] = gdf.apply(confidence_class, axis=1)

    table = pd.DataFrame(gdf.drop(columns="geometry"))
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_geojson).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    gdf.to_file(args.output_geojson, driver="GeoJSON")

    insar = table[table["height_source"] == "insar"].copy()
    counts = table["confidence_class"].value_counts().sort_index()
    summary = {
        "buildings": int(len(table)),
        "insar_buildings": int(len(insar)),
        "confidence_counts": {str(k): int(v) for k, v in counts.items()},
        "insar_median_height_m": float(insar["height_insar_m"].median()),
        "insar_mae_to_prior_m_diagnostic_only": float(insar["abs_err_to_prior_m"].mean()),
        "high_or_medium_confidence_buildings": int(insar["confidence_class"].isin(["high_confidence", "medium_confidence"]).sum()),
        "review_buildings": int(insar["confidence_class"].str.startswith("review").sum()),
        "figure": args.figure,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=220)
    class_order = [
        "high_confidence",
        "medium_confidence",
        "review_low_coherence",
        "review_prior_gap",
        "review_large_gap",
        "prior_only",
    ]
    colors = {
        "high_confidence": "#117733",
        "medium_confidence": "#44aa99",
        "review_low_coherence": "#ddcc77",
        "review_prior_gap": "#cc6677",
        "review_large_gap": "#882255",
        "prior_only": "#d9d9d9",
    }
    plot_counts = table["confidence_class"].value_counts().reindex(class_order).fillna(0)
    axes[0, 0].bar(plot_counts.index, plot_counts.values, color=[colors[c] for c in plot_counts.index])
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].set_title("Confidence QC counts")
    axes[0, 1].scatter(insar["height_prior_m"], insar["height_insar_m"], c=insar["confidence_class"].map(colors), s=18, alpha=0.75)
    max_lim = max(float(insar["height_prior_m"].max()), float(insar["height_insar_m"].max()), 1.0)
    axes[0, 1].plot([0, max_lim], [0, max_lim], color="0.25", lw=0.8, ls="--")
    axes[0, 1].set_xlabel("height prior, m")
    axes[0, 1].set_ylabel("source-range height, m")
    axes[0, 1].set_title("Height diagnostic")
    axes[1, 0].scatter(insar["median_coh"], insar["abs_err_to_prior_m"], c=insar["confidence_class"].map(colors), s=18, alpha=0.75)
    axes[1, 0].set_xlabel("median coherence")
    axes[1, 0].set_ylabel("abs difference to prior, m")
    axes[1, 0].set_title("Coherence vs diagnostic gap")
    gdf.plot(ax=axes[1, 1], color="#eeeeee", linewidth=0.03, edgecolor="#aaaaaa")
    for cls in class_order[:-1]:
        sub = gdf[gdf["confidence_class"] == cls]
        if not sub.empty:
            sub.plot(ax=axes[1, 1], color=colors[cls], edgecolor="#222222", linewidth=0.04, label=cls)
    axes[1, 1].set_axis_off()
    axes[1, 1].legend(frameon=False, fontsize=6)
    axes[1, 1].set_title("Confidence map")
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
