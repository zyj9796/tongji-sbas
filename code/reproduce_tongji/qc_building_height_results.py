#!/usr/bin/env python3
"""QC the GAMMA/LGR building height candidate results."""

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


def classify(row: pd.Series, mismatch_abs: float, mismatch_ratio: float, low_coh: float) -> str:
    if row["height_source"] != "insar":
        return "prior_only"
    prior = row["height_prior_m"]
    insar = row["height_insar_m"]
    coh = row.get("median_coh", np.nan)
    if pd.notna(insar) and float(insar) < 0:
        return "invalid_negative_height"
    if pd.notna(coh) and coh < low_coh:
        return "low_coherence"
    if pd.notna(prior) and pd.notna(insar):
        diff = abs(insar - prior)
        ratio = diff / max(float(prior), 1.0)
        if diff >= mismatch_abs and ratio >= mismatch_ratio:
            return "prior_mismatch"
    return "ok"


def plot_qc(gdf: gpd.GeoDataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=220)
    insar = gdf[gdf["height_source"] == "insar"].copy()
    axes[0, 0].hist(insar["height_insar_m"].dropna(), bins=45, color="#4477aa", edgecolor="white")
    axes[0, 0].set_title("GAMMA/InSAR candidate heights")
    axes[0, 0].set_xlabel("m")
    axes[0, 1].scatter(insar["height_prior_m"], insar["height_insar_m"], s=8, alpha=0.55, color="#117733")
    lim = max(float(insar["height_prior_m"].max()), float(insar["height_insar_m"].max()), 1.0)
    axes[0, 1].plot([0, lim], [0, lim], color="#333333", linewidth=0.8)
    axes[0, 1].set_title("Prior vs GAMMA/InSAR")
    axes[0, 1].set_xlabel("height_prior_m")
    axes[0, 1].set_ylabel("height_insar_m")
    status_order = ["ok", "prior_mismatch", "low_coherence", "invalid_negative_height", "prior_only"]
    counts = gdf["height_qc_class"].value_counts().reindex(status_order).fillna(0)
    axes[1, 0].bar(counts.index, counts.values, color=["#117733", "#cc6677", "#ddcc77", "#882255", "#aaaaaa"])
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_title("QC class counts")
    gdf.plot(ax=axes[1, 1], color="#eeeeee", linewidth=0.03, edgecolor="#999999")
    colors = {"ok": "#117733", "prior_mismatch": "#cc6677", "low_coherence": "#ddcc77", "invalid_negative_height": "#882255"}
    for cls, color in colors.items():
        sub = gdf[gdf["height_qc_class"] == cls]
        if not sub.empty:
            sub.plot(ax=axes[1, 1], color=color, linewidth=0.04, edgecolor="#333333", label=cls)
    axes[1, 1].set_title("QC map")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].tick_params(axis="x", labelrotation=20, labelsize=7)
    axes[1, 1].tick_params(axis="y", labelsize=7)
    for ax in axes.ravel():
        ax.grid(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_building_height_insar.geojson")
    parser.add_argument("--mismatch-abs", type=float, default=15.0)
    parser.add_argument("--mismatch-ratio", type=float, default=0.6)
    parser.add_argument("--low-coh", type=float, default=0.4)
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_qc.csv")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_qc.geojson")
    parser.add_argument("--summary", default="results/metadata/building_height_qc_summary.json")
    parser.add_argument("--figure", default="results/pic_all/22_building_height_qc.png")
    args = parser.parse_args()

    gdf = gpd.read_file(args.buildings)
    gdf["height_abs_diff_m"] = np.where(
        gdf["height_source"] == "insar",
        (gdf["height_insar_m"] - gdf["height_prior_m"]).abs(),
        np.nan,
    )
    gdf["height_rel_diff"] = gdf["height_abs_diff_m"] / gdf["height_prior_m"].clip(lower=1.0)
    gdf["height_qc_class"] = gdf.apply(lambda row: classify(row, args.mismatch_abs, args.mismatch_ratio, args.low_coh), axis=1)
    table = pd.DataFrame(gdf.drop(columns="geometry"))
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_geojson).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    gdf.to_file(args.output_geojson, driver="GeoJSON")
    plot_qc(gdf, Path(args.figure))
    insar = table[table["height_source"] == "insar"]
    summary = {
        "buildings": int(len(table)),
        "insar_buildings": int(len(insar)),
        "prior_only_buildings": int((table["height_source"] == "prior").sum()),
        "qc_counts": {str(k): int(v) for k, v in table["height_qc_class"].value_counts().sort_index().items()},
        "mismatch_abs_threshold_m": args.mismatch_abs,
        "mismatch_ratio_threshold": args.mismatch_ratio,
        "low_coherence_threshold": args.low_coh,
        "insar_height_median_m": float(insar["height_insar_m"].median()),
        "insar_height_p98_m": float(insar["height_insar_m"].quantile(0.98)),
        "figure": args.figure,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
