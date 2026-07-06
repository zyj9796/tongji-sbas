#!/usr/bin/env python3
"""Diagnose height-reference variants without fitting to the shapefile prior."""

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


def iqr_median(values: pd.Series) -> float:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return np.nan
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
    if keep.empty:
        keep = vals
    return float(keep.median())


def metrics(df: pd.DataFrame, col: str) -> dict[str, float | int]:
    valid = np.isfinite(df[col]) & np.isfinite(df["height_prior_m"])
    sub = df.loc[valid].copy()
    if sub.empty:
        return {"n": 0}
    err = sub[col] - sub["height_prior_m"]
    return {
        "n": int(len(sub)),
        "median_m": float(sub[col].median()),
        "p05_m": float(sub[col].quantile(0.05)),
        "p95_m": float(sub[col].quantile(0.95)),
        "negative_count": int((sub[col] < 0).sum()),
        "mae_to_prior_diag_m": float(err.abs().mean()),
        "median_abs_to_prior_diag_m": float(err.abs().median()),
        "bias_to_prior_diag_m": float(err.mean()),
        "rmse_to_prior_diag_m": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
    }


def plot(df: pd.DataFrame, comparison: pd.DataFrame, figure: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=240)
    cols = [
        ("current_dsm_plus_residual_minus_ground_m", "current sign"),
        ("opposite_residual_dsm_minus_ground_m", "opposite sign"),
        ("dsm_only_minus_ground_m", "DSM only"),
    ]
    for col, label in cols:
        axes[0].hist(df[col].dropna(), bins=24, alpha=0.55, label=label)
    axes[0].axvline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Height distributions")
    axes[0].set_xlabel("height m")
    axes[0].legend(frameon=False, fontsize=8)

    comparison.set_index("variant")[["mae_to_prior_diag_m", "median_abs_to_prior_diag_m"]].plot.bar(
        ax=axes[1], color=["#66c2a5", "#fc8d62"]
    )
    axes[1].set_title("Diagnostic difference to shp prior")
    axes[1].set_ylabel("m")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].scatter(
        df["height_prior_m"],
        df["current_dsm_plus_residual_minus_ground_m"],
        s=26,
        alpha=0.75,
        label="current",
        color="#4477aa",
    )
    axes[2].scatter(
        df["height_prior_m"],
        df["opposite_residual_dsm_minus_ground_m"],
        s=26,
        alpha=0.75,
        label="opposite",
        color="#cc6677",
    )
    lim = float(np.nanmax([df["height_prior_m"].max(), df["opposite_residual_dsm_minus_ground_m"].max(), 1.0]))
    axes[2].plot([0, lim], [0, lim], color="#222222", linewidth=0.8)
    axes[2].set_title("Reference-sign diagnostic")
    axes[2].set_xlabel("shp height prior m")
    axes[2].set_ylabel("candidate height m")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default="work/height/height_points_paper_subset.csv")
    parser.add_argument("--buildings", default="results/geodata/tongji_buildings_normalized.geojson")
    parser.add_argument("--ground-dem-m", type=float, default=4.0)
    parser.add_argument("--output-csv", default="results/tables/height_reference_variant_diagnostic_paper_subset.csv")
    parser.add_argument("--comparison-csv", default="results/tables/height_reference_variant_comparison_paper_subset.csv")
    parser.add_argument("--summary", default="results/metadata/height_reference_variant_diagnostic_paper_subset_summary.json")
    parser.add_argument("--figure", default="results/pic_all/33_height_reference_variant_diagnostic_paper_subset.png")
    args = parser.parse_args()

    points = pd.read_csv(args.points)
    points["current_dsm_plus_residual_minus_ground_m"] = points["reference_elevation_median_m"] + points["dem_error_median_m"] - args.ground_dem_m
    points["opposite_residual_dsm_minus_ground_m"] = points["reference_elevation_median_m"] - points["dem_error_median_m"] - args.ground_dem_m
    points["dsm_only_minus_ground_m"] = points["reference_elevation_median_m"] - args.ground_dem_m
    points["signed_residual_only_m"] = points["dem_error_median_m"]
    points["opposite_residual_only_m"] = -points["dem_error_median_m"]

    rows = []
    variant_cols = [
        "current_dsm_plus_residual_minus_ground_m",
        "opposite_residual_dsm_minus_ground_m",
        "dsm_only_minus_ground_m",
        "signed_residual_only_m",
        "opposite_residual_only_m",
    ]
    for uid, grp in points.groupby("uid"):
        row = {
            "uid": int(uid),
            "height_point_count": int(len(grp)),
            "pixel_count_used_sum": int(grp["pixel_count_used"].sum()),
            "coh_median": float(grp["coh_mean"].median()),
        }
        for col in variant_cols:
            row[col] = iqr_median(grp[col])
        rows.append(row)
    out = pd.DataFrame(rows)

    buildings = gpd.read_file(args.buildings)[["uid", "height_prior_m", "geometry"]]
    out = buildings.merge(out, on="uid", how="inner")
    comparison_rows = []
    for col in variant_cols:
        row = {"variant": col}
        row.update(metrics(out, col))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out.drop(columns="geometry")).to_csv(args.output_csv, index=False)
    comparison.to_csv(args.comparison_csv, index=False)
    plot(pd.DataFrame(out.drop(columns="geometry")), comparison, Path(args.figure))

    summary = {
        "points": args.points,
        "buildings": int(len(out)),
        "ground_dem_m": args.ground_dem_m,
        "note": "Diagnostic only: shapefile height is used for scoring variants, not for fitting.",
        "comparison": comparison.to_dict(orient="records"),
        "output_csv": args.output_csv,
        "comparison_csv": args.comparison_csv,
        "figure": args.figure,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
