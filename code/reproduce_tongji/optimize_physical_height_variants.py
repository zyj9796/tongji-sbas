#!/usr/bin/env python3
"""Build non-fitted physical height variants and visualize diagnostics."""

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


VARIANTS = {
    "median_current_m": "height_m",
    "p95_roof_m": "building_height_p95_m",
    "p05_roof_m": "building_height_p05_m",
}


def robust_median(series: pd.Series) -> float:
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        return np.nan
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
    if keep.empty:
        keep = vals
    return float(keep.median())


def aggregate_points(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, grp in points.groupby("uid"):
        row = {
            "uid": int(uid),
            "physical_n_height_points": int(len(grp)),
            "physical_n_pixels_used_total": int(grp["pixel_count_used"].sum()),
            "physical_n_islands": int(grp["island_id"].nunique()),
            "physical_median_coh": float(grp["coh_mean"].median()) if grp["coh_mean"].notna().any() else np.nan,
            "physical_median_reference_elevation_m": robust_median(grp["reference_elevation_median_m"]),
            "physical_median_dem_error_m": robust_median(grp["dem_error_median_m"]),
        }
        for out_col, source_col in VARIANTS.items():
            row[out_col] = robust_median(grp[source_col])
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame, col: str) -> dict[str, float | int]:
    valid = np.isfinite(df[col]) & np.isfinite(df["height_prior_m"])
    sub = df.loc[valid].copy()
    if sub.empty:
        return {"n": 0}
    err = sub[col] - sub["height_prior_m"]
    return {
        "n": int(len(sub)),
        "median_m": float(sub[col].median()),
        "p02_m": float(sub[col].quantile(0.02)),
        "p98_m": float(sub[col].quantile(0.98)),
        "negative_count": int((sub[col] < 0).sum()),
        "mae_to_prior_diag_m": float(err.abs().mean()),
        "rmse_to_prior_diag_m": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
        "bias_to_prior_diag_m": float(err.mean()),
        "median_abs_to_prior_diag_m": float(err.abs().median()),
    }


def classify(row: pd.Series) -> str:
    h = row.get("p95_roof_m")
    if not np.isfinite(h):
        return "no_insar"
    if h < 0:
        return "invalid_negative_height"
    if row.get("physical_median_coh", np.nan) < 0.4:
        return "low_coherence"
    prior = row.get("height_prior_m", np.nan)
    if np.isfinite(prior) and abs(h - prior) > max(20.0, 0.6 * max(prior, 1.0)):
        return "prior_diagnostic_gap"
    return "ok_physical_p95"


def plot_outputs(gdf: gpd.GeoDataFrame, comparison: pd.DataFrame, figure: Path) -> None:
    ins = gdf[gdf["height_source_physical"].eq("insar_p95")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=220)

    axes[0, 0].hist(ins["median_current_m"].dropna(), bins=45, alpha=0.7, color="#4477aa", label="median")
    axes[0, 0].hist(ins["p95_roof_m"].dropna(), bins=45, alpha=0.55, color="#228833", label="p95 roof")
    axes[0, 0].set_title("InSAR height distributions")
    axes[0, 0].set_xlabel("height m")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(ins["height_prior_m"], ins["median_current_m"], s=8, alpha=0.35, color="#4477aa", label="median")
    axes[0, 1].scatter(ins["height_prior_m"], ins["p95_roof_m"], s=8, alpha=0.35, color="#228833", label="p95 roof")
    lim = float(np.nanmax([ins["height_prior_m"].max(), ins["p95_roof_m"].max(), 1.0]))
    axes[0, 1].plot([0, lim], [0, lim], color="#222222", linewidth=0.8)
    axes[0, 1].set_title("Diagnostic comparison to shp height")
    axes[0, 1].set_xlabel("shp height prior m")
    axes[0, 1].set_ylabel("InSAR candidate m")
    axes[0, 1].legend(frameon=False)

    err = ins["p95_roof_m"] - ins["height_prior_m"]
    axes[0, 2].hist(err.dropna(), bins=45, color="#aa3377", edgecolor="white")
    axes[0, 2].axvline(0, color="#222222", linewidth=0.8)
    axes[0, 2].set_title("p95 roof minus shp prior")
    axes[0, 2].set_xlabel("m")

    comparison.set_index("variant")[["mae_to_prior_diag_m", "rmse_to_prior_diag_m"]].plot.bar(
        ax=axes[1, 0], color=["#66c2a5", "#fc8d62"]
    )
    axes[1, 0].set_title("Prior diagnostic error by variant")
    axes[1, 0].set_ylabel("m")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(frameon=False)

    gdf.plot(ax=axes[1, 1], color="#eeeeee", linewidth=0.03, edgecolor="#bbbbbb")
    ins.plot(ax=axes[1, 1], column="p95_roof_m", cmap="viridis", linewidth=0.04, edgecolor="#333333", legend=True)
    axes[1, 1].set_title("Physical p95 InSAR heights")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")

    cls_counts = gdf["height_qc_physical"].value_counts()
    axes[1, 2].bar(cls_counts.index, cls_counts.values, color="#88ccee")
    axes[1, 2].set_title("Physical p95 QC counts")
    axes[1, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].set_ylabel("buildings")

    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_building_height_qc.geojson")
    parser.add_argument("--points", default="work/height/height_points.csv")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_physical_p95.csv")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_physical_p95.geojson")
    parser.add_argument("--comparison-csv", default="results/tables/tongji_building_height_variant_comparison.csv")
    parser.add_argument("--summary", default="results/metadata/building_height_physical_p95_summary.json")
    parser.add_argument("--figure", default="results/pic_all/29_building_height_physical_p95_diagnostics.png")
    args = parser.parse_args()

    buildings = gpd.read_file(args.buildings)
    points = pd.read_csv(args.points)
    agg = aggregate_points(points)
    out = buildings.merge(agg, on="uid", how="left")
    out["height_physical_p95_m"] = out["p95_roof_m"]
    out["height_source_physical"] = np.where(np.isfinite(out["height_physical_p95_m"]), "insar_p95", "no_insar")
    out["height_qc_physical"] = out.apply(classify, axis=1)
    out["height_abs_diff_prior_diag_m"] = (out["height_physical_p95_m"] - out["height_prior_m"]).abs()
    out["height_signed_diff_prior_diag_m"] = out["height_physical_p95_m"] - out["height_prior_m"]

    comparison_rows = []
    for variant in ["median_current_m", "p95_roof_m", "p05_roof_m"]:
        row = {"variant": variant}
        row.update(metrics(out, variant))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_geojson).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out.drop(columns="geometry")).to_csv(args.output_csv, index=False)
    out.to_file(args.output_geojson, driver="GeoJSON")
    comparison.to_csv(args.comparison_csv, index=False)
    plot_outputs(out, comparison, Path(args.figure))

    summary = {
        "input_buildings": args.buildings,
        "input_points": args.points,
        "method": "non-fitted physical variant: use building footprint p95 of DSM_RDC + signed LGR residual - 4 m, then robust median across building height points",
        "note": "The shapefile height is used only for diagnostics/QC, not for fitting, stretching, or calibration.",
        "buildings": int(len(out)),
        "insar_p95_buildings": int(out["height_source_physical"].eq("insar_p95").sum()),
        "no_insar_buildings": int(out["height_source_physical"].eq("no_insar").sum()),
        "qc_counts": {str(k): int(v) for k, v in out["height_qc_physical"].value_counts().sort_index().items()},
        "variant_metrics": comparison.to_dict(orient="records"),
        "output_csv": args.output_csv,
        "output_geojson": args.output_geojson,
        "comparison_csv": args.comparison_csv,
        "figure": args.figure,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
