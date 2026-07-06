#!/usr/bin/env python3
"""Aggregate clean-vector roof-only InSAR height points to building polygons."""

from __future__ import annotations

import argparse
import json
import math
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
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def robust_median(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    keep = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    if keep.size == 0:
        keep = arr
    return float(np.median(keep))


HEIGHT_STAT_COLUMNS = {
    "grubbs_top": "building_height_grubbs_top_m",
    "max": "building_height_max_m",
    "median": "height_m",
    "p75": "building_height_p75_m",
    "p85": "building_height_p85_m",
    "p90": "building_height_p90_m",
    "p95": "building_height_p95_m",
}


def source_height_column(points: pd.DataFrame, statistic: str) -> str:
    col = HEIGHT_STAT_COLUMNS[statistic]
    if col in points.columns:
        return col
    if statistic != "p95" and "building_height_p95_m" in points.columns:
        return "building_height_p95_m"
    if "height_m" in points.columns:
        return "height_m"
    raise ValueError(f"No usable height column found for statistic {statistic!r}")


def diagnostic_metrics(gdf: gpd.GeoDataFrame, cols: list[str]) -> list[dict[str, float | int | str]]:
    rows = []
    prior = pd.to_numeric(gdf.get("height_prior_m"), errors="coerce")
    for col in cols:
        if col not in gdf.columns:
            continue
        vals = pd.to_numeric(gdf[col], errors="coerce")
        valid = np.isfinite(vals) & np.isfinite(prior)
        if not bool(valid.any()):
            rows.append({"variant": col, "n": 0})
            continue
        err = vals[valid] - prior[valid]
        rows.append(
            {
                "variant": col,
                "n": int(valid.sum()),
                "median_m": float(vals[valid].median()),
                "p05_m": float(vals[valid].quantile(0.05)),
                "p95_m": float(vals[valid].quantile(0.95)),
                "mae_to_height_diag_m": float(err.abs().mean()),
                "median_abs_to_height_diag_m": float(err.abs().median()),
                "bias_to_height_diag_m": float(err.mean()),
                "rmse_to_height_diag_m": float(np.sqrt(np.mean(err.to_numpy() ** 2))),
            }
        )
    return rows


def aggregate_points(points: pd.DataFrame, height_statistic: str) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["clean_id"])
    selected_col = source_height_column(points, height_statistic)
    rows = []
    for uid, grp in points.groupby("uid", dropna=True):
        selected_row = None
        if height_statistic == "max":
            selected_values = pd.to_numeric(grp[selected_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            height_selected = float(selected_values.max()) if not selected_values.empty else math.nan
            selected_row = grp.loc[selected_values.idxmax()] if not selected_values.empty else None
        else:
            height_selected = robust_median(grp[selected_col])
        height_med = robust_median(grp["height_m"])
        prior_gap = np.nan
        if selected_row is not None:
            max_height_reliable = bool(selected_row.get("max_height_reliable", True))
            max_height_reject_outlier = bool(selected_row.get("max_height_reject_outlier", False))
            max_height_grubbs_p = float(selected_row.get("max_height_grubbs_p", np.nan))
            max_height_grubbs_g = float(selected_row.get("max_height_grubbs_g", np.nan))
            max_height_grubbs_gcrit = float(selected_row.get("max_height_grubbs_gcrit", np.nan))
        else:
            reliable_values = grp.get("max_height_reliable", pd.Series([True] * len(grp), index=grp.index))
            reject_values = grp.get("max_height_reject_outlier", pd.Series([False] * len(grp), index=grp.index))
            max_height_reliable = bool(pd.Series(reliable_values).fillna(True).all())
            max_height_reject_outlier = bool(pd.Series(reject_values).fillna(False).any())
            max_height_grubbs_p = robust_median(grp["max_height_grubbs_p"]) if "max_height_grubbs_p" in grp else np.nan
            max_height_grubbs_g = robust_median(grp["max_height_grubbs_g"]) if "max_height_grubbs_g" in grp else np.nan
            max_height_grubbs_gcrit = robust_median(grp["max_height_grubbs_gcrit"]) if "max_height_grubbs_gcrit" in grp else np.nan
        rows.append(
            {
                "clean_id": int(uid),
                "height_insar_m": height_selected,
                "height_statistic": height_statistic,
                "height_source_column": selected_col,
                "height_iqr_median_m": height_med,
                "building_height_p05_m": robust_median(grp["building_height_p05_m"]),
                "building_height_p25_m": robust_median(grp["building_height_p25_m"]) if "building_height_p25_m" in grp else np.nan,
                "building_height_p50_m": robust_median(grp["building_height_p50_m"]) if "building_height_p50_m" in grp else np.nan,
                "building_height_p75_m": robust_median(grp["building_height_p75_m"]) if "building_height_p75_m" in grp else np.nan,
                "building_height_p85_m": robust_median(grp["building_height_p85_m"]) if "building_height_p85_m" in grp else np.nan,
                "building_height_p90_m": robust_median(grp["building_height_p90_m"]) if "building_height_p90_m" in grp else np.nan,
                "building_height_p95_m": robust_median(grp["building_height_p95_m"]) if "building_height_p95_m" in grp else np.nan,
                "building_height_max_m": float(pd.to_numeric(grp["building_height_max_m"], errors="coerce").max()) if "building_height_max_m" in grp else np.nan,
                "building_height_grubbs_top_m": float(pd.to_numeric(grp["building_height_grubbs_top_m"], errors="coerce").max()) if "building_height_grubbs_top_m" in grp else np.nan,
                "grubbs_top_removed_count": int(pd.to_numeric(grp["grubbs_top_removed_count"], errors="coerce").fillna(0).sum()) if "grubbs_top_removed_count" in grp else 0,
                "grubbs_top_remaining_count": int(pd.to_numeric(grp["grubbs_top_remaining_count"], errors="coerce").fillna(0).sum()) if "grubbs_top_remaining_count" in grp else 0,
                "grubbs_top_last_p": robust_median(grp["grubbs_top_last_p"]) if "grubbs_top_last_p" in grp else np.nan,
                "grubbs_top_reliable": bool(pd.Series(grp["grubbs_top_reliable"]).fillna(False).astype(bool).any()) if "grubbs_top_reliable" in grp else False,
                "max_height_reliable": max_height_reliable,
                "max_height_reject_outlier": max_height_reject_outlier,
                "max_height_grubbs_p": max_height_grubbs_p,
                "max_height_grubbs_g": max_height_grubbs_g,
                "max_height_grubbs_gcrit": max_height_grubbs_gcrit,
                "dem_error_median_m": robust_median(grp["dem_error_median_m"]),
                "roof_elevation_median_m": robust_median(grp["roof_elevation_median_m"]),
                "roof_elevation_p75_m": robust_median(grp["roof_elevation_p75_m"]) if "roof_elevation_p75_m" in grp else np.nan,
                "roof_elevation_p85_m": robust_median(grp["roof_elevation_p85_m"]) if "roof_elevation_p85_m" in grp else np.nan,
                "roof_elevation_p90_m": robust_median(grp["roof_elevation_p90_m"]) if "roof_elevation_p90_m" in grp else np.nan,
                "roof_elevation_p95_m": robust_median(grp["roof_elevation_p95_m"]),
                "reference_elevation_median_m": robust_median(grp["reference_elevation_median_m"]),
                "pixel_count_used": int(grp["pixel_count_used"].sum()),
                "point_count": int(len(grp)),
                "island_count": int(grp["island_id"].nunique()),
                "max_island_uid_count": int(grp["island_uid_count"].max()),
                "median_coherence": robust_median(grp["coh_mean"]),
                "median_amplitude_dispersion": robust_median(grp["amplitude_dispersion"]),
                "lgr_rmse_rad": robust_median(grp["lgr_rmse_rad"]) if "lgr_rmse_rad" in grp else np.nan,
                "valid_pairs_median": robust_median(grp["valid_pairs_median"]) if "valid_pairs_median" in grp else np.nan,
                "bperp_span_median": robust_median(grp["bperp_span_median"]) if "bperp_span_median" in grp else np.nan,
                "source_touying_fids": ",".join(str(int(v)) for v in sorted(grp["touying_fid"].dropna().unique())),
                "source_islands": ",".join(str(int(v)) for v in sorted(grp["island_id"].dropna().unique())),
                "prior_gap_m": prior_gap,
            }
        )
    return pd.DataFrame(rows)


def add_qc(gdf: gpd.GeoDataFrame, prior_gap_review: bool, omit_height_field: bool = False) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["has_insar_height"] = np.isfinite(gdf["height_insar_m"])
    if omit_height_field:
        gdf["height_prior_m"] = np.nan
        gdf["prior_gap_m"] = np.nan
    else:
        gdf["height_prior_m"] = pd.to_numeric(gdf.get("height"), errors="coerce")
        gdf["prior_gap_m"] = gdf["height_insar_m"] - gdf["height_prior_m"]
    gdf["review_multi_clean_id_island"] = gdf["max_island_uid_count"].fillna(0).astype(float) > 1
    gdf["review_negative_height"] = gdf["height_insar_m"] < 0
    gdf["review_extreme_height"] = gdf["height_insar_m"] > 120
    max_reliable = gdf["max_height_reliable"] if "max_height_reliable" in gdf.columns else pd.Series(True, index=gdf.index)
    gdf["review_unreliable_max_height"] = max_reliable.fillna(True).astype(bool).eq(False)
    gdf["review_large_prior_gap"] = False if omit_height_field else (gdf["prior_gap_m"].abs() > 40)
    review_cols = [
        "review_multi_clean_id_island",
        "review_negative_height",
        "review_extreme_height",
        "review_unreliable_max_height",
    ]
    if prior_gap_review:
        review_cols.append("review_large_prior_gap")
    gdf["qc_review"] = gdf[review_cols].any(axis=1)
    gdf["height_source"] = np.where(gdf["has_insar_height"], "insar_roof_only_strict", "no_strict_insar_solution")
    return gdf


def plot_height_map(gdf: gpd.GeoDataFrame, out_png: Path | None, out_svg: Path | None, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), dpi=260)
    base = gdf.copy()
    ok = base[base["has_insar_height"]].copy()
    nodata = base[~base["has_insar_height"]].copy()
    if not nodata.empty:
        nodata.plot(ax=ax, color="#f1f3f4", edgecolor="#c8cdd2", linewidth=0.25)
    if not ok.empty:
        vmax = float(np.nanpercentile(ok["height_insar_m"], 95))
        vmax = max(vmax, 10.0)
        norm = Normalize(vmin=0.0, vmax=vmax)
        ok.plot(
            ax=ax,
            column="height_insar_m",
            cmap="viridis",
            norm=norm,
            edgecolor="#202124",
            linewidth=0.35,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.08)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("InSAR building height (m)")
    review = base[base["qc_review"] & base["has_insar_height"]]
    if not review.empty:
        review.boundary.plot(ax=ax, color="#d73027", linewidth=0.9)

    label_gdf = ok.to_crs(3857) if ok.crs and ok.crs.to_epsg() == 4326 else ok
    full_for_labels = base.to_crs(3857) if base.crs and base.crs.to_epsg() == 4326 else base
    area_cut = np.nanpercentile(full_for_labels.geometry.area, 72) if len(full_for_labels) else 0
    label_gdf = label_gdf[label_gdf.geometry.area >= area_cut].copy()
    if len(label_gdf) > 130:
        label_gdf = label_gdf.sort_values("pixel_count_used", ascending=False).head(130)
    label_plot = label_gdf.to_crs(base.crs) if label_gdf.crs != base.crs else label_gdf
    for _, row in label_plot.iterrows():
        pt = row.geometry.representative_point()
        ax.text(
            pt.x,
            pt.y,
            f"{row['height_insar_m']:.0f}",
            ha="center",
            va="center",
            fontsize=4.6,
            color="black",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()
    ax.text(
        0.01,
        0.01,
        "Legend: colored buildings = strict InSAR height; gray buildings = no strict InSAR solution;\n"
        "red outline = QC review / needs manual check; labels are height in m; no shp-height filling.",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#c8cdd2", "linewidth": 0.6, "alpha": 0.86, "pad": 4},
    )
    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)


def plot_diagnostics(gdf: gpd.GeoDataFrame, out_png: Path | None, out_svg: Path | None) -> None:
    base = gdf.copy()
    plot = base.to_crs(3857) if base.crs and base.crs.to_epsg() == 4326 else base
    plot = plot.copy()
    plot["solution_class"] = np.where(plot["has_insar_height"], 1, 0)
    plot["qc_class"] = 0
    plot.loc[plot["has_insar_height"], "qc_class"] = 1
    plot.loc[plot["review_multi_clean_id_island"] & plot["has_insar_height"], "qc_class"] = 2
    plot.loc[(plot["review_negative_height"] | plot["review_extreme_height"]) & plot["has_insar_height"], "qc_class"] = 3
    plot["quality_class"] = 0
    good = (
        plot["has_insar_height"]
        & (plot["median_coherence"] >= 0.8)
        & (plot["median_amplitude_dispersion"] <= 0.35)
        & (plot["pixel_count_used"] >= 40)
        & ((pd.to_numeric(plot.get("lgr_rmse_rad"), errors="coerce") <= 1.0) if "lgr_rmse_rad" in plot.columns else True)
        & (~plot["qc_review"])
    )
    usable = plot["has_insar_height"] & ~good & (~plot["qc_review"])
    review = plot["has_insar_height"] & plot["qc_review"]
    plot.loc[usable, "quality_class"] = 1
    plot.loc[good, "quality_class"] = 2
    plot.loc[review, "quality_class"] = 3
    plot["height_class"] = 0
    ok = plot[plot["has_insar_height"]].copy()
    if not ok.empty:
        bins = [-np.inf, 10, 20, 35, 50, np.inf]
        plot.loc[plot["has_insar_height"], "height_class"] = pd.cut(
            plot.loc[plot["has_insar_height"], "height_insar_m"],
            bins=bins,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=230)
    axes = axes.ravel()
    plot.plot(ax=axes[0], column="solution_class", categorical=True, cmap=ListedColormap(["#eeeeee", "#1b9e77"]), linewidth=0.05, edgecolor="#666666")
    axes[0].set_title("Strict InSAR coverage")
    axes[0].text(0.02, 0.02, "gray: no solution\nteal: solved", transform=axes[0].transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})

    plot.plot(ax=axes[1], column="qc_class", categorical=True, cmap=ListedColormap(["#eeeeee", "#1b9e77", "#fdbf6f", "#d73027"]), linewidth=0.05, edgecolor="#666666")
    axes[1].set_title("QC review category")
    axes[1].text(0.02, 0.02, "teal: accepted\norange: multi-ID island\nred: internal QC review", transform=axes[1].transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})

    plot.plot(ax=axes[2], column="quality_class", categorical=True, cmap=ListedColormap(["#eeeeee", "#fee08b", "#1a9850", "#d73027"]), linewidth=0.05, edgecolor="#666666")
    axes[2].set_title("Quality gate class")
    axes[2].text(0.02, 0.02, "yellow: usable\ngreen: high quality\nred: review", transform=axes[2].transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})

    plot.plot(ax=axes[3], column="height_class", categorical=True, cmap=ListedColormap(["#eeeeee", "#d9f0a3", "#78c679", "#238443", "#2b8cbe", "#253494"]), linewidth=0.05, edgecolor="#666666")
    axes[3].set_title("Height class map")
    axes[3].text(0.02, 0.02, "<10, 10-20, 20-35,\n35-50, >=50 m", transform=axes[3].transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})

    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.geojson")
    parser.add_argument("--points", default="work/height/height_points_clean_equal_height_roof_only_paper_DA040_coh075_full.csv")
    parser.add_argument("--out-csv", default="results/tables/tongji_building_height_clean_equal_height_roof_only_paper_DA040_coh075_full_insar_only.csv")
    parser.add_argument("--out-geojson", default="results/geodata/tongji_building_height_clean_equal_height_roof_only_paper_DA040_coh075_full_insar_only.geojson")
    parser.add_argument("--summary", default="results/metadata/full_area_clean_equal_height_roof_only_paper_DA040_coh075_insar_only_summary.json")
    parser.add_argument("--height-map-png", default="results/pic_all/png/current_strict_clean_equal_height_full/106_full_area_clean_equal_height_roof_only_paper_DA040_coh075_height_labeled.png")
    parser.add_argument("--height-map-svg", default="results/pic_all/svg/current_strict_clean_equal_height_full/106_full_area_clean_equal_height_roof_only_paper_DA040_coh075_height_labeled.svg")
    parser.add_argument("--diagnostic-png", default="results/pic_all/png/current_strict_clean_equal_height_full/107_full_area_clean_equal_height_roof_only_paper_DA040_coh075_diagnostics.png")
    parser.add_argument("--diagnostic-svg", default="results/pic_all/svg/current_strict_clean_equal_height_full/107_full_area_clean_equal_height_roof_only_paper_DA040_coh075_diagnostics.svg")
    parser.add_argument("--height-statistic", choices=sorted(HEIGHT_STAT_COLUMNS), default="p90", help="InSAR-only building-height statistic to aggregate across source-range roof pixels.")
    parser.add_argument("--prior-gap-review", action="store_true", help="Include large difference from the shapefile height field in qc_review. By default height is diagnostic only.")
    parser.add_argument("--omit-height-field", action="store_true", help="Do not compute or report any comparison to the building height attribute.")
    args = parser.parse_args()

    buildings = gpd.read_file(args.buildings)
    points = pd.read_csv(args.points) if Path(args.points).exists() else pd.DataFrame()
    agg = aggregate_points(points, args.height_statistic)
    out = buildings.merge(agg, on="clean_id", how="left")
    out = add_qc(out, args.prior_gap_review, omit_height_field=args.omit_height_field)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_geojson).parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="geometry").to_csv(args.out_csv, index=False)
    out.to_file(args.out_geojson, driver="GeoJSON")

    plot_height_map(
        out,
        Path(args.height_map_png) if args.height_map_png else None,
        Path(args.height_map_svg) if args.height_map_svg else None,
        "Clean equal-height vectors: roof-only strict InSAR building height",
    )
    if args.diagnostic_png or args.diagnostic_svg:
        plot_diagnostics(
            out,
            Path(args.diagnostic_png) if args.diagnostic_png else None,
            Path(args.diagnostic_svg) if args.diagnostic_svg else None,
        )

    ok = out[out["has_insar_height"]].copy()
    summary = {
        "method": "Clean equal-height vector roof-only projection; paper-like unwrap; SBAS/LGR DEM-residual inversion; building height = selected InSAR-only roof statistic from DSM_RDC + residual - 4 m bare DEM. No shp-height filling or fitting.",
        "height_statistic": args.height_statistic,
        "height_source_column": source_height_column(points, args.height_statistic) if not points.empty else "",
        "prior_gap_review_enabled": bool(args.prior_gap_review),
        "building_vector": args.buildings,
        "height_points": args.points,
        "total_clean_buildings": int(len(out)),
        "buildings_with_strict_insar_height": int(out["has_insar_height"].sum()),
        "buildings_without_strict_insar_height": int((~out["has_insar_height"]).sum()),
        "height_source": "InSAR-only for solved buildings; NaN for unsolved buildings",
        "height_insar_m_median": float(ok["height_insar_m"].median()) if not ok.empty else math.nan,
        "height_insar_m_p05": float(ok["height_insar_m"].quantile(0.05)) if not ok.empty else math.nan,
        "height_insar_m_p95": float(ok["height_insar_m"].quantile(0.95)) if not ok.empty else math.nan,
        "height_iqr_median_m_median": float(ok["height_iqr_median_m"].median()) if not ok.empty else math.nan,
        "median_coherence": float(ok["median_coherence"].median()) if not ok.empty else math.nan,
        "median_amplitude_dispersion": float(ok["median_amplitude_dispersion"].median()) if not ok.empty else math.nan,
        "median_lgr_rmse_rad": float(ok["lgr_rmse_rad"].median()) if "lgr_rmse_rad" in ok.columns and not ok.empty else math.nan,
        "median_valid_pairs": float(ok["valid_pairs_median"].median()) if "valid_pairs_median" in ok.columns and not ok.empty else math.nan,
        "median_bperp_span_m": float(ok["bperp_span_median"].median()) if "bperp_span_median" in ok.columns and not ok.empty else math.nan,
        "qc_review_count": int(out["qc_review"].sum()),
        "review_multi_clean_id_island_count": int(out["review_multi_clean_id_island"].sum()),
        "review_negative_height_count": int(out["review_negative_height"].sum()),
        "review_extreme_height_count": int(out["review_extreme_height"].sum()),
        "review_unreliable_max_height_count": int(out["review_unreliable_max_height"].sum()),
        "review_large_prior_gap_count": int(out["review_large_prior_gap"].sum()),
        "max_height_reliable_count": int(out.get("max_height_reliable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "max_height_reject_outlier_count": int(out.get("max_height_reject_outlier", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "diagnostic_variant_metrics_vs_height_field": []
        if args.omit_height_field
        else diagnostic_metrics(
            out,
            [
                "height_iqr_median_m",
                "building_height_p75_m",
                "building_height_p85_m",
                "building_height_p90_m",
                "building_height_p95_m",
                "building_height_max_m",
                "building_height_grubbs_top_m",
                "height_insar_m",
            ],
        ),
        "height_field_use": "not_read_for_comparison_or_quality_control" if args.omit_height_field else "diagnostic_only_not_fitted_not_used_for_default_qc",
        "outputs": {
            "csv": args.out_csv,
            "geojson": args.out_geojson,
            "height_map_png": args.height_map_png or None,
            "height_map_svg": args.height_map_svg or None,
            "diagnostic_png": args.diagnostic_png or None,
            "diagnostic_svg": args.diagnostic_svg or None,
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
