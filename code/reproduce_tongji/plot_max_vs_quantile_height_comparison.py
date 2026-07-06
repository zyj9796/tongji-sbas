#!/usr/bin/env python3
"""Compare max-height inversion against the previous quantile-height product."""

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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default)
    if series.dtype == bool:
        return series.fillna(default)
    text = series.astype("string").str.lower().str.strip()
    out = text.map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    return out.fillna(default).astype(bool)


def add_colorbar(fig, ax, cmap, norm, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)


def plot_height_panel(ax, gdf: gpd.GeoDataFrame, column: str, title: str, norm: Normalize) -> None:
    nodata = gdf[~gdf["has_both"]]
    ok = gdf[gdf["has_both"]]
    if not nodata.empty:
        nodata.plot(ax=ax, color="#f0f2f3", edgecolor="#d5dadd", linewidth=0.18)
    if not ok.empty:
        ok.plot(ax=ax, column=column, cmap="viridis", norm=norm, edgecolor="#333333", linewidth=0.18)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantile-geojson",
        default="results/geodata/tongji_building_height_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_insar_only.geojson",
    )
    parser.add_argument(
        "--max-geojson",
        default="results/geodata/tongji_building_height_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_max_grubbs_insar_only.geojson",
    )
    parser.add_argument("--diff-threshold-m", type=float, default=10.0)
    parser.add_argument("--large-diff-threshold-m", type=float, default=20.0)
    parser.add_argument(
        "--out-png",
        default="results/pic_all/png/current_strict_clean_equal_height_full/138_max_vs_p90_height_comparison_marked.png",
    )
    parser.add_argument(
        "--out-svg",
        default="results/pic_all/svg/current_strict_clean_equal_height_full/138_max_vs_p90_height_comparison_marked.svg",
    )
    parser.add_argument(
        "--summary",
        default="results/metadata/max_vs_p90_height_comparison_summary.json",
    )
    args = parser.parse_args()

    q = gpd.read_file(args.quantile_geojson)
    m = gpd.read_file(args.max_geojson)
    q_cols = ["clean_id", "height_insar_m", "building_height_p90_m"]
    m_cols = [
        "clean_id",
        "height_insar_m",
        "building_height_max_m",
        "max_height_reliable",
        "max_height_reject_outlier",
        "max_height_grubbs_p",
        "review_unreliable_max_height",
    ]
    q_attr = pd.DataFrame(q[[c for c in q_cols if c in q.columns]].drop(columns=[], errors="ignore")).rename(
        columns={"height_insar_m": "height_p90_product_m"}
    )
    m_attr = pd.DataFrame(m[[c for c in m_cols if c in m.columns]].drop(columns=[], errors="ignore")).rename(
        columns={"height_insar_m": "height_max_product_m"}
    )
    geom = m[["clean_id", "geometry"]].copy()
    out = geom.merge(q_attr, on="clean_id", how="left").merge(m_attr, on="clean_id", how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=m.crs)
    out["has_both"] = np.isfinite(out["height_p90_product_m"]) & np.isfinite(out["height_max_product_m"])
    out["delta_max_minus_p90_m"] = out["height_max_product_m"] - out["height_p90_product_m"]
    out["abs_delta_m"] = out["delta_max_minus_p90_m"].abs()
    if "review_unreliable_max_height" in out.columns:
        out["max_height_reliable"] = ~as_bool(out["review_unreliable_max_height"], default=False)
    else:
        out["max_height_reliable"] = as_bool(out.get("max_height_reliable", pd.Series(True, index=out.index)), default=False)
    out["max_height_reject_outlier"] = as_bool(out.get("max_height_reject_outlier", pd.Series(False, index=out.index)), default=False)
    out["change_class"] = "no strict InSAR solution"
    out.loc[out["has_both"] & (out["abs_delta_m"] <= args.diff_threshold_m), "change_class"] = f"<= {args.diff_threshold_m:.0f} m"
    out.loc[
        out["has_both"] & (out["abs_delta_m"] > args.diff_threshold_m) & (out["abs_delta_m"] <= args.large_diff_threshold_m),
        "change_class",
    ] = f"{args.diff_threshold_m:.0f}-{args.large_diff_threshold_m:.0f} m"
    out.loc[out["has_both"] & (out["abs_delta_m"] > args.large_diff_threshold_m), "change_class"] = f"> {args.large_diff_threshold_m:.0f} m"
    out.loc[out["has_both"] & (~out["max_height_reliable"]), "change_class"] = "max rejected by Grubbs"

    solved = out[out["has_both"]].copy()
    changed = solved[solved["abs_delta_m"] > args.diff_threshold_m].copy()
    large_changed = solved[solved["abs_delta_m"] > args.large_diff_threshold_m].copy()
    unreliable = solved[~solved["max_height_reliable"]].copy()

    vmax = float(np.nanpercentile(solved[["height_p90_product_m", "height_max_product_m"]].to_numpy(), 95)) if not solved.empty else 1.0
    vmax = max(vmax, 10.0)
    height_norm = Normalize(vmin=0.0, vmax=vmax)
    delta_lim = float(np.nanpercentile(solved["delta_max_minus_p90_m"], 95)) if not solved.empty else 1.0
    delta_lim = max(delta_lim, args.large_diff_threshold_m)
    delta_norm = Normalize(vmin=0.0, vmax=delta_lim)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 12.0), dpi=260)
    ax = axes.ravel()
    plot_height_panel(ax[0], out, "height_p90_product_m", "Previous quantile result: p90 height", height_norm)
    plot_height_panel(ax[1], out, "height_max_product_m", "New result: maximum height", height_norm)
    for axis in ax[:2]:
        if not changed.empty:
            changed.boundary.plot(ax=axis, color="#ff7f00", linewidth=0.7)
        if not unreliable.empty:
            unreliable.boundary.plot(ax=axis, color="#d7191c", linewidth=1.1)
    add_colorbar(fig, ax[1], "viridis", height_norm, "height (m)")

    nodata = out[~out["has_both"]]
    if not nodata.empty:
        nodata.plot(ax=ax[2], color="#f0f2f3", edgecolor="#d5dadd", linewidth=0.18)
    if not solved.empty:
        solved.plot(ax=ax[2], column="delta_max_minus_p90_m", cmap="magma", norm=delta_norm, edgecolor="#333333", linewidth=0.16)
    if not large_changed.empty:
        large_changed.boundary.plot(ax=ax[2], color="#00bfc4", linewidth=0.95)
    if not unreliable.empty:
        unreliable.boundary.plot(ax=ax[2], color="#d7191c", linewidth=1.15)
    ax[2].set_title("Difference map: max - p90", fontsize=12)
    ax[2].set_axis_off()
    add_colorbar(fig, ax[2], "magma", delta_norm, "max - p90 (m)")

    class_order = [
        "no strict InSAR solution",
        f"<= {args.diff_threshold_m:.0f} m",
        f"{args.diff_threshold_m:.0f}-{args.large_diff_threshold_m:.0f} m",
        f"> {args.large_diff_threshold_m:.0f} m",
        "max rejected by Grubbs",
    ]
    class_colors = ["#eeeeee", "#1a9850", "#fee08b", "#f46d43", "#d7191c"]
    class_to_code = {name: i for i, name in enumerate(class_order)}
    class_plot = out.copy()
    class_plot["change_code"] = class_plot["change_class"].map(class_to_code).fillna(0).astype(int)
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(class_order) + 0.5, 1), cmap.N)
    class_plot.plot(ax=ax[3], column="change_code", categorical=True, cmap=cmap, norm=norm, edgecolor="#666666", linewidth=0.12)
    ax[3].set_title("Marked changed buildings", fontsize=12)
    ax[3].set_axis_off()
    handles = [Patch(facecolor=color, edgecolor="#666666", label=label) for label, color in zip(class_order, class_colors)]
    handles.extend(
        [
            Line2D([0], [0], color="#ff7f00", lw=1.5, label=f"outlined: |delta| > {args.diff_threshold_m:.0f} m"),
            Line2D([0], [0], color="#00bfc4", lw=1.5, label=f"cyan outline: |delta| > {args.large_diff_threshold_m:.0f} m"),
            Line2D([0], [0], color="#d7191c", lw=1.8, label="red outline: Grubbs rejects max"),
        ]
    )
    ax[3].legend(handles=handles, loc="lower left", fontsize=7.4, frameon=True, framealpha=0.9)

    fig.suptitle("Maximum-height inversion compared with previous p90 quantile product", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)

    summary = {
        "quantile_geojson": args.quantile_geojson,
        "max_geojson": args.max_geojson,
        "solved_both": int(len(solved)),
        "changed_gt_threshold": int(len(changed)),
        "changed_gt_large_threshold": int(len(large_changed)),
        "max_rejected_by_grubbs": int(len(unreliable)),
        "delta_threshold_m": args.diff_threshold_m,
        "large_delta_threshold_m": args.large_diff_threshold_m,
        "delta_median_m": float(solved["delta_max_minus_p90_m"].median()) if not solved.empty else np.nan,
        "delta_p05_m": float(solved["delta_max_minus_p90_m"].quantile(0.05)) if not solved.empty else np.nan,
        "delta_p95_m": float(solved["delta_max_minus_p90_m"].quantile(0.95)) if not solved.empty else np.nan,
        "outputs": {"png": str(out_png), "svg": str(out_svg)},
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
