#!/usr/bin/env python3
"""Plot spatial changes between two InSAR-only building-height products."""

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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def setup_font() -> None:
    if Path(FONT_PATH).exists():
        fontManager.addfont(FONT_PATH)
        matplotlib.rcParams.update(
            {
                "font.family": "Noto Sans CJK JP",
                "axes.unicode_minus": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
            }
        )


def add_colorbar(fig, ax, cmap, norm, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)


def plot_height(ax, gdf: gpd.GeoDataFrame, column: str, title: str, norm: Normalize) -> None:
    empty = gdf[~np.isfinite(gdf[column])]
    valid = gdf[np.isfinite(gdf[column])]
    if not empty.empty:
        empty.plot(ax=ax, color="#f1f3f4", edgecolor="#d9dee2", linewidth=0.12)
    if not valid.empty:
        valid.plot(ax=ax, column=column, cmap="viridis", norm=norm, edgecolor="#303030", linewidth=0.12)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def main() -> None:
    setup_font()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--label-old", default="old")
    parser.add_argument("--label-new", default="new")
    parser.add_argument("--diff-threshold-m", type=float, default=2.0)
    parser.add_argument("--large-diff-threshold-m", type=float, default=5.0)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-svg", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--chinese-labels", action="store_true")
    args = parser.parse_args()

    old = gpd.read_file(args.old)
    new = gpd.read_file(args.new)
    old_attr = pd.DataFrame(old[["clean_id", "height_insar_m"]]).rename(columns={"height_insar_m": "old_height_m"})
    new_attr = pd.DataFrame(new[["clean_id", "height_insar_m"]]).rename(columns={"height_insar_m": "new_height_m"})
    geom = new[["clean_id", "geometry"]].copy()
    out = geom.merge(old_attr, on="clean_id", how="left").merge(new_attr, on="clean_id", how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=new.crs)
    out["has_old"] = np.isfinite(out["old_height_m"])
    out["has_new"] = np.isfinite(out["new_height_m"])
    out["has_both"] = out["has_old"] & out["has_new"]
    out["delta_new_old_m"] = out["new_height_m"] - out["old_height_m"]
    out["abs_delta_m"] = out["delta_new_old_m"].abs()

    out["change_class"] = "no solution"
    out.loc[out["has_old"] & ~out["has_new"], "change_class"] = "lost"
    out.loc[~out["has_old"] & out["has_new"], "change_class"] = "new"
    out.loc[out["has_both"] & (out["abs_delta_m"] <= args.diff_threshold_m), "change_class"] = "stable"
    out.loc[
        out["has_both"] & (out["abs_delta_m"] > args.diff_threshold_m) & (out["abs_delta_m"] <= args.large_diff_threshold_m),
        "change_class",
    ] = "changed"
    out.loc[out["has_both"] & (out["abs_delta_m"] > args.large_diff_threshold_m), "change_class"] = "large change"

    both = out[out["has_both"]].copy()
    heights = out[["old_height_m", "new_height_m"]].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(heights, 95)) if np.isfinite(heights).any() else 10.0
    height_norm = Normalize(vmin=0, vmax=max(vmax, 10.0))
    delta_v = float(np.nanpercentile(both["abs_delta_m"], 98)) if not both.empty else args.large_diff_threshold_m
    delta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max(delta_v, args.large_diff_threshold_m), vmax=max(delta_v, args.large_diff_threshold_m))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.2), dpi=260)
    ax = axes.ravel()
    plot_height(ax[0], out, "old_height_m", args.label_old, height_norm)
    plot_height(ax[1], out, "new_height_m", args.label_new, height_norm)
    add_colorbar(fig, ax[1], "viridis", height_norm, "建筑高度（米）" if args.chinese_labels else "height (m)")

    no_delta = out[~out["has_both"]]
    if not no_delta.empty:
        no_delta.plot(ax=ax[2], color="#f1f3f4", edgecolor="#d9dee2", linewidth=0.12)
    if not both.empty:
        both.plot(ax=ax[2], column="delta_new_old_m", cmap="coolwarm", norm=delta_norm, edgecolor="#303030", linewidth=0.12)
    changed = out[out["change_class"].isin(["changed", "large change", "new", "lost"])]
    large = out[out["change_class"] == "large change"]
    if not changed.empty:
        changed.boundary.plot(ax=ax[2], color="#f28e2b", linewidth=0.75)
    if not large.empty:
        large.boundary.plot(ax=ax[2], color="#d62728", linewidth=1.05)
    ax[2].set_title(f"{args.label_new} - {args.label_old}", fontsize=11)
    ax[2].set_axis_off()
    add_colorbar(fig, ax[2], "coolwarm", delta_norm, "高度差（米）" if args.chinese_labels else "delta (m)")

    order = ["no solution", "stable", "changed", "large change", "new", "lost"]
    label_map = {
        "no solution": "无严格InSAR解",
        "stable": "稳定",
        "changed": "变化>阈值",
        "large change": "大变化",
        "new": "新增解",
        "lost": "失解",
    }
    colors = ["#eeeeee", "#2ca25f", "#fee08b", "#d73027", "#2b8cbe", "#7b3294"]
    codes = {name: idx for idx, name in enumerate(order)}
    out["change_code"] = out["change_class"].map(codes).fillna(0).astype(int)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(order) + 0.5, 1), cmap.N)
    out.plot(ax=ax[3], column="change_code", categorical=True, cmap=cmap, norm=norm, edgecolor="#666666", linewidth=0.1)
    ax[3].set_title("高度变化分类" if args.chinese_labels else "Marked differences", fontsize=11)
    ax[3].set_axis_off()
    handles = [
        Patch(facecolor=color, edgecolor="#666666", label=label_map[label] if args.chinese_labels else label)
        for label, color in zip(order, colors)
    ]
    ax[3].legend(handles=handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.92)

    fig.suptitle("InSAR建筑高度空间变化对比" if args.chinese_labels else "InSAR-only building-height spatial change", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)

    summary = {
        "old": args.old,
        "new": args.new,
        "label_old": args.label_old,
        "label_new": args.label_new,
        "old_count": int(out["has_old"].sum()),
        "new_count": int(out["has_new"].sum()),
        "common_count": int(out["has_both"].sum()),
        "new_only_count": int((~out["has_old"] & out["has_new"]).sum()),
        "lost_count": int((out["has_old"] & ~out["has_new"]).sum()),
        "changed_gt_threshold_count": int((out["has_both"] & (out["abs_delta_m"] > args.diff_threshold_m)).sum()),
        "changed_gt_large_threshold_count": int((out["has_both"] & (out["abs_delta_m"] > args.large_diff_threshold_m)).sum()),
        "delta_threshold_m": args.diff_threshold_m,
        "large_delta_threshold_m": args.large_diff_threshold_m,
        "delta_median_m": float(both["delta_new_old_m"].median()) if not both.empty else None,
        "delta_p05_m": float(both["delta_new_old_m"].quantile(0.05)) if not both.empty else None,
        "delta_p95_m": float(both["delta_new_old_m"].quantile(0.95)) if not both.empty else None,
        "height_field_use": "not_used",
        "outputs": {"png": str(out_png), "svg": str(out_svg), "json": args.summary},
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
