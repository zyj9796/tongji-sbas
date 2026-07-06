#!/usr/bin/env python3
"""Build a hypothesis-guided likely roof-top height product."""

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
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    text = series.astype("string").str.lower().str.strip()
    mapped = text.map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    return mapped.fillna(default).astype(bool)


def choose_likely_height(row: pd.Series, tail_gap_m: float, tail_ratio: float, selection_mode: str) -> tuple[float, str, str]:
    p75 = row.get("building_height_p75_m", np.nan)
    p90 = row.get("building_height_p90_m", np.nan)
    p95 = row.get("building_height_p95_m", np.nan)
    hmax = row.get("building_height_max_m", np.nan)
    grubbs_top = row.get("building_height_grubbs_top_m", np.nan)
    has = bool(row.get("has_insar_height", False))
    if not has or not np.isfinite(p90):
        return np.nan, "no_strict_insar_solution", "no_solution"

    max_reliable = bool(row.get("max_height_reliable_bool", False))
    if max_reliable and np.isfinite(hmax):
        return float(hmax), "max_accept_grubbs", "high"

    if selection_mode in {"descending_grubbs", "descending_grubbs_p95_floor", "top_down_grubbs"}:
        grubbs_top_reliable = bool(row.get("grubbs_top_reliable_bool", False))
        removed = row.get("grubbs_top_removed_count", np.nan)
        if grubbs_top_reliable and np.isfinite(grubbs_top):
            if selection_mode == "top_down_grubbs":
                if np.isfinite(removed) and float(removed) <= 2:
                    return float(grubbs_top), "max_reject_use_top_down_grubbs", "medium"
                return float(grubbs_top), "max_reject_use_top_down_grubbs_many_removed", "review"
            if selection_mode == "descending_grubbs_p95_floor" and np.isfinite(p95) and p95 > grubbs_top:
                if np.isfinite(removed) and float(removed) <= 2:
                    return float(p95), "grubbs_top_below_p95_use_p95_floor", "medium"
                return float(p95), "grubbs_top_below_p95_use_p95_floor_many_removed", "review"
            if np.isfinite(removed) and float(removed) <= 2:
                return float(grubbs_top), "max_reject_use_iterative_grubbs_top", "medium"
            return float(grubbs_top), "max_reject_use_iterative_grubbs_top_many_removed", "review"
        if selection_mode == "top_down_grubbs":
            return np.nan, "no_stable_top_after_grubbs", "no_solution"
        if np.isfinite(p95):
            return float(p95), "no_grubbs_top_fallback_p95", "review"

    if np.isfinite(p95):
        gap_95_90 = float(p95 - p90)
        core_gap = float(p90 - p75) if np.isfinite(p75) else np.nan
        stable_by_abs = gap_95_90 <= tail_gap_m
        stable_by_ratio = np.isfinite(core_gap) and core_gap > 0 and gap_95_90 <= tail_ratio * core_gap
        if stable_by_abs or stable_by_ratio:
            return float(p95), "max_reject_use_stable_p95", "medium"

    return float(p90), "max_reject_p95_unstable_use_p90", "review"


def add_colorbar(fig, ax, cmap, norm, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.8%", pad=0.06)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)


def setup_chinese_font() -> FontProperties:
    fontManager.addfont(FONT_PATH)
    prop = FontProperties(fname=FONT_PATH)
    matplotlib.rcParams.update(
        {
            "font.family": prop.get_name(),
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return prop


def plot_outputs(gdf: gpd.GeoDataFrame, out_png: Path | None, out_svg: Path | None, chinese_labels: bool = False) -> None:
    font = setup_chinese_font() if chinese_labels and Path(FONT_PATH).exists() else None
    solved = gdf[gdf["has_likely_top_height"]].copy()
    nodata = gdf[~gdf["has_likely_top_height"]].copy()
    vmax = float(np.nanpercentile(solved["height_likely_top_m"], 95)) if not solved.empty else 10.0
    vmax = max(vmax, 10.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 8.0), dpi=260)
    if not nodata.empty:
        nodata.plot(ax=axes[0], color="#f0f2f3", edgecolor="#d5dadd", linewidth=0.18)
    if not solved.empty:
        solved.plot(
            ax=axes[0],
            column="height_likely_top_m",
            cmap="viridis",
            norm=norm,
            edgecolor="#333333",
            linewidth=0.18,
        )
    review = solved[solved["likely_top_reliability"].eq("review")]
    medium = solved[solved["likely_top_reliability"].eq("medium")]
    if not medium.empty:
        medium.boundary.plot(ax=axes[0], color="#ffb000", linewidth=0.75)
    if not review.empty:
        review.boundary.plot(ax=axes[0], color="#d7191c", linewidth=1.0)
    axes[0].set_title("逐级检验后的建筑顶面高度" if chinese_labels else "Hypothesis-guided likely roof-top height", fontsize=13, fontproperties=font)
    axes[0].set_axis_off()
    add_colorbar(fig, axes[0], "viridis", norm, "建筑高度（米）" if chinese_labels else "height (m)")

    classes = ["no_solution", "high", "medium", "review"]
    colors = ["#eeeeee", "#1a9850", "#fee08b", "#d7191c"]
    code = {name: i for i, name in enumerate(classes)}
    cls = gdf.copy()
    cls["rel_code"] = cls["likely_top_reliability"].map(code).fillna(0).astype(int)
    cls.plot(
        ax=axes[1],
        column="rel_code",
        categorical=True,
        cmap=ListedColormap(colors),
        edgecolor="#666666",
        linewidth=0.12,
    )
    axes[1].set_title("顶点选择可靠性", fontsize=13, fontproperties=font) if chinese_labels else axes[1].set_title("Selection reliability and fallback class", fontsize=13)
    axes[1].set_axis_off()
    if chinese_labels:
        handles = [
            Patch(facecolor="#eeeeee", edgecolor="#666666", label="无严格InSAR解"),
            Patch(facecolor="#1a9850", edgecolor="#666666", label="高可靠：最高点通过检验"),
            Patch(facecolor="#fee08b", edgecolor="#666666", label="中可靠：剔除异常后取顶点"),
            Patch(facecolor="#d7191c", edgecolor="#666666", label="需复核：顶端序列不稳定"),
        ]
        axes[1].legend(handles=handles, loc="lower left", prop=font, fontsize=8, frameon=True, framealpha=0.9)
        fig.suptitle("建筑顶面高度：从最高点向下逐级异常检验", fontsize=14, fontproperties=font)
    else:
        handles = [
            Patch(facecolor="#eeeeee", edgecolor="#666666", label="no strict InSAR solution"),
            Patch(facecolor="#1a9850", edgecolor="#666666", label="high: max accepted by Grubbs"),
            Patch(facecolor="#fee08b", edgecolor="#666666", label="medium: tested top fallback"),
            Patch(facecolor="#d7191c", edgecolor="#666666", label="review: weak/unstable top fallback"),
        ]
        axes[1].legend(handles=handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
        fig.suptitle("Most likely building top height from max-value hypothesis testing", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/geodata/tongji_building_height_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_max_grubbs_insar_only.geojson",
    )
    parser.add_argument("--tail-gap-m", type=float, default=8.0)
    parser.add_argument("--tail-ratio", type=float, default=0.75)
    parser.add_argument(
        "--selection-mode",
        choices=["stable_quantile", "descending_grubbs", "descending_grubbs_p95_floor", "top_down_grubbs"],
        default="stable_quantile",
        help="stable_quantile uses p95/p90 after rejecting max; descending_grubbs uses the highest value left after iterative Grubbs testing; descending_grubbs_p95_floor prevents many-removal cases from falling below p95; top_down_grubbs never falls back to quantiles.",
    )
    parser.add_argument(
        "--out-geojson",
        default="results/geodata/tongji_building_height_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_likely_top_insar_only.geojson",
    )
    parser.add_argument(
        "--out-csv",
        default="results/tables/tongji_building_height_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_likely_top_insar_only.csv",
    )
    parser.add_argument(
        "--out-png",
        default="results/pic_all/png/current_strict_clean_equal_height_full/141_full_area_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_likely_top_height.png",
    )
    parser.add_argument(
        "--out-svg",
        default="results/pic_all/svg/current_strict_clean_equal_height_full/141_full_area_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_likely_top_height.svg",
    )
    parser.add_argument(
        "--summary",
        default="results/metadata/full_area_clean_equal_height_roof_only_full_area_128_paper_opt_coh080_DA035_likely_top_insar_only_summary.json",
    )
    parser.add_argument("--omit-height-field", action="store_true", help="Do not compute or report any comparison to the building height attribute.")
    parser.add_argument("--chinese-labels", action="store_true", help="Use Chinese titles, legends, and colorbar labels in the output figure.")
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    gdf = gdf.copy()
    gdf["has_insar_height"] = as_bool(gdf.get("has_insar_height", pd.Series(False, index=gdf.index)), default=False)
    gdf["max_height_reliable_bool"] = as_bool(gdf.get("max_height_reliable", pd.Series(False, index=gdf.index)), default=False)
    gdf["grubbs_top_reliable_bool"] = as_bool(gdf.get("grubbs_top_reliable", pd.Series(False, index=gdf.index)), default=False)
    choices = gdf.apply(lambda row: choose_likely_height(row, args.tail_gap_m, args.tail_ratio, args.selection_mode), axis=1)
    gdf["height_likely_top_m"] = [v[0] for v in choices]
    gdf["likely_top_method"] = [v[1] for v in choices]
    gdf["likely_top_reliability"] = [v[2] for v in choices]
    gdf["has_likely_top_height"] = np.isfinite(gdf["height_likely_top_m"])
    gdf["height_insar_m"] = gdf["height_likely_top_m"]
    gdf["height_statistic"] = "likely_top_hypothesis_guided"
    gdf["height_source_column"] = gdf["likely_top_method"]
    if args.omit_height_field:
        gdf["likely_top_minus_height_diag_m"] = np.nan
    else:
        gdf["likely_top_minus_height_diag_m"] = gdf["height_likely_top_m"] - pd.to_numeric(gdf.get("height"), errors="coerce")

    out_geojson = Path(args.out_geojson)
    out_csv = Path(args.out_csv)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    pd.DataFrame(gdf.drop(columns="geometry")).to_csv(out_csv, index=False)
    if args.out_png or args.out_svg:
        plot_outputs(
            gdf,
            Path(args.out_png) if args.out_png else None,
            Path(args.out_svg) if args.out_svg else None,
            chinese_labels=args.chinese_labels,
        )

    solved = gdf[gdf["has_likely_top_height"]].copy()
    err = solved["likely_top_minus_height_diag_m"].dropna()
    summary = {
        "input": args.input,
        "method": "Hypothesis-guided top-height estimate. stable_quantile uses max when Grubbs accepts it, then stable p95/p90. descending_grubbs starts from 100% and uses the highest value retained after iterative Grubbs testing. descending_grubbs_p95_floor protects against over-removal by keeping the result no lower than p95 and marking many-removal cases for review. top_down_grubbs starts from the maximum and iteratively removes only statistically rejected high outliers, with no quantile fallback.",
        "selection_mode": args.selection_mode,
        "tail_gap_m": args.tail_gap_m,
        "tail_ratio": args.tail_ratio,
        "buildings_total": int(len(gdf)),
        "buildings_with_likely_top_height": int(gdf["has_likely_top_height"].sum()),
        "method_counts": {str(k): int(v) for k, v in gdf["likely_top_method"].value_counts(dropna=False).sort_index().items()},
        "reliability_counts": {str(k): int(v) for k, v in gdf["likely_top_reliability"].value_counts(dropna=False).sort_index().items()},
        "height_likely_top_m_median": float(solved["height_likely_top_m"].median()) if not solved.empty else math.nan,
        "height_likely_top_m_p05": float(solved["height_likely_top_m"].quantile(0.05)) if not solved.empty else math.nan,
        "height_likely_top_m_p95": float(solved["height_likely_top_m"].quantile(0.95)) if not solved.empty else math.nan,
        "height_field_use": "not_read_for_comparison_or_quality_control" if args.omit_height_field else "diagnostic_only_not_fitted_or_used_for_quality_control",
        "diagnostic_mae_to_height_m": None if args.omit_height_field else (float(err.abs().mean()) if not err.empty else math.nan),
        "diagnostic_bias_to_height_m": None if args.omit_height_field else (float(err.mean()) if not err.empty else math.nan),
        "diagnostic_rmse_to_height_m": None if args.omit_height_field else (float(np.sqrt(np.mean(err.to_numpy() ** 2))) if not err.empty else math.nan),
        "outputs": {
            "geojson": args.out_geojson,
            "csv": args.out_csv,
            "png": args.out_png or None,
            "svg": args.out_svg or None,
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
