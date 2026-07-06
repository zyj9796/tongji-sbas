#!/usr/bin/env python3
"""Chinese visual QA figures for the 128 full-area height inversion branch."""

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
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def setup_style() -> FontProperties:
    fontManager.addfont(FONT_PATH)
    prop = FontProperties(fname=FONT_PATH)
    matplotlib.rcParams.update(
        {
            "font.family": prop.get_name(),
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    return prop


def stretch(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    lo, hi = np.percentile(arr[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def save(fig: plt.Figure, png: Path, svg: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=280, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def sar_axis(ax: plt.Axes, amp: np.ndarray, title: str) -> None:
    ax.imshow(amp, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("SAR距离向列")
    ax.set_ylabel("SAR方位向行")


def fig_projection_and_masks(args: argparse.Namespace, font: FontProperties) -> None:
    amp = stretch(np.load(args.amp_npy).astype(np.float32))
    projection = gpd.read_file(args.projection_geojson)
    shifted = projection[(projection["final_row_shift"].astype(float) != 0) | (projection["final_col_shift"].astype(float) != 0)]
    unchanged = projection.drop(shifted.index)
    fid_mask = np.load(args.fid_mask)
    island = np.load(args.island_label)
    da = np.load(args.amplitude_dispersion_npy).astype(np.float32)
    valid_da = np.isfinite(da) & (da <= args.max_amplitude_dispersion)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220, sharex=True, sharey=True)
    axes = axes.ravel()

    sar_axis(axes[0], amp, "步骤1：128投影校正叠加到SAR平均幅度图")
    if not unchanged.empty:
        unchanged.boundary.plot(ax=axes[0], color="#00bcd4", linewidth=0.22, alpha=0.72)
    if not shifted.empty:
        shifted.boundary.plot(ax=axes[0], color="#ffeb3b", linewidth=0.34, alpha=0.95)
    axes[0].legend(
        handles=[
            Patch(facecolor="none", edgecolor="#00bcd4", label="未移动屋顶轮廓"),
            Patch(facecolor="none", edgecolor="#ffeb3b", label="已校正屋顶轮廓"),
        ],
        loc="lower left",
        prop=font,
        framealpha=0.86,
    )

    sar_axis(axes[1], amp, "步骤2：建筑栅格掩膜")
    uid_overlay = np.ma.masked_where(fid_mask <= 0, fid_mask % 251)
    axes[1].imshow(uid_overlay, cmap="turbo", alpha=0.55, origin="upper", interpolation="nearest")
    axes[1].legend(handles=[Patch(color="#34a853", label="屋顶mask像素")], loc="lower left", prop=font, framealpha=0.86)

    sar_axis(axes[2], amp, "步骤3：连通岛分割")
    island_overlay = np.ma.masked_where(island <= 0, island % 251)
    axes[2].imshow(island_overlay, cmap="nipy_spectral", alpha=0.58, origin="upper", interpolation="nearest")
    axes[2].legend(handles=[Patch(color="#7e57c2", label="独立屋顶岛")], loc="lower left", prop=font, framealpha=0.86)

    sar_axis(axes[3], amp, f"步骤4：振幅离散度筛选（DA≤{args.max_amplitude_dispersion:.2f}）")
    da_overlay = np.ma.masked_where(~valid_da, valid_da.astype(float))
    axes[3].imshow(da_overlay, cmap=ListedColormap(["#1a9850"]), alpha=0.42, origin="upper", interpolation="nearest")
    axes[3].legend(handles=[Patch(color="#1a9850", label="满足DA阈值的候选像素")], loc="lower left", prop=font, framealpha=0.86)

    fig.suptitle("128投影分支：从矢量投影到反演候选像素", fontsize=14)
    fig.tight_layout()
    save(fig, Path(args.out_png_dir) / "132_128投影_mask_岛分割_中文图例.png", Path(args.out_svg_dir) / "132_128投影_mask_岛分割_中文图例.svg")


def fig_island_inversion(args: argparse.Namespace, font: FontProperties) -> None:
    amp = stretch(np.load(args.amp_npy).astype(np.float32))
    label = np.load(args.island_label)
    islands = pd.read_csv(args.islands_csv)
    heights = pd.read_csv(args.island_heights)
    island_ids = set(islands["island_id"].astype(int))
    processed = set(heights["island_id"].dropna().astype(int)) if not heights.empty else set()
    solved = set(heights.loc[heights["height_m"].notna(), "island_id"].astype(int)) if not heights.empty else set()

    status = np.zeros(label.shape, dtype=np.uint8)
    for island_id in island_ids:
        status[label == island_id] = 1
    for island_id in processed:
        status[label == island_id] = 2
    for island_id in solved:
        status[label == island_id] = 3

    height_raster = np.full(label.shape, np.nan, dtype=np.float32)
    for row in heights.dropna(subset=["height_m"]).itertuples(index=False):
        height_raster[label == int(row.island_id)] = float(row.height_m)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=230, sharex=True, sharey=True)
    sar_axis(axes[0], amp, "步骤5：岛级处理状态")
    axes[0].imshow(status, cmap=ListedColormap(["#00000000", "#bdbdbd", "#fdae61", "#1b9e77"]), vmin=0, vmax=3, alpha=0.76, origin="upper", interpolation="nearest")
    axes[0].legend(
        handles=[
            Patch(color="#bdbdbd", label="屋顶岛"),
            Patch(color="#fdae61", label="已处理"),
            Patch(color="#1b9e77", label="已解出高度"),
        ],
        loc="lower left",
        prop=font,
        framealpha=0.88,
    )

    sar_axis(axes[1], amp, "步骤6：岛级高度结果")
    im = axes[1].imshow(height_raster, cmap="viridis", alpha=0.82, origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.036, pad=0.02)
    cbar.set_label("岛级建筑高度（米）")

    axes[2].set_axis_off()
    axes[2].set_title("步骤7：反演数量核查")
    stats = [
        ("屋顶岛总数", len(island_ids)),
        ("进入反演岛数", len(processed)),
        ("有高度岛数", len(solved)),
        ("高度点数量", int(pd.read_csv(args.height_points).shape[0])),
    ]
    y = 0.80
    for name, value in stats:
        axes[2].text(0.10, y, name, fontproperties=font, fontsize=11, transform=axes[2].transAxes)
        axes[2].text(0.82, y, f"{value}", fontproperties=font, fontsize=15, weight="bold", ha="right", transform=axes[2].transAxes)
        axes[2].add_patch(plt.Rectangle((0.07, y - 0.04), 0.80, 0.10, fill=False, edgecolor="#c8cdd2", linewidth=0.9, transform=axes[2].transAxes))
        y -= 0.17
    axes[2].text(
        0.08,
        0.10,
        f"严格参数：相干系数≥{args.min_coherence:.2f}，DA≤{args.max_amplitude_dispersion:.2f}，"
        f"最少{args.min_pairs}个有效干涉对，最少{args.min_pixels}个有效像素。",
        fontproperties=font,
        fontsize=9,
        wrap=True,
        transform=axes[2].transAxes,
    )

    fig.suptitle("128投影分支：像素LGR反演状态与岛级高度", fontsize=14)
    fig.tight_layout()
    save(fig, Path(args.out_png_dir) / "133_128岛级反演状态_高度_中文图例.png", Path(args.out_svg_dir) / "133_128岛级反演状态_高度_中文图例.svg")


def fig_building_height(args: argparse.Namespace, font: FontProperties) -> None:
    gdf = gpd.read_file(args.building_height_geojson)
    ok = gdf[gdf["has_insar_height"]].copy()
    nodata = gdf[~gdf["has_insar_height"]].copy()
    fig, ax = plt.subplots(figsize=(11, 10), dpi=260)
    if not nodata.empty:
        nodata.plot(ax=ax, color="#eeeeee", edgecolor="#bdbdbd", linewidth=0.18)
    if not ok.empty:
        vmax = max(10.0, float(np.nanpercentile(ok["height_insar_m"], 95)))
        norm = Normalize(vmin=0.0, vmax=vmax)
        ok.plot(ax=ax, column="height_insar_m", cmap="viridis", norm=norm, edgecolor="#202124", linewidth=0.25)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.08)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("建筑高度（米）")
    review = gdf[gdf["qc_review"] & gdf["has_insar_height"]]
    if not review.empty:
        review.boundary.plot(ax=ax, color="#d73027", linewidth=0.8)
    ax.legend(
        handles=[
            Patch(facecolor=plt.cm.viridis(0.68), edgecolor="#202124", label="InSAR反演高度"),
            Patch(facecolor="#eeeeee", edgecolor="#bdbdbd", label="无严格InSAR解"),
            Patch(facecolor="none", edgecolor="#d73027", label="需要人工复核"),
        ],
        loc="lower left",
        prop=font,
        framealpha=0.88,
    )
    ax.set_title("步骤8：建筑矢量高度图（128投影反演）", fontsize=13)
    ax.set_axis_off()
    ax.text(
        0.01,
        0.01,
        "说明：只显示InSAR严格解；未用shp原始高度补全。红色边界表示内部质检需人工复核，主要包括多建筑混岛、异常高度或顶端检验不稳定。",
        fontproperties=font,
        fontsize=8,
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#c8cdd2", "alpha": 0.88, "pad": 4},
    )
    fig.tight_layout()
    save(fig, Path(args.out_png_dir) / "134_128建筑高度矢量图_中文图例.png", Path(args.out_svg_dir) / "134_128建筑高度矢量图_中文图例.svg")


def fig_building_qc(args: argparse.Namespace, font: FontProperties) -> None:
    gdf = gpd.read_file(args.building_height_geojson)
    plot = gdf.to_crs(3857) if gdf.crs and gdf.crs.to_epsg() == 4326 else gdf.copy()
    plot["覆盖类别"] = np.where(plot["has_insar_height"], 1, 0)
    plot["复核类别"] = 0
    plot.loc[plot["has_insar_height"], "复核类别"] = 1
    plot.loc[plot["review_multi_clean_id_island"] & plot["has_insar_height"], "复核类别"] = 2
    unreliable = plot.get("review_unreliable_max_height", pd.Series(False, index=plot.index)).fillna(False).astype(bool)
    plot.loc[(plot["review_negative_height"] | plot["review_extreme_height"] | unreliable) & plot["has_insar_height"], "复核类别"] = 3
    plot["质量类别"] = 0
    good = plot["has_insar_height"] & (plot["median_coherence"] >= 0.85) & (plot["median_amplitude_dispersion"] <= 0.30) & (~plot["qc_review"])
    usable = plot["has_insar_height"] & (~good) & (~plot["qc_review"])
    review = plot["has_insar_height"] & plot["qc_review"]
    plot.loc[usable, "质量类别"] = 1
    plot.loc[good, "质量类别"] = 2
    plot.loc[review, "质量类别"] = 3
    plot["高度分级"] = 0
    bins = [-np.inf, 10, 20, 35, 50, np.inf]
    if plot["has_insar_height"].any():
        plot.loc[plot["has_insar_height"], "高度分级"] = pd.cut(
            plot.loc[plot["has_insar_height"], "height_insar_m"],
            bins=bins,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=240)
    axes = axes.ravel()
    plot.plot(ax=axes[0], column="覆盖类别", categorical=True, cmap=ListedColormap(["#eeeeee", "#1b9e77"]), linewidth=0.04, edgecolor="#777777")
    axes[0].set_title("步骤9a：严格解覆盖")
    axes[0].legend(handles=[Patch(color="#eeeeee", label="无解"), Patch(color="#1b9e77", label="有解")], loc="lower left", prop=font, framealpha=0.86)

    plot.plot(ax=axes[1], column="复核类别", categorical=True, cmap=ListedColormap(["#eeeeee", "#1b9e77", "#fdae61", "#d73027"]), linewidth=0.04, edgecolor="#777777")
    axes[1].set_title("步骤9b：人工复核类别")
    axes[1].legend(
        handles=[
            Patch(color="#1b9e77", label="通过"),
            Patch(color="#fdae61", label="多建筑岛"),
            Patch(color="#d73027", label="高度异常或顶端不稳定"),
        ],
        loc="lower left",
        prop=font,
        framealpha=0.86,
    )

    plot.plot(ax=axes[2], column="质量类别", categorical=True, cmap=ListedColormap(["#eeeeee", "#fee08b", "#1a9850", "#d73027"]), linewidth=0.04, edgecolor="#777777")
    axes[2].set_title("步骤9c：质量门控")
    axes[2].legend(handles=[Patch(color="#fee08b", label="可用"), Patch(color="#1a9850", label="高质量"), Patch(color="#d73027", label="复核")], loc="lower left", prop=font, framealpha=0.86)

    plot.plot(ax=axes[3], column="高度分级", categorical=True, cmap=ListedColormap(["#eeeeee", "#d9f0a3", "#78c679", "#238443", "#2b8cbe", "#253494"]), linewidth=0.04, edgecolor="#777777")
    axes[3].set_title("步骤9d：高度分级")
    axes[3].legend(
        handles=[
            Patch(color="#d9f0a3", label="<10 m"),
            Patch(color="#78c679", label="10-20 m"),
            Patch(color="#238443", label="20-35 m"),
            Patch(color="#2b8cbe", label="35-50 m"),
            Patch(color="#253494", label="≥50 m"),
        ],
        loc="lower left",
        prop=font,
        framealpha=0.86,
    )

    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("128投影分支：建筑尺度结果质检", fontsize=14)
    fig.tight_layout()
    save(fig, Path(args.out_png_dir) / "135_128建筑结果质检_中文图例.png", Path(args.out_svg_dir) / "135_128建筑结果质检_中文图例.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-geojson", default="work/projection/20200708_clean_equal_height_roof_projection_sar_full_area_layer_constrained.geojson")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--fid-mask", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--islands-csv", default="work/masks/islands_clean_equal_height_roof_only_full_area_128.csv")
    parser.add_argument("--amplitude-dispersion-npy", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    parser.add_argument("--max-amplitude-dispersion", type=float, default=0.40)
    parser.add_argument("--min-coherence", type=float, default=0.75)
    parser.add_argument("--min-pairs", type=int, default=12)
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--island-heights", default="work/height/island_pixel_lgr_heights_clean_equal_height_roof_only_full_area_128_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv")
    parser.add_argument("--height-points", default="work/height/height_points_clean_equal_height_roof_only_full_area_128_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv")
    parser.add_argument("--building-height-geojson", default="results/geodata/tongji_building_height_paper_lit_clean_equal_height_roof_only_full_area_128_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson")
    parser.add_argument("--out-png-dir", default="results/pic_all/png/current_strict_clean_equal_height_full")
    parser.add_argument("--out-svg-dir", default="results/pic_all/svg/current_strict_clean_equal_height_full")
    parser.add_argument("--summary", default="results/metadata/full_area_128_chinese_visual_qc_summary.json")
    args = parser.parse_args()

    font = setup_style()
    fig_projection_and_masks(args, font)
    fig_island_inversion(args, font)
    fig_building_height(args, font)
    fig_building_qc(args, font)
    outputs = {
        "projection_masks_png": str(Path(args.out_png_dir) / "132_128投影_mask_岛分割_中文图例.png"),
        "projection_masks_svg": str(Path(args.out_svg_dir) / "132_128投影_mask_岛分割_中文图例.svg"),
        "island_inversion_png": str(Path(args.out_png_dir) / "133_128岛级反演状态_高度_中文图例.png"),
        "island_inversion_svg": str(Path(args.out_svg_dir) / "133_128岛级反演状态_高度_中文图例.svg"),
        "building_height_png": str(Path(args.out_png_dir) / "134_128建筑高度矢量图_中文图例.png"),
        "building_height_svg": str(Path(args.out_svg_dir) / "134_128建筑高度矢量图_中文图例.svg"),
        "building_qc_png": str(Path(args.out_png_dir) / "135_128建筑结果质检_中文图例.png"),
        "building_qc_svg": str(Path(args.out_svg_dir) / "135_128建筑结果质检_中文图例.svg"),
    }
    summary = {"outputs": outputs, "font": FONT_PATH, "note": "Chinese visual QA figures for each major step of the 128 full-area projection height inversion branch."}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
