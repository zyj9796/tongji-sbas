#!/usr/bin/env python3
"""Create per-building diagnostics for QC/review buildings."""

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
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def setup_font() -> FontProperties | None:
    if Path(FONT_PATH).exists():
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
    return None


def stretch(arr: np.ndarray) -> np.ndarray:
    ok = np.isfinite(arr)
    if not np.any(ok):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[ok], [2, 98])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)


def parse_ids(value) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            pass
    return out


def fmt(v, nd=2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_building_height_cleanid_split_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson")
    parser.add_argument("--topstats", default="results/geodata/tongji_building_height_cleanid_split_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topstats_insar_only.geojson")
    parser.add_argument("--height-points", default="work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_split_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--uid-mask", default="work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_split.npy")
    parser.add_argument("--old-island-label", default="work/masks/island_label_clean_equal_height_roof_only_full_area_128.npy")
    parser.add_argument("--out-dir", default="results/diagnostics/cleanid_split_red_buildings")
    parser.add_argument("--summary", default="results/metadata/cleanid_split_red_building_diagnostics_summary.json")
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--max-buildings", type=int, default=0)
    args = parser.parse_args()

    font = setup_font()
    gdf = gpd.read_file(args.buildings)
    topstats = gpd.read_file(args.topstats)
    points = pd.read_csv(args.height_points)
    amp = stretch(np.load(args.amp_npy).astype(np.float32))
    uid_mask = np.load(args.uid_mask).astype(np.int32)
    island_label = np.load(args.island_label).astype(np.int32)
    old_island_label = np.load(args.old_island_label).astype(np.int32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    review = gdf[gdf["has_insar_height"] & (gdf["qc_review"] | gdf["likely_top_reliability"].eq("review"))].copy()
    review = review.sort_values(["likely_top_reliability", "clean_id"], ascending=[False, True])
    if args.max_buildings > 0:
        review = review.head(args.max_buildings).copy()
    top_by_id = topstats.set_index("clean_id")

    rows = []
    for row in review.itertuples(index=False):
        clean_id = int(row.clean_id)
        mask = uid_mask == clean_id
        if not np.any(mask):
            continue
        rr, cc = np.nonzero(mask)
        r0 = max(0, int(rr.min()) - args.margin)
        r1 = min(uid_mask.shape[0], int(rr.max()) + args.margin + 1)
        c0 = max(0, int(cc.min()) - args.margin)
        c1 = min(uid_mask.shape[1], int(cc.max()) + args.margin + 1)
        crop_amp = amp[r0:r1, c0:c1]
        crop_mask = uid_mask[r0:r1, c0:c1] == clean_id
        crop_island = island_label[r0:r1, c0:c1]
        crop_old = old_island_label[r0:r1, c0:c1]
        current_island_ids = [int(v) for v in np.unique(island_label[mask]) if int(v) > 0]
        old_island_ids = [int(v) for v in np.unique(old_island_label[mask]) if int(v) > 0]

        top = top_by_id.loc[clean_id] if clean_id in top_by_id.index else None
        point = points[points["uid"].astype(int).eq(clean_id)]
        source_island = parse_ids(getattr(row, "source_islands", ""))
        point = point[point["island_id"].astype(int).isin(source_island)] if source_island else point
        if point.empty:
            point = points[points["uid"].astype(int).eq(clean_id)]
        p = point.iloc[0] if not point.empty else pd.Series(dtype=float)

        max_h = float(getattr(row, "building_height_max_m", np.nan))
        kept_h = float(getattr(row, "height_insar_m", np.nan))
        removed = int(float(getattr(row, "grubbs_top_removed_count", 0) or 0))
        rejected_top = max_h if removed > 0 and np.isfinite(max_h) and abs(max_h - kept_h) > 1e-6 else np.nan

        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), dpi=220)
        axes = axes.ravel()
        axes[0].imshow(crop_amp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0].contour(crop_mask.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
        axes[0].set_title("SAR平均幅度 + 当前clean_id轮廓", fontproperties=font)
        axes[0].set_axis_off()

        axes[1].imshow(crop_amp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].imshow(np.ma.masked_where(~crop_mask, crop_mask), cmap=ListedColormap(["#2ca25f"]), alpha=0.55, interpolation="nearest")
        axes[1].set_title("参与反演的clean_id mask", fontproperties=font)
        axes[1].set_axis_off()

        axes[2].imshow(crop_amp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        split_show = np.ma.masked_where(crop_island <= 0, crop_island % 251)
        axes[2].imshow(split_show, cmap="nipy_spectral", alpha=0.62, interpolation="nearest")
        axes[2].set_title("clean_id拆分后的岛编号", fontproperties=font)
        axes[2].set_axis_off()

        axes[3].set_axis_off()
        lines = [
            f"clean_id: {clean_id}",
            f"当前split岛: {','.join(map(str, current_island_ids)) or 'NA'}",
            f"原始混岛: {','.join(map(str, old_island_ids)) or 'NA'}",
            f"最终保留顶点高度: {fmt(kept_h)} m",
            f"原最高点高度: {fmt(max_h)} m",
            f"被剔除最高点: {fmt(rejected_top)} m" if np.isfinite(rejected_top) else "被剔除最高点: 无",
            f"剔除顶端点数: {removed}",
            f"相干系数中位数: {fmt(getattr(row, 'median_coherence', np.nan), 3)}",
            f"振幅离散度DA: {fmt(getattr(row, 'median_amplitude_dispersion', np.nan), 3)}",
            f"LGR RMSE: {fmt(getattr(row, 'lgr_rmse_rad', np.nan), 3)} rad",
            f"有效干涉对: {fmt(getattr(row, 'valid_pairs_median', np.nan), 1)}",
            f"Bperp跨度: {fmt(getattr(row, 'bperp_span_median', np.nan), 1)} m",
            f"可靠性: {getattr(row, 'likely_top_reliability', 'NA')}",
            f"选择规则: {getattr(row, 'likely_top_method', 'NA')}",
        ]
        y = 0.96
        for text in lines:
            axes[3].text(0.02, y, text, transform=axes[3].transAxes, va="top", fontsize=9.2, fontproperties=font)
            y -= 0.065
        fig.suptitle(f"红线建筑诊断：clean_id {clean_id}", fontproperties=font, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out_png = out_dir / f"clean_id_{clean_id:04d}_diagnostic.png"
        fig.savefig(out_png)
        plt.close(fig)

        rows.append(
            {
                "clean_id": clean_id,
                "diagnostic_png": str(out_png),
                "height_insar_m": kept_h,
                "building_height_max_m": max_h,
                "rejected_top_height_m": rejected_top,
                "grubbs_top_removed_count": removed,
                "likely_top_reliability": getattr(row, "likely_top_reliability", ""),
                "likely_top_method": getattr(row, "likely_top_method", ""),
                "qc_review": bool(getattr(row, "qc_review", False)),
                "review_unreliable_max_height": bool(getattr(row, "review_unreliable_max_height", False)),
                "review_multi_clean_id_island": bool(getattr(row, "review_multi_clean_id_island", False)),
                "median_coherence": getattr(row, "median_coherence", np.nan),
                "median_amplitude_dispersion": getattr(row, "median_amplitude_dispersion", np.nan),
                "lgr_rmse_rad": getattr(row, "lgr_rmse_rad", np.nan),
                "valid_pairs_median": getattr(row, "valid_pairs_median", np.nan),
                "bperp_span_median": getattr(row, "bperp_span_median", np.nan),
                "split_islands": ",".join(map(str, current_island_ids)),
                "old_islands": ",".join(map(str, old_island_ids)),
                "pixel_count_used": getattr(row, "pixel_count_used", np.nan),
            }
        )

    manifest = out_dir / "diagnostic_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    summary = {
        "input_buildings": args.buildings,
        "review_buildings_plotted": int(len(rows)),
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "note": "Rejected top height is the original maximum when top-down Grubbs removed at least one upper outlier; individual removed lower-order top values are summarized by grubbs_top_removed_count.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
