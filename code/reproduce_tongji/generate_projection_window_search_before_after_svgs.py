#!/usr/bin/env python3
"""Generate two standalone, matched SVGs before/after local window search."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import box

from refine_projection_local_window_search import sar_display


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amplitude", default="work/mli/20200708_rslc_amplitude.npy")
    p.add_argument(
        "--before",
        default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson",
    )
    p.add_argument(
        "--after",
        default="work/zjc_original_reproduction/20200708_all_building_adaptive_window_roof_projection_sar.geojson",
    )
    p.add_argument(
        "--metrics",
        default="work/zjc_original_reproduction/20200708_all_building_local_window_projection_metrics.csv",
    )
    p.add_argument("--output-dir", default="picall")
    return p.parse_args()


def draw(
    amplitude: np.ndarray,
    projection: gpd.GeoDataFrame,
    changed_ids: set[int],
    visible_count: int,
    output: Path,
    title: str,
    state_text: str,
) -> None:
    roofs = projection[projection["surface"].eq("roof")]
    changed = roofs[roofs["clean_id"].astype(int).isin(changed_ids)]
    unchanged = roofs[~roofs["clean_id"].astype(int).isin(changed_ids)]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.imshow(sar_display(amplitude), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    unchanged.boundary.plot(ax=ax, color="#00D5E8", linewidth=0.30, alpha=0.72, zorder=4)
    changed.boundary.plot(ax=ax, color="#E83E8C", linewidth=0.62, alpha=0.98, zorder=5)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title(title, pad=7, fontweight="bold")
    ax.text(
        0.012,
        0.985,
        f"可见屋顶：{visible_count}栋\n{state_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "#111111", "edgecolor": "#C8CDD2", "alpha": 0.80, "pad": 2.3},
    )
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#00D5E8", label="本轮未调整屋顶"),
            Patch(facecolor="none", edgecolor="#E83E8C", label="本轮调整屋顶"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": title})
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    amplitude = np.load(args.amplitude).astype(np.float32)
    before = gpd.read_file(args.before).set_crs(None, allow_override=True)
    after = gpd.read_file(args.after).set_crs(None, allow_override=True)
    metrics = pd.read_csv(args.metrics)
    changed_ids = set(metrics.loc[metrics["local_window_accepted"].eq(True), "clean_id"].astype(int))
    footprint = box(0, 0, amplitude.shape[1] - 1, amplitude.shape[0] - 1)
    visible_count = int(before[before["surface"].eq("roof")].geometry.intersects(footprint).sum())
    output_dir = Path(args.output_dir)
    draw(
        amplitude,
        before,
        changed_ids,
        visible_count,
        output_dir / "18a_局部窗口搜索前建筑投影.svg",
        "自适应局部窗口搜索前的屋顶投影",
        f"待调整屋顶：{len(changed_ids)}栋（洋红色）",
    )
    draw(
        amplitude,
        after,
        changed_ids,
        visible_count,
        output_dir / "18b_局部窗口搜索后建筑投影.svg",
        "自适应局部窗口搜索后的屋顶投影",
        f"已调整屋顶：{len(changed_ids)}栋（洋红色）",
    )
    print(
        {
            "outputs": [
                str(output_dir / "18a_局部窗口搜索前建筑投影.svg"),
                str(output_dir / "18b_局部窗口搜索后建筑投影.svg"),
            ],
            "visible_roofs": visible_count,
            "highlighted_changed_buildings": len(changed_ids),
            "matched_display_and_extent": True,
        }
    )


if __name__ == "__main__":
    main()
