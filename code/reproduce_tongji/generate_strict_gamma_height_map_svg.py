#!/usr/bin/env python3
"""Generate the standalone Chinese SVG map for strict GAMMA-SBAS heights."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="results/geodata/tongji_building_height_prior_roof_init_gamma100_paper_strict.geojson",
    )
    p.add_argument("--height-column", default="insar_height_m")
    p.add_argument("--output", default="picall/17_屋顶核心区SBAS建筑高度分布.svg")
    p.add_argument("--map-crs", default="EPSG:32651")
    p.add_argument("--title", default="同济区域严格 GAMMA-SBAS 建筑高度")
    p.add_argument(
        "--method-note",
        default="先验仅辅助屋顶R-D定位与整周初始化，最终高度由GAMMA mb_pt重新估计",
    )
    args = p.parse_args()

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 12,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    source = gpd.read_file(args.input)
    if args.height_column not in source:
        raise ValueError(f"Missing height column: {args.height_column}")
    source[args.height_column] = pd.to_numeric(source[args.height_column], errors="coerce")
    if "filled_from_prior" in source:
        filled = pd.to_numeric(source["filled_from_prior"], errors="coerce").fillna(0)
        if bool((filled != 0).any()):
            raise RuntimeError("Input contains prior-filled heights; refusing to draw")
    solved = source[args.height_column].notna()
    if int(solved.sum()) == 0:
        raise RuntimeError("Input contains no solved building heights")

    gdf = source.to_crs(args.map_crs)
    solved_gdf = gdf[solved].copy()
    unsolved_gdf = gdf[~solved].copy()
    values = solved_gdf[args.height_column].to_numpy(float)
    vmax = max(20.0, 5.0 * math.ceil(float(np.nanpercentile(values, 99)) / 5.0))
    norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9.2, 8.4), constrained_layout=True)
    unsolved_gdf.plot(ax=ax, facecolor="#ECEFF1", edgecolor="#AEB5BD", linewidth=0.22, zorder=1)
    solved_gdf.plot(
        ax=ax,
        column=args.height_column,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.24,
        zorder=2,
    )

    for row in solved_gdf.itertuples(index=False):
        value = float(getattr(row, args.height_column))
        point = row.geometry.representative_point()
        text_color = "#111111" if norm(value) >= 0.60 else "#FFFFFF"
        ax.text(
            point.x,
            point.y,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=3.15,
            color=text_color,
            zorder=4,
            clip_on=True,
        )

    minx, miny, maxx, maxy = gdf.total_bounds
    width, height = maxx - minx, maxy - miny
    pad_x, pad_y = width * 0.025, height * 0.035
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(args.title, pad=8, fontweight="bold")

    ax.text(
        0.015,
        0.018,
        f"严格有解：{len(solved_gdf)} 栋　无解：{len(unsolved_gdf)} 栋\n"
        "数值为建筑高度（m）；无解建筑不填充\n"
        + args.method_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "linewidth": 0.55, "alpha": 0.93, "pad": 3.2},
        zorder=6,
    )

    ax.annotate(
        "北",
        xy=(0.956, 0.925),
        xytext=(0.956, 0.855),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#202124", "lw": 1.0},
        zorder=6,
    )

    scale_m = 200.0
    scale_x0 = minx + 0.76 * width
    scale_y = miny + 0.045 * height
    ax.plot([scale_x0, scale_x0 + scale_m], [scale_y, scale_y], color="#202124", lw=1.5, zorder=6)
    ax.plot([scale_x0, scale_x0], [scale_y - 4, scale_y + 4], color="#202124", lw=1.0, zorder=6)
    ax.plot([scale_x0 + scale_m, scale_x0 + scale_m], [scale_y - 4, scale_y + 4], color="#202124", lw=1.0, zorder=6)
    ax.text(scale_x0 + scale_m / 2, scale_y + 9, "200 m", ha="center", va="bottom", fontsize=7, zorder=6)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.032,
        pad=0.012,
        shrink=0.80,
        extend="max" if float(np.nanmax(values)) > vmax else "neither",
    )
    colorbar.set_label("建筑高度（m）")
    colorbar.outline.set_linewidth(0.55)
    ax.legend(
        handles=[Patch(facecolor="#ECEFF1", edgecolor="#AEB5BD", linewidth=0.5, label="无严格解")],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#B8BEC5",
        fontsize=7,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": args.title})
    plt.close(fig)
    print(
        {
            "output": str(output),
            "features": int(len(gdf)),
            "solved": int(len(solved_gdf)),
            "unsolved": int(len(unsolved_gdf)),
            "labels": int(len(solved_gdf)),
            "color_limit_m": float(vmax),
            "maximum_label_height_m": float(np.nanmax(values)),
        }
    )


if __name__ == "__main__":
    main()
