#!/usr/bin/env python3
"""Generate separate Chinese SVG figures for the strict Tianjin reproduction.

Figure contract
---------------
Conclusion: reported building heights come from GAMMA-MCF + SBAS only; Floor is
not used for projection correction, pixel selection, inversion, or filling.
Archetype: separate image plates and geographic result maps.
Export: SVG only, editable Chinese text, rasterized dense image/point layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np


WIDTH = 10_000
LINES = 7_000
STEP = 4


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans CN", "DejaVu Sans"],
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def amplitude_plate(path: Path) -> np.ndarray:
    """Globally enhance one amplitude scene without local/feature-wise edits."""
    raw = np.memmap(path, dtype=">f4", mode="r", shape=(LINES, WIDTH))
    h = LINES // STEP
    w = WIDTH // STEP
    block = np.asarray(raw[: h * STEP, : w * STEP], dtype=np.float32)
    block = np.log1p(block).reshape(h, STEP, w, STEP).mean(axis=(1, 3))
    finite = np.isfinite(block)
    lo, hi = np.quantile(block[finite], (0.015, 0.995))
    plate = np.clip((block - lo) / (hi - lo), 0, 1)
    # A mild global gamma gives the reference-like separation of dark urban
    # background and bright double-bounce features; no local operation is used.
    return np.power(plate, 0.78, dtype=np.float32)


def radar_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("距离向像元（列）")
    ax.set_ylabel("方位向像元（行）")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(LINES, 0)
    ax.set_aspect("equal")


def show_amplitude(ax: plt.Axes, amplitude: np.ndarray) -> None:
    ax.imshow(
        amplitude,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, WIDTH, LINES, 0),
        interpolation="nearest",
        rasterized=True,
    )


def downsample_uid_boundaries(path: Path) -> tuple[np.ndarray, np.ndarray]:
    uid = np.memmap(path, dtype=">f4", mode="r", shape=(LINES, WIDTH))[::STEP, ::STEP]
    uid = np.asarray(uid, dtype=np.int32)
    occupied = uid > 0
    boundary = np.zeros_like(occupied)
    boundary[1:, :] |= (uid[1:, :] != uid[:-1, :]) & (occupied[1:, :] | occupied[:-1, :])
    boundary[:, 1:] |= (uid[:, 1:] != uid[:, :-1]) & (occupied[:, 1:] | occupied[:, :-1])
    return occupied, boundary


def radar_label_positions(pixel: np.lib.npyio.NpzFile) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    uid = pixel["building_uid"].astype(np.int64)
    for building in np.unique(uid):
        m = uid == building
        result[int(building)] = (
            float(np.median(pixel["col"][m])),
            float(np.median(pixel["row"][m])),
        )
    return result


def add_height_labels(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    heights: dict[int, float],
    color: str = "white",
    fontsize: float = 3.8,
) -> None:
    halo = [pe.withStroke(linewidth=1.25, foreground="#111111")]
    for uid, (x, y) in positions.items():
        value = heights.get(uid)
        if value is None or not np.isfinite(value):
            continue
        ax.text(
            x,
            y,
            f"{value:.0f}",
            color=color,
            fontsize=fontsize,
            ha="center",
            va="center",
            path_effects=halo,
            clip_on=True,
            zorder=8,
        )


def figure_amplitude(amplitude: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.15), constrained_layout=True)
    show_amplitude(ax, amplitude)
    radar_axes(ax, "天津研究区单景雷达幅度（2023年10月7日）")
    ax.text(
        0.012, 0.018, "全局 log(1+A) 拉伸；未进行局部选择性增强",
        transform=ax.transAxes, color="white", fontsize=6,
        path_effects=[pe.withStroke(linewidth=1.4, foreground="black")],
    )
    save_svg(fig, out)


def figure_vector_projection(
    amplitude: np.ndarray,
    uid_rdc: Path,
    pixel: np.lib.npyio.NpzFile,
    heights: dict[int, float],
    out: Path,
) -> None:
    occupied, boundary = downsample_uid_boundaries(uid_rdc)
    overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
    overlay[occupied] = (0.05, 0.75, 0.95, 0.10)
    overlay[boundary] = (0.00, 0.93, 1.00, 0.88)
    fig, ax = plt.subplots(figsize=(10.8, 7.7), constrained_layout=True)
    show_amplitude(ax, amplitude)
    ax.imshow(
        overlay, extent=(0, WIDTH, LINES, 0), interpolation="nearest", rasterized=True
    )
    positions = radar_label_positions(pixel)
    add_height_labels(ax, positions, heights)
    radar_axes(ax, "全部建筑矢量投影至雷达坐标（有解建筑标注反演高度）")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#00edff", lw=1.2, label="全部建筑地面足迹投影边界"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#111111", markersize=4, label="数字：有解建筑高度（m）"),
        ],
        loc="lower right", facecolor="white", framealpha=0.88, fontsize=6,
    )
    save_svg(fig, out)


def figure_search_and_selected(
    amplitude: np.ndarray,
    search: np.lib.npyio.NpzFile,
    quality: np.lib.npyio.NpzFile,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.15), constrained_layout=True)
    show_amplitude(ax, amplitude)
    ax.scatter(
        search["col"], search["row"], s=0.10, c="#ff9d2e", alpha=0.18,
        linewidths=0, rasterized=True, label="建筑独立搜索带",
    )
    m = quality["paper_quality_selected"]
    ax.scatter(
        quality["col"][m], quality["row"][m], s=0.55, c="#00e5ff", alpha=0.90,
        linewidths=0, rasterized=True, label="DA≤0.4 且平均相干性≥0.75",
    )
    radar_axes(ax, "建筑独立搜索带与论文阈值筛选像元")
    ax.legend(loc="lower right", facecolor="white", framealpha=0.9, fontsize=6,
              markerscale=6)
    save_svg(fig, out)


def figure_pixel_height(
    amplitude: np.ndarray,
    pixel: np.lib.npyio.NpzFile,
    out: Path,
) -> None:
    value = pixel["dem_error_or_height_above_anchor_m"].astype(float)
    lo, hi = np.nanquantile(value, (0.02, 0.98))
    lim = max(abs(lo), abs(hi), 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.15), constrained_layout=True)
    show_amplitude(ax, amplitude)
    sc = ax.scatter(
        pixel["col"], pixel["row"], c=value, s=1.2, cmap="coolwarm",
        norm=Normalize(-lim, lim), linewidths=0, rasterized=True,
    )
    radar_axes(ax, "GAMMA-MCF 解缠与 SBAS 反演的像元相对高差")
    cb = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.018)
    cb.set_label("相对于建筑固定参考像元的高差（m）")
    save_svg(fig, out)


def figure_building_height_map(gdf: gpd.GeoDataFrame, out: Path) -> None:
    solved = gdf[gdf["recommended_building_height_m"].notna()].copy()
    base = gdf[gdf["recommended_building_height_m"].isna()]
    vmax = float(np.nanquantile(solved["recommended_building_height_m"], 0.98))
    fig, ax = plt.subplots(figsize=(8.0, 7.3), constrained_layout=True)
    base.plot(ax=ax, facecolor="#e8eaed", edgecolor="#c7cbd1", linewidth=0.06,
              rasterized=True)
    solved.plot(
        ax=ax, column="recommended_building_height_m", cmap="viridis", vmin=0,
        vmax=vmax, edgecolor="#18222b", linewidth=0.20,
    )
    norm = Normalize(0, vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.015)
    cb.set_label("建筑高度（m，SBAS像元高差P95−P05）")
    for row in solved.itertuples():
        point = row.geometry.representative_point()
        ax.text(
            point.x, point.y, f"{row.recommended_building_height_m:.0f}",
            ha="center", va="center", fontsize=3.25, color="white",
            path_effects=[pe.withStroke(linewidth=1.0, foreground="#151515")],
            clip_on=True,
        )
    ax.set_title("天津研究区建筑高度反演图（有解建筑均标注数值）", fontsize=10, pad=8)
    ax.set_xlabel("经度（°E）")
    ax.set_ylabel("纬度（°N）")
    mean_lat = float((gdf.total_bounds[1] + gdf.total_bounds[3]) / 2)
    ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))
    ax.text(
        0.012, 0.015, "灰色：无解（保持空值）；颜色与数字：纯InSAR反演高度",
        transform=ax.transAxes, fontsize=6, color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    save_svg(fig, out)


def figure_quality_map(gdf: gpd.GeoDataFrame, out: Path) -> None:
    status_order = ["无解", "低可信", "通过"]
    colors = ["#d8dadd", "#e69f00", "#2a9d8f"]
    code = {v: i for i, v in enumerate(status_order)}
    plot = gdf.copy()
    plot["status_code"] = plot["solution_status"].map(code).astype(int)
    fig, ax = plt.subplots(figsize=(8.0, 7.3), constrained_layout=True)
    plot.plot(
        ax=ax, column="status_code", cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
        edgecolor="#ffffff", linewidth=0.04, rasterized=True,
    )
    ax.set_title("建筑高度解算状态", fontsize=10, pad=8)
    ax.set_xlabel("经度（°E）")
    ax.set_ylabel("纬度（°N）")
    mean_lat = float((gdf.total_bounds[1] + gdf.total_bounds[3]) / 2)
    ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor=c,
                   markeredgecolor="none", markersize=6, label=s)
            for s, c in zip(status_order, colors)
        ],
        loc="lower right", fontsize=6,
    )
    save_svg(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=Path, required=True)
    parser.add_argument("--uid-rdc", type=Path, required=True)
    parser.add_argument("--search-points", type=Path, required=True)
    parser.add_argument("--quality-points", type=Path, required=True)
    parser.add_argument("--pixel-height", type=Path, required=True)
    parser.add_argument("--building-height", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    amplitude = amplitude_plate(args.amplitude)
    search = np.load(args.search_points)
    quality = np.load(args.quality_points)
    pixel = np.load(args.pixel_height)
    gdf = gpd.read_file(args.building_height, layer="building_height")
    solved = gdf[gdf["recommended_building_height_m"].notna()]
    heights = dict(zip(
        solved["building_uid"].astype(int), solved["recommended_building_height_m"].astype(float)
    ))

    figure_amplitude(amplitude, args.output_dir / "01_天津研究区单景雷达幅度.svg")
    figure_vector_projection(
        amplitude, args.uid_rdc, pixel, heights,
        args.output_dir / "02_全部建筑矢量投影至雷达坐标.svg",
    )
    figure_search_and_selected(
        amplitude, search, quality,
        args.output_dir / "03_建筑独立搜索带与高质量像元.svg",
    )
    figure_pixel_height(
        amplitude, pixel,
        args.output_dir / "04_GAMMA解缠与SBAS像元相对高差.svg",
    )
    figure_building_height_map(gdf, args.output_dir / "05_建筑高度反演图.svg")
    figure_quality_map(gdf, args.output_dir / "06_建筑高度解算状态.svg")
    print(f"wrote 6 SVG figures to {args.output_dir}")


if __name__ == "__main__":
    main()
