#!/usr/bin/env python3
"""Update all standalone SVGs that depend on the strict building solution."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon


def style() -> None:
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
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def sar_display(amplitude: np.ndarray) -> np.ndarray:
    transformed = np.power(np.maximum(amplitude, 0.0), 0.70)
    valid = np.isfinite(transformed)
    lo, hi = np.nanpercentile(transformed[valid], [1.0, 99.4])
    return np.clip((transformed - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.svg", format="svg", metadata={"Title": name})
    plt.close(fig)
    print(f"saved {output_dir / (name + '.svg')}")


def parts(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def best_zoom(mask: np.ndarray, window_rows: int = 140, window_cols: int = 180) -> tuple[int, int, int, int]:
    density = ndimage.uniform_filter(mask.astype(np.float32), size=(window_rows, window_cols), mode="constant")
    row, col = np.unravel_index(int(np.argmax(density)), density.shape)
    r0 = int(np.clip(row - window_rows // 2, 0, mask.shape[0] - window_rows))
    c0 = int(np.clip(col - window_cols // 2, 0, mask.shape[1] - window_cols))
    return r0, r0 + window_rows, c0, c0 + window_cols


def draw_study_area(gdf: gpd.GeoDataFrame, solved: np.ndarray, output_dir: Path) -> None:
    metric = gdf.to_crs("EPSG:32651")
    selected = metric[solved]
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    metric.plot(ax=ax, facecolor="#ECEFF1", edgecolor="#AEB5BD", linewidth=0.22)
    selected.plot(ax=ax, facecolor="#2A9D8F", edgecolor="#165B50", linewidth=0.32)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("同济区域建筑与双冗余网络稳定解范围", pad=7, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#2A9D8F", edgecolor="#165B50", label=f"双网络稳定解（{int(solved.sum())}栋）"),
            Patch(facecolor="#ECEFF1", edgecolor="#AEB5BD", label=f"无解（{int((~solved).sum())}栋）"),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fontsize=7,
    )
    ax.text(
        0.015,
        0.018,
        "48对与36对等权GAMMA解差异不超过4 m；无解不填充",
        transform=ax.transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "alpha": 0.92, "pad": 2.5},
    )
    save(fig, output_dir, "01_同济建筑轮廓与严格有解建筑")


def draw_roof_masks(
    mask: np.ndarray, amplitude: np.ndarray, output_dir: Path, solved_count: int
) -> None:
    lo, hi = np.nanpercentile(amplitude[np.isfinite(amplitude)], [3, 99.5])
    r0, r1, c0, c1 = best_zoom(mask)

    fig, ax = plt.subplots(figsize=(6.8, 4.9), constrained_layout=True)
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], colors=["#D62728"], linewidths=0.48)
    ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="#2A9D8F", linewidth=1.0))
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title(f"{solved_count}栋严格有解建筑的屋顶核心区", pad=6, fontweight="bold")
    ax.text(
        0.015,
        0.975,
        f"严格屋顶核心像元：{int(mask.sum()):,}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "alpha": 0.88, "pad": 2.0},
    )
    save(fig, output_dir, "10_建筑孤岛全景")

    fig, ax = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], colors=["#D62728"], linewidths=0.75)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("严格有解屋顶核心区局部放大", pad=6, fontweight="bold")
    ax.text(
        0.02,
        0.97,
        "红色轮廓：腐蚀并剔除跨楼冲突后的屋顶核心",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    save(fig, output_dir, "11_建筑孤岛局部放大")


def draw_roof_height(
    owner: np.ndarray,
    height_lookup: dict[int, float],
    amplitude: np.ndarray,
    output_dir: Path,
    vmax: float,
    solved_count: int,
) -> None:
    raster = np.full(owner.shape, np.nan, dtype=np.float32)
    for clean_id, height in height_lookup.items():
        raster[owner == clean_id] = height
    norm = colors.Normalize(0.0, vmax, clip=True)
    fig, ax = plt.subplots(figsize=(7.1, 5.0), constrained_layout=True)
    ax.imshow(sar_display(amplitude), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest", alpha=0.78)
    image = ax.imshow(np.ma.masked_invalid(raster), cmap="viridis", norm=norm, origin="upper", interpolation="nearest", alpha=0.95)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("双冗余网络稳定屋顶核心区建筑高度", pad=6, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.02, extend="max")
    cbar.set_label("建筑高度（m）")
    cbar.outline.set_linewidth(0.55)
    ax.text(
        0.015,
        0.975,
        f"仅显示{solved_count}栋双网络稳定建筑；无解像元透明",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#B8BEC5", "alpha": 0.90, "pad": 2.3},
    )
    save(fig, output_dir, "12_屋顶核心区SBAS高度图")


def add_extrusion(ax, polygon: Polygon, height: float, x0: float, y0: float, norm, cmap) -> None:
    xy = np.asarray(polygon.exterior.coords)
    if len(xy) < 4:
        return
    x = (xy[:, 0] - x0) / 1000.0
    y = (xy[:, 1] - y0) / 1000.0
    z = np.full_like(x, height)
    color = cmap(norm(height))
    ax.add_collection3d(
        Poly3DCollection([list(zip(x, y, z))], facecolors=[color], edgecolors="white", linewidths=0.18, alpha=0.96)
    )
    sides = []
    for i in range(len(x) - 1):
        sides.append([(x[i], y[i], 0), (x[i + 1], y[i + 1], 0), (x[i + 1], y[i + 1], height), (x[i], y[i], height)])
    ax.add_collection3d(
        Poly3DCollection(sides, facecolors=[colors.to_rgba(color, 0.70)], edgecolors="#5C6570", linewidths=0.10)
    )


def draw_lod1(
    gdf: gpd.GeoDataFrame,
    solved: np.ndarray,
    output_dir: Path,
    vmax: float,
    solved_count: int,
) -> None:
    metric = gdf.to_crs("EPSG:32651")
    selected = metric[solved]
    minx, miny, maxx, maxy = metric.total_bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    norm = colors.Normalize(0.0, vmax, clip=True)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(10.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.015, 0.03, 0.82, 0.91])
    for geometry in metric.geometry:
        for polygon in parts(geometry):
            xy = np.asarray(polygon.exterior.coords)
            ax.plot((xy[:, 0] - cx) / 1000.0, (xy[:, 1] - cy) / 1000.0, np.zeros(len(xy)), color="#C5CBD2", linewidth=0.16, alpha=0.68)
    for row in selected.itertuples(index=False):
        height = float(row.insar_height_m)
        for polygon in parts(row.geometry):
            add_extrusion(ax, polygon.simplify(0.35), height, cx, cy, norm, cmap)
    ax.view_init(elev=34, azim=-58)
    ax.set_xlabel("东向偏移（km）", labelpad=4)
    ax.set_ylabel("北向偏移（km）", labelpad=4)
    ax.set_zlabel("")
    ax.set_zticks([])
    ax.set_box_aspect((1.72, 1.22, 0.56))
    ax.grid(False)
    ax.set_title(f"{solved_count}栋双冗余网络稳定建筑三维重建", pad=2, fontweight="bold")
    cax = fig.add_axes([0.86, 0.19, 0.025, 0.61])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max")
    cbar.set_label("建筑高度（m）")
    fig.text(0.50, 0.025, "仅拉伸48对与36对网络一致的建筑；无解建筑保持地面轮廓", ha="center", fontsize=7)
    save(fig, output_dir, "14_严格有解建筑三维重建")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", default="results/geodata/tongji_building_height_prior_roof_init_gamma100_paper_strict.geojson")
    p.add_argument("--roof-owner", default="work/roof_sbas_optimized/roof_core_clean_id_mask.npy")
    p.add_argument("--amplitude", default="work/mli/mean_crop_bmp_amplitude.npy")
    p.add_argument("--output-dir", default="picall")
    p.add_argument(
        "--height-dependent-only",
        action="store_true",
        help="Update Figures 01, 12 and 14 but preserve pre-inversion island Figures 10 and 11",
    )
    args = p.parse_args()

    style()
    gdf = gpd.read_file(args.result)
    gdf["insar_height_m"] = pd.to_numeric(gdf["insar_height_m"], errors="coerce")
    if "filled_from_prior" in gdf:
        filled = pd.to_numeric(gdf["filled_from_prior"], errors="coerce").fillna(0)
        if bool((filled != 0).any()):
            raise RuntimeError("Prior-filled values are forbidden")
    solved = gdf["insar_height_m"].notna().to_numpy()
    solved_count = int(solved.sum())
    if solved_count == 0:
        raise RuntimeError("Input contains no strict solutions")
    owner = np.load(args.roof_owner).astype(np.int64)
    amplitude = np.load(args.amplitude).astype(float)
    accepted_ids = gdf.loc[solved, "clean_id"].astype(int).to_numpy()
    mask = np.isin(owner, accepted_ids)
    height_lookup = dict(zip(gdf.loc[solved, "clean_id"].astype(int), gdf.loc[solved, "insar_height_m"].astype(float)))
    values = np.asarray(list(height_lookup.values()), dtype=float)
    vmax = max(20.0, 5.0 * math.ceil(float(np.nanpercentile(values, 99)) / 5.0))
    output_dir = Path(args.output_dir)

    draw_study_area(gdf, solved, output_dir)
    if not args.height_dependent_only:
        draw_roof_masks(mask, amplitude, output_dir, solved_count)
    draw_roof_height(owner, height_lookup, amplitude, output_dir, vmax, solved_count)
    draw_lod1(gdf, solved, output_dir, vmax, solved_count)
    print(
        {
            "updated_figures": [1, 12, 14] if args.height_dependent_only else [1, 10, 11, 12, 14],
            "solved_buildings": int(solved.sum()),
            "unsolved_buildings": int((~solved).sum()),
            "strict_roof_core_pixels": int(mask.sum()),
            "height_color_limit_m": float(vmax),
        }
    )


if __name__ == "__main__":
    main()
