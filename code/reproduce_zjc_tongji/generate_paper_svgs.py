#!/usr/bin/env python3
"""Generate SVG-only Tongji figures following the visual logic of the ZJC paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm
from shapely.geometry import MultiPolygon, Polygon, box


PROJECT = Path(__file__).resolve().parents[2]

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Noto Serif CJK SC", "Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.1,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

BLUE = "#2878B5"
ORANGE = "#F28E2B"
RED = "#D62728"
GREEN = "#2A9D8F"
GRAY = "#6B7280"


def save_svg(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.svg"
    fig.savefig(target, format="svg")
    plt.close(fig)
    print(f"saved {target}")


def panel_label(ax: plt.Axes, label: str, image_background: bool = False) -> None:
    kwargs = {}
    if image_background:
        kwargs["bbox"] = {"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.5}
    ax.text(0.015, 0.985, label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", **kwargs)


def derive_acquisition_baselines(pairs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(pairs["master"].astype(str)) | set(pairs["slave"].astype(str)))
    index = {date: idx for idx, date in enumerate(dates)}
    a = np.zeros((len(pairs), len(dates)), dtype=float)
    for row_idx, row in enumerate(pairs.itertuples(index=False)):
        a[row_idx, index[str(row.master)]] = -1.0
        a[row_idx, index[str(row.slave)]] = 1.0
    reduced = a[:, 1:]
    solution, *_ = np.linalg.lstsq(reduced, pairs["bperp_m"].to_numpy(float), rcond=None)
    values = np.r_[0.0, solution]
    values -= np.mean(values)
    return pd.DataFrame({"date": pd.to_datetime(dates), "bperp_m": values, "key": dates})


def geometry_parts(geom):
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def figure_study_area(buildings: gpd.GeoDataFrame, selected_uids: set[int], amplitude: np.ndarray, out_dir: Path) -> None:
    selected = buildings[buildings["clean_id"].astype(int).isin(selected_uids)]
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    buildings.plot(ax=ax, facecolor="#D9DEE5", edgecolor="#6B7280", linewidth=0.25)
    selected.plot(ax=ax, facecolor=ORANGE, edgecolor="#8C4A00", linewidth=0.45)
    ax.set_xlabel("经度（°）")
    ax.set_ylabel("纬度（°）")
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
    ax.text(0.985, 0.02, f"全部建筑：{len(buildings)}\n严格SBAS有解：{len(selected)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    save_svg(fig, out_dir, "01_同济建筑轮廓与严格有解建筑")

    lo, hi = np.nanpercentile(amplitude[np.isfinite(amplitude)], [2, 99.5])
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.annotate("距离向", xy=(0.92, 0.95), xytext=(0.68, 0.95), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.4}, color=GREEN, va="center")
    save_svg(fig, out_dir, "02_同济雷达平均幅度")


def figure_all_building_projections(projection: gpd.GeoDataFrame, amplitude: np.ndarray, out_dir: Path) -> None:
    """Plot the paper's ground-to-rooftop radar support for every building."""
    valid = projection[projection.geometry.notna() & ~projection.geometry.is_empty].copy()
    if "surface" not in valid.columns or not {"bottom", "roof", "layover"}.issubset(set(valid["surface"])):
        raise RuntimeError("Paper-style projection requires bottom, roof, and layover geometries")
    # These geometries are expressed directly in SAR pixel coordinates although
    # GeoPandas assigns the GeoJSON default CRS. Remove it before plotting.
    valid = valid.set_crs(None, allow_override=True)
    supports = valid[valid["surface"] == "layover"]
    ground = valid[valid["surface"] == "bottom"]
    roofs = valid[valid["surface"] == "roof"]
    height, width = amplitude.shape
    image_footprint = box(0, 0, width, height)
    visible_count = int(supports.geometry.intersects(image_footprint).sum())
    lo, hi = np.nanpercentile(amplitude[np.isfinite(amplitude)], [2, 99.5])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    supports.plot(ax=ax, facecolor="#F28E2B", edgecolor="none", alpha=0.18)
    ground.boundary.plot(ax=ax, color="#00A6D6", linewidth=0.26, alpha=0.82)
    roofs.boundary.plot(ax=ax, color=RED, linewidth=0.32, alpha=0.90)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.text(
        0.015,
        0.985,
        f"全部建筑：{len(supports)} 栋\n影像范围内可见：{visible_count} 栋",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 2.5},
    )
    ax.legend(
        handles=[
            Patch(facecolor="#F28E2B", edgecolor="none", alpha=0.35, label="GAMMA校正建筑体支持区"),
            Patch(facecolor="none", edgecolor="#00A6D6", linewidth=0.8, label="GAMMA地面投影轮廓"),
            Patch(facecolor="none", edgecolor=RED, linewidth=0.8, label="先验高度屋顶投影轮廓"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.88,
    )
    save_svg(fig, out_dir, "18_全部建筑高度辅助投影至雷达影像")


def figure_baseline(pairs: pd.DataFrame, out_dir: Path) -> None:
    nodes = derive_acquisition_baselines(pairs)
    lookup = nodes.set_index("key")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for row in pairs.itertuples(index=False):
        x = [pd.to_datetime(str(row.master)), pd.to_datetime(str(row.slave))]
        y = [lookup.loc[str(row.master), "bperp_m"], lookup.loc[str(row.slave), "bperp_m"]]
        ax.plot(x, y, color="#4B5563", linewidth=0.65, alpha=0.75, zorder=1)
    ax.scatter(nodes["date"], nodes["bperp_m"], s=21, facecolor=RED, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axhline(0, color="#BBBBBB", linewidth=0.65, zorder=0)
    ax.set_xlabel("成像日期")
    ax.set_ylabel("相对垂直基线（m）")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
    ax.grid(True, color="#E5E7EB", linewidth=0.55, linestyle="--")
    ax.text(0.02, 0.97, f"{len(nodes)} 景影像，{len(pairs)} 个干涉对", transform=ax.transAxes, va="top")
    save_svg(fig, out_dir, "03_干涉对时空基线网络")


def largest_buffer_overlaps(projection: gpd.GeoDataFrame, image_shape: tuple[int, int], count: int = 3) -> list[tuple[int, int, object]]:
    valid = projection[projection.geometry.notna() & ~projection.geometry.is_empty].copy().reset_index(drop=True)
    # Projection coordinates are SAR pixels although the source GeoJSON carries
    # an EPSG:4326 tag. Buffer shapely objects directly so 4 means four pixels.
    expanded = gpd.GeoSeries([geom.buffer(4.0) for geom in valid.geometry], index=valid.index, crs=None)
    left, right = expanded.sindex.query(expanded, predicate="intersects")
    candidates = []
    for i, j in zip(left, right):
        id_col = "clean_id" if "clean_id" in valid.columns else "fid"
        if i >= j or int(valid.iloc[i][id_col]) == int(valid.iloc[j][id_col]):
            continue
        x0, y0, x1, y1 = expanded.iloc[i].union(expanded.iloc[j]).bounds
        if x1 <= 0 or x0 >= image_shape[1] or y1 <= 0 or y0 >= image_shape[0]:
            continue
        inter = expanded.iloc[i].intersection(expanded.iloc[j])
        if not inter.is_empty and inter.area > 5:
            candidates.append((float(inter.area), int(i), int(j), inter))
    candidates.sort(reverse=True)
    chosen = []
    used: set[int] = set()
    for _, i, j, inter in candidates:
        if i in used or j in used:
            continue
        chosen.append((i, j, inter))
        used.update([i, j])
        if len(chosen) == count:
            break
    return [(i, j, inter, valid) for i, j, inter in chosen]


def figure_overlap(projection: gpd.GeoDataFrame, amplitude: np.ndarray, out_dir: Path) -> None:
    examples = largest_buffer_overlaps(projection, amplitude.shape, 3)
    if not examples:
        raise RuntimeError("No projected-building overlap examples found")
    chinese_numbers = ["一", "二", "三"]
    for row_idx, (i, j, inter, valid) in enumerate(examples):
        g1 = valid.geometry.iloc[i].buffer(4.0)
        g2 = valid.geometry.iloc[j].buffer(4.0)
        bounds = np.array([g1.union(g2).bounds], dtype=float).ravel()
        x0, y0, x1, y1 = bounds
        pad = 8
        for col_idx in range(2):
            fig, ax = plt.subplots(figsize=(5.4, 3.8))
            ax.imshow(amplitude, cmap="gray", origin="upper", interpolation="nearest", vmin=np.nanpercentile(amplitude, 5), vmax=np.nanpercentile(amplitude, 99.5))
            if col_idx == 0:
                gpd.GeoSeries([g1]).plot(ax=ax, facecolor=(1, 0, 0, 0.14), edgecolor=RED, linewidth=1.0)
                gpd.GeoSeries([g2]).plot(ax=ax, facecolor=(0.12, 0.45, 0.78, 0.16), edgecolor=BLUE, linewidth=1.0)
                gpd.GeoSeries([inter]).plot(ax=ax, facecolor=(0.2, 0.2, 0.2, 0.36), edgecolor="#222222", hatch="////", linewidth=0.8)
            else:
                gpd.GeoSeries([g1.difference(inter)]).plot(ax=ax, facecolor=(1, 0, 0, 0.20), edgecolor=RED, linewidth=0.9)
                gpd.GeoSeries([g2.difference(inter)]).plot(ax=ax, facecolor=(0.12, 0.45, 0.78, 0.20), edgecolor=BLUE, linewidth=0.9)
            ax.set_xlim(max(0, x0 - pad), min(amplitude.shape[1], x1 + pad))
            ax.set_ylim(min(amplitude.shape[0], y1 + pad), max(0, y0 - pad))
            ax.set_xticks([])
            ax.set_yticks([])
            state = "排除前" if col_idx == 0 else "排除后"
            ax.text(0.02, 0.98, f"重叠{state}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5})
            save_svg(fig, out_dir, f"{4 + row_idx * 2 + col_idx:02d}_建筑重叠示例{chinese_numbers[row_idx]}_{state}")


def figure_islands(label: np.ndarray, selected_mask: np.ndarray, amplitude: np.ndarray, out_dir: Path) -> None:
    lo, hi = np.nanpercentile(amplitude, [3, 99.5])
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    ax.contour(selected_mask.astype(float), levels=[0.5], colors=[RED], linewidths=0.55)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    rr, cc = np.nonzero(selected_mask)
    if rr.size:
        center_r, center_c = int(np.median(rr)), int(np.median(cc))
        ax.add_patch(plt.Rectangle((center_c - 90, center_r - 70), 180, 140, fill=False, edgecolor=GREEN, linewidth=1.1))
    save_svg(fig, out_dir, "10_建筑孤岛全景")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.imshow(amplitude, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    ax.contour(selected_mask.astype(float), levels=[0.5], colors=[RED], linewidths=0.7)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    if rr.size:
        ax.set_xlim(max(0, center_c - 90), min(label.shape[1], center_c + 90))
        ax.set_ylim(min(label.shape[0], center_r + 70), max(0, center_r - 70))
    save_svg(fig, out_dir, "11_建筑孤岛局部放大")


def island_height_raster(label: np.ndarray, islands: pd.DataFrame) -> np.ndarray:
    raster = np.full(label.shape, np.nan, dtype=np.float32)
    for row in islands.dropna(subset=["height_m"]).itertuples(index=False):
        raster[label == int(row.island_id)] = float(row.height_m)
    return raster


def figure_height_comparison(local_map: np.ndarray, global_map: np.ndarray, out_dir: Path) -> None:
    entries = [
        ("12_屋顶核心区SBAS高度图", local_map, "屋顶核心区SBAS高度（m）", 99),
        ("13_全局解缠高度图", global_map, "全局解缠高度范围（m）", 97),
    ]
    for name, data, colorbar_label, upper_percentile in entries:
        values = data[np.isfinite(data)]
        vmax = max(10.0, float(np.nanpercentile(values, upper_percentile))) if values.size else 100.0
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=vmax, origin="upper", interpolation="nearest")
        ax.set_xlabel("距离向像元")
        ax.set_ylabel("方位向像元")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(colorbar_label)
        save_svg(fig, out_dir, name)


def add_extruded_polygon(ax, polygon: Polygon, height: float, x0: float, y0: float, color_value: float, norm_obj, cmap) -> None:
    xy = np.asarray(polygon.exterior.coords)
    if len(xy) < 4:
        return
    x = (xy[:, 0] - x0) / 1000.0
    y = (xy[:, 1] - y0) / 1000.0
    z0 = np.zeros_like(x)
    z1 = np.full_like(x, height)
    color = cmap(norm_obj(color_value))
    top = Poly3DCollection([list(zip(x, y, z1))], facecolors=[color], edgecolors="#7A4A15", linewidths=0.25, alpha=0.95)
    ax.add_collection3d(top)
    side_faces = []
    for idx in range(len(x) - 1):
        side_faces.append([(x[idx], y[idx], 0), (x[idx + 1], y[idx + 1], 0), (x[idx + 1], y[idx + 1], height), (x[idx], y[idx], height)])
    sides = Poly3DCollection(side_faces, facecolors=[colors.to_rgba(color, 0.72)], edgecolors="#855A2A", linewidths=0.15)
    ax.add_collection3d(sides)


def figure_lod1(buildings: gpd.GeoDataFrame, result_geo: gpd.GeoDataFrame, out_dir: Path) -> None:
    metric = buildings.to_crs(3857)
    solved = result_geo.dropna(subset=["height_insar_m"]).to_crs(3857)
    x0, y0, x1, y1 = metric.total_bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.02, 0.02, 0.80, 0.95])
    for geom in metric.geometry:
        for part in geometry_parts(geom):
            xy = np.asarray(part.exterior.coords)
            ax.plot((xy[:, 0] - cx) / 1000, (xy[:, 1] - cy) / 1000, np.zeros(len(xy)), color="#C9D1D9", linewidth=0.18, alpha=0.7)
    vals = solved["height_insar_m"].to_numpy(float)
    vmax = max(20.0, float(np.nanpercentile(vals, 97))) if vals.size else 100.0
    norm_obj = colors.Normalize(0, vmax)
    cmap = plt.get_cmap("YlOrBr")
    for row in solved.itertuples(index=False):
        height = float(row.height_insar_m)
        for part in geometry_parts(row.geometry):
            add_extruded_polygon(ax, part.simplify(0.4), height, cx, cy, height, norm_obj, cmap)
    ax.view_init(elev=33, azim=-58)
    ax.set_xlabel("东向偏移（km）")
    ax.set_ylabel("北向偏移（km）")
    ax.set_zlabel("")
    ax.set_zticks([])
    ax.set_box_aspect((1.7, 1.2, 0.55))
    ax.grid(False)
    cax = fig.add_axes([0.86, 0.18, 0.025, 0.64])
    fig.colorbar(ScalarMappable(norm=norm_obj, cmap=cmap), cax=cax, label="屋顶核心区SBAS高度（m）")
    save_svg(fig, out_dir, "14_严格有解建筑三维重建")


def validation_data(building_results: pd.DataFrame) -> pd.DataFrame:
    df = building_results.dropna(subset=["zjc_height_range_m", "vector_height_prior_m"]).copy()
    df = df[(df["zjc_height_range_m"] > 0) & (df["zjc_height_range_m"] < 500) & (df["vector_height_prior_m"] > 0)]
    df["residual_m"] = df["zjc_height_range_m"] - df["vector_height_prior_m"]
    df["absolute_error_m"] = df["residual_m"].abs()
    return df


def metrics_text(df: pd.DataFrame) -> tuple[float, float, float]:
    x = df["vector_height_prior_m"].to_numpy(float)
    y = df["zjc_height_range_m"].to_numpy(float)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(df) > 1 else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))
    return r2, rmse, mae


def figure_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    r2, rmse, mae = metrics_text(df)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    sc = ax.scatter(df["vector_height_prior_m"], df["zjc_height_range_m"], c=df["floor_prior"], cmap="viridis", s=28, edgecolor="white", linewidth=0.35)
    maximum = max(float(df["vector_height_prior_m"].max()), float(df["zjc_height_range_m"].max())) * 1.05
    ax.plot([0, maximum], [0, maximum], linestyle="--", color="#333333", linewidth=0.9, label="1:1 reference")
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_xlabel("Vector prior height (m)")
    ax.set_ylabel("ZJC InSAR height range (m)")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.96, f"N = {len(df)}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} m\nMAE = {mae:.2f} m", transform=ax.transAxes, va="top")
    fig.colorbar(sc, ax=ax, label="Number of floors")
    fig.text(0.50, 0.006, "Consistency only; vector prior is not independent LiDAR truth", ha="center", va="bottom", fontsize=7, color="#555555")
    fig.subplots_adjust(bottom=0.14)
    save_svg(fig, out_dir, "07_height_prior_consistency_scatter")


def figure_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    residual = df["residual_m"].to_numpy(float)
    mu, sigma = float(np.mean(residual)), float(np.std(residual, ddof=1))
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.hist(residual, bins=max(8, int(np.sqrt(len(residual)))), density=True, color="#78ADD2", edgecolor="#315B7D", linewidth=0.55)
    if sigma > 0:
        xx = np.linspace(float(np.min(residual)), float(np.max(residual)), 300)
        ax.plot(xx, norm.pdf(xx, mu, sigma), color=RED, label="Normal fit")
    ax.set_xlabel("Residual relative to vector prior (m)")
    ax.set_ylabel("Probability density")
    ax.text(0.04, 0.96, f"$\\mu$ = {mu:.2f} m\n$\\sigma$ = {sigma:.2f} m", transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    save_svg(fig, out_dir, "08_residual_frequency_distribution")


def figure_residual(df: pd.DataFrame, out_dir: Path) -> None:
    residual = df["residual_m"].to_numpy(float)
    mu, sigma = float(np.mean(residual)), float(np.std(residual, ddof=1))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(df["vector_height_prior_m"], residual, color="#5DA5DA", s=23, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#333333", linewidth=0.75)
    ax.axhline(mu - 1.96 * sigma, color=RED, linestyle="--", linewidth=0.85)
    ax.axhline(mu + 1.96 * sigma, color=RED, linestyle="--", linewidth=0.85, label="95% interval")
    ax.set_xlabel("Vector prior height (m)")
    ax.set_ylabel("Residual (m)")
    ax.legend(frameon=False)
    ax.text(0.04, 0.05, f"$\\mu$ = {mu:.2f} m\n$\\sigma$ = {sigma:.2f} m", transform=ax.transAxes, va="bottom")
    save_svg(fig, out_dir, "09_residual_vs_vector_prior")


def figure_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.5))
    ax.boxplot(df["residual_m"], widths=0.42, patch_artist=True, boxprops={"facecolor": "#E5E7EB", "edgecolor": "#222222"}, medianprops={"color": ORANGE, "linewidth": 1.5}, flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": GRAY, "markersize": 4})
    ax.axhline(0, color=RED, linestyle="--", linewidth=0.85)
    ax.set_xticklabels(["Overall error"])
    ax.set_ylabel("Residual relative to vector prior (m)")
    ax.grid(axis="y", color="#E5E7EB", linestyle="--", linewidth=0.55)
    save_svg(fig, out_dir, "10_overall_error_boxplot")


def figure_cdf(df: pd.DataFrame, out_dir: Path) -> None:
    err = np.sort(df["absolute_error_m"].to_numpy(float))
    prob = np.arange(1, len(err) + 1) / len(err)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(err, prob, color=BLUE, linewidth=1.7)
    palette = [GREEN, ORANGE, RED]
    for q, color in zip([0.5, 0.75, 0.9], palette):
        value = float(np.quantile(err, q))
        ax.plot([value, value], [0, q], color=color, linestyle=":", linewidth=1.0)
        ax.plot([0, value], [q, q], color=color, linestyle=":", linewidth=1.0)
        ax.text(value, min(0.98, q + 0.035), f"{int(q * 100)}%: {value:.2f} m", color=color, ha="center", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Absolute difference from vector prior (m)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, color="#E5E7EB", linestyle="--", linewidth=0.5)
    save_svg(fig, out_dir, "11_absolute_error_cdf")


def polygon_collection(gdf: gpd.GeoDataFrame, value_col: str, norm_obj, cmap) -> PatchCollection:
    patches = []
    facecolors = []
    for row in gdf.itertuples(index=False):
        value = getattr(row, value_col)
        if not np.isfinite(value):
            continue
        for part in geometry_parts(row.geometry):
            patches.append(MplPolygon(np.asarray(part.exterior.coords), closed=True))
            facecolors.append(cmap(norm_obj(float(value))))
    return PatchCollection(patches, facecolor=facecolors, edgecolor="white", linewidth=0.16)


def figure_map_comparison(buildings: gpd.GeoDataFrame, old_result_geo: gpd.GeoDataFrame, optimized_geo: gpd.GeoDataFrame, out_dir: Path) -> None:
    base = buildings[["clean_id", "height", "geometry"]].copy()
    base["vector_height_prior_m"] = pd.to_numeric(base["height"], errors="coerce")
    merged = base.merge(old_result_geo.drop(columns="geometry")[["clean_id", "global_height_range_m"]], on="clean_id", how="left")
    merged = merged.merge(optimized_geo.drop(columns="geometry")[["clean_id", "height_insar_m"]], on="clean_id", how="left")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=buildings.crs)
    physical_values = np.r_[merged["vector_height_prior_m"].dropna(), merged["height_insar_m"].dropna()]
    vmax = max(20.0, float(np.nanpercentile(physical_values, 99)))
    norm_obj = colors.Normalize(0, vmax)
    cmap = plt.get_cmap("viridis")
    entries = [
        ("vector_height_prior_m", "15_投影初始化先验高度分布"),
        ("global_height_range_m", "16_全局解缠高度分布"),
        ("height_insar_m", "17_屋顶核心区SBAS建筑高度分布"),
    ]
    for col, name in entries:
        is_strict_height = col == "height_insar_m"
        fig, ax = plt.subplots(figsize=(9.2, 8.4) if is_strict_height else (5.2, 4.8))
        buildings.boundary.plot(ax=ax, color="#D1D5DB", linewidth=0.18)
        collection = polygon_collection(merged, col, norm_obj, cmap)
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_axis_off()
        if col == "vector_height_prior_m":
            ax.text(0.02, 0.02, "仅用于首次GAMMA屋顶投影初始化", transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
        elif col == "global_height_range_m":
            ax.text(0.02, 0.02, "超过统一色标上限的值按上限显示", transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
        elif is_strict_height:
            solved_count = int(merged[col].notna().sum())
            solved = merged.dropna(subset=[col])
            for row in solved.itertuples(index=False):
                value = float(getattr(row, col))
                point = row.geometry.representative_point()
                normalized = float(norm_obj(value))
                text_color = "#111111" if normalized >= 0.58 else "#FFFFFF"
                ax.text(
                    point.x,
                    point.y,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=3.0,
                    color=text_color,
                    zorder=5,
                    clip_on=True,
                )
            ax.text(0.02, 0.02, f"严格有解：{solved_count}栋；标注为高度（m）；无解建筑不填充", transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9}, zorder=6)
        fig.colorbar(ScalarMappable(norm=norm_obj, cmap=cmap), ax=ax, fraction=0.035, pad=0.015, label="高度（m）")
        save_svg(fig, out_dir, name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default="work/zjc_original_reproduction")
    p.add_argument("--output-dir", default="picall")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--projection", default="work/projection/20200708_clean_equal_height_roof_projection_sar.geojson")
    p.add_argument("--paper-projection", default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson")
    p.add_argument("--amplitude", default="work/mli/mean_crop_bmp_amplitude.npy")
    p.add_argument("--pairs", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--fid-map", default="results/tables/clean_equal_height_fid_uid_map.csv")
    p.add_argument("--optimized-island-label", default="work/roof_sbas_optimized/roof_core_island_label.npy")
    p.add_argument("--optimized-fid-mask", default="work/roof_sbas_optimized/roof_core_clean_id_mask.npy")
    p.add_argument("--optimized-island-heights", default="work/roof_sbas_optimized/roof_core_sbas_islands.csv")
    p.add_argument("--optimized-buildings", default="results/geodata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.geojson")
    p.add_argument("--optimized-summary", default="results/metadata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT
    resolve = lambda value: root / value
    work_dir = resolve(args.work_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(work_dir / "zjc_reproduction_arrays.npz")
    island_results = pd.read_csv(work_dir / "zjc_island_results.csv")
    building_results = pd.read_csv(work_dir / "zjc_building_results.csv")
    buildings = gpd.read_file(resolve(args.buildings))
    projection = gpd.read_file(resolve(args.projection))
    paper_projection = gpd.read_file(resolve(args.paper_projection))
    amplitude = np.load(resolve(args.amplitude)).astype(float)
    pairs = pd.read_csv(resolve(args.pairs))
    fid_map = pd.read_csv(resolve(args.fid_map))
    optimized_label = np.load(resolve(args.optimized_island_label))
    optimized_mask = np.load(resolve(args.optimized_fid_mask)) > 0
    optimized_islands = pd.read_csv(resolve(args.optimized_island_heights))
    optimized_geo = gpd.read_file(resolve(args.optimized_buildings))
    selected_uids = set(optimized_geo.loc[optimized_geo["height_insar_m"].notna(), "clean_id"].astype(int))

    building_results = building_results.merge(fid_map[["touying_fid", "uid"]], on="touying_fid", how="left")
    building_results["clean_id"] = building_results["uid"].astype("Int64")
    result_geo = buildings[["clean_id", "geometry"]].merge(building_results, on="clean_id", how="inner")
    result_geo = gpd.GeoDataFrame(result_geo, geometry="geometry", crs=buildings.crs)

    figure_study_area(buildings, selected_uids, amplitude, out_dir)
    figure_all_building_projections(paper_projection, amplitude, out_dir)
    figure_baseline(pairs, out_dir)
    figure_overlap(paper_projection[paper_projection["surface"] == "roof"].copy(), amplitude, out_dir)
    figure_islands(optimized_label, optimized_mask, amplitude, out_dir)
    figure_height_comparison(island_height_raster(optimized_label, optimized_islands), arrays["global_height_map"], out_dir)
    figure_lod1(buildings, optimized_geo, out_dir)
    figure_map_comparison(buildings, result_geo, optimized_geo, out_dir)

    svg_files = sorted(out_dir.glob("*.svg"))
    manifest = {
        "svg_count": len(svg_files),
        "files": [path.name for path in svg_files],
        "source_summary": json.loads(resolve(args.optimized_summary).read_text(encoding="utf-8")),
        "zjc_reference_summary": json.loads((work_dir / "summary.json").read_text(encoding="utf-8")),
        "第17图严格高度数值标签数": int(optimized_geo["height_insar_m"].notna().sum()),
        "说明": "按用户要求，各图均为单独SVG输出，不合并排版；文件名及图内可见文字均使用中文。第10至12、14、17图已切换为屋顶核心区、稳定地面定标、时序多初值模糊度搜索和质量加权中位数的260栋严格SBAS结果，768栋无解保持空值；第15图仅表示首次GAMMA投影初始化高度。第18图采用GAMMA coord_to_sarpix与DIFF_par投影及训练/验证时相SAR校正；height不进入反演、填充、校正目标或质量控制；不生成误差统计类图件。",
    }
    (work_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
