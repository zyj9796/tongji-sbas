#!/usr/bin/env python3
"""Refine a GAMMA building projection against persistent SAR image features.

The input GAMMA R-D geometry remains the physical starting point.  This script
only estimates a small residual affine transform and conservative per-building
translations from multi-temporal SAR amplitude edges.  Alternating acquisitions
form frozen training and validation stacks; a correction is accepted only when
it improves both stacks.  Vector height is never used as a matching target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, optimize
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from skimage import exposure, feature, filters
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--projection",
        default="work/zjc_original_reproduction/20200708_all_building_gamma_corrected_projection_sar.geojson",
    )
    p.add_argument("--amplitude", default="work/mli/mean_crop_bmp_amplitude.npy")
    p.add_argument("--bmp-dir", default="data/tongji_rslc")
    p.add_argument(
        "--output",
        default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson",
    )
    p.add_argument(
        "--metrics",
        default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_metrics.csv",
    )
    p.add_argument(
        "--summary",
        default="results/metadata/tongji_gamma_projection_sar_feature_refinement_summary.json",
    )
    p.add_argument(
        "--figure",
        default="",
        help="Optional legacy projection-overlay figure; empty keeps RSLC-derived Figure 02 untouched",
    )
    p.add_argument("--projection-figure", default="picall/18_全部建筑高度辅助投影至雷达影像.svg")
    p.add_argument("--local-limit", type=int, default=3)
    p.add_argument("--local-min-gain", type=float, default=0.012)
    p.add_argument("--seed", type=int, default=20260823)
    return p.parse_args()


def normalized_log(image: np.ndarray) -> np.ndarray:
    data = np.log1p(np.maximum(np.asarray(image, dtype=np.float32), 0.0))
    valid = np.isfinite(data)
    lo, hi = np.nanpercentile(data[valid], [1.0, 99.7])
    return np.clip((data - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0).astype(np.float32)


def enhance(image: np.ndarray) -> np.ndarray:
    base = normalized_log(image)
    local = exposure.equalize_adapthist(base, kernel_size=(31, 31), clip_limit=0.012)
    smooth = filters.gaussian(local, sigma=1.15, preserve_range=True)
    sharp = np.clip(local + 0.85 * (local - smooth), 0.0, 1.0)
    return np.clip(0.25 * base + 0.75 * sharp, 0.0, 1.0).astype(np.float32)


def feature_strength(image: np.ndarray) -> np.ndarray:
    visual = enhance(image)
    grad = filters.scharr(visual).astype(np.float32)
    grad /= max(float(np.nanpercentile(grad, 99.5)), 1.0e-6)
    grad = np.clip(grad, 0.0, 1.0)
    edges = feature.canny(visual, sigma=1.0, low_threshold=0.06, high_threshold=0.18)
    edge_support = ndimage.gaussian_filter(edges.astype(np.float32), sigma=0.85)
    edge_support /= max(float(edge_support.max()), 1.0e-6)
    return np.clip(0.70 * grad + 0.30 * edge_support, 0.0, 1.0).astype(np.float32)


def mildly_enhanced_amplitude(image: np.ndarray) -> np.ndarray:
    """Preserve the original dark SAR appearance with only global enhancement."""
    data = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(data)
    lo, hi = np.nanpercentile(data[valid], [2.0, 99.5])
    base = np.clip((data - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)
    smooth = filters.gaussian(base, sigma=0.70, preserve_range=True)
    sharpened = np.clip(base + 0.32 * (base - smooth), 0.0, 1.0)
    # Gamma > 1 keeps the black/grey tonal hierarchy and prevents whitening.
    return np.power(sharpened, 1.08).astype(np.float32)


def load_temporal_split(directory: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    paths = sorted(directory.glob("*.crop.bmp"))
    if len(paths) < 6:
        raise RuntimeError("至少需要6景时相BMP进行训练/验证拆分")
    train_paths, validation_paths = paths[::2], paths[1::2]

    def robust_mean(items: list[Path]) -> np.ndarray:
        stack = []
        for path in items:
            arr = np.asarray(Image.open(path), dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            if arr.shape != shape:
                raise ValueError(f"影像尺寸不一致：{path} {arr.shape} != {shape}")
            stack.append(normalized_log(arr))
        return np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    return robust_mean(train_paths), robust_mean(validation_paths), [p.name for p in train_paths], [p.name for p in validation_paths]


def polygon_parts(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def sampled_boundary(geometry, spacing: float = 1.25) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for poly in polygon_parts(geometry):
        ring = poly.exterior
        length = float(ring.length)
        count = max(8, int(math.ceil(length / spacing)))
        for distance in np.linspace(0.0, length, count, endpoint=False):
            point = ring.interpolate(float(distance))
            points.append((float(point.x), float(point.y)))
    return np.asarray(points, dtype=np.float64)


def affine_coefficients(params: np.ndarray, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    tx, ty, rotation_deg, scale_x_delta, scale_y_delta, shear = map(float, params)
    theta = math.radians(rotation_deg)
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    scale_shear = np.asarray([[1.0 + scale_x_delta, shear], [0.0, 1.0 + scale_y_delta]])
    matrix = rotation @ scale_shear
    centre = np.asarray(center, dtype=float)
    offset = centre + np.asarray([tx, ty]) - matrix @ centre
    return matrix, offset


def transform_points(points: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points @ matrix.T + offset[None, :]


def sample_image(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(
        image,
        [points[:, 1], points[:, 0]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def balanced_score(image: np.ndarray, points: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> float:
    values = sample_image(image, points)
    totals = np.add.reduceat(values, starts)
    per_building = totals / counts
    if len(per_building) < 20:
        return float("-inf")
    lower, upper = np.nanpercentile(per_building, [10, 90])
    trimmed = per_building[(per_building >= lower) & (per_building <= upper)]
    return float(0.55 * np.nanmedian(per_building) + 0.45 * np.nanmean(trimmed))


def build_boundary_stack(roofs: gpd.GeoDataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    footprint = box(0, 0, shape[1] - 1, shape[0] - 1)
    arrays: list[np.ndarray] = []
    ids: list[int] = []
    for row in roofs.itertuples(index=False):
        if row.geometry is None or row.geometry.is_empty or not row.geometry.intersects(footprint):
            continue
        points = sampled_boundary(row.geometry)
        if len(points) < 8:
            continue
        arrays.append(points)
        ids.append(int(row.clean_id))
    counts = np.asarray([len(x) for x in arrays], dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    return np.vstack(arrays), starts, counts, ids


def optimize_global(
    feature_train: np.ndarray,
    points: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    center: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    bounds = [(-4.0, 4.0), (-4.0, 4.0), (-0.45, 0.45), (-0.006, 0.006), (-0.006, 0.006), (-0.004, 0.004)]

    def objective(params: np.ndarray) -> float:
        matrix, offset = affine_coefficients(params, center)
        score = balanced_score(feature_train, transform_points(points, matrix, offset), starts, counts)
        regularization = (
            0.00006 * (params[0] ** 2 + params[1] ** 2)
            + 0.002 * params[2] ** 2
            + 0.35 * (params[3] ** 2 + params[4] ** 2 + params[5] ** 2)
        )
        return -score + regularization

    result = optimize.differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=8,
        maxiter=24,
        polish=True,
        workers=1,
        updating="immediate",
        tol=2.0e-4,
    )
    identity = np.zeros(6, dtype=float)
    base = -objective(identity)
    best = -objective(result.x)
    return np.asarray(result.x, dtype=float), {"base_score": float(base), "best_score": float(best), "gain": float(best - base)}


def shapely_affine(geometry, matrix: np.ndarray, offset: np.ndarray):
    return affinity.affine_transform(
        geometry,
        [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], offset[0], offset[1]],
    )


def local_score(feature_map: np.ndarray, points: np.ndarray, dr: int, dc: int) -> float:
    shifted = points + np.asarray([dc, dr], dtype=float)
    return float(np.mean(sample_image(feature_map, shifted)))


def choose_local_shift(
    train_feature: np.ndarray,
    validation_feature: np.ndarray,
    geometry,
    limit: int,
    minimum_gain: float,
) -> dict[str, object]:
    points = sampled_boundary(geometry)
    base_train = local_score(train_feature, points, 0, 0)
    base_validation = local_score(validation_feature, points, 0, 0)

    def best(image: np.ndarray) -> tuple[float, int, int]:
        candidates = []
        for dr in range(-limit, limit + 1):
            for dc in range(-limit, limit + 1):
                penalty = 0.0007 * (dr * dr + dc * dc)
                candidates.append((local_score(image, points, dr, dc) - penalty, dr, dc))
        return max(candidates)

    train_best, train_dr, train_dc = best(train_feature)
    validation_best, validation_dr, validation_dc = best(validation_feature)
    train_gain = train_best - base_train
    validation_gain_at_train = local_score(validation_feature, points, train_dr, train_dc) - base_validation
    agrees = abs(train_dr - validation_dr) <= 1 and abs(train_dc - validation_dc) <= 1
    boundary_hit = max(abs(train_dr), abs(train_dc)) == limit
    accepted = bool(
        len(points) >= 8
        and (train_dr != 0 or train_dc != 0)
        and train_gain >= minimum_gain
        and validation_gain_at_train >= minimum_gain
        and agrees
        and not boundary_hit
    )
    return {
        "local_row_shift": int(train_dr if accepted else 0),
        "local_col_shift": int(train_dc if accepted else 0),
        "local_shift_accepted": accepted,
        "train_edge_score_before": float(base_train),
        "train_edge_score_after": float(local_score(train_feature, points, train_dr, train_dc) if accepted else base_train),
        "validation_edge_score_before": float(base_validation),
        "validation_edge_score_after": float(local_score(validation_feature, points, train_dr, train_dc) if accepted else base_validation),
        "train_best_row_shift": int(train_dr),
        "train_best_col_shift": int(train_dc),
        "validation_best_row_shift": int(validation_dr),
        "validation_best_col_shift": int(validation_dc),
        "temporal_shift_agreement": bool(agrees),
        "search_boundary_hit": bool(boundary_hit),
    }


def plot_figure(amplitude: np.ndarray, roofs: gpd.GeoDataFrame, output: Path, summary_note: str) -> None:
    # Figure 02 intentionally remains the original single SAR image plate.
    # Projection overlays belong to Figure 18; adding them here obscures the
    # amplitude texture the panel is intended to document.
    visual = mildly_enhanced_amplitude(amplitude)
    fig, ax = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    ax.imshow(visual, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.annotate(
        "距离向",
        xy=(0.92, 0.95),
        xytext=(0.68, 0.95),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#2A9D8F", "lw": 1.4},
        color="#2A9D8F",
        va="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "同济雷达平均幅度"})
    plt.close(fig)


def plot_projection_figure(
    amplitude: np.ndarray,
    projection: gpd.GeoDataFrame,
    output: Path,
    visible_count: int,
) -> None:
    visual = enhance(amplitude)
    supports = projection[projection["surface"].eq("layover")]
    bottoms = projection[projection["surface"].eq("bottom")]
    roofs = projection[projection["surface"].eq("roof")]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.imshow(visual, cmap="gray", vmin=0.02, vmax=0.98, origin="upper", interpolation="nearest")
    supports.plot(ax=ax, facecolor="#F28E2B", edgecolor="none", alpha=0.10, zorder=2)
    bottoms.boundary.plot(ax=ax, color="#2878B5", linewidth=0.24, alpha=0.72, zorder=3)
    roofs.boundary.plot(ax=ax, color="#00D5E8", linewidth=0.34, alpha=0.90, zorder=4)
    ax.set_xlim(0, amplitude.shape[1] - 1)
    ax.set_ylim(amplitude.shape[0] - 1, 0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("SAR特征约束后的建筑体投影", pad=7, fontweight="bold")
    ax.text(
        0.012,
        0.985,
        f"全部建筑：{len(roofs)}栋\n影像范围内可见屋顶：{visible_count}栋",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#AAB5BA", "alpha": 0.88, "pad": 2.2},
    )
    ax.legend(
        handles=[
            Patch(facecolor="#F28E2B", edgecolor="none", alpha=0.25, label="建筑体投影支持区"),
            Patch(facecolor="none", edgecolor="#2878B5", label="地面投影轮廓"),
            Patch(facecolor="none", edgecolor="#00D5E8", label="特征校正屋顶轮廓"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "SAR特征约束后的建筑体投影"})
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 11,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    amplitude = np.load(args.amplitude).astype(np.float32)
    train_amp, validation_amp, train_names, validation_names = load_temporal_split(Path(args.bmp_dir), amplitude.shape)
    train_feature = feature_strength(train_amp)
    validation_feature = feature_strength(validation_amp)
    projection = gpd.read_file(args.projection).set_crs(None, allow_override=True)
    roofs = projection[projection["surface"].eq("roof")].copy()
    points, starts, counts, visible_ids = build_boundary_stack(roofs, amplitude.shape)
    center = ((amplitude.shape[1] - 1) / 2.0, (amplitude.shape[0] - 1) / 2.0)
    params, train_global = optimize_global(train_feature, points, starts, counts, center, args.seed)
    matrix, offset = affine_coefficients(params, center)
    transformed_points = transform_points(points, matrix, offset)
    validation_base = balanced_score(validation_feature, points, starts, counts)
    validation_candidate = balanced_score(validation_feature, transformed_points, starts, counts)
    global_accepted = bool(train_global["gain"] > 0.002 and validation_candidate - validation_base > 0.002)
    if not global_accepted:
        params = np.zeros(6, dtype=float)
        matrix, offset = affine_coefficients(params, center)

    refined = projection.copy()
    refined["geometry"] = refined.geometry.map(lambda geom: shapely_affine(geom, matrix, offset))
    refined["sar_feature_global_affine_accepted"] = np.int8(global_accepted)
    metrics: list[dict[str, object]] = []
    roof_index = refined.index[refined["surface"].eq("roof")]
    for ordinal, idx in enumerate(roof_index, start=1):
        row = refined.loc[idx]
        result = choose_local_shift(
            train_feature,
            validation_feature,
            row.geometry,
            args.local_limit,
            args.local_min_gain,
        )
        clean_id = int(row.clean_id)
        dr, dc = int(result["local_row_shift"]), int(result["local_col_shift"])
        same_building = refined["clean_id"].astype(int).eq(clean_id)
        if dr or dc:
            refined.loc[same_building, "geometry"] = refined.loc[same_building, "geometry"].map(
                lambda geom: affinity.translate(geom, xoff=dc, yoff=dr)
            )
        refined.loc[same_building, "sar_feature_local_row_shift"] = dr
        refined.loc[same_building, "sar_feature_local_col_shift"] = dc
        refined.loc[same_building, "sar_feature_local_accepted"] = np.int8(bool(result["local_shift_accepted"]))
        metrics.append({"clean_id": clean_id, **result})
        if ordinal % 250 == 0 or ordinal == len(roof_index):
            print(f"SAR特征局部校正：{ordinal}/{len(roof_index)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    refined.to_file(output, driver="GeoJSON")
    metrics_table = pd.DataFrame(metrics)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(metrics_path, index=False)

    corrected_roofs = refined[refined["surface"].eq("roof")].copy()
    footprint = box(0, 0, amplitude.shape[1], amplitude.shape[0])
    visible = int(corrected_roofs.geometry.intersects(footprint).sum())
    accepted_local = int(metrics_table["local_shift_accepted"].sum())
    validation_gain = float(validation_candidate - validation_base) if global_accepted else 0.0
    summary = {
        "method": "GAMMA R-D initialization + frozen temporal SAR feature affine/local residual refinement",
        "input_projection": args.projection,
        "output_projection": args.output,
        "height_used_as_matching_target_or_fill": False,
        "training_images": train_names,
        "validation_images": validation_names,
        "visible_roofs": visible,
        "global_affine": {
            "accepted": global_accepted,
            "parameters_tx_ty_rotation_sx_sy_shear": [float(x) for x in params],
            "train_score_before": train_global["base_score"],
            "train_score_after": train_global["best_score"] if global_accepted else train_global["base_score"],
            "train_gain": train_global["gain"] if global_accepted else 0.0,
            "validation_score_before": float(validation_base),
            "validation_score_after": float(validation_candidate) if global_accepted else float(validation_base),
            "validation_gain": validation_gain,
        },
        "local_shift_limit_px": args.local_limit,
        "local_min_gain": args.local_min_gain,
        "local_corrections_accepted": accepted_local,
        "local_corrections_rejected_or_zero": int(len(metrics_table) - accepted_local),
        "validation_mean_edge_score_before": float(metrics_table["validation_edge_score_before"].mean()),
        "validation_mean_edge_score_after": float(metrics_table["validation_edge_score_after"].mean()),
        "metrics_csv": args.metrics,
        "figure_svg": args.figure or None,
        "projection_figure_svg": args.projection_figure,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    note = (
        f"多时相留出验证；可见屋顶{visible}栋；局部残差校正{accepted_local}栋\n"
        "GAMMA R-D为几何初值；先验高度不作为影像匹配目标或缺失值填充"
    )
    if args.figure:
        plot_figure(amplitude, corrected_roofs, Path(args.figure), note)
    plot_projection_figure(amplitude, refined, Path(args.projection_figure), visible)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
