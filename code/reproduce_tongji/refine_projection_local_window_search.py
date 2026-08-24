#!/usr/bin/env python3
"""Refine each building projection by a gated local SAR window search.

The latest GAMMA R-D/SAR-refined projection is the physical initial value.
For each visible roof, a small translation window is searched using the
20200708 full-precision RSLC boundary and local contrast. Persistent temporal
edges provide an independent acceptance gate. The formal output contains only
roof surfaces. Vector height sets only the adaptive window radius and is never
read as a score, target, result filter, or fill value.
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
from matplotlib.patches import Patch
from scipy import ndimage
from shapely import affinity
from shapely.geometry import box
from skimage.draw import polygon as draw_polygon

from extract_gamma_differential_island_observations import read_float

from refine_gamma_projection_with_sar_features import (
    feature_strength,
    load_temporal_split,
    sample_image,
    sampled_boundary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--projection",
        default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson",
    )
    p.add_argument("--amplitude", default="work/mli/20200708_rslc_amplitude.npy")
    p.add_argument("--bmp-dir", default="data/tongji_rslc")
    p.add_argument("--minimum-search-limit", type=int, default=3)
    p.add_argument("--maximum-search-limit", type=int, default=9)
    p.add_argument("--window-base", type=float, default=1.5)
    p.add_argument("--window-size-factor", type=float, default=0.045)
    p.add_argument("--window-height-factor", type=float, default=0.035)
    p.add_argument("--minimum-primary-gain", type=float, default=0.008)
    p.add_argument("--minimum-validation-gain", type=float, default=0.003)
    p.add_argument("--maximum-split-disagreement", type=int, default=2)
    p.add_argument("--pairs-csv", default=None)
    p.add_argument("--interferogram-dir", default=None)
    p.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--paper-quality-gate", action="store_true")
    p.add_argument(
        "--output",
        default="work/zjc_original_reproduction/20200708_all_building_adaptive_window_roof_projection_sar.geojson",
    )
    p.add_argument(
        "--metrics",
        default="work/zjc_original_reproduction/20200708_all_building_local_window_projection_metrics.csv",
    )
    p.add_argument(
        "--summary",
        default="results/metadata/tongji_projection_local_window_search_summary.json",
    )
    p.add_argument("--figure", default="picall/18_全部建筑高度辅助投影至雷达影像.svg")
    return p.parse_args()


def sar_display(amplitude: np.ndarray) -> np.ndarray:
    transformed = np.power(np.maximum(amplitude, 0.0), 0.70)
    valid = np.isfinite(transformed)
    lo, hi = np.nanpercentile(transformed[valid], [1.0, 99.4])
    return np.clip((transformed - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0).astype(np.float32)


def raster_mask(geometry, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for part in parts:
        if part.is_empty:
            continue
        xy = np.asarray(part.exterior.coords)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=shape)
        mask[rr, cc] = True
        for ring in part.interiors:
            hole = np.asarray(ring.coords)
            rr, cc = draw_polygon(hole[:, 1], hole[:, 0], shape=shape)
            mask[rr, cc] = False
    return mask


def raster_points(geometry, shape: tuple[int, int], maximum: int = 300) -> np.ndarray:
    mask = raster_mask(geometry, shape)
    rr, cc = np.nonzero(mask)
    if rr.size > maximum:
        take = np.linspace(0, rr.size - 1, maximum, dtype=int)
        rr, cc = rr[take], cc[take]
    return np.column_stack([cc, rr]).astype(np.float64)


def paper_quality_mask(args: argparse.Namespace, shape: tuple[int, int]) -> np.ndarray | None:
    if not args.paper_quality_gate:
        return None
    if not args.pairs_csv or not args.interferogram_dir:
        raise ValueError("--paper-quality-gate requires --pairs-csv and --interferogram-dir")
    pairs = pd.read_csv(args.pairs_csv)
    coherence_sum = np.zeros(shape, dtype=np.float64)
    valid_count = np.zeros(shape, dtype=np.int16)
    coherent_count = np.zeros(shape, dtype=np.int16)
    for row in pairs.itertuples(index=False):
        pair = f"{row.master}_{row.slave}"
        coherence = read_float(Path(args.interferogram_dir) / pair / f"{pair}.cc", shape[0], shape[1])
        finite = np.isfinite(coherence)
        coherence_sum += np.where(finite, coherence, 0.0)
        valid_count += finite.astype(np.int16)
        coherent_count += (finite & (coherence >= 0.55)).astype(np.int16)
    coherence_mean = np.divide(
        coherence_sum,
        valid_count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    da = np.load(args.amplitude_dispersion).astype(np.float64)
    return np.isfinite(da) & (da <= 0.40) & (coherence_mean >= 0.75) & (coherent_count >= 12)


def score_components(
    feature_map: np.ndarray,
    display: np.ndarray,
    boundary: np.ndarray,
    interior: np.ndarray,
    dr: int,
    dc: int,
) -> tuple[float, float, float]:
    shift = np.asarray([dc, dr], dtype=np.float64)
    edge = float(np.mean(sample_image(feature_map, boundary + shift)))
    if interior.size:
        inside = sample_image(display, interior + shift)
        # A translated annulus is approximated by four two-pixel offsets from
        # the same roof samples; this keeps the metric local and size-balanced.
        outside = np.concatenate(
            [
                sample_image(display, interior + shift + np.asarray(offset, dtype=float))
                for offset in ((-3, 0), (3, 0), (0, -3), (0, 3))
            ]
        )
        contrast = float(np.nanpercentile(inside, 65) - np.nanmedian(outside))
    else:
        contrast = 0.0
    combined = 0.82 * edge + 0.18 * np.clip(0.5 + contrast, 0.0, 1.0)
    return combined, edge, contrast


def choose_shift(
    reference_feature: np.ndarray,
    reference_display: np.ndarray,
    validation_feature: np.ndarray,
    geometry,
    shape: tuple[int, int],
    limit: int,
    minimum_primary_gain: float,
    minimum_validation_gain: float,
    maximum_disagreement: int,
) -> dict[str, object]:
    boundary = sampled_boundary(geometry, spacing=1.0)
    interior = raster_points(geometry, shape)
    candidates: list[dict[str, float | int]] = []
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            primary, edge, contrast = score_components(
                reference_feature, reference_display, boundary, interior, dr, dc
            )
            validation_edge = float(
                np.mean(sample_image(validation_feature, boundary + np.asarray([dc, dr], dtype=float)))
            )
            penalty = 0.00045 * (dr * dr + dc * dc)
            candidates.append(
                {
                    "dr": dr,
                    "dc": dc,
                    "primary": primary,
                    "edge": edge,
                    "contrast": contrast,
                    "validation": validation_edge,
                    "objective": 0.72 * primary + 0.28 * validation_edge - penalty,
                }
            )
    base = next(row for row in candidates if row["dr"] == 0 and row["dc"] == 0)
    best = max(candidates, key=lambda row: float(row["objective"]))
    validation_best = max(candidates, key=lambda row: float(row["validation"]) - 0.00045 * (row["dr"] ** 2 + row["dc"] ** 2))
    primary_gain = float(best["primary"] - base["primary"])
    validation_gain = float(best["validation"] - base["validation"])
    disagreement = max(abs(int(best["dr"]) - int(validation_best["dr"])), abs(int(best["dc"]) - int(validation_best["dc"])))
    boundary_hit = max(abs(int(best["dr"])), abs(int(best["dc"]))) == limit
    accepted = bool(
        len(boundary) >= 8
        and (int(best["dr"]) != 0 or int(best["dc"]) != 0)
        and primary_gain >= minimum_primary_gain
        and validation_gain >= minimum_validation_gain
        and disagreement <= maximum_disagreement
        and not boundary_hit
    )
    return {
        "local_window_row_shift": int(best["dr"] if accepted else 0),
        "local_window_col_shift": int(best["dc"] if accepted else 0),
        "local_window_accepted": accepted,
        "primary_score_before": float(base["primary"]),
        "primary_score_after": float(best["primary"] if accepted else base["primary"]),
        "validation_score_before": float(base["validation"]),
        "validation_score_after": float(best["validation"] if accepted else base["validation"]),
        "primary_gain_candidate": primary_gain,
        "validation_gain_candidate": validation_gain,
        "candidate_row_shift": int(best["dr"]),
        "candidate_col_shift": int(best["dc"]),
        "validation_best_row_shift": int(validation_best["dr"]),
        "validation_best_col_shift": int(validation_best["dc"]),
        "split_shift_disagreement": int(disagreement),
        "search_boundary_hit": boundary_hit,
        "reference_edge_before": float(base["edge"]),
        "reference_edge_after": float(best["edge"] if accepted else base["edge"]),
        "reference_contrast_before": float(base["contrast"]),
        "reference_contrast_after": float(best["contrast"] if accepted else base["contrast"]),
    }


def adaptive_search_limit(geometry, height_m: float, args: argparse.Namespace) -> tuple[int, float]:
    minx, miny, maxx, maxy = geometry.bounds
    maximum_dimension = max(float(maxx - minx), float(maxy - miny))
    raw = math.ceil(
        args.window_base
        + args.window_size_factor * maximum_dimension
        + args.window_height_factor * max(float(height_m), 0.0)
    )
    return int(np.clip(raw, args.minimum_search_limit, args.maximum_search_limit)), maximum_dimension


def draw_figure(
    amplitude: np.ndarray,
    projection: gpd.GeoDataFrame,
    output: Path,
    accepted_count: int,
    visible_count: int,
    minimum_limit: int,
    maximum_limit: int,
) -> None:
    roofs = projection[projection["surface"].eq("roof")]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.imshow(sar_display(amplitude), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    roofs.boundary.plot(ax=ax, color="#00D5E8", linewidth=0.42, alpha=0.95, zorder=4)
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("距离向像元")
    ax.set_ylabel("方位向像元")
    ax.set_title("自适应局部窗口校正后的屋顶投影", pad=7, fontweight="bold")
    ax.text(
        0.012,
        0.985,
        f"可见屋顶：{visible_count}栋\n局部窗口调整：{accepted_count}栋（±{minimum_limit}–{maximum_limit}像元）",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "#111111", "edgecolor": "#C8CDD2", "alpha": 0.80, "pad": 2.3},
    )
    ax.legend(
        handles=[Patch(facecolor="none", edgecolor="#00D5E8", label="自适应窗口校正屋顶轮廓")],
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        fontsize=7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "自适应局部窗口校正后的屋顶投影"})
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
    reference_display = sar_display(amplitude)
    reference_feature = feature_strength(amplitude)
    _train_amp, validation_amp, _train_names, validation_names = load_temporal_split(Path(args.bmp_dir), amplitude.shape)
    # The frozen validation half excludes the 20200708 reference acquisition,
    # so a shift found on the master scene cannot validate itself.
    persistent_validation = feature_strength(validation_amp)
    phase_quality = paper_quality_mask(args, amplitude.shape)

    projection = gpd.read_file(args.projection).set_crs(None, allow_override=True)
    refined = projection.copy()
    roofs = projection[projection["surface"].eq("roof")]
    footprint = box(0, 0, amplitude.shape[1] - 1, amplitude.shape[0] - 1)
    metrics: list[dict[str, object]] = []
    visible_count = 0
    accepted_count = 0
    for ordinal, row in enumerate(roofs.itertuples(index=False), start=1):
        clean_id = int(row.clean_id)
        if row.geometry is None or row.geometry.is_empty or not row.geometry.intersects(footprint):
            result = {"local_window_row_shift": 0, "local_window_col_shift": 0, "local_window_accepted": False, "outside_scene": True}
        else:
            visible_count += 1
            height_for_window = float(getattr(row, "height_prior_m", 0.0))
            search_limit, maximum_dimension = adaptive_search_limit(row.geometry, height_for_window, args)
            result = choose_shift(
                reference_feature,
                reference_display,
                persistent_validation,
                row.geometry,
                amplitude.shape,
                search_limit,
                args.minimum_primary_gain,
                args.minimum_validation_gain,
                args.maximum_split_disagreement,
            )
            candidate_geometry = affinity.translate(
                row.geometry,
                xoff=int(result["candidate_col_shift"]),
                yoff=int(result["candidate_row_shift"]),
            )
            if phase_quality is not None:
                before_core = ndimage.binary_erosion(raster_mask(row.geometry, amplitude.shape), iterations=1)
                after_core = ndimage.binary_erosion(raster_mask(candidate_geometry, amplitude.shape), iterations=1)
                quality_before = int((before_core & phase_quality).sum())
                quality_after = int((after_core & phase_quality).sum())
                phase_gate_passed = quality_after >= quality_before
                result["paper_quality_points_before"] = quality_before
                result["paper_quality_points_after"] = quality_after
                result["paper_quality_point_gain"] = quality_after - quality_before
                result["paper_quality_gate_passed"] = phase_gate_passed
                if bool(result["local_window_accepted"]) and not phase_gate_passed:
                    result["local_window_row_shift"] = 0
                    result["local_window_col_shift"] = 0
                    result["local_window_accepted"] = False
            result["search_limit_pixels"] = search_limit
            result["roof_maximum_dimension_pixels"] = maximum_dimension
            result["height_for_window_m"] = height_for_window
            result["height_used_only_for_window_size"] = True
            result["outside_scene"] = False
        dr = int(result["local_window_row_shift"])
        dc = int(result["local_window_col_shift"])
        if bool(result["local_window_accepted"]):
            accepted_count += 1
            same = refined["clean_id"].astype(int).eq(clean_id)
            refined.loc[same, "geometry"] = refined.loc[same, "geometry"].map(
                lambda geom: affinity.translate(geom, xoff=dc, yoff=dr)
            )
        same = refined["clean_id"].astype(int).eq(clean_id)
        refined.loc[same, "local_window_row_shift"] = dr
        refined.loc[same, "local_window_col_shift"] = dc
        refined.loc[same, "local_window_accepted"] = np.int8(bool(result["local_window_accepted"]))
        metrics.append({"clean_id": clean_id, **result})
        if ordinal % 200 == 0 or ordinal == len(roofs):
            print(f"逐栋局部窗口搜索：{ordinal}/{len(roofs)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    roof_output = refined[refined["surface"].eq("roof")].copy()
    roof_output.to_file(output, driver="GeoJSON")
    metrics_table = pd.DataFrame(metrics)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(metrics_path, index=False)
    draw_figure(
        amplitude,
        roof_output,
        Path(args.figure),
        accepted_count,
        visible_count,
        args.minimum_search_limit,
        args.maximum_search_limit,
    )

    accepted = metrics_table[metrics_table["local_window_accepted"].eq(True)]
    summary = {
        "method": "per-building adaptive local translation window on top of GAMMA R-D + prior SAR refinement; window size from projected roof size and height; 20200708 RSLC boundary/contrast primary score; persistent temporal edge validation gate",
        "input_projection": args.projection,
        "output_projection": args.output,
        "height_used_as_matching_target_filter_or_fill": False,
        "height_used_only_to_set_search_window": True,
        "output_surface": "roof_only",
        "reference_amplitude": args.amplitude,
        "validation_images": validation_names,
        "adaptive_window_formula": "ceil(base + size_factor * roof_max_dimension_px + height_factor * height_m), clipped to limits",
        "window_base": args.window_base,
        "window_size_factor": args.window_size_factor,
        "window_height_factor": args.window_height_factor,
        "minimum_search_limit_pixels": args.minimum_search_limit,
        "maximum_search_limit_pixels": args.maximum_search_limit,
        "minimum_primary_gain": args.minimum_primary_gain,
        "minimum_validation_gain": args.minimum_validation_gain,
        "maximum_split_shift_disagreement_pixels": args.maximum_split_disagreement,
        "paper_quality_gate": bool(args.paper_quality_gate),
        "paper_quality_definition": (
            "DA<=0.40; all-pair mean coherence>=0.75; at least 12 pairs coherence>=0.55; accepted shift may not reduce count"
            if args.paper_quality_gate
            else None
        ),
        "visible_roofs": visible_count,
        "local_window_corrections_accepted": accepted_count,
        "unchanged_or_outside": int(len(roofs) - accepted_count),
        "mean_primary_score_gain_accepted": float((accepted["primary_score_after"] - accepted["primary_score_before"]).mean()) if len(accepted) else 0.0,
        "mean_validation_score_gain_accepted": float((accepted["validation_score_after"] - accepted["validation_score_before"]).mean()) if len(accepted) else 0.0,
        "median_accepted_shift_pixels": float(np.median(np.hypot(accepted["local_window_row_shift"], accepted["local_window_col_shift"]))) if len(accepted) else 0.0,
        "search_limit_distribution_visible": metrics_table.loc[metrics_table["outside_scene"].eq(False), "search_limit_pixels"].value_counts().sort_index().astype(int).to_dict(),
        "metrics_csv": args.metrics,
        "figure_svg": args.figure,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
