#!/usr/bin/env python3
"""Pixel-wise LGR DEM-error inversion inside each island, then island height."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import t as student_t
from skimage.restoration import unwrap_phase

from benchmark_paper_unwrap import paper_like_unwrap
from extract_gamma_differential_island_observations import read_fcomplex, read_float
from inventory_data import parse_gamma_par


def design(bperp: np.ndarray, dt_days: np.ndarray, wavelength: float, range_m: float, inc_deg: float) -> np.ndarray:
    return np.column_stack(
        [
            4.0 * math.pi * bperp / (wavelength * range_m * math.sin(math.radians(inc_deg))),
            4.0 * math.pi * dt_days / wavelength,
        ]
    )


def solve_pixels(
    phases: np.ndarray,
    cohs: np.ndarray,
    a: np.ndarray,
    bperp: np.ndarray,
    min_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pairs, n_pix = phases.shape
    dem = np.full(n_pix, np.nan, dtype=np.float32)
    rate = np.full(n_pix, np.nan, dtype=np.float32)
    rmse = np.full(n_pix, np.nan, dtype=np.float32)
    n_valid = np.zeros(n_pix, dtype=np.int16)
    bperp_span = np.full(n_pix, np.nan, dtype=np.float32)
    for j in range(n_pix):
        valid = np.isfinite(phases[:, j]) & np.isfinite(cohs[:, j]) & (cohs[:, j] > 0)
        n_valid[j] = int(np.sum(valid))
        if n_valid[j] < min_pairs:
            continue
        w = np.clip(cohs[valid, j], 0.05, 1.0)
        aw = a[valid] * w[:, None]
        yw = phases[valid, j] * w
        coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        pred = a[valid] @ coef
        dem[j] = coef[0]
        rate[j] = coef[1] * 365.0
        rmse[j] = float(np.sqrt(np.mean((phases[valid, j] - pred) ** 2)))
        bperp_valid = bperp[valid]
        bperp_span[j] = float(np.nanmax(bperp_valid) - np.nanmin(bperp_valid))
    return dem, rate, rmse, n_valid, bperp_span


def solve_wrapped_pixels_multistart(
    phases: np.ndarray,
    cohs: np.ndarray,
    a: np.ndarray,
    bperp: np.ndarray,
    min_pairs: int,
    dem_seed_min_m: float,
    dem_seed_max_m: float,
    dem_seed_step_m: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve wrapped SBAS observations without an island-local zero-height gauge."""
    n_pairs, n_pix = phases.shape
    dem = np.full(n_pix, np.nan, dtype=np.float32)
    rate = np.full(n_pix, np.nan, dtype=np.float32)
    rmse = np.full(n_pix, np.nan, dtype=np.float32)
    n_valid = np.zeros(n_pix, dtype=np.int16)
    bperp_span = np.full(n_pix, np.nan, dtype=np.float32)
    seeds = np.arange(dem_seed_min_m, dem_seed_max_m + 0.5 * dem_seed_step_m, dem_seed_step_m)
    for j in range(n_pix):
        valid = np.isfinite(phases[:, j]) & np.isfinite(cohs[:, j]) & (cohs[:, j] > 0)
        n_valid[j] = int(valid.sum())
        if n_valid[j] < min_pairs:
            continue
        av = a[valid]
        wrapped = phases[valid, j].astype(np.float64)
        weights = np.clip(cohs[valid, j].astype(np.float64), 0.05, 1.0) ** 2
        aw = av * np.sqrt(weights)[:, None]
        best: tuple[float, float, np.ndarray] | None = None
        for seed in seeds:
            coef = np.array([seed, 0.0], dtype=np.float64)
            unwrapped = wrapped.copy()
            for _ in range(max(1, iterations)):
                pred = av @ coef
                ambiguity = np.rint((pred - wrapped) / (2.0 * np.pi))
                unwrapped = wrapped + 2.0 * np.pi * ambiguity
                coef_new, *_ = np.linalg.lstsq(aw, unwrapped * np.sqrt(weights), rcond=None)
                if np.max(np.abs(coef_new - coef)) < 1e-7:
                    coef = coef_new
                    break
                coef = coef_new
            residual = unwrapped - av @ coef
            score = float(np.sqrt(np.average(residual**2, weights=weights)))
            candidate = (score, abs(float(coef[0])), coef)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            continue
        score, _, coef = best
        dem[j] = float(coef[0])
        rate[j] = float(coef[1] * 365.0)
        rmse[j] = score
        bp = bperp[valid]
        bperp_span[j] = float(np.nanmax(bp) - np.nanmin(bp))
    return dem, rate, rmse, n_valid, bperp_span


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    vals = values[valid]
    wts = weights[valid]
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cumulative = np.cumsum(wts) - 0.5 * wts
    cumulative /= np.sum(wts)
    return float(np.interp(quantile, cumulative, vals))


def plot_qc(df: pd.DataFrame, label: np.ndarray, islands_info: pd.DataFrame, out: Path | None = None, out_svg: Path | None = None) -> None:
    status = np.zeros_like(label, dtype=np.uint8)
    island_ids = set(islands_info["island_id"].dropna().astype(int).tolist())
    processed_ids = set(df["island_id"].dropna().astype(int).tolist()) if not df.empty else set()
    solved_ids = set(df.loc[df["height_m"].notna(), "island_id"].astype(int).tolist()) if not df.empty else set()
    for island_id in island_ids:
        status[label == island_id] = 1
    for island_id in processed_ids:
        status[label == island_id] = 2
    for island_id in solved_ids:
        status[label == island_id] = 3

    height_raster = np.full(label.shape, np.nan, dtype=np.float32)
    if not df.empty:
        height_map = df.dropna(subset=["height_m"]).set_index("island_id")["height_m"].to_dict()
        for island_id, height in height_map.items():
            height_raster[label == int(island_id)] = float(height)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=220)
    cmap = ListedColormap(["#ffffff", "#d9d9d9", "#fdbf6f", "#1b9e77"])
    axes[0].imshow(status, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[0].set_title("Island processing status")
    axes[0].set_axis_off()
    axes[0].text(
        0.02,
        0.02,
        "gray: roof island\norange: processed\nteal: solved",
        transform=axes[0].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )

    im = axes[1].imshow(height_raster, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Solved island height map")
    axes[1].set_axis_off()
    fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02, label="height (m)")

    metrics = [
        ("roof islands", len(island_ids)),
        ("processed", len(processed_ids)),
        ("solved", len(solved_ids)),
        ("height points", int(df["pixel_count_used"].notna().sum()) if not df.empty else 0),
    ]
    axes[2].set_axis_off()
    axes[2].set_title("Step accounting")
    y = 0.84
    for name, value in metrics:
        axes[2].text(0.08, y, name, fontsize=10, color="#333333", transform=axes[2].transAxes)
        axes[2].text(0.78, y, f"{value}", fontsize=13, weight="bold", ha="right", transform=axes[2].transAxes)
        axes[2].add_patch(plt.Rectangle((0.06, y - 0.035), 0.78, 0.085, fill=False, edgecolor="#c8cdd2", linewidth=0.9, transform=axes[2].transAxes))
        y -= 0.16
    axes[2].text(
        0.06,
        0.08,
        "Visualization avoids scatter/line statistics; it tracks spatial processing state and solved roof islands.",
        fontsize=8,
        color="#444444",
        transform=axes[2].transAxes,
        wrap=True,
    )
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
    plt.close(fig)


def iqr_median(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    keep = clean[(clean >= q1 - 1.5 * iqr) & (clean <= q3 + 1.5 * iqr)]
    if keep.size == 0:
        keep = clean
    return float(np.median(keep))


def grubbs_max_test(values: np.ndarray, alpha: float = 0.05) -> dict[str, float | bool | int]:
    clean = values[np.isfinite(values)]
    n = int(clean.size)
    if n < 3:
        return {
            "max_height_grubbs_n": n,
            "max_height_grubbs_g": np.nan,
            "max_height_grubbs_gcrit": np.nan,
            "max_height_grubbs_p": np.nan,
            "max_height_reject_outlier": False,
            "max_height_reliable": n > 0,
        }
    std = float(np.std(clean, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return {
            "max_height_grubbs_n": n,
            "max_height_grubbs_g": 0.0,
            "max_height_grubbs_gcrit": np.nan,
            "max_height_grubbs_p": 1.0,
            "max_height_reject_outlier": False,
            "max_height_reliable": True,
        }
    g = float((np.max(clean) - np.mean(clean)) / std)
    tcrit = float(student_t.ppf(1.0 - alpha / n, n - 2))
    gcrit = float((n - 1) / math.sqrt(n) * math.sqrt(tcrit**2 / (n - 2 + tcrit**2)))
    denom = (n - 1) ** 2 - n * g**2
    if denom > 0:
        tval = math.sqrt((n * (n - 2) * g**2) / denom)
        pval = min(1.0, float(n * (1.0 - student_t.cdf(tval, n - 2))))
    else:
        pval = 0.0
    reject = bool(g > gcrit)
    return {
        "max_height_grubbs_n": n,
        "max_height_grubbs_g": g,
        "max_height_grubbs_gcrit": gcrit,
        "max_height_grubbs_p": pval,
        "max_height_reject_outlier": reject,
        "max_height_reliable": not reject,
    }


def iterative_grubbs_top(values: np.ndarray, alpha: float = 0.05) -> dict[str, float | int | bool]:
    clean = values[np.isfinite(values)].astype(np.float64)
    removed = 0
    last = grubbs_max_test(clean, alpha=alpha)
    while clean.size >= 3:
        last = grubbs_max_test(clean, alpha=alpha)
        if not bool(last["max_height_reject_outlier"]):
            break
        clean = np.delete(clean, int(np.argmax(clean)))
        removed += 1
    if clean.size == 0:
        top = np.nan
    else:
        top = float(np.max(clean))
    return {
        "building_height_grubbs_top_m": top,
        "grubbs_top_removed_count": int(removed),
        "grubbs_top_remaining_count": int(clean.size),
        "grubbs_top_last_g": float(last.get("max_height_grubbs_g", np.nan)),
        "grubbs_top_last_gcrit": float(last.get("max_height_grubbs_gcrit", np.nan)),
        "grubbs_top_last_p": float(last.get("max_height_grubbs_p", np.nan)),
        "grubbs_top_reliable": bool(clean.size > 0),
    }


def summarize_height(
    dem_error_values: np.ndarray,
    reference_values: np.ndarray,
    ground_dem_m: float,
    residual_sign: float,
    min_pixels: int,
    grubbs_alpha: float,
    quality_weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    dem_error = residual_sign * dem_error_values
    valid = np.isfinite(dem_error) & np.isfinite(reference_values) & (reference_values > -1000.0)
    if quality_weights is None:
        quality_weights = np.ones_like(dem_error, dtype=np.float64)
    else:
        quality_weights = np.asarray(quality_weights, dtype=np.float64)
        valid &= np.isfinite(quality_weights) & (quality_weights > 0)
    valid_count = int(np.sum(valid))
    dem_error = dem_error[valid]
    reference = reference_values[valid]
    weights = quality_weights[valid]
    if valid_count < min_pixels:
        return {
            "pixel_count_used": valid_count,
            "height_m": np.nan,
            "building_height_weighted_median_m": np.nan,
            "effective_pixel_count": 0.0,
            "dem_error_p05_m": np.nan,
            "dem_error_median_m": np.nan,
            "dem_error_p95_m": np.nan,
            "reference_elevation_median_m": np.nan,
            "roof_elevation_p05_m": np.nan,
            "roof_elevation_median_m": np.nan,
            "roof_elevation_p75_m": np.nan,
            "roof_elevation_p85_m": np.nan,
            "roof_elevation_p90_m": np.nan,
            "roof_elevation_p95_m": np.nan,
            "building_height_p05_m": np.nan,
            "building_height_p25_m": np.nan,
            "building_height_p50_m": np.nan,
            "building_height_p75_m": np.nan,
            "building_height_p85_m": np.nan,
            "building_height_p90_m": np.nan,
            "building_height_p95_m": np.nan,
            "building_height_max_m": np.nan,
            "max_height_grubbs_n": valid_count,
            "max_height_grubbs_g": np.nan,
            "max_height_grubbs_gcrit": np.nan,
            "max_height_grubbs_p": np.nan,
            "max_height_reject_outlier": False,
            "max_height_reliable": False,
            "building_height_grubbs_top_m": np.nan,
            "grubbs_top_removed_count": 0,
            "grubbs_top_remaining_count": 0,
            "grubbs_top_last_g": np.nan,
            "grubbs_top_last_gcrit": np.nan,
            "grubbs_top_last_p": np.nan,
            "grubbs_top_reliable": False,
        }
    roof_elevation = reference + dem_error
    building_height = roof_elevation - ground_dem_m
    weighted_height = weighted_quantile(building_height, weights, 0.5)
    effective_n = float(np.sum(weights) ** 2 / np.sum(weights**2)) if np.sum(weights**2) > 0 else 0.0
    max_test = grubbs_max_test(building_height, alpha=grubbs_alpha)
    top_test = iterative_grubbs_top(building_height, alpha=grubbs_alpha)
    dem_p05, dem_med, dem_p95 = np.percentile(dem_error, [5, 50, 95])
    roof_p05, roof_med, roof_p75, roof_p85, roof_p90, roof_p95 = np.percentile(roof_elevation, [5, 50, 75, 85, 90, 95])
    h_p05, h_p25, h_p50, h_p75, h_p85, h_p90, h_p95 = np.percentile(building_height, [5, 25, 50, 75, 85, 90, 95])
    return {
        "pixel_count_used": valid_count,
        "height_m": weighted_height,
        "building_height_weighted_median_m": weighted_height,
        "effective_pixel_count": effective_n,
        "dem_error_p05_m": float(dem_p05),
        "dem_error_median_m": float(dem_med),
        "dem_error_p95_m": float(dem_p95),
        "reference_elevation_median_m": float(np.median(reference)),
        "roof_elevation_p05_m": float(roof_p05),
        "roof_elevation_median_m": float(roof_med),
        "roof_elevation_p75_m": float(roof_p75),
        "roof_elevation_p85_m": float(roof_p85),
        "roof_elevation_p90_m": float(roof_p90),
        "roof_elevation_p95_m": float(roof_p95),
        "building_height_p05_m": float(h_p05),
        "building_height_p25_m": float(h_p25),
        "building_height_p50_m": float(h_p50),
        "building_height_p75_m": float(h_p75),
        "building_height_p85_m": float(h_p85),
        "building_height_p90_m": float(h_p90),
        "building_height_p95_m": float(h_p95),
        "building_height_max_m": float(np.max(building_height)),
        **max_test,
        **top_test,
    }


def unwrap_patch(
    phase_patch: np.ndarray,
    valid: np.ndarray,
    method: str,
    wavelength_m: float,
    coherence_patch: np.ndarray | None = None,
    amplitude_dispersion_patch: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if method == "paper":
        unw, info = paper_like_unwrap(
            phase_patch,
            valid,
            wavelength_m,
            coherence_patch=coherence_patch,
            amplitude_dispersion_patch=amplitude_dispersion_patch,
        )
        return unw.astype(np.float32), str(info.get("status", "unknown"))
    unw = unwrap_phase(np.ma.array(phase_patch, mask=~valid)).filled(np.nan)
    return unw.astype(np.float32), "skimage"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--intf-root", default="work/gamma_sbas/intf")
    parser.add_argument("--island-label", default="work/masks/island_label_touying_blue_bottom.npy")
    parser.add_argument("--fid-mask", default="work/masks/building_fid_mask_touying_blue_bottom.npy")
    parser.add_argument("--islands-csv", default="work/masks/islands_touying_blue_bottom.csv")
    parser.add_argument("--fid-uid-map", default="results/tables/touying_fid_uid_map.csv")
    parser.add_argument("--reference-height-rdc", default="work/gamma_sbas/dem/20200708_dsm_rdc.hgt")
    parser.add_argument("--ground-dem-m", type=float, default=4.0)
    parser.add_argument("--residual-sign", type=float, default=-1.0)
    parser.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--min-coherence", type=float, default=0.2)
    parser.add_argument("--amplitude-dispersion-npy", default="", help="Optional amplitude-dispersion raster; pixels above --max-amplitude-dispersion are excluded before unwrapping.")
    parser.add_argument("--max-amplitude-dispersion", type=float, default=0.0, help="Use only pixels with amplitude dispersion <= this value; 0 disables DA screening.")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--max-pixel-rmse-rad", type=float, default=1.25, help="Reject pixels whose LGR phase-model RMSE exceeds this value.")
    parser.add_argument("--min-bperp-span-m", type=float, default=80.0, help="Reject pixels whose valid observations do not span this perpendicular-baseline range.")
    parser.add_argument("--min-physical-height-m", type=float, default=-1.0, help="Reject pixel ambiguity branches below this building height; set a very low value to disable.")
    parser.add_argument("--max-physical-height-m", type=float, default=120.0, help="Reject pixel ambiguity branches above this building height; this is an InSAR physical QC bound, not a fill value.")
    parser.add_argument("--grubbs-alpha", type=float, default=0.05, help="Significance level for one-sided iterative Grubbs high-outlier rejection.")
    parser.add_argument("--unwrap-method", choices=["skimage", "paper", "temporal_multistart"], default="skimage")
    parser.add_argument("--stable-reference-mask", default="", help="Optional stable-ground boolean NPY. Its per-pair circular phase center is removed before roof inversion.")
    parser.add_argument("--temporal-dem-seed-min-m", type=float, default=-80.0)
    parser.add_argument("--temporal-dem-seed-max-m", type=float, default=80.0)
    parser.add_argument("--temporal-dem-seed-step-m", type=float, default=5.0)
    parser.add_argument("--temporal-unwrap-iterations", type=int, default=8)
    parser.add_argument("--max-islands", type=int, default=0, help="Process only the first N islands after sorting; 0 means all.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Process only the first N interferometric pairs; 0 means all.")
    parser.add_argument("--island-id-file", default="", help="Optional CSV containing island_id values to process.")
    parser.add_argument("--output-islands", default="work/height/island_pixel_lgr_heights.csv")
    parser.add_argument("--output-points", default="work/height/height_points.csv")
    parser.add_argument("--summary", default="results/metadata/pixel_lgr_building_heights_summary.json")
    parser.add_argument("--figure", default="results/pic_all/20_pixel_lgr_building_heights.png")
    parser.add_argument("--figure-svg", default="")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    if args.max_pairs > 0:
        pairs = pairs.head(args.max_pairs).copy()
    islands_info = pd.read_csv(args.islands_csv)
    if args.island_id_file:
        island_filter = pd.read_csv(args.island_id_file)
        if "island_id" not in island_filter.columns:
            raise ValueError("--island-id-file must contain an island_id column")
        keep_ids = set(island_filter["island_id"].dropna().astype(int).tolist())
        islands_info = islands_info[islands_info["island_id"].astype(int).isin(keep_ids)].copy()
    if args.max_islands > 0:
        islands_info = islands_info.sort_values("pixel_count", ascending=False).head(args.max_islands).copy()
    fid_map = pd.read_csv(args.fid_uid_map)
    fid_to_uid = dict(zip(fid_map["touying_fid"].astype(int), fid_map["uid"].astype(int)))
    label = np.load(args.island_label)
    fid_mask = np.load(args.fid_mask)
    amp_disp = None
    if args.amplitude_dispersion_npy:
        amp_disp = np.load(args.amplitude_dispersion_npy).astype(np.float32)
        if amp_disp.shape != label.shape:
            raise ValueError(f"amplitude dispersion shape {amp_disp.shape} does not match island label shape {label.shape}")
    reference_height = read_float(Path(args.reference_height_rdc), args.rows, args.cols)
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    a = design(
        pairs["bperp_m"].to_numpy(dtype=np.float64),
        pairs["dt_days"].to_numpy(dtype=np.float64),
        wavelength,
        float(par["center_range_slc"]),
        float(par["incidence_angle"]),
    )
    phase_stack = []
    coh_stack = []
    for row in pairs.itertuples(index=False):
        pair = f"{row.master}_{row.slave}"
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        phase_stack.append(np.angle(diff).astype(np.float32))
        coh_stack.append(cc.astype(np.float32))
    phase_stack = np.stack(phase_stack, axis=0)
    coh_stack = np.stack(coh_stack, axis=0)
    reference_offsets = np.zeros(len(pairs), dtype=np.float64)
    reference_pixel_counts = np.zeros(len(pairs), dtype=np.int32)
    if args.stable_reference_mask:
        reference_mask = np.load(args.stable_reference_mask).astype(bool)
        if reference_mask.shape != label.shape:
            raise ValueError(f"stable reference shape {reference_mask.shape} does not match label shape {label.shape}")
        for k in range(len(pairs)):
            valid_ref = reference_mask & np.isfinite(phase_stack[k]) & np.isfinite(coh_stack[k]) & (coh_stack[k] >= args.min_coherence)
            reference_pixel_counts[k] = int(valid_ref.sum())
            if reference_pixel_counts[k] < 20:
                raise ValueError(f"pair {k} has only {reference_pixel_counts[k]} stable reference pixels")
            z = np.sum(coh_stack[k][valid_ref].astype(np.float64) * np.exp(1j * phase_stack[k][valid_ref].astype(np.float64)))
            reference_offsets[k] = float(np.angle(z))
            phase_stack[k] = np.angle(np.exp(1j * (phase_stack[k] - reference_offsets[k]))).astype(np.float32)

    rows_out = []
    points = []
    for island_index, info in enumerate(islands_info.itertuples(index=False), start=1):
        island_id = int(info.island_id)
        if args.unwrap_method == "paper" and (island_index == 1 or island_index % 10 == 0 or island_index == len(islands_info)):
            print(f"processing island {island_index}/{len(islands_info)} id={island_id}", flush=True)
        keep = label == island_id
        if int(np.sum(keep)) < args.min_pixels:
            continue
        rr, cc_idx = np.nonzero(keep)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc_idx.min()), int(cc_idx.max()) + 1
        patch_keep = keep[r0:r1, c0:c1]
        if amp_disp is not None and args.max_amplitude_dispersion > 0:
            da_patch = amp_disp[r0:r1, c0:c1]
            patch_keep = patch_keep & np.isfinite(da_patch) & (da_patch <= args.max_amplitude_dispersion)
        if int(np.sum(patch_keep)) < args.min_pixels:
            continue
        pix = np.nonzero(patch_keep.ravel())[0]
        n_pix = len(pix)
        phases = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
        cohs = np.full((len(pairs), n_pix), np.nan, dtype=np.float32)
        for k in range(len(pairs)):
            p = phase_stack[k, r0:r1, c0:c1]
            c = coh_stack[k, r0:r1, c0:c1]
            valid = patch_keep & np.isfinite(p) & np.isfinite(c) & (c >= args.min_coherence)
            if int(np.sum(valid)) < args.min_pixels:
                continue
            valid_vec = valid.ravel()[pix]
            coh_vec = c.ravel()[pix]
            if args.unwrap_method == "temporal_multistart":
                p_vec = p.ravel()[pix]
                phases[k, valid_vec] = p_vec[valid_vec]
                cohs[k, valid_vec] = coh_vec[valid_vec]
            else:
                try:
                    unw, _unwrap_status = unwrap_patch(
                        p,
                        valid,
                        args.unwrap_method,
                        wavelength,
                        coherence_patch=c,
                        amplitude_dispersion_patch=(da_patch if amp_disp is not None else None),
                    )
                except Exception:
                    continue
                unw_vec = unw.ravel()[pix]
                phases[k, valid_vec] = unw_vec[valid_vec]
                cohs[k, valid_vec] = coh_vec[valid_vec]
        if args.unwrap_method == "temporal_multistart":
            dem, rate, pixel_rmse, n_valid, bperp_span = solve_wrapped_pixels_multistart(
                phases,
                cohs,
                a,
                pairs["bperp_m"].to_numpy(dtype=np.float64),
                args.min_pairs,
                args.temporal_dem_seed_min_m,
                args.temporal_dem_seed_max_m,
                args.temporal_dem_seed_step_m,
                args.temporal_unwrap_iterations,
            )
        else:
            dem, rate, pixel_rmse, n_valid, bperp_span = solve_pixels(
                phases,
                cohs,
                a,
                pairs["bperp_m"].to_numpy(dtype=np.float64),
                args.min_pairs,
            )
        reference_vec = reference_height[r0:r1, c0:c1].ravel()[pix]
        physical_height = reference_vec + args.residual_sign * dem - args.ground_dem_m
        reject_pixel = (
            (pixel_rmse > args.max_pixel_rmse_rad)
            | (n_valid < args.min_pairs)
            | (bperp_span < args.min_bperp_span_m)
            | ~np.isfinite(physical_height)
            | (physical_height < args.min_physical_height_m)
            | (physical_height > args.max_physical_height_m)
        )
        dem[reject_pixel] = np.nan
        rate[~np.isfinite(dem)] = np.nan
        pixel_rmse[~np.isfinite(dem)] = np.nan
        bperp_span[~np.isfinite(dem)] = np.nan
        n_valid[~np.isfinite(dem)] = 0
        amp_disp_vec = amp_disp[r0:r1, c0:c1].ravel()[pix] if amp_disp is not None else None
        coherence_quality = np.full(n_pix, np.nan, dtype=np.float32)
        coherence_columns = np.any(np.isfinite(cohs), axis=0)
        coherence_quality[coherence_columns] = np.nanmedian(cohs[:, coherence_columns], axis=0)
        da_quality = amp_disp_vec if amp_disp_vec is not None else np.zeros(n_pix, dtype=np.float32)
        quality_weights = (
            np.clip(coherence_quality, 0.0, 1.0) ** 2
            * np.clip(1.0 - da_quality, 0.05, 1.0) ** 2
            * np.clip(n_valid / max(len(pairs), 1), 0.0, 1.0)
            / np.maximum(pixel_rmse, 0.20)
        )
        quality_weights[~np.isfinite(dem)] = np.nan
        island_height = summarize_height(dem, reference_vec, args.ground_dem_m, args.residual_sign, args.min_pixels, args.grubbs_alpha, quality_weights)
        primary_fid = int(info.primary_uid)
        uid = fid_to_uid.get(primary_fid)
        row_out = {
            "island_id": island_id,
            "primary_touying_fid": primary_fid,
            "uid": uid,
            "uid_count": int(info.uid_count),
            "pixel_count": int(info.pixel_count),
            "pixel_count_used": island_height["pixel_count_used"],
            "height_m": island_height["height_m"],
            "building_height_weighted_median_m": island_height["building_height_weighted_median_m"],
            "effective_pixel_count": island_height["effective_pixel_count"],
            "dem_error_p05_m": island_height["dem_error_p05_m"],
            "dem_error_median_m": island_height["dem_error_median_m"],
            "dem_error_p95_m": island_height["dem_error_p95_m"],
            "reference_elevation_median_m": island_height["reference_elevation_median_m"],
            "roof_elevation_p05_m": island_height["roof_elevation_p05_m"],
            "roof_elevation_median_m": island_height["roof_elevation_median_m"],
            "roof_elevation_p75_m": island_height["roof_elevation_p75_m"],
            "roof_elevation_p85_m": island_height["roof_elevation_p85_m"],
            "roof_elevation_p90_m": island_height["roof_elevation_p90_m"],
            "roof_elevation_p95_m": island_height["roof_elevation_p95_m"],
            "building_height_p05_m": island_height["building_height_p05_m"],
            "building_height_p25_m": island_height["building_height_p25_m"],
            "building_height_p50_m": island_height["building_height_p50_m"],
            "building_height_p75_m": island_height["building_height_p75_m"],
            "building_height_p85_m": island_height["building_height_p85_m"],
            "building_height_p90_m": island_height["building_height_p90_m"],
            "building_height_p95_m": island_height["building_height_p95_m"],
            "building_height_max_m": island_height["building_height_max_m"],
            "max_height_grubbs_n": island_height["max_height_grubbs_n"],
            "max_height_grubbs_g": island_height["max_height_grubbs_g"],
            "max_height_grubbs_gcrit": island_height["max_height_grubbs_gcrit"],
            "max_height_grubbs_p": island_height["max_height_grubbs_p"],
            "max_height_reject_outlier": island_height["max_height_reject_outlier"],
            "max_height_reliable": island_height["max_height_reliable"],
            "building_height_grubbs_top_m": island_height["building_height_grubbs_top_m"],
            "grubbs_top_removed_count": island_height["grubbs_top_removed_count"],
            "grubbs_top_remaining_count": island_height["grubbs_top_remaining_count"],
            "grubbs_top_last_g": island_height["grubbs_top_last_g"],
            "grubbs_top_last_gcrit": island_height["grubbs_top_last_gcrit"],
            "grubbs_top_last_p": island_height["grubbs_top_last_p"],
            "grubbs_top_reliable": island_height["grubbs_top_reliable"],
            "median_coherence": float(np.nanmedian(cohs)) if np.isfinite(cohs).any() else np.nan,
            "median_amplitude_dispersion": float(np.nanmedian(amp_disp_vec)) if amp_disp_vec is not None and np.isfinite(amp_disp_vec).any() else np.nan,
            "median_pixel_lgr_rmse_rad": float(np.nanmedian(pixel_rmse)) if np.isfinite(pixel_rmse).any() else np.nan,
            "median_valid_pairs": float(np.nanmedian(n_valid[n_valid > 0])) if np.any(n_valid > 0) else np.nan,
            "median_bperp_span_m": float(np.nanmedian(bperp_span)) if np.isfinite(bperp_span).any() else np.nan,
            "method": "gamma_pixel_lgr_residual_plus_dsm_minus_ground",
        }
        rows_out.append(row_out)
        fid_patch = fid_mask[r0:r1, c0:c1].ravel()[pix].astype(np.int64)
        for fid in sorted(int(v) for v in np.unique(fid_patch) if int(v) > 0):
            building_uid = fid_to_uid.get(fid)
            if building_uid is None:
                continue
            building_keep = fid_patch == fid
            building_summary = summarize_height(
                dem[building_keep],
                reference_vec[building_keep],
                args.ground_dem_m,
                args.residual_sign,
                args.min_pixels,
                args.grubbs_alpha,
                quality_weights[building_keep],
            )
            if not np.isfinite(building_summary["height_m"]):
                continue
            building_coh = cohs[:, building_keep]
            building_da = amp_disp_vec[building_keep] if amp_disp_vec is not None else None
            building_rmse = pixel_rmse[building_keep]
            building_n_valid = n_valid[building_keep]
            building_bperp_span = bperp_span[building_keep]
            points.append(
                {
                    "uid": int(building_uid),
                    "touying_fid": int(fid),
                    "island_id": island_id,
                    "island_uid_count": int(info.uid_count),
                    "height_m": building_summary["height_m"],
                    "building_height_weighted_median_m": building_summary["building_height_weighted_median_m"],
                    "effective_pixel_count": building_summary["effective_pixel_count"],
                    "pixel_count_used": building_summary["pixel_count_used"],
                    "dem_error_p05_m": building_summary["dem_error_p05_m"],
                    "dem_error_median_m": building_summary["dem_error_median_m"],
                    "dem_error_p95_m": building_summary["dem_error_p95_m"],
                    "reference_elevation_median_m": building_summary["reference_elevation_median_m"],
                    "roof_elevation_p05_m": building_summary["roof_elevation_p05_m"],
                    "roof_elevation_median_m": building_summary["roof_elevation_median_m"],
                    "roof_elevation_p75_m": building_summary["roof_elevation_p75_m"],
                    "roof_elevation_p85_m": building_summary["roof_elevation_p85_m"],
                    "roof_elevation_p90_m": building_summary["roof_elevation_p90_m"],
                    "roof_elevation_p95_m": building_summary["roof_elevation_p95_m"],
                    "building_height_p05_m": building_summary["building_height_p05_m"],
                    "building_height_p25_m": building_summary["building_height_p25_m"],
                    "building_height_p50_m": building_summary["building_height_p50_m"],
                    "building_height_p75_m": building_summary["building_height_p75_m"],
                    "building_height_p85_m": building_summary["building_height_p85_m"],
                    "building_height_p90_m": building_summary["building_height_p90_m"],
                    "building_height_p95_m": building_summary["building_height_p95_m"],
                    "building_height_max_m": building_summary["building_height_max_m"],
                    "max_height_grubbs_n": building_summary["max_height_grubbs_n"],
                    "max_height_grubbs_g": building_summary["max_height_grubbs_g"],
                    "max_height_grubbs_gcrit": building_summary["max_height_grubbs_gcrit"],
                    "max_height_grubbs_p": building_summary["max_height_grubbs_p"],
                    "max_height_reject_outlier": building_summary["max_height_reject_outlier"],
                    "max_height_reliable": building_summary["max_height_reliable"],
                    "building_height_grubbs_top_m": building_summary["building_height_grubbs_top_m"],
                    "grubbs_top_removed_count": building_summary["grubbs_top_removed_count"],
                    "grubbs_top_remaining_count": building_summary["grubbs_top_remaining_count"],
                    "grubbs_top_last_g": building_summary["grubbs_top_last_g"],
                    "grubbs_top_last_gcrit": building_summary["grubbs_top_last_gcrit"],
                    "grubbs_top_last_p": building_summary["grubbs_top_last_p"],
                    "grubbs_top_reliable": building_summary["grubbs_top_reliable"],
                    "coh_mean": float(np.nanmedian(building_coh)) if np.isfinite(building_coh).any() else np.nan,
                    "amplitude_dispersion": float(np.nanmedian(building_da)) if building_da is not None and np.isfinite(building_da).any() else "",
                    "lgr_rmse_rad": float(np.nanmedian(building_rmse)) if np.isfinite(building_rmse).any() else np.nan,
                    "valid_pairs_median": float(np.nanmedian(building_n_valid[building_n_valid > 0])) if np.any(building_n_valid > 0) else np.nan,
                    "bperp_span_median": float(np.nanmedian(building_bperp_span)) if np.isfinite(building_bperp_span).any() else np.nan,
                    "method": "gamma_pixel_lgr_building_fid_residual_plus_dsm_minus_ground",
                }
            )
    out = pd.DataFrame(rows_out)
    pts = pd.DataFrame(points)
    Path(args.output_islands).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_points).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_islands, index=False)
    pts.to_csv(args.output_points, index=False)
    if args.figure or args.figure_svg:
        plot_qc(
            out,
            label,
            islands_info,
            Path(args.figure) if args.figure else None,
            Path(args.figure_svg) if args.figure_svg else None,
        )
    summary = {
        "islands_total": int(len(islands_info)),
        "islands_processed": int(len(out)),
        "islands_with_height": int(out["height_m"].notna().sum()) if not out.empty else 0,
        "height_points": int(len(pts)),
        "height_point_buildings": int(pts["uid"].nunique()) if not pts.empty else 0,
        "height_points_from_multi_fid_islands": int((pts["island_uid_count"] > 1).sum()) if not pts.empty else 0,
        "height_point_buildings_from_multi_fid_islands": int(pts.loc[pts["island_uid_count"] > 1, "uid"].nunique()) if not pts.empty else 0,
        "reference_height_rdc": args.reference_height_rdc,
        "ground_dem_m": args.ground_dem_m,
            "residual_sign": args.residual_sign,
        "unwrap_method": args.unwrap_method,
        "stable_reference_mask": args.stable_reference_mask or None,
        "stable_reference_min_pixels_per_pair": int(reference_pixel_counts.min()) if args.stable_reference_mask else 0,
        "stable_reference_max_pixels_per_pair": int(reference_pixel_counts.max()) if args.stable_reference_mask else 0,
        "reference_phase_offsets_rad": reference_offsets.tolist() if args.stable_reference_mask else [],
        "height_prior_used_for_inversion_or_fill": False,
        "building_estimator": "quality-weighted median of roof pixels",
        "min_coherence": args.min_coherence,
        "amplitude_dispersion_npy": args.amplitude_dispersion_npy,
        "max_amplitude_dispersion": args.max_amplitude_dispersion,
        "min_pairs": args.min_pairs,
        "max_pixel_rmse_rad": args.max_pixel_rmse_rad,
        "min_bperp_span_m": args.min_bperp_span_m,
        "physical_height_bounds_m": [args.min_physical_height_m, args.max_physical_height_m],
        "physical_height_bounds_use_vector_height": False,
        "grubbs_alpha": args.grubbs_alpha,
        "max_islands": args.max_islands,
        "max_pairs": args.max_pairs,
        "island_id_file": args.island_id_file,
        "output_islands": args.output_islands,
        "output_points": args.output_points,
        "figure": args.figure or None,
        "figure_svg": args.figure_svg or None,
        "method": "Roof-only pixel SBAS/LGR DEM-error inversion. Optional stable-ground circular phase anchoring removes the per-pair phase datum; temporal_multistart avoids an island-local zero-height gauge. Building height is the quality-weighted median of reference DSM/HGT plus signed DEM residual minus ground elevation. Vector height is never read for inversion or filling.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
