#!/usr/bin/env python3
"""High-coherence pixel SBAS LOS deformation monitoring from GAMMA differential interferograms."""

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
from matplotlib.colors import TwoSlopeNorm

from extract_gamma_differential_island_observations import read_float, read_fcomplex
from inventory_data import parse_gamma_par


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def build_design(pairs: pd.DataFrame, dates: list[str], reference_date: str, wavelength_m: float) -> np.ndarray:
    date_to_col = {date: i for i, date in enumerate(d for d in dates if d != reference_date)}
    a = np.zeros((len(pairs), len(dates) - 1), dtype=np.float64)
    k = 4.0 * math.pi / wavelength_m
    for row_idx, row in enumerate(pairs.itertuples(index=False)):
        master = str(row.master)
        slave = str(row.slave)
        if slave != reference_date:
            a[row_idx, date_to_col[slave]] += k
        if master != reference_date:
            a[row_idx, date_to_col[master]] -= k
    return a


def solve_chunk(
    phases_wrapped: np.ndarray,
    coherence: np.ndarray,
    a: np.ndarray,
    iterations: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve weighted SBAS time series for a chunk of pixels.

    phases_wrapped and coherence are shaped (n_pairs, n_pixels).
    Returns displacement coefficients (n_unknown, n_pixels), final predicted
    phases, residual phases, and integer ambiguity counts.
    """
    n_pairs, n_pix = phases_wrapped.shape
    n_unknown = a.shape[1]
    weights = np.clip(coherence, 0.03, 1.0) ** 2
    y_wrapped = phases_wrapped.astype(np.float64, copy=False)
    y_unwrapped = y_wrapped.copy()
    coef = np.zeros((n_unknown, n_pix), dtype=np.float64)
    eye = np.eye(n_unknown, dtype=np.float64) * ridge
    ambiguity = np.zeros_like(y_wrapped, dtype=np.int16)

    for _ in range(max(1, iterations)):
        for j in range(n_pix):
            w = weights[:, j]
            aw = a * w[:, None]
            normal = a.T @ aw + eye
            rhs = a.T @ (w * y_unwrapped[:, j])
            try:
                coef[:, j] = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                coef[:, j] = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        pred = a @ coef
        ambiguity = np.rint((pred - y_wrapped) / (2.0 * np.pi)).astype(np.int16)
        y_unwrapped = y_wrapped + 2.0 * np.pi * ambiguity

    pred = a @ coef
    residual = y_unwrapped - pred
    return coef, pred, residual, ambiguity


def robust_limits(arr: np.ndarray, qlo: float = 2.0, qhi: float = 98.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_map(ax: plt.Axes, img: np.ndarray, title: str, cmap: str, center_zero: bool = False) -> None:
    lo, hi = robust_limits(img)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if center_zero and lo < 0.0 < hi else None
    im = ax.imshow(img, cmap=cmap, vmin=None if norm else lo, vmax=None if norm else hi, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_maps(
    mean_coh: np.ndarray,
    valid_mask: np.ndarray,
    velocity_mm_yr: np.ndarray,
    final_mm: np.ndarray,
    rmse_rad: np.ndarray,
    residual_mad_rad: np.ndarray,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=220)
    add_map(axes[0, 0], np.where(valid_mask, mean_coh, np.nan), "Mean coherence, monitored pixels", "viridis")
    add_map(axes[0, 1], velocity_mm_yr, "LOS velocity, mm/yr", "RdBu_r", center_zero=True)
    add_map(axes[0, 2], final_mm, "Cumulative LOS displacement, mm", "RdBu_r", center_zero=True)
    add_map(axes[1, 0], rmse_rad, "SBAS phase RMSE, rad", "magma")
    add_map(axes[1, 1], residual_mad_rad, "Residual MAD, rad", "magma")
    axes[1, 2].hist(velocity_mm_yr[np.isfinite(velocity_mm_yr)], bins=80, color="#4477aa", edgecolor="white")
    axes[1, 2].set_title("Velocity distribution")
    axes[1, 2].set_xlabel("mm/yr")
    axes[1, 2].set_ylabel("pixels")
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_timeseries(ts: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=220)
    x = np.arange(len(ts))
    ax.fill_between(x, ts["p10_mm"], ts["p90_mm"], color="#88ccee", alpha=0.35, label="10-90%")
    ax.plot(x, ts["median_mm"], color="black", lw=1.8, label="median")
    ax.plot(x, ts["p25_mm"], color="#4477aa", lw=0.9, alpha=0.7, label="25/75%")
    ax.plot(x, ts["p75_mm"], color="#4477aa", lw=0.9, alpha=0.7)
    step = max(1, len(ts) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(ts["date"].iloc[::step], rotation=35, ha="right")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_ylabel("LOS displacement, mm")
    ax.set_title("High-coherence SBAS displacement time series")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--intf-root", required=True)
    parser.add_argument("--reference-par", required=True)
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--reference-date")
    parser.add_argument("--min-mean-coherence", type=float, default=0.58)
    parser.add_argument("--min-p20-coherence", type=float, default=0.35)
    parser.add_argument("--closure-mean-abs")
    parser.add_argument("--max-closure-mean-abs-rad", type=float, default=1.2)
    parser.add_argument("--max-rmse-rad", type=float, default=1.25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--no-reference-median-center", action="store_true")
    parser.add_argument("--reference-mask")
    parser.add_argument("--min-reference-pixels", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ["arrays", "figures", "tables", "metadata"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    dates = sorted(set(pairs["master"].astype(str)).union(set(pairs["slave"].astype(str))))
    reference_date = args.reference_date or dates[0]
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    a = build_design(pairs, dates, reference_date, wavelength)

    phase_stack = np.empty((len(pairs), args.rows, args.cols), dtype=np.float32)
    coh_stack = np.empty_like(phase_stack)
    pair_qc = []
    for idx, row in enumerate(pairs.itertuples(index=False)):
        master = str(row.master)
        slave = str(row.slave)
        pair = f"{master}_{slave}"
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        phase = np.angle(diff).astype(np.float32)
        valid = np.isfinite(phase) & np.isfinite(cc) & (np.abs(diff) > 0.0)
        phase_stack[idx] = np.where(valid, phase, np.nan)
        coh_stack[idx] = np.where(valid, cc.astype(np.float32), np.nan)
        pair_qc.append(
            {
                "master": master,
                "slave": slave,
                "dt_days": int(row.dt_days),
                "bperp_m": float(row.bperp_m),
                "valid_fraction": float(np.mean(valid)),
                "mean_coherence": float(np.nanmean(np.where(valid, cc, np.nan))),
                "median_coherence": float(np.nanmedian(np.where(valid, cc, np.nan))),
            }
        )

    mean_coh = np.nanmean(coh_stack, axis=0)
    p20_coh = np.nanpercentile(coh_stack, 20, axis=0)
    finite_all = np.all(np.isfinite(phase_stack), axis=0) & np.all(np.isfinite(coh_stack), axis=0)
    initial_mask = finite_all & (mean_coh >= args.min_mean_coherence) & (p20_coh >= args.min_p20_coherence)
    closure_mean_abs = None
    if args.closure_mean_abs:
        closure_mean_abs = np.load(args.closure_mean_abs)
        if closure_mean_abs.shape != initial_mask.shape:
            raise ValueError(f"closure map shape {closure_mean_abs.shape} does not match {initial_mask.shape}")
        initial_mask &= np.isfinite(closure_mean_abs) & (closure_mean_abs <= args.max_closure_mean_abs_rad)
    flat_idx = np.flatnonzero(initial_mask.ravel())
    n_pix = int(flat_idx.size)
    n_dates = len(dates)
    coef_all = np.full((n_dates - 1, n_pix), np.nan, dtype=np.float32)
    rmse_all = np.full(n_pix, np.nan, dtype=np.float32)
    mad_all = np.full(n_pix, np.nan, dtype=np.float32)
    max_ambiguity_all = np.zeros(n_pix, dtype=np.int16)

    phase_flat = phase_stack.reshape(len(pairs), -1)[:, flat_idx]
    coh_flat = coh_stack.reshape(len(pairs), -1)[:, flat_idx]
    for start in range(0, n_pix, args.chunk_size):
        end = min(start + args.chunk_size, n_pix)
        coef, _pred, residual, ambiguity = solve_chunk(
            phase_flat[:, start:end],
            coh_flat[:, start:end],
            a,
            args.iterations,
            args.ridge,
        )
        coef_all[:, start:end] = coef.astype(np.float32)
        rmse_all[start:end] = np.sqrt(np.mean(residual**2, axis=0)).astype(np.float32)
        med = np.median(residual, axis=0)
        mad_all[start:end] = np.median(np.abs(residual - med[None, :]), axis=0).astype(np.float32)
        max_ambiguity_all[start:end] = np.max(np.abs(ambiguity), axis=0).astype(np.int16)

    quality_mask_flat = rmse_all <= args.max_rmse_rad
    monitored_idx = flat_idx[quality_mask_flat]
    coef_good = coef_all[:, quality_mask_flat]
    days = np.array([(pd.to_datetime(d) - pd.to_datetime(reference_date)).days for d in dates], dtype=np.float64)
    date_to_coef = {date: i for i, date in enumerate(d for d in dates if d != reference_date)}
    disp = np.zeros((n_dates, coef_good.shape[1]), dtype=np.float64)
    for row_idx, date in enumerate(dates):
        if date != reference_date:
            disp[row_idx] = coef_good[date_to_coef[date]]
    shape = (args.rows, args.cols)
    reference_column_mask = None
    if args.reference_mask:
        reference_mask = np.load(args.reference_mask).astype(bool)
        if reference_mask.shape != shape:
            raise ValueError(f"reference mask shape {reference_mask.shape} does not match {shape}")
        reference_column_mask = reference_mask.ravel()[monitored_idx]
        if int(reference_column_mask.sum()) < args.min_reference_pixels:
            raise ValueError(
                f"reference mask has {int(reference_column_mask.sum())} monitored pixels, "
                f"below --min-reference-pixels {args.min_reference_pixels}"
            )
    reference_values = disp[:, reference_column_mask] if reference_column_mask is not None else disp
    reference_median_m = np.nanmedian(reference_values, axis=1) if reference_values.size else np.zeros(n_dates)
    if not args.no_reference_median_center and disp.size:
        disp = disp - reference_median_m[:, None]

    centered_days = days - days.mean()
    denom = np.sum(centered_days**2)
    velocity_m_day = (centered_days[:, None] * disp).sum(axis=0) / denom if denom > 0 else np.zeros(disp.shape[1])
    velocity_mm_yr_flat = velocity_m_day * 365.25 * 1000.0
    final_mm_flat = disp[-1] * 1000.0

    valid_mask = np.zeros(args.rows * args.cols, dtype=bool)
    valid_mask[monitored_idx] = True
    valid_mask = valid_mask.reshape(shape)
    velocity = np.full(args.rows * args.cols, np.nan, dtype=np.float32)
    final = np.full_like(velocity, np.nan)
    rmse = np.full_like(velocity, np.nan)
    mad = np.full_like(velocity, np.nan)
    max_amb = np.full(args.rows * args.cols, -1, dtype=np.int16)
    velocity[monitored_idx] = velocity_mm_yr_flat.astype(np.float32)
    final[monitored_idx] = final_mm_flat.astype(np.float32)
    rmse[flat_idx] = rmse_all
    mad[flat_idx] = mad_all
    max_amb[flat_idx] = max_ambiguity_all

    velocity = velocity.reshape(shape)
    final = final.reshape(shape)
    rmse = rmse.reshape(shape)
    mad = mad.reshape(shape)
    max_amb = max_amb.reshape(shape)

    disp_stack = np.full((n_dates, args.rows * args.cols), np.nan, dtype=np.float32)
    disp_stack[:, monitored_idx] = (disp * 1000.0).astype(np.float32)
    disp_stack = disp_stack.reshape((n_dates, args.rows, args.cols))

    np.save(out_dir / "arrays/mean_coherence.npy", mean_coh.astype(np.float32))
    np.save(out_dir / "arrays/monitored_pixel_mask.npy", valid_mask)
    np.save(out_dir / "arrays/los_velocity_mm_per_year.npy", velocity)
    np.save(out_dir / "arrays/cumulative_los_displacement_mm.npy", final)
    np.save(out_dir / "arrays/phase_rmse_rad.npy", rmse)
    np.save(out_dir / "arrays/residual_mad_rad.npy", mad)
    np.save(out_dir / "arrays/max_phase_ambiguity_count.npy", max_amb)
    np.save(out_dir / "arrays/los_displacement_timeseries_mm.npy", disp_stack)
    if closure_mean_abs is not None:
        np.save(out_dir / "arrays/phase_closure_mean_abs_rad.npy", closure_mean_abs.astype(np.float32))

    ts_rows = []
    for i, date in enumerate(dates):
        vals = disp_stack[i][valid_mask]
        ts_rows.append(
            {
                "date": date,
                "days_since_reference": int(days[i]),
                "n_pixels": int(np.isfinite(vals).sum()),
                "median_mm": float(np.nanmedian(vals)),
                "p10_mm": float(np.nanpercentile(vals, 10)),
                "p25_mm": float(np.nanpercentile(vals, 25)),
                "p75_mm": float(np.nanpercentile(vals, 75)),
                "p90_mm": float(np.nanpercentile(vals, 90)),
            }
        )
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(out_dir / "tables/high_coherence_sbas_timeseries_summary.csv", index=False)
    pd.DataFrame(pair_qc).to_csv(out_dir / "tables/pair_quality_summary.csv", index=False)

    plot_maps(mean_coh, valid_mask, velocity, final, rmse, mad, out_dir / "figures/high_quality_sbas_deformation_maps")
    plot_timeseries(ts, out_dir / "figures/high_quality_sbas_displacement_timeseries")

    summary = {
        "pairs_csv": args.pairs_csv,
        "intf_root": args.intf_root,
        "reference_par": args.reference_par,
        "reference_date": reference_date,
        "dates": dates,
        "n_pairs": int(len(pairs)),
        "n_dates": int(len(dates)),
        "rows": args.rows,
        "cols": args.cols,
        "candidate_pixels": int(n_pix),
        "monitored_pixels": int(valid_mask.sum()),
        "min_mean_coherence": args.min_mean_coherence,
        "min_p20_coherence": args.min_p20_coherence,
        "closure_mean_abs": args.closure_mean_abs,
        "max_closure_mean_abs_rad": args.max_closure_mean_abs_rad if args.closure_mean_abs else None,
        "max_rmse_rad": args.max_rmse_rad,
        "iterations": args.iterations,
        "reference_median_centered": not args.no_reference_median_center,
        "reference_mask": args.reference_mask,
        "reference_pixels": int(reference_column_mask.sum()) if reference_column_mask is not None else int(valid_mask.sum()),
        "reference_median_m": [float(x) for x in reference_median_m],
        "velocity_median_mm_yr": float(np.nanmedian(velocity)),
        "velocity_p05_mm_yr": float(np.nanpercentile(velocity, 5)),
        "velocity_p95_mm_yr": float(np.nanpercentile(velocity, 95)),
        "final_displacement_median_mm": float(np.nanmedian(final)),
        "final_displacement_p05_mm": float(np.nanpercentile(final, 5)),
        "final_displacement_p95_mm": float(np.nanpercentile(final, 95)),
        "phase_rmse_median_rad": float(np.nanmedian(rmse[valid_mask])),
        "phase_rmse_p95_rad": float(np.nanpercentile(rmse[valid_mask], 95)),
        "wavelength_m": wavelength,
        "method": "High-coherence pixel SBAS LOS time-series inversion from GAMMA DSM differential interferograms; iterative temporal ambiguity update; no building height field used.",
    }
    (out_dir / "metadata/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
