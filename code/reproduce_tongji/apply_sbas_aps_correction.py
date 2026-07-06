#!/usr/bin/env python3
"""Apply empirical APS/long-wavelength correction to an SBAS LOS time series."""

from __future__ import annotations

import argparse
import json
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


def design_matrix(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], degree: int) -> np.ndarray:
    y = (rows.astype(float) - (shape[0] - 1) / 2.0) / max(shape[0], 1)
    x = (cols.astype(float) - (shape[1] - 1) / 2.0) / max(shape[1], 1)
    terms = [np.ones_like(x), x, y]
    if degree >= 2:
        terms.extend([x * x, x * y, y * y])
    return np.column_stack(terms)


def robust_fit(a: np.ndarray, z: np.ndarray, iterations: int, huber_k: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(z)
    a = a[valid]
    z = z[valid]
    if len(z) < a.shape[1] * 5:
        return np.full(a.shape[1], np.nan), np.zeros(0, dtype=bool)
    weights = np.ones(len(z), dtype=float)
    keep = np.ones(len(z), dtype=bool)
    coef = np.zeros(a.shape[1], dtype=float)
    for _ in range(max(1, iterations)):
        aw = a * weights[:, None]
        coef, *_ = np.linalg.lstsq(aw, z * weights, rcond=None)
        resid = z - a @ coef
        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med))
        sigma = 1.4826 * mad if mad > 1e-6 else np.nanstd(resid)
        if not np.isfinite(sigma) or sigma <= 1e-6:
            break
        u = np.abs(resid - med) / (huber_k * sigma)
        weights = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-6))
        keep = u <= 2.5
    return coef, keep


def robust_limits(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, [2, 98])
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_map(ax: plt.Axes, img: np.ndarray, title: str, cmap: str = "RdBu_r") -> None:
    lo, hi = robust_limits(img)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if lo < 0 < hi else None
    im = ax.imshow(img, cmap=cmap, vmin=None if norm else lo, vmax=None if norm else hi, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_overview(
    before_final: np.ndarray,
    after_final: np.ndarray,
    aps_final: np.ndarray,
    before_vel: np.ndarray,
    after_vel: np.ndarray,
    ref_mask: np.ndarray,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=220)
    add_map(axes[0, 0], before_final, "Before APS: final LOS, mm")
    add_map(axes[0, 1], aps_final, "Estimated APS/ramp: final date, mm")
    add_map(axes[0, 2], after_final, "After APS: final LOS, mm")
    add_map(axes[1, 0], before_vel, "Before APS: velocity, mm/yr")
    add_map(axes[1, 1], after_vel - before_vel, "Velocity correction, mm/yr")
    add_map(axes[1, 2], after_vel, "After APS: velocity, mm/yr")
    for ax in axes.flat:
        rr, cc = np.nonzero(ref_mask)
        if len(rr):
            step = max(1, len(rr) // 2500)
            ax.scatter(cc[::step], rr[::step], s=0.8, c="black", alpha=0.18, linewidths=0)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_timeseries(before: pd.DataFrame, after: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=220)
    x = np.arange(len(before))
    ax.fill_between(x, before["p10_mm"], before["p90_mm"], color="#88ccee", alpha=0.25, label="before 10-90%")
    ax.plot(x, before["median_mm"], color="#4477aa", lw=1.3, label="before median")
    ax.fill_between(x, after["p10_mm"], after["p90_mm"], color="#ddcc77", alpha=0.25, label="after 10-90%")
    ax.plot(x, after["median_mm"], color="black", lw=1.6, label="after median")
    step = max(1, len(before) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(before["date"].iloc[::step], rotation=35, ha="right")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.set_ylabel("LOS displacement, mm")
    ax.set_title("SBAS time series before/after APS correction")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def velocity_from_stack(stack: np.ndarray, dates: list[str], mask: np.ndarray) -> np.ndarray:
    days = np.array([(pd.to_datetime(d) - pd.to_datetime(dates[0])).days for d in dates], dtype=float)
    centered = days - days.mean()
    denom = np.sum(centered**2)
    flat = stack.reshape(stack.shape[0], -1)
    vel = np.full(flat.shape[1], np.nan, dtype=np.float32)
    idx = np.flatnonzero(mask.ravel())
    if idx.size and denom > 0:
        vel[idx] = ((centered[:, None] * flat[:, idx]).sum(axis=0) / denom * 365.25).astype(np.float32)
    return vel.reshape(mask.shape)


def summarize_ts(stack: np.ndarray, dates: list[str], mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, date in enumerate(dates):
        vals = stack[i][mask]
        rows.append(
            {
                "date": date,
                "median_mm": float(np.nanmedian(vals)),
                "p10_mm": float(np.nanpercentile(vals, 10)),
                "p25_mm": float(np.nanpercentile(vals, 25)),
                "p75_mm": float(np.nanpercentile(vals, 75)),
                "p90_mm": float(np.nanpercentile(vals, 90)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--degree", type=int, default=1, choices=[1, 2])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--huber-k", type=float, default=1.5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    for sub in ["arrays", "figures", "tables", "metadata"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "metadata/summary.json").read_text(encoding="utf-8"))
    dates = [str(d) for d in summary["dates"]]
    stack = np.load(run_dir / "arrays/los_displacement_timeseries_mm.npy").astype(np.float32)
    mask = np.load(run_dir / "arrays/monitored_pixel_mask.npy").astype(bool)
    ref_mask = np.load(args.reference_mask).astype(bool) & mask
    shape = mask.shape
    rr, cc = np.nonzero(ref_mask)
    a_ref = design_matrix(rr, cc, shape, args.degree)
    grid_r, grid_c = np.indices(shape)
    a_grid = design_matrix(grid_r.ravel(), grid_c.ravel(), shape, args.degree)

    aps_stack = np.zeros_like(stack, dtype=np.float32)
    fit_rows = []
    for i, date in enumerate(dates):
        vals = stack[i][ref_mask].astype(float)
        coef, keep = robust_fit(a_ref, vals, args.iterations, args.huber_k)
        if np.any(~np.isfinite(coef)):
            surface = np.zeros(shape, dtype=np.float32)
            used = 0
            rms = np.nan
        else:
            surface = (a_grid @ coef).reshape(shape).astype(np.float32)
            surface -= np.nanmedian(surface[ref_mask])
            used = int(keep.sum()) if keep.size else int(ref_mask.sum())
            resid = vals - a_ref @ coef
            rms = float(np.sqrt(np.nanmean(resid**2)))
        aps_stack[i] = surface
        fit_rows.append(
            {
                "date": date,
                "reference_pixels": int(ref_mask.sum()),
                "robust_used_pixels": used,
                "fit_rms_mm": rms,
                "aps_p05_mm": float(np.nanpercentile(surface[mask], 5)),
                "aps_p95_mm": float(np.nanpercentile(surface[mask], 95)),
            }
        )

    corrected = stack - aps_stack
    # Keep the stable reference datum exactly centered after surface removal.
    ref_median = np.nanmedian(corrected[:, ref_mask], axis=1)
    corrected = corrected - ref_median[:, None, None].astype(np.float32)

    before_vel = np.load(run_dir / "arrays/los_velocity_mm_per_year.npy")
    after_vel = velocity_from_stack(corrected, dates, mask)
    final_before = np.load(run_dir / "arrays/cumulative_los_displacement_mm.npy")
    final_after = corrected[-1]

    np.save(out_dir / "arrays/los_displacement_timeseries_mm.npy", corrected.astype(np.float32))
    np.save(out_dir / "arrays/aps_correction_timeseries_mm.npy", aps_stack.astype(np.float32))
    np.save(out_dir / "arrays/los_velocity_mm_per_year.npy", after_vel.astype(np.float32))
    np.save(out_dir / "arrays/cumulative_los_displacement_mm.npy", final_after.astype(np.float32))
    np.save(out_dir / "arrays/monitored_pixel_mask.npy", mask)
    np.save(out_dir / "arrays/stable_reference_mask.npy", ref_mask)

    before_ts = summarize_ts(stack, dates, mask)
    after_ts = summarize_ts(corrected, dates, mask)
    before_ts.to_csv(out_dir / "tables/timeseries_before_aps.csv", index=False)
    after_ts.to_csv(out_dir / "tables/timeseries_after_aps.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out_dir / "tables/aps_fit_summary.csv", index=False)

    plot_overview(
        np.where(mask, final_before, np.nan),
        np.where(mask, final_after, np.nan),
        np.where(mask, aps_stack[-1], np.nan),
        np.where(mask, before_vel, np.nan),
        np.where(mask, after_vel, np.nan),
        ref_mask,
        out_dir / "figures/aps_correction_overview",
    )
    plot_timeseries(before_ts, after_ts, out_dir / "figures/aps_corrected_timeseries")

    payload = {
        "source_run": args.run_dir,
        "reference_mask": args.reference_mask,
        "degree": args.degree,
        "reference_pixels": int(ref_mask.sum()),
        "monitored_pixels": int(mask.sum()),
        "before_velocity_median_mm_yr": float(np.nanmedian(before_vel[mask])),
        "after_velocity_median_mm_yr": float(np.nanmedian(after_vel[mask])),
        "velocity_correction_median_mm_yr": float(np.nanmedian((after_vel - before_vel)[mask])),
        "velocity_correction_p05_mm_yr": float(np.nanpercentile((after_vel - before_vel)[mask], 5)),
        "velocity_correction_p95_mm_yr": float(np.nanpercentile((after_vel - before_vel)[mask], 95)),
        "before_final_p05_mm": float(np.nanpercentile(final_before[mask], 5)),
        "before_final_p95_mm": float(np.nanpercentile(final_before[mask], 95)),
        "after_final_p05_mm": float(np.nanpercentile(final_after[mask], 5)),
        "after_final_p95_mm": float(np.nanpercentile(final_after[mask], 95)),
        "aps_final_p05_mm": float(np.nanpercentile(aps_stack[-1][mask], 5)),
        "aps_final_p95_mm": float(np.nanpercentile(aps_stack[-1][mask], 95)),
        "note": "Empirical long-wavelength APS/ramp correction fitted on stable reference pixels in SAR row/column coordinates.",
    }
    (out_dir / "metadata/summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
