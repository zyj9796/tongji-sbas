#!/usr/bin/env python3
"""Invert island-level SBAS displacement time series and DEM residuals."""

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

from inventory_data import parse_gamma_par


def dem_phase_coeff(bperp_m: np.ndarray, wavelength_m: float, range_m: float, incidence_deg: float) -> np.ndarray:
    return (4.0 * math.pi * bperp_m) / (wavelength_m * range_m * math.sin(math.radians(incidence_deg)))


def solve_island(
    group: pd.DataFrame,
    dates: list[str],
    reference_date: str,
    wavelength_m: float,
    range_m: float,
    incidence_deg: float,
    min_pairs: int,
) -> tuple[dict, pd.DataFrame]:
    island_id = int(group["island_id"].iloc[0])
    g = group[np.isfinite(group["mean_phase_rad"]) & np.isfinite(group["bperp_m"])].copy()
    if len(g) < min_pairs:
        summary = {
            "island_id": island_id,
            "n_pairs": int(len(g)),
            "dem_error_m": np.nan,
            "rmse_phase_rad": np.nan,
            "mean_coherence": np.nan,
            "status": "insufficient_pairs",
        }
        return summary, pd.DataFrame()

    date_to_col = {date: i for i, date in enumerate(d for d in dates if d != reference_date)}
    n_unknown = 1 + len(date_to_col)
    a = np.zeros((len(g), n_unknown), dtype=float)
    a[:, 0] = dem_phase_coeff(g["bperp_m"].to_numpy(dtype=float), wavelength_m, range_m, incidence_deg)
    defo_coeff = 4.0 * math.pi / wavelength_m
    for row_idx, row in enumerate(g.itertuples(index=False)):
        master = str(row.master)
        slave = str(row.slave)
        if slave != reference_date:
            a[row_idx, 1 + date_to_col[slave]] += defo_coeff
        if master != reference_date:
            a[row_idx, 1 + date_to_col[master]] -= defo_coeff

    y = g["mean_phase_rad"].to_numpy(dtype=float)
    weights = np.clip(g["mean_coherence"].to_numpy(dtype=float), 0.05, 1.0)
    aw = a * weights[:, None]
    yw = y * weights
    coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    pred = a @ coef
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    disp_rows = []
    for date in dates:
        disp = 0.0 if date == reference_date else float(coef[1 + date_to_col[date]])
        disp_rows.append(
            {
                "island_id": island_id,
                "date": date,
                "los_displacement_m": disp,
                "dem_error_m": float(coef[0]),
                "n_pairs": int(len(g)),
                "rmse_phase_rad": rmse,
                "mean_coherence": float(g["mean_coherence"].mean()),
                "status": "sbas_timeseries_dem_error",
            }
        )

    summary = {
        "island_id": island_id,
        "n_pairs": int(len(g)),
        "dem_error_m": float(coef[0]),
        "rmse_phase_rad": rmse,
        "mean_coherence": float(g["mean_coherence"].mean()),
        "status": "sbas_timeseries_dem_error",
    }
    return summary, pd.DataFrame(disp_rows)


def plot_timeseries(ts: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=220)
    ok = summary[summary["status"] == "sbas_timeseries_dem_error"]
    if ts.empty or ok.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[1].text(0.5, 0.5, "No solved islands", ha="center", va="center")
    else:
        pivot = ts.pivot(index="date", columns="island_id", values="los_displacement_m").sort_index()
        x = np.arange(len(pivot.index))
        axes[0].plot(x, pivot.to_numpy() * 1000.0, color="0.75", lw=0.4, alpha=0.25)
        axes[0].plot(x, pivot.median(axis=1).to_numpy() * 1000.0, color="black", lw=1.6, label="median")
        axes[0].fill_between(
            x,
            pivot.quantile(0.1, axis=1).to_numpy() * 1000.0,
            pivot.quantile(0.9, axis=1).to_numpy() * 1000.0,
            color="#88ccee",
            alpha=0.35,
            label="10-90%",
        )
        axes[0].set_title("Island LOS displacement time series")
        axes[0].set_xticks(x[:: max(1, len(x) // 6)])
        axes[0].set_xticklabels(pivot.index[:: max(1, len(x) // 6)], rotation=35, ha="right")
        axes[0].set_ylabel("LOS displacement, mm")
        axes[0].legend(frameon=False)

        lo, hi = ok["dem_error_m"].quantile([0.02, 0.98])
        axes[1].hist(ok["dem_error_m"].clip(lo, hi), bins=50, color="#4477aa", edgecolor="white")
        axes[1].set_title("SBAS DEM residual")
        axes[1].set_xlabel("m, 2-98% clipped")

        axes[2].scatter(ok["n_pairs"], ok["rmse_phase_rad"], c=ok["mean_coherence"], s=12, alpha=0.65, cmap="viridis")
        axes[2].set_title("SBAS fit quality")
        axes[2].set_xlabel("valid interferogram count")
        axes[2].set_ylabel("phase RMSE, rad")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def island_map(label: np.ndarray, values: pd.Series) -> np.ndarray:
    out = np.full(label.shape, np.nan, dtype=np.float32)
    for island_id, value in values.items():
        if np.isfinite(value):
            out[label == int(island_id)] = float(value)
    return out


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_map(ax: plt.Axes, image: np.ndarray, title: str, cmap: str, center_zero: bool = False) -> None:
    lo, hi = robust_limits(image)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if center_zero and lo < 0 < hi else None
    im = ax.imshow(image, cmap=cmap, vmin=None if norm else lo, vmax=None if norm else hi, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_spatial(ts: pd.DataFrame, summary: pd.DataFrame, label: np.ndarray, out: Path) -> None:
    ok = summary[summary["status"] == "sbas_timeseries_dem_error"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=220)
    if ok.empty or ts.empty:
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 1].text(0.5, 0.5, "No solved islands", ha="center", va="center")
    else:
        by_id = ok.set_index("island_id")
        pivot = ts.pivot(index="date", columns="island_id", values="los_displacement_m").sort_index()
        final_disp = pivot.iloc[-1] * 1000.0
        disp_range = (pivot.max(axis=0) - pivot.min(axis=0)) * 1000.0
        add_map(axes[0, 0], island_map(label, by_id["dem_error_m"]), "SBAS DEM residual, m", "RdBu_r", True)
        add_map(axes[0, 1], island_map(label, by_id["rmse_phase_rad"]), "Phase model RMSE, rad", "magma")
        add_map(axes[0, 2], island_map(label, by_id["n_pairs"]), "Valid interferogram count", "viridis")
        add_map(axes[1, 0], island_map(label, final_disp), "Final LOS displacement, mm", "RdBu_r", True)
        add_map(axes[1, 1], island_map(label, disp_range), "LOS displacement range, mm", "YlGnBu")
        add_map(axes[1, 2], island_map(label, by_id["mean_coherence"]), "Mean coherence", "viridis")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--reference-par", required=True)
    parser.add_argument("--reference-date")
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--output-timeseries", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--island-label")
    parser.add_argument("--spatial-figure")
    args = parser.parse_args()

    obs = pd.read_csv(args.observations)
    pairs = pd.read_csv(args.pairs_csv)
    dates = sorted(set(pairs["master"].astype(str)).union(set(pairs["slave"].astype(str))))
    reference_date = args.reference_date or dates[0]
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    range_m = float(par["center_range_slc"])
    incidence = float(par["incidence_angle"])

    summaries = []
    time_series = []
    for _, group in obs.groupby("island_id", sort=True):
        summary, ts = solve_island(group, dates, reference_date, wavelength, range_m, incidence, args.min_pairs)
        summaries.append(summary)
        if not ts.empty:
            time_series.append(ts)

    summary_df = pd.DataFrame(summaries)
    ts_df = pd.concat(time_series, ignore_index=True) if time_series else pd.DataFrame()
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_timeseries).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary, index=False)
    ts_df.to_csv(args.output_timeseries, index=False)
    plot_timeseries(ts_df, summary_df, Path(args.figure))
    if args.island_label and args.spatial_figure:
        plot_spatial(ts_df, summary_df, np.load(args.island_label), Path(args.spatial_figure))

    ok = summary_df[summary_df["status"] == "sbas_timeseries_dem_error"]
    payload = {
        "observations": args.observations,
        "pairs_csv": args.pairs_csv,
        "reference_date": reference_date,
        "dates": dates,
        "islands_total": int(len(summary_df)),
        "islands_solved": int(len(ok)),
        "time_series_rows": int(len(ts_df)),
        "min_pairs": args.min_pairs,
        "dem_error_median_m": float(ok["dem_error_m"].median()) if not ok.empty else None,
        "dem_error_p05_m": float(ok["dem_error_m"].quantile(0.05)) if not ok.empty else None,
        "dem_error_p95_m": float(ok["dem_error_m"].quantile(0.95)) if not ok.empty else None,
        "phase_rmse_median_rad": float(ok["rmse_phase_rad"].median()) if not ok.empty else None,
        "wavelength_m": wavelength,
        "range_m": range_m,
        "incidence_deg": incidence,
        "note": "Island-level SBAS inversion with unknown DEM residual and per-date LOS displacement; no building height field is used.",
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
