#!/usr/bin/env python3
"""Estimate island DEM error with the thesis LGR least-squares model."""

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

from inventory_data import parse_gamma_par


def design_coefficients(bperp_m: np.ndarray, dt_days: np.ndarray, wavelength_m: float, range_m: float, incidence_deg: float) -> np.ndarray:
    fen_mu = range_m * math.sin(math.radians(incidence_deg))
    return np.column_stack(
        [
            (4.0 * math.pi * bperp_m) / (wavelength_m * fen_mu),
            (4.0 * math.pi * dt_days) / wavelength_m,
        ]
    )


def solve_one(group: pd.DataFrame, baselines: pd.DataFrame, min_pairs: int, wavelength_m: float, range_m: float, incidence_deg: float) -> dict:
    if {"bperp_m", "dt_days"}.issubset(group.columns):
        merged = group.copy()
    else:
        merged = group.merge(baselines[["master", "slave", "dt_days", "bperp_m"]], on=["master", "slave"], how="inner")
    merged = merged[np.isfinite(merged["mean_phase_rad"]) & np.isfinite(merged["bperp_m"])]
    if len(merged) < min_pairs:
        return {
            "island_id": int(group["island_id"].iloc[0]),
            "n_pairs": int(len(merged)),
            "dem_error_m": np.nan,
            "rate_m_per_year": np.nan,
            "rmse_phase_rad": np.nan,
            "status": "insufficient_pairs",
        }
    a = design_coefficients(
        merged["bperp_m"].to_numpy(dtype=float),
        merged["dt_days"].to_numpy(dtype=float),
        wavelength_m,
        range_m,
        incidence_deg,
    )
    y = merged["mean_phase_rad"].to_numpy(dtype=float)
    weights = np.clip(merged["mean_coherence"].to_numpy(dtype=float), 0.05, 1.0)
    aw = a * weights[:, None]
    yw = y * weights
    coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    pred = a @ coef
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    # Mirrors my_study/my_muilt_hight/LGR_demerror_est.m:
    # W(1) is DEM error term and W(2)*365 is annual deformation rate.
    return {
        "island_id": int(group["island_id"].iloc[0]),
        "n_pairs": int(len(merged)),
        "dem_error_m": float(coef[0]),
        "rate_m_per_year": float(coef[1] * 365.0),
        "rmse_phase_rad": rmse,
        "mean_coherence": float(merged["mean_coherence"].mean()),
        "bperp_span_m": float(merged["bperp_m"].max() - merged["bperp_m"].min()),
        "status": "diagnostic_raw_phase_lgr",
    }


def plot_results(df: pd.DataFrame, out: Path) -> None:
    ok = df[df["status"] == "diagnostic_raw_phase_lgr"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    if ok.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[1].text(0.5, 0.5, "No solved islands", ha="center", va="center")
    else:
        clipped_dem = ok["dem_error_m"].clip(ok["dem_error_m"].quantile(0.02), ok["dem_error_m"].quantile(0.98))
        axes[0].hist(clipped_dem, bins=50, color="#4477aa", edgecolor="white")
        axes[0].set_title("Diagnostic DEM error")
        axes[0].set_xlabel("m, 2-98% clipped")
        axes[1].hist(ok["rmse_phase_rad"], bins=50, color="#cc6677", edgecolor="white")
        axes[1].set_title("Phase model RMSE")
        axes[1].set_xlabel("rad")
        axes[2].scatter(ok["bperp_span_m"], ok["rmse_phase_rad"], s=6, alpha=0.45, color="#117733")
        axes[2].set_title("Baseline span vs RMSE")
        axes[2].set_xlabel("Bperp span m")
        axes[2].set_ylabel("RMSE rad")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", default="work/height/unwrapped_differential_island_phase_observations.csv")
    parser.add_argument("--baselines", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--allow-raw-diagnostic", action="store_true")
    parser.add_argument("--output-csv", default="work/height/island_dem_error_lgr_gamma.csv")
    parser.add_argument("--summary", default="results/metadata/island_dem_error_lgr_gamma_summary.json")
    parser.add_argument("--figure", default="results/pic_all/19_island_dem_error_lgr_gamma.png")
    args = parser.parse_args()

    if "raw_island_phase" in args.observations and not args.allow_raw_diagnostic:
        raise SystemExit("Refusing to run on raw preview phases without --allow-raw-diagnostic.")
    obs = pd.read_csv(args.observations)
    baselines = pd.read_csv(args.baselines)
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    range_m = float(par["center_range_slc"])
    incidence = float(par["incidence_angle"])
    rows = [
        solve_one(group, baselines, args.min_pairs, wavelength, range_m, incidence)
        for _, group in obs.groupby("island_id", sort=True)
    ]
    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    plot_results(out, Path(args.figure))
    ok = out[out["status"] == "diagnostic_raw_phase_lgr"]
    summary = {
        "observations": args.observations,
        "baselines": args.baselines,
        "output_csv": args.output_csv,
        "islands_total": int(len(out)),
        "islands_solved": int(len(ok)),
        "min_pairs": args.min_pairs,
        "wavelength_m": wavelength,
        "range_m": range_m,
        "incidence_deg": incidence,
        "note": "LGR DEM-error estimate from GAMMA differential interferograms and GAMMA baseline table.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
