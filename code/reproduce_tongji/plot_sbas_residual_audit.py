#!/usr/bin/env python3
"""Build spatial and residual diagnostics for the island-level SBAS DEM audit."""

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

from estimate_island_dem_error_lgr import design_coefficients
from inventory_data import parse_gamma_par


def island_map(label: np.ndarray, values: pd.Series) -> np.ndarray:
    out = np.full(label.shape, np.nan, dtype=np.float32)
    for island_id, value in values.items():
        if np.isfinite(value):
            out[label == int(island_id)] = float(value)
    return out


def robust_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_image(ax: plt.Axes, image: np.ndarray, title: str, cmap: str, center_zero: bool = False) -> None:
    lo, hi = robust_limits(image)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi) if center_zero and lo < 0 < hi else None
    im = ax.imshow(image, cmap=cmap, vmin=None if norm else lo, vmax=None if norm else hi, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def compute_residuals(
    obs: pd.DataFrame,
    dem: pd.DataFrame,
    wavelength_m: float,
    range_m: float,
    incidence_deg: float,
) -> pd.DataFrame:
    solved = dem[dem["status"] == "diagnostic_raw_phase_lgr"][
        ["island_id", "dem_error_m", "rate_m_per_year", "rmse_phase_rad", "n_pairs"]
    ]
    merged = obs.merge(solved, on="island_id", how="inner")
    merged = merged[np.isfinite(merged["mean_phase_rad"]) & np.isfinite(merged["bperp_m"])]
    if merged.empty:
        return merged
    a = design_coefficients(
        merged["bperp_m"].to_numpy(dtype=float),
        merged["dt_days"].to_numpy(dtype=float),
        wavelength_m,
        range_m,
        incidence_deg,
    )
    coef = np.column_stack(
        [
            merged["dem_error_m"].to_numpy(dtype=float),
            merged["rate_m_per_year"].to_numpy(dtype=float) / 365.0,
        ]
    )
    merged = merged.copy()
    merged["pred_phase_rad"] = np.sum(a * coef, axis=1)
    merged["residual_phase_rad"] = merged["mean_phase_rad"] - merged["pred_phase_rad"]
    return merged


def plot_overview(label: np.ndarray, dem: pd.DataFrame, residuals: pd.DataFrame, out_base: Path) -> None:
    solved = dem[dem["status"] == "diagnostic_raw_phase_lgr"].copy()
    by_id = solved.set_index("island_id")
    dem_map = island_map(label, by_id["dem_error_m"])
    rmse_map = island_map(label, by_id["rmse_phase_rad"])
    npair_map = island_map(label, by_id["n_pairs"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=220)
    add_image(axes[0, 0], dem_map, "SBAS DEM residual, m", "RdBu_r", center_zero=True)
    add_image(axes[0, 1], rmse_map, "Phase model RMSE, rad", "magma")
    add_image(axes[0, 2], npair_map, "Valid interferogram count", "viridis")

    if solved.empty:
        for ax in axes[1]:
            ax.set_axis_off()
    else:
        lo, hi = solved["dem_error_m"].quantile([0.02, 0.98])
        axes[1, 0].hist(solved["dem_error_m"].clip(lo, hi), bins=50, color="#4477aa", edgecolor="white")
        axes[1, 0].set_title("DEM residual distribution")
        axes[1, 0].set_xlabel("m, 2-98% clipped")
        axes[1, 0].set_ylabel("islands")

        axes[1, 1].scatter(
            residuals["bperp_m"],
            residuals["residual_phase_rad"],
            c=residuals["mean_coherence"],
            s=8,
            alpha=0.35,
            cmap="viridis",
        )
        axes[1, 1].axhline(0.0, color="black", lw=0.8)
        axes[1, 1].set_title("Residual phase vs perpendicular baseline")
        axes[1, 1].set_xlabel("Bperp, m")
        axes[1, 1].set_ylabel("observed - model, rad")

        axes[1, 2].scatter(
            residuals["dt_days"],
            residuals["residual_phase_rad"],
            c=residuals["mean_coherence"],
            s=8,
            alpha=0.35,
            cmap="viridis",
        )
        axes[1, 2].axhline(0.0, color="black", lw=0.8)
        axes[1, 2].set_title("Residual phase vs temporal baseline")
        axes[1, 2].set_xlabel("dt, days")
        axes[1, 2].set_ylabel("observed - model, rad")

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_examples(obs_resid: pd.DataFrame, dem: pd.DataFrame, out_base: Path, max_examples: int) -> list[int]:
    solved = dem[dem["status"] == "diagnostic_raw_phase_lgr"].copy()
    if solved.empty or obs_resid.empty:
        return []
    candidates = solved.sort_values(["rmse_phase_rad", "n_pairs"], ascending=[True, False]).head(max_examples * 4)
    ids = candidates["island_id"].head(max_examples).astype(int).tolist()
    if not ids:
        return []
    ncols = min(3, len(ids))
    nrows = int(math.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), dpi=220, squeeze=False)
    for ax, island_id in zip(axes.flat, ids):
        g = obs_resid[obs_resid["island_id"] == island_id].sort_values("dt_days")
        ax.scatter(g["dt_days"], g["mean_phase_rad"], c=g["bperp_m"], cmap="RdBu_r", s=24, label="observed")
        ax.plot(g["dt_days"], g["pred_phase_rad"], color="black", lw=1.2, label="model")
        info = solved[solved["island_id"] == island_id].iloc[0]
        ax.set_title(f"island {island_id}: DEM {info.dem_error_m:.2f} m, RMSE {info.rmse_phase_rad:.2f} rad")
        ax.set_xlabel("dt, days")
        ax.set_ylabel("phase, rad")
        ax.axhline(0.0, color="0.75", lw=0.8)
    for ax in axes.flat[len(ids) :]:
        ax.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--dem-errors", required=True)
    parser.add_argument("--island-label", required=True)
    parser.add_argument("--reference-par", required=True)
    parser.add_argument("--overview", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-examples", type=int, default=9)
    args = parser.parse_args()

    obs = pd.read_csv(args.observations)
    dem = pd.read_csv(args.dem_errors)
    label = np.load(args.island_label)
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    range_m = float(par["center_range_slc"])
    incidence = float(par["incidence_angle"])
    residuals = compute_residuals(obs, dem, wavelength, range_m, incidence)

    plot_overview(label, dem, residuals, Path(args.overview))
    example_ids = plot_examples(residuals, dem, Path(args.examples), args.max_examples)

    solved = dem[dem["status"] == "diagnostic_raw_phase_lgr"].copy()
    summary = {
        "observations": args.observations,
        "dem_errors": args.dem_errors,
        "islands_solved": int(len(solved)),
        "observation_rows_used": int(len(residuals)),
        "dem_error_median_m": float(solved["dem_error_m"].median()) if not solved.empty else None,
        "dem_error_p05_m": float(solved["dem_error_m"].quantile(0.05)) if not solved.empty else None,
        "dem_error_p95_m": float(solved["dem_error_m"].quantile(0.95)) if not solved.empty else None,
        "phase_rmse_median_rad": float(solved["rmse_phase_rad"].median()) if not solved.empty else None,
        "phase_residual_median_rad": float(residuals["residual_phase_rad"].median()) if not residuals.empty else None,
        "phase_residual_mad_rad": float(np.median(np.abs(residuals["residual_phase_rad"] - residuals["residual_phase_rad"].median())))
        if not residuals.empty
        else None,
        "example_island_ids": example_ids,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
