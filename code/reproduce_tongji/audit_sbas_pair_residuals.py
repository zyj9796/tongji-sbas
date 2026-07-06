#!/usr/bin/env python3
"""Audit per-interferogram residuals from a pixel SBAS deformation run."""

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

from extract_gamma_differential_island_observations import read_float, read_fcomplex
from inventory_data import parse_gamma_par


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def robust_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def plot_pair_stats(stats: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=220)
    order = np.arange(len(stats))
    stats_sorted = stats.sort_values("residual_mad_rad", ascending=False).reset_index(drop=True)
    axes[0, 0].bar(order, stats_sorted["residual_mad_rad"], color="#cc6677")
    axes[0, 0].set_title("Pair residual MAD")
    axes[0, 0].set_xlabel("pairs sorted")
    axes[0, 0].set_ylabel("rad")

    axes[0, 1].scatter(stats["mean_coherence"], stats["residual_mad_rad"], c=stats["abs_bperp_m"], cmap="viridis", s=36)
    axes[0, 1].set_title("Residual vs coherence")
    axes[0, 1].set_xlabel("mean coherence")
    axes[0, 1].set_ylabel("residual MAD, rad")

    axes[1, 0].scatter(stats["dt_days"], stats["residual_mad_rad"], c=stats["abs_bperp_m"], cmap="magma", s=36)
    axes[1, 0].set_title("Residual vs temporal baseline")
    axes[1, 0].set_xlabel("days")
    axes[1, 0].set_ylabel("residual MAD, rad")

    worst = stats_sorted.head(12)
    axes[1, 1].barh(range(len(worst)), worst["residual_mad_rad"], color="#4477aa")
    axes[1, 1].set_yticks(range(len(worst)))
    axes[1, 1].set_yticklabels(worst["pair"], fontsize=7)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Worst residual pairs")
    axes[1, 1].set_xlabel("rad")

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--intf-root", required=True)
    parser.add_argument("--reference-par", required=True)
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--out-base", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pairs = pd.read_csv(args.pairs_csv)
    dates = sorted(set(pairs["master"].astype(str)).union(set(pairs["slave"].astype(str))))
    date_to_idx = {d: i for i, d in enumerate(dates)}
    disp_mm = np.load(run_dir / "arrays/los_displacement_timeseries_mm.npy")
    mask = np.load(run_dir / "arrays/monitored_pixel_mask.npy")
    if disp_mm.shape[1:] != mask.shape:
        raise ValueError(f"displacement shape {disp_mm.shape} does not match mask {mask.shape}")
    par = parse_gamma_par(Path(args.reference_par))
    wavelength = 299792458.0 / float(par["radar_frequency"])
    phase_scale = 4.0 * math.pi / wavelength

    rows = []
    flat_mask = mask.ravel()
    for pair_row in pairs.itertuples(index=False):
        master = str(pair_row.master)
        slave = str(pair_row.slave)
        pair = f"{master}_{slave}"
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        observed = np.angle(diff).ravel()[flat_mask]
        cohs = cc.ravel()[flat_mask]
        dm = disp_mm[date_to_idx[master]].ravel()[flat_mask] / 1000.0
        ds = disp_mm[date_to_idx[slave]].ravel()[flat_mask] / 1000.0
        predicted = phase_scale * (ds - dm)
        residual = wrap_phase(observed - predicted)
        valid = np.isfinite(residual) & np.isfinite(cohs) & (np.abs(diff).ravel()[flat_mask] > 0)
        residual = residual[valid]
        cohs = cohs[valid]
        rows.append(
            {
                "pair": pair,
                "master": master,
                "slave": slave,
                "dt_days": int(pair_row.dt_days),
                "bperp_m": float(pair_row.bperp_m),
                "abs_bperp_m": float(abs(pair_row.bperp_m)),
                "n_pixels": int(residual.size),
                "mean_coherence": float(np.nanmean(cohs)) if residual.size else np.nan,
                "residual_median_rad": float(np.nanmedian(residual)) if residual.size else np.nan,
                "residual_mad_rad": robust_mad(residual),
                "residual_abs_median_rad": float(np.nanmedian(np.abs(residual))) if residual.size else np.nan,
                "residual_abs_p90_rad": float(np.nanpercentile(np.abs(residual), 90)) if residual.size else np.nan,
                "residual_abs_p95_rad": float(np.nanpercentile(np.abs(residual), 95)) if residual.size else np.nan,
            }
        )

    stats = pd.DataFrame(rows).sort_values("residual_mad_rad", ascending=False)
    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_base.with_suffix(".csv"), index=False)
    plot_pair_stats(stats, out_base)
    summary = {
        "run_dir": args.run_dir,
        "pairs": int(len(stats)),
        "monitored_pixels": int(mask.sum()),
        "residual_mad_median_rad": float(stats["residual_mad_rad"].median()),
        "residual_mad_p90_rad": float(stats["residual_mad_rad"].quantile(0.9)),
        "residual_abs_p95_median_rad": float(stats["residual_abs_p95_rad"].median()),
        "worst_pairs": stats.head(10).to_dict(orient="records"),
        "outputs": {
            "csv": str(out_base.with_suffix(".csv")),
            "png": str(out_base.with_suffix(".png")),
            "svg": str(out_base.with_suffix(".svg")),
        },
    }
    out_base.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
