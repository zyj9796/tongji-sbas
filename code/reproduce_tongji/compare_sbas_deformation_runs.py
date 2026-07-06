#!/usr/bin/env python3
"""Compare two SBAS deformation monitoring runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(path: Path) -> dict:
    return {
        "summary": json.loads((path / "metadata/summary.json").read_text(encoding="utf-8")),
        "mask": np.load(path / "arrays/monitored_pixel_mask.npy"),
        "vel": np.load(path / "arrays/los_velocity_mm_per_year.npy"),
        "final": np.load(path / "arrays/cumulative_los_displacement_mm.npy"),
        "rmse": np.load(path / "arrays/phase_rmse_rad.npy"),
        "coh": np.load(path / "arrays/mean_coherence.npy"),
    }


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [2, 98])
    if lo == hi or abs(hi - lo) < 1e-3:
        med = float(np.nanmedian(finite))
        lo = med - 1.0
        hi = med + 1.0
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--opt-run", required=True)
    parser.add_argument("--out-base", required=True)
    args = parser.parse_args()

    base = load_run(Path(args.base_run))
    opt = load_run(Path(args.opt_run))
    common = base["mask"] & opt["mask"]
    dropped = base["mask"] & ~opt["mask"]
    added = opt["mask"] & ~base["mask"]
    vel_diff = np.where(common, opt["vel"] - base["vel"], np.nan)
    rmse_diff = np.where(common, opt["rmse"] - base["rmse"], np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=220)
    status = np.zeros(base["mask"].shape, dtype=np.float32)
    status[base["mask"]] = 1
    status[opt["mask"]] = 2
    status[common] = 3
    im = axes[0, 0].imshow(status, cmap="viridis", vmin=0, vmax=3)
    axes[0, 0].set_title("Pixel selection: 1 base, 2 opt, 3 common")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

    lo, hi = robust_limits(vel_diff)
    im = axes[0, 1].imshow(vel_diff, cmap="RdBu_r", vmin=lo, vmax=hi)
    axes[0, 1].set_title("Velocity difference opt-base, mm/yr")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)

    lo, hi = robust_limits(rmse_diff)
    im = axes[0, 2].imshow(rmse_diff, cmap="RdBu_r", vmin=lo, vmax=hi)
    axes[0, 2].set_title("RMSE difference opt-base, rad")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)

    axes[1, 0].hist(base["rmse"][base["mask"]], bins=70, alpha=0.55, label="base", color="#4477aa")
    axes[1, 0].hist(opt["rmse"][opt["mask"]], bins=70, alpha=0.55, label="opt", color="#cc6677")
    axes[1, 0].set_title("Phase RMSE")
    axes[1, 0].set_xlabel("rad")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].hist(base["vel"][base["mask"]], bins=80, alpha=0.55, label="base", color="#4477aa")
    axes[1, 1].hist(opt["vel"][opt["mask"]], bins=80, alpha=0.55, label="opt", color="#cc6677")
    axes[1, 1].set_title("Velocity")
    axes[1, 1].set_xlabel("mm/yr")
    axes[1, 1].legend(frameon=False)

    axes[1, 2].scatter(base["vel"][common], opt["vel"][common], s=2, alpha=0.12, color="#117733")
    axes[1, 2].axline((0, 0), slope=1, color="black", lw=0.8)
    axes[1, 2].set_title("Common-pixel velocity agreement")
    axes[1, 2].set_xlabel("base, mm/yr")
    axes[1, 2].set_ylabel("opt, mm/yr")

    fig.tight_layout()
    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)

    payload = {
        "base_run": args.base_run,
        "opt_run": args.opt_run,
        "base_pixels": int(base["mask"].sum()),
        "opt_pixels": int(opt["mask"].sum()),
        "common_pixels": int(common.sum()),
        "dropped_pixels": int(dropped.sum()),
        "added_pixels": int(added.sum()),
        "base_rmse_median_rad": float(np.nanmedian(base["rmse"][base["mask"]])),
        "opt_rmse_median_rad": float(np.nanmedian(opt["rmse"][opt["mask"]])),
        "base_rmse_p95_rad": float(np.nanpercentile(base["rmse"][base["mask"]], 95)),
        "opt_rmse_p95_rad": float(np.nanpercentile(opt["rmse"][opt["mask"]], 95)),
        "common_velocity_diff_median_mm_yr": float(np.nanmedian(vel_diff)),
        "common_velocity_diff_mad_mm_yr": float(np.nanmedian(np.abs(vel_diff - np.nanmedian(vel_diff)))),
        "common_rmse_diff_median_rad": float(np.nanmedian(rmse_diff)),
    }
    out_base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
