#!/usr/bin/env python3
"""Spatially unwrap GAMMA differential phase inside each building island."""

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
from skimage.restoration import unwrap_phase

from extract_gamma_differential_island_observations import read_fcomplex, read_float


def unwrap_island_phase(phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ma = np.ma.array(phase, mask=~mask)
    unw = unwrap_phase(ma)
    out = np.asarray(unw.filled(np.nan), dtype=np.float32)
    return out


def plot_qc(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    axes[0].hist(df["mean_coherence"], bins=45, color="#4477aa", edgecolor="white")
    axes[0].set_title("Unwrapped island coherence")
    axes[0].set_xlabel("mean coherence")
    axes[1].hist(df["phase_std_rad"].clip(0, df["phase_std_rad"].quantile(0.98)), bins=45, color="#cc6677", edgecolor="white")
    axes[1].set_title("Within-island phase std")
    axes[1].set_xlabel("rad, 98% clipped")
    per_pair = df.groupby(["master", "slave"]).size().reset_index(name="n")
    axes[2].bar(range(len(per_pair)), per_pair["n"], color="#44aa99")
    axes[2].set_title("Unwrapped islands per pair")
    axes[2].set_xticks([])
    axes[2].set_ylabel("count")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--intf-root", default="work/gamma_sbas/intf")
    parser.add_argument("--island-label", default="work/masks/island_label_touying_blue_bottom.npy")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--min-coherence", type=float, default=0.2)
    parser.add_argument("--output-csv", default="work/height/unwrapped_differential_island_phase_observations.csv")
    parser.add_argument("--summary", default="results/metadata/unwrapped_differential_island_phase_observations_summary.json")
    parser.add_argument("--figure", default="results/pic_all/18_gamma_unwrapped_island_observations.png")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    islands = np.load(args.island_label)
    island_ids = [int(x) for x in np.unique(islands) if x > 0]
    rows_out = []
    for pair_row in pairs.itertuples(index=False):
        master = str(pair_row.master)
        slave = str(pair_row.slave)
        pair = f"{master}_{slave}"
        pair_dir = Path(args.intf_root) / pair
        diff = read_fcomplex(pair_dir / f"{pair}.diff", args.rows, args.cols)
        cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
        phase = np.angle(diff)
        for island_id in island_ids:
            keep = islands == island_id
            if int(np.sum(keep)) < args.min_pixels:
                continue
            valid = keep & np.isfinite(phase) & np.isfinite(cc) & (cc >= args.min_coherence) & (np.abs(diff) > 0)
            if int(np.sum(valid)) < args.min_pixels:
                continue
            rr, cc_idx = np.nonzero(keep)
            r0, r1 = int(rr.min()), int(rr.max()) + 1
            c0, c1 = int(cc_idx.min()), int(cc_idx.max()) + 1
            patch_phase = phase[r0:r1, c0:c1]
            patch_valid = valid[r0:r1, c0:c1]
            patch_cc = cc[r0:r1, c0:c1]
            try:
                patch_unw = unwrap_island_phase(patch_phase, patch_valid)
            except Exception:
                continue
            vals = patch_unw[patch_valid]
            cohs = patch_cc[patch_valid]
            vals = vals[np.isfinite(vals)]
            if vals.size < args.min_pixels:
                continue
            rows_out.append(
                {
                    "master": master,
                    "slave": slave,
                    "island_id": island_id,
                    "pixel_count": int(np.sum(keep)),
                    "valid_pixel_count": int(vals.size),
                    "mean_phase_rad": float(np.mean(vals)),
                    "median_phase_rad": float(np.median(vals)),
                    "phase_std_rad": float(np.std(vals)),
                    "mean_coherence": float(np.mean(cohs)),
                    "median_coherence": float(np.median(cohs)),
                    "bperp_m": float(pair_row.bperp_m),
                    "dt_days": int(pair_row.dt_days),
                }
            )
    df = pd.DataFrame(rows_out)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    plot_qc(df, Path(args.figure))
    summary = {
        "pairs": int(pairs.shape[0]),
        "islands_total": int(len(island_ids)),
        "observation_rows": int(len(df)),
        "islands_with_observations": int(df["island_id"].nunique()) if not df.empty else 0,
        "min_pixels": args.min_pixels,
        "min_coherence": args.min_coherence,
        "output_csv": args.output_csv,
        "figure": args.figure,
        "note": "Spatial unwrap within each island using GAMMA differential interferograms.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
