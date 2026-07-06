#!/usr/bin/env python3
"""Extract island-level phase observations from GAMMA differential interferograms."""

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


def read_fcomplex(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=">f4")
    if raw.size != rows * cols * 2:
        raise ValueError(f"{path} size {raw.size}, expected {rows * cols * 2}")
    raw = raw.reshape(rows, cols, 2)
    return raw[:, :, 0] + 1j * raw[:, :, 1]


def read_float(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=">f4")
    if raw.size != rows * cols:
        raise ValueError(f"{path} size {raw.size}, expected {rows * cols}")
    return raw.reshape(rows, cols)


def circular_mean(phases: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    z = np.sum(weights * np.exp(1j * phases))
    denom = np.sum(weights)
    if denom <= 0:
        return float("nan"), float("nan")
    return float(np.angle(z)), float(np.abs(z) / denom)


def plot_qc(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    axes[0].hist(df["mean_coherence"], bins=45, color="#4477aa", edgecolor="white")
    axes[0].set_title("GAMMA diff island coherence")
    axes[0].set_xlabel("mean coherence")
    axes[1].hist(df["phase_resultant_length"], bins=45, color="#44aa99", edgecolor="white")
    axes[1].set_title("Phase concentration")
    axes[1].set_xlabel("resultant length")
    per_pair = df.groupby(["master", "slave"]).agg(n=("island_id", "count"), coh=("mean_coherence", "mean")).reset_index()
    axes[2].bar(range(len(per_pair)), per_pair["coh"], color="#cc6677")
    axes[2].set_title("Mean coherence per pair")
    axes[2].set_xlabel("pair")
    axes[2].set_ylabel("coherence")
    axes[2].set_xticks([])
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
    parser.add_argument("--min-coherence", type=float, default=0.0)
    parser.add_argument("--output-csv", default="work/height/differential_island_phase_observations.csv")
    parser.add_argument("--summary", default="results/metadata/differential_island_phase_observations_summary.json")
    parser.add_argument("--figure", default="results/pic_all/17_gamma_differential_island_observations.png")
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
            phases = phase[keep]
            cohs = cc[keep]
            valid = np.isfinite(phases) & np.isfinite(cohs) & (np.abs(diff[keep]) > 0)
            if args.min_coherence > 0:
                valid &= cohs >= args.min_coherence
            if int(np.sum(valid)) < args.min_pixels:
                continue
            mean_phase, phase_r = circular_mean(phases[valid], np.clip(cohs[valid], 0.05, 1.0))
            rows_out.append(
                {
                    "master": master,
                    "slave": slave,
                    "island_id": island_id,
                    "pixel_count": int(np.sum(keep)),
                    "valid_pixel_count": int(np.sum(valid)),
                    "mean_phase_rad": mean_phase,
                    "phase_resultant_length": phase_r,
                    "mean_coherence": float(np.mean(cohs[valid])),
                    "median_coherence": float(np.median(cohs[valid])),
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
        "output_csv": args.output_csv,
        "figure": args.figure,
        "note": "GAMMA SLC_intf + ADF + phase_sim_orb(hgt=-) + sub_phase differential observations.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
