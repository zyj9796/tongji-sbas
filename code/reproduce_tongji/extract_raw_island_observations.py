#!/usr/bin/env python3
"""Extract island-level raw interferometric phase observations.

This is a diagnostic interface for the later unwrapping/height inversion step.
Inputs are the raw preview interferograms produced by
`build_interferogram_previews.py`; they are not flat-earth removed or
DEM-differenced and must not be treated as final height observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_RE = re.compile(r"(?P<master>\d{8})_(?P<slave>\d{8})_raw_intf_preview\.npz$")


def circular_mean(phases: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    if phases.size == 0:
        return float("nan"), float("nan")
    if weights is None:
        weights = np.ones_like(phases, dtype=np.float64)
    z = np.sum(weights * np.exp(1j * phases))
    denom = np.sum(weights)
    if denom <= 0:
        return float("nan"), float("nan")
    return float(np.angle(z)), float(np.abs(z) / denom)


def extract(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    islands = np.load(args.island_label)
    island_ids = np.asarray(sorted(int(x) for x in np.unique(islands) if x > 0), dtype=np.int32)
    if island_ids.size == 0:
        raise RuntimeError("No islands found")
    rows = []
    npz_paths = sorted(Path(args.intf_dir).glob("*_raw_intf_preview.npz"))
    for npz_path in npz_paths:
        match = PAIR_RE.match(npz_path.name)
        if match is None:
            continue
        data = np.load(npz_path)
        phase = data["phase"]
        coh = data["coherence"]
        if phase.shape != islands.shape:
            raise RuntimeError(f"Shape mismatch for {npz_path}: phase={phase.shape}, islands={islands.shape}")
        for island_id in island_ids:
            keep = islands == island_id
            if not np.any(keep):
                continue
            phases = phase[keep]
            cohs = coh[keep]
            valid = np.isfinite(phases) & np.isfinite(cohs)
            if args.min_coherence > 0:
                valid &= cohs >= args.min_coherence
            if int(np.sum(valid)) < args.min_pixels:
                continue
            mean_phase, phase_r = circular_mean(phases[valid], cohs[valid])
            rows.append(
                {
                    "master": match.group("master"),
                    "slave": match.group("slave"),
                    "island_id": int(island_id),
                    "pixel_count": int(np.sum(keep)),
                    "valid_pixel_count": int(np.sum(valid)),
                    "mean_phase_rad": mean_phase,
                    "phase_resultant_length": phase_r,
                    "mean_coherence": float(np.mean(cohs[valid])),
                    "median_coherence": float(np.median(cohs[valid])),
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "island_label": args.island_label,
        "intf_dir": args.intf_dir,
        "pairs_available": int(df[["master", "slave"]].drop_duplicates().shape[0]) if not df.empty else 0,
        "islands_total": int(island_ids.size),
        "observation_rows": int(len(df)),
        "islands_with_observations": int(df["island_id"].nunique()) if not df.empty else 0,
        "min_pixels": args.min_pixels,
        "min_coherence": args.min_coherence,
        "note": "Raw preview interferograms only; not flat-earth removed or DEM differenced.",
    }
    return df, summary


def plot_summary(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=220)
    if df.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[1].text(0.5, 0.5, "No observations", ha="center", va="center")
    else:
        axes[0].hist(df["mean_coherence"], bins=40, color="#4477aa", edgecolor="white")
        axes[0].set_title("Island mean coherence")
        axes[0].set_xlabel("coherence")
        axes[1].hist(df["phase_resultant_length"], bins=40, color="#44aa99", edgecolor="white")
        axes[1].set_title("Circular phase concentration")
        axes[1].set_xlabel("resultant length")
        per_pair = df.groupby(["master", "slave"]).size().reset_index(name="n")
        labels = [f"{r.master}-{r.slave}" for r in per_pair.itertuples()]
        axes[2].bar(range(len(per_pair)), per_pair["n"], color="#cc6677")
        axes[2].set_title("Island observations per pair")
        axes[2].set_xlabel("pair")
        axes[2].set_ylabel("count")
        axes[2].set_xticks(range(len(per_pair)))
        axes[2].set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--island-label", default="work/masks/island_label_touying_blue_bottom.npy")
    parser.add_argument("--intf-dir", default="work/intf_quicklooks")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--min-coherence", type=float, default=0.0)
    parser.add_argument("--output-csv", default="work/height/raw_island_phase_observations.csv")
    parser.add_argument("--summary", default="results/metadata/raw_island_phase_observations_summary.json")
    parser.add_argument("--figure", default="results/pic_all/14_raw_island_phase_observations.png")
    args = parser.parse_args()

    df, summary = extract(args)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_summary(df, Path(args.figure))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
