#!/usr/bin/env python3
"""Plot GAMMA FCOMPLEX interferogram and FLOAT coherence quicklook."""

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


def stretch(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[valid], [2, 98])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", default="work/gamma_sbas/intf/20200525_20200616")
    parser.add_argument("--rows", type=int, default=630)
    parser.add_argument("--cols", type=int, default=900)
    parser.add_argument("--out", default="results/pic_all/16_gamma_pair_interferogram_quicklook.png")
    parser.add_argument("--summary", default="results/metadata/gamma_pair_interferogram_quicklook_summary.json")
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    pair = pair_dir.name
    intf = read_fcomplex(pair_dir / f"{pair}.int", args.rows, args.cols)
    filt = read_fcomplex(pair_dir / f"{pair}.adf", args.rows, args.cols)
    cc = read_float(pair_dir / f"{pair}.cc", args.rows, args.cols)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=220)
    axes[0].imshow(stretch(np.abs(intf)), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("GAMMA int amplitude")
    axes[1].imshow(np.angle(intf), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("GAMMA int phase")
    axes[2].imshow(np.angle(filt), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("ADF filtered phase")
    im = axes[3].imshow(cc, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("ADF coherence")
    for ax in axes:
        ax.set_xlabel("Range column")
        ax.set_ylabel("Azimuth row")
    fig.colorbar(im, ax=axes[3], fraction=0.035, pad=0.02)
    fig.suptitle(pair)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    summary = {
        "pair": pair,
        "pair_dir": str(pair_dir),
        "rows": args.rows,
        "cols": args.cols,
        "mean_coherence": float(np.nanmean(cc)),
        "median_coherence": float(np.nanmedian(cc)),
        "figure": args.out,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
