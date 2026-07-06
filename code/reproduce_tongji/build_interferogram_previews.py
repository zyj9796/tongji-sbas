#!/usr/bin/env python3
"""Build quick interferogram phase/coherence previews from SCOMPLEX RSLC files.

These previews are diagnostic only. They are not flat-earth removed or DEM
differenced and must not be used as the final thesis-method height input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from inventory_data import parse_gamma_par


def read_scomplex(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=">i2")
    expected = rows * cols * 2
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} int16 values, expected {expected}")
    raw = raw.astype(np.float32).reshape(rows, cols, 2)
    return raw[:, :, 0] + 1j * raw[:, :, 1]


def local_coherence(master: np.ndarray, slave: np.ndarray, win: int) -> np.ndarray:
    intf = master * np.conj(slave)
    num_r = uniform_filter(np.real(intf), size=win)
    num_i = uniform_filter(np.imag(intf), size=win)
    p1 = uniform_filter(np.abs(master) ** 2, size=win)
    p2 = uniform_filter(np.abs(slave) ** 2, size=win)
    return np.sqrt(num_r * num_r + num_i * num_i) / np.sqrt(np.maximum(p1 * p2, 1e-6))


def stretch_amp(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    p2, p98 = np.percentile(arr[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    return np.clip((arr - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)


def read_pairs(path: Path, limit: int) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def plot_pair(out: Path, amp: np.ndarray, phase: np.ndarray, coh: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=220)
    axes[0].imshow(stretch_amp(amp), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Master amplitude")
    im1 = axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Raw interferometric phase")
    im2 = axes[2].imshow(coh, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Local coherence")
    for ax in axes:
        ax.set_xlabel("Range column")
        ax.set_ylabel("Azimuth row")
    fig.suptitle(title)
    fig.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.02)
    fig.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs.csv")
    parser.add_argument("--reference-date", default="20200708")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--coh-window", type=int, default=7)
    parser.add_argument("--out-dir", default="work/intf_quicklooks")
    parser.add_argument("--pic-dir", default="results/pic_all")
    parser.add_argument("--summary", default="results/metadata/interferogram_preview_summary.json")
    args = parser.parse_args()

    rslc_dir = Path(args.rslc_dir)
    par = parse_gamma_par(rslc_dir / f"{args.reference_date}.rslc.par")
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    pairs = read_pairs(Path(args.pairs_csv), args.limit)
    out_dir = Path(args.out_dir)
    pic_dir = Path(args.pic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for idx, pair in enumerate(pairs, start=1):
        master_date = pair["master"]
        slave_date = pair["slave"]
        master = read_scomplex(rslc_dir / f"{master_date}.rslc", rows, cols)
        slave = read_scomplex(rslc_dir / f"{slave_date}.rslc", rows, cols)
        intf = master * np.conj(slave)
        phase = np.angle(intf).astype(np.float32)
        coh = local_coherence(master, slave, args.coh_window).astype(np.float32)
        amp = np.abs(master).astype(np.float32)
        npz_path = out_dir / f"{master_date}_{slave_date}_raw_intf_preview.npz"
        np.savez_compressed(npz_path, phase=phase, coherence=coh)
        png_path = pic_dir / f"09_intf_preview_{idx:02d}_{master_date}_{slave_date}.png"
        plot_pair(png_path, amp, phase, coh, f"{master_date}-{slave_date}, dt={pair.get('dt_days')} days")
        summary_rows.append(
            {
                "master": master_date,
                "slave": slave_date,
                "dt_days": int(pair.get("dt_days", 0)),
                "phase_png": str(png_path),
                "npz": str(npz_path),
                "mean_coherence": float(np.nanmean(coh)),
                "median_coherence": float(np.nanmedian(coh)),
            }
        )

    summary = {
        "rows": rows,
        "cols": cols,
        "pairs_processed": len(summary_rows),
        "coh_window": args.coh_window,
        "items": summary_rows,
        "note": "Diagnostic raw interferograms from SCOMPLEX RSLC. Not flat-earth removed or DEM differenced.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
