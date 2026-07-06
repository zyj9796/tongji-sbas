#!/usr/bin/env python3
"""Build an amplitude-dispersion raster from cropped SAR BMP quicklooks."""

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
from PIL import Image


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/tongji_rslc")
    parser.add_argument("--pattern", default="*.crop.bmp")
    parser.add_argument("--output-npy", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    parser.add_argument("--output-png", default="work/mli/amplitude_dispersion_crop_bmp.png")
    parser.add_argument("--summary", default="results/metadata/amplitude_dispersion_crop_bmp_summary.json")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No images matched {Path(args.input_dir) / args.pattern}")

    arrays: list[np.ndarray] = []
    skipped: list[dict[str, object]] = []
    shape = None
    for path in paths:
        arr = read_gray(path)
        if shape is None:
            shape = arr.shape
        if arr.shape != shape:
            skipped.append({"path": str(path), "shape": list(arr.shape), "reason": "shape_mismatch"})
            continue
        arrays.append(arr)
    if not arrays:
        raise RuntimeError("No compatible amplitude images were available")

    stack = np.stack(arrays, axis=0).astype(np.float32)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0)
    da = std / np.maximum(mean, 1.0)
    da = da.astype(np.float32)

    out_npy = Path(args.output_npy)
    out_png = Path(args.output_png)
    summary_path = Path(args.summary)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, da)

    valid = np.isfinite(da)
    lo, hi = np.percentile(da[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    show = np.clip((da - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)
    im = ax.imshow(show, cmap="magma_r", interpolation="nearest")
    ax.set_title("Amplitude dispersion proxy from SAR crop BMP stack")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="2-98% stretched DA")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    summary = {
        "input_dir": args.input_dir,
        "pattern": args.pattern,
        "image_count": len(paths),
        "used_count": len(arrays),
        "skipped_count": len(skipped),
        "shape_rows_cols": list(shape) if shape else None,
        "da_min": float(np.nanmin(da)),
        "da_p25": float(np.nanpercentile(da, 25)),
        "da_median": float(np.nanmedian(da)),
        "da_p75": float(np.nanpercentile(da, 75)),
        "da_p95": float(np.nanpercentile(da, 95)),
        "output_npy": str(out_npy),
        "output_png": str(out_png),
        "note": "Amplitude dispersion is computed from crop BMP quicklooks as std(amplitude)/mean(amplitude). It is a proxy for the thesis DA criterion, not a calibrated SLC amplitude product.",
        "skipped": skipped,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
