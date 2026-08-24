#!/usr/bin/env python3
"""Compute 21-scene amplitude dispersion and 48-pair mean coherence on island pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WIDTH = 10_000
LINES = 7_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--island-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--da-max", type=float, default=0.4)
    parser.add_argument("--coherence-min", type=float, default=0.75)
    args = parser.parse_args()

    points = np.load(args.island_points)
    keep = points["label"] > 0
    row = points["row"][keep].astype(np.int32)
    col = points["col"][keep].astype(np.int32)
    flat = row.astype(np.int64) * WIDTH + col

    dates = sorted(path.stem for path in args.crop_root.glob("*.slc"))
    amplitudes = np.empty((len(dates), len(flat)), dtype=np.float32)
    for index, date in enumerate(dates):
        if date == args.reference:
            path = args.crop_root / f"{date}.slc"
        else:
            path = args.rslc_root / f"{date}.rslc"
        raw = np.memmap(path, dtype=">i2", mode="r", shape=(WIDTH * LINES, 2))
        selected = raw[flat].astype(np.float32)
        amplitudes[index] = np.hypot(selected[:, 0], selected[:, 1])
        print(f"amplitude {index + 1}/{len(dates)} {date}", flush=True)

    amplitude_mean = amplitudes.mean(axis=0)
    amplitude_std = amplitudes.std(axis=0, ddof=1)
    amplitude_dispersion = np.divide(
        amplitude_std,
        amplitude_mean,
        out=np.full_like(amplitude_mean, np.nan),
        where=amplitude_mean > 0,
    )

    pair_paths = sorted(args.pair_root.glob("*.npz"))
    if len(pair_paths) != 48:
        raise RuntimeError(f"expected 48 pair observations, found {len(pair_paths)}")
    coherence_stack = np.empty((len(pair_paths), len(flat)), dtype=np.float32)
    for index, path in enumerate(pair_paths):
        with np.load(path) as pair:
            if not np.array_equal(pair["row"], row) or not np.array_equal(pair["col"], col):
                raise RuntimeError(f"point ordering mismatch: {path}")
            coherence_stack[index] = pair["coherence"]
    mean_coherence = np.nanmean(coherence_stack, axis=0)
    valid_pair_count = np.isfinite(coherence_stack).sum(axis=0).astype(np.int16)
    selected = (
        np.isfinite(amplitude_dispersion)
        & np.isfinite(mean_coherence)
        & (amplitude_dispersion <= args.da_max)
        & (mean_coherence >= args.coherence_min)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row=row,
        col=col,
        label=points["label"][keep].astype(np.int32),
        building_uid=points["building_uid"][keep].astype(np.int32),
        amplitude_mean=amplitude_mean,
        amplitude_std=amplitude_std,
        amplitude_dispersion=amplitude_dispersion,
        mean_coherence=mean_coherence,
        valid_pair_count=valid_pair_count,
        paper_quality_selected=selected,
    )
    labels = points["label"][keep].astype(np.int32)
    selected_labels = np.unique(labels[selected])
    summary = {
        "scene_count": len(dates),
        "pair_count": len(pair_paths),
        "island_pixel_count": int(len(flat)),
        "amplitude_dispersion_threshold_max": args.da_max,
        "mean_coherence_threshold_min": args.coherence_min,
        "selected_pixel_count": int(selected.sum()),
        "selected_pixel_fraction": float(selected.mean()),
        "total_island_count": int(np.unique(labels).size),
        "islands_with_at_least_one_selected_pixel": int(selected_labels.size),
        "amplitude_dispersion_quantiles": {
            str(q): float(np.nanquantile(amplitude_dispersion, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "mean_coherence_quantiles": {
            str(q): float(np.nanquantile(mean_coherence, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
