#!/usr/bin/env python3
"""Compute 21-scene amplitude statistics at an arbitrary radar point set."""

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
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    args = parser.parse_args()

    points = np.load(args.points)
    keep = points["label"] > 0
    row = points["row"][keep].astype(np.int32)
    col = points["col"][keep].astype(np.int32)
    flat = row.astype(np.int64) * WIDTH + col
    dates = sorted(path.stem for path in args.crop_root.glob("*.slc"))
    amplitude = np.empty((len(dates), len(flat)), dtype=np.float32)
    for index, date in enumerate(dates):
        path = args.crop_root / f"{date}.slc" if date == args.reference else args.rslc_root / f"{date}.rslc"
        raw = np.memmap(path, dtype=">i2", mode="r", shape=(WIDTH * LINES, 2))
        value = raw[flat].astype(np.float32)
        amplitude[index] = np.hypot(value[:, 0], value[:, 1])
        print(f"[{index + 1}/{len(dates)}] {date}", flush=True)
    mean = amplitude.mean(axis=0)
    std = amplitude.std(axis=0, ddof=1)
    dispersion = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean > 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, row=row, col=col, label=points["label"][keep],
        building_uid=points["building_uid"][keep], amplitude_mean=mean,
        amplitude_std=std, amplitude_dispersion=dispersion,
    )
    summary = {
        "scene_count": len(dates), "point_count": int(len(row)),
        "amplitude_dispersion_quantiles": {
            str(q): float(np.nanquantile(dispersion, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
