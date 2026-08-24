#!/usr/bin/env python3
"""Select spatially distributed stable non-building pixels for phase referencing only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WIDTH = 10_000
LINES = 7_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-uid-rdc", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--mli-grid-step", type=int, default=2)
    parser.add_argument("--da-max", type=float, default=0.25)
    args = parser.parse_args()

    # Candidate locations are centers of 2x6 multilook cells on a regular sparse grid.
    mli_rows, mli_cols = np.meshgrid(
        np.arange(4, 1166, args.mli_grid_step, dtype=np.int32),
        np.arange(4, 5000, args.mli_grid_step, dtype=np.int32),
        indexing="ij",
    )
    mli_row = mli_rows.ravel()
    mli_col = mli_cols.ravel()
    row = mli_row * 6 + 2
    col = mli_col * 2
    inside = row < LINES
    row, col, mli_row, mli_col = row[inside], col[inside], mli_row[inside], mli_col[inside]

    uid = np.memmap(args.building_uid_rdc, dtype=">f4", mode="r", shape=(LINES, WIDTH))
    ground = np.ones(len(row), dtype=bool)
    # Require a 7x7 full-resolution neighborhood free of mapped buildings.
    for dr in (-3, 0, 3):
        for dc in (-3, 0, 3):
            ground &= uid[row + dr, col + dc] == 0
    row, col, mli_row, mli_col = row[ground], col[ground], mli_row[ground], mli_col[ground]
    flat = row.astype(np.int64) * WIDTH + col

    dates = sorted(path.stem for path in args.crop_root.glob("*.slc"))
    amplitudes = np.empty((len(dates), len(flat)), dtype=np.float32)
    for index, date in enumerate(dates):
        path = (
            args.crop_root / f"{date}.slc"
            if date == args.reference
            else args.rslc_root / f"{date}.rslc"
        )
        raw = np.memmap(path, dtype=">i2", mode="r", shape=(WIDTH * LINES, 2))
        values = raw[flat].astype(np.float32)
        amplitudes[index] = np.hypot(values[:, 0], values[:, 1])
    mean = amplitudes.mean(axis=0)
    da = amplitudes.std(axis=0, ddof=1) / np.maximum(mean, np.finfo(np.float32).tiny)
    # A median-amplitude floor avoids noise-only pixels; it is a phase-reference QA rule,
    # not a building-height prior or building-observation threshold.
    amplitude_floor = float(np.nanmedian(mean))
    stable = np.isfinite(da) & (da <= args.da_max) & (mean >= amplitude_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row=row[stable],
        col=col[stable],
        mli_row=mli_row[stable],
        mli_col=mli_col[stable],
        amplitude_mean=mean[stable],
        amplitude_dispersion=da[stable],
    )
    summary = {
        "purpose": "phase reference only; points never enter building height aggregation",
        "mli_grid_step": args.mli_grid_step,
        "nonbuilding_candidate_count": int(len(row)),
        "amplitude_dispersion_max": args.da_max,
        "amplitude_mean_floor": amplitude_floor,
        "selected_reference_pixel_count": int(stable.sum()),
        "selected_fraction": float(stable.mean()),
        "row_range": [int(row[stable].min()), int(row[stable].max())],
        "col_range": [int(col[stable].min()), int(col[stable].max())],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
