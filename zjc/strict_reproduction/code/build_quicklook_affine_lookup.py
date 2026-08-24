#!/usr/bin/env python3
"""Build a GAMMA big-endian FCOMPLEX lookup from a validated quicklook affine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=5000)
    parser.add_argument("--lines", type=int, default=1166)
    parser.add_argument("--range-looks", type=int, default=2)
    parser.add_argument("--azimuth-looks", type=int, default=6)
    parser.add_argument("--reference-range-origin", type=int, default=2400)
    parser.add_argument("--reference-azimuth-origin", type=int, default=10000)
    args = parser.parse_args()

    records = json.loads(args.offsets.read_text(encoding="utf-8"))
    record = next(item for item in records if item["date"] == args.date)
    matrix = np.asarray(record["affine_reference_to_date_downsampled"], dtype=np.float64)
    source_x0 = float(record["range_offset_px"])
    source_y0 = float(record["azimuth_offset_px"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(args.width, dtype=np.float64) * args.range_looks + (args.range_looks - 1) / 2
    x += args.reference_range_origin
    with args.output.open("wb") as stream:
        for y_index in range(args.lines):
            y = y_index * args.azimuth_looks + (args.azimuth_looks - 1) / 2
            y += args.reference_azimuth_origin
            # The affine was estimated on images sampled every 16 native pixels.
            source_x = matrix[0, 0] * x + matrix[0, 1] * y + 16.0 * matrix[0, 2]
            source_y = matrix[1, 0] * x + matrix[1, 1] * y + 16.0 * matrix[1, 2]
            source_x -= source_x0
            source_y -= source_y0
            lookup_x = (source_x - (args.range_looks - 1) / 2) / args.range_looks
            lookup_y = (source_y - (args.azimuth_looks - 1) / 2) / args.azimuth_looks
            row = (lookup_x + 1j * lookup_y).astype(">c8")
            stream.write(row.tobytes())

    expected = args.width * args.lines * 8
    if args.output.stat().st_size != expected:
        raise RuntimeError(f"lookup size mismatch: {args.output.stat().st_size} != {expected}")
    print(
        f"{args.date}: {args.width} x {args.lines}, {expected} bytes, "
        f"inliers={record['ransac_inliers']}/{record['feature_matches']}"
    )


if __name__ == "__main__":
    main()
