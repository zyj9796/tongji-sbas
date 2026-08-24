#!/usr/bin/env python3
"""Prepare an exact GAMMA SLC/DEM subwindow containing all target points."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


FULL_WIDTH = 10_000
FULL_LINES = 7_000


def gamma_environment() -> dict[str, str]:
    env = os.environ.copy()
    compat = Path("/tmp/gamma_gdal_compat")
    compat.mkdir(parents=True, exist_ok=True)
    link = compat / "libgdal.so.26"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to("/lib/libgdal.so.30")
    env["LD_LIBRARY_PATH"] = str(compat)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--height-rdc", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-crop-root", type=Path, required=True)
    parser.add_argument("--output-rslc-root", type=Path, required=True)
    parser.add_argument("--output-height", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--alignment", type=int, default=4)
    args = parser.parse_args()

    points = np.load(args.points)
    row, col = points["row"], points["col"]
    roff = max(int(col.min()) - args.margin, 0)
    loff = max(int(row.min()) - args.margin, 0)
    rend = min(int(col.max()) + args.margin + 1, FULL_WIDTH)
    lend = min(int(row.max()) + args.margin + 1, FULL_LINES)
    roff = (roff // args.alignment) * args.alignment
    loff = (loff // args.alignment) * args.alignment
    rend = min(((rend + args.alignment - 1) // args.alignment) * args.alignment, FULL_WIDTH)
    lend = min(((lend + args.alignment - 1) // args.alignment) * args.alignment, FULL_LINES)
    width, lines = rend - roff, lend - loff
    args.output_crop_root.mkdir(parents=True, exist_ok=True)
    args.output_rslc_root.mkdir(parents=True, exist_ok=True)
    env = gamma_environment()
    dates = sorted(path.stem for path in args.crop_root.glob("*.slc"))
    records = []
    for index, date in enumerate(dates, start=1):
        if date == args.reference:
            source = args.crop_root / f"{date}.slc"
            source_par = args.crop_root / f"{date}.slc.par"
            output = args.output_crop_root / f"{date}.slc"
            output_par = args.output_crop_root / f"{date}.slc.par"
        else:
            source = args.rslc_root / f"{date}.rslc"
            source_par = args.rslc_root / f"{date}.rslc.par"
            output = args.output_rslc_root / f"{date}.rslc"
            output_par = args.output_rslc_root / f"{date}.rslc.par"
        expected = width * lines * 4
        if not (output.exists() and output.stat().st_size == expected and output_par.exists()):
            command = [
                "SLC_copy", str(source), str(source_par), str(output), str(output_par),
                "4", "1.0", str(roff), str(width), str(loff), str(lines), "0", "0",
            ]
            result = subprocess.run(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if result.returncode or output.stat().st_size != expected:
                raise RuntimeError(f"SLC_copy failed for {date}")
        records.append({"date": date, "bytes": output.stat().st_size})
        print(f"[{index}/{len(dates)}] {date}", flush=True)

    height = np.memmap(args.height_rdc, dtype=">f4", mode="r", shape=(FULL_LINES, FULL_WIDTH))
    args.output_height.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(height[loff:lend, roff:rend], dtype=">f4").tofile(args.output_height)
    result = {
        "global_range_offset": roff, "global_azimuth_offset": loff,
        "width": width, "lines": lines, "margin_px": args.margin,
        "alignment_px": args.alignment,
        "target_global_row_range": [int(row.min()), int(row.max())],
        "target_global_col_range": [int(col.min()), int(col.max())],
        "full_pixel_count": FULL_WIDTH * FULL_LINES,
        "window_pixel_count": width * lines,
        "retained_fraction": width * lines / (FULL_WIDTH * FULL_LINES),
        "slc_records": records,
        "policy": "exact array subset; GAMMA SLC_copy updates timing/range metadata; margin protects ADF support",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "slc_records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
