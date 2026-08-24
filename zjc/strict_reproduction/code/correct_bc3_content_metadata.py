#!/usr/bin/env python3
"""Correct BC3 crop metadata when raster content is displaced inside a product.

The source orbit/state vectors are retained.  Only the image time and slant-range
origin are shifted by the independently measured quicklook displacement so that
range-Doppler geometry describes the pixels actually stored in the SLC.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = (
    "start_time",
    "center_time",
    "near_range_slc",
    "center_range_slc",
    "far_range_slc",
)


def value(text: str, key: str) -> float:
    match = re.search(rf"(?m)^{re.escape(key)}:\s+([-+0-9.eE]+)", text)
    if not match:
        raise KeyError(f"{key} missing from parameter file")
    return float(match.group(1))


def replace_value(text: str, key: str, new_value: float) -> str:
    pattern = rf"(?m)^({re.escape(key)}:\s+)([-+0-9.eE]+)(.*)$"
    replacement = rf"\g<1>{new_value:.9f}\g<3>"
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise KeyError(f"could not replace {key}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--par-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--nominal-range-offset", type=int, default=2400)
    parser.add_argument("--nominal-azimuth-offset", type=int, default=10000)
    args = parser.parse_args()

    offsets = {
        item["date"]: item
        for item in json.loads(args.offsets.read_text(encoding="utf-8"))
    }
    audit = []
    for date in args.dates:
        item = offsets[date]
        par = args.par_root / f"{date}.slc.par"
        source_text = par.read_text(encoding="utf-8", errors="strict")
        line_time = value(source_text, "azimuth_line_time")
        range_spacing = value(source_text, "range_pixel_spacing")
        az_delta = args.nominal_azimuth_offset - int(item["azimuth_offset_px"])
        range_delta = args.nominal_range_offset - int(item["range_offset_px"])
        time_shift = az_delta * line_time
        range_shift = range_delta * range_spacing
        before = {key: value(source_text, key) for key in FIELDS}
        after = dict(before)
        for key in ("start_time", "center_time"):
            after[key] += time_shift
        for key in ("near_range_slc", "center_range_slc", "far_range_slc"):
            after[key] += range_shift
        corrected = source_text
        for key in FIELDS:
            corrected = replace_value(corrected, key, after[key])
        par.write_text(corrected, encoding="utf-8")
        audit.append(
            {
                "date": date,
                "method": "quicklook_RANSAC_displacement_to_range_Doppler_metadata",
                "source_pixels_unchanged": True,
                "orbit_state_vectors_unchanged": True,
                "adaptive_crop_range_azimuth_px": [
                    int(item["range_offset_px"]),
                    int(item["azimuth_offset_px"]),
                ],
                "nominal_metadata_crop_range_azimuth_px": [
                    args.nominal_range_offset,
                    args.nominal_azimuth_offset,
                ],
                "time_shift_seconds": time_shift,
                "slant_range_shift_metres": range_shift,
                "before": before,
                "after": after,
                "feature_matches": item["feature_matches"],
                "ransac_inliers": item["ransac_inliers"],
                "inlier_fraction": item["inlier_fraction"],
            }
        )
        print(f"{date}: time {time_shift:+.9f} s, range {range_shift:+.6f} m")

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
