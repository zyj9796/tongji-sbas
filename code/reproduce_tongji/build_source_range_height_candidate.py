#!/usr/bin/env python3
"""Build a source-code-style height candidate from paper-method height points.

The thesis text describes IQR filtering followed by a median height. Several
bundled MATLAB scripts, however, compute island height as max(height)-min(height).
This script records that source-code-style diagnostic without using shapefile
height to fit or rescale the result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True)
    parser.add_argument("--output-points", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    input_path = Path(args.points)
    output_path = Path(args.output_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            h05 = number(row.get("building_height_p05_m"))
            h95 = number(row.get("building_height_p95_m"))
            d05 = number(row.get("dem_error_p05_m"))
            d95 = number(row.get("dem_error_p95_m"))
            if h05 is None or h95 is None:
                continue
            row["height_source_range_m"] = f"{h95 - h05:.9f}"
            if d05 is not None and d95 is not None:
                row["dem_error_source_range_m"] = f"{d95 - d05:.9f}"
            else:
                row["dem_error_source_range_m"] = ""
            row["source_range_note"] = "p95_minus_p05_building_height_from_pixel_lgr; source-code max-min analogue"
            rows.append(row)

    extra_fields = ["height_source_range_m", "dem_error_source_range_m", "source_range_note"]
    fields_out = fields + [field for field in extra_fields if field not in fields]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields_out)
        writer.writeheader()
        writer.writerows(rows)

    values = [number(row["height_source_range_m"]) for row in rows]
    values = [v for v in values if v is not None]
    summary = {
        "input_points": str(input_path),
        "output_points": str(output_path),
        "rows": len(rows),
        "height_field": "height_source_range_m",
        "method": "Source-code-style robust range diagnostic: building_height_p95_m - building_height_p05_m. No shapefile height fitting or scaling is applied.",
        "min_height_m": min(values) if values else None,
        "median_height_m": sorted(values)[len(values) // 2] if values else None,
        "max_height_m": max(values) if values else None,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
