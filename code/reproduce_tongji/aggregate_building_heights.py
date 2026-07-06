#!/usr/bin/env python3
"""Aggregate pixel-level InSAR heights back to normalized WGS84 buildings.

Expected points CSV columns:

- uid: building UID from `prepare_buildings_geojson.py`
- height_m: relative building height sample in meters
- optional island_id, coh_mean, amplitude_dispersion

The aggregation follows the thesis rule: remove IQR outliers and use the median
as the building-level height. This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def robust_height(values: list[float], iqr_multiplier: float) -> dict[str, Any]:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return {"height": None, "n_used": 0, "iqr": None, "q1": None, "q3": None}
    q1 = percentile(clean, 0.25)
    q3 = percentile(clean, 0.75)
    iqr = q3 - q1
    lo = q1 - iqr_multiplier * iqr
    hi = q3 + iqr_multiplier * iqr
    kept = [v for v in clean if lo <= v <= hi]
    if not kept:
        kept = clean
    return {
        "height": float(median(kept)),
        "n_used": len(kept),
        "iqr": float(iqr),
        "q1": float(q1),
        "q3": float(q3),
    }


def read_points(path: Path, height_field: str) -> dict[int, list[dict[str, Any]]]:
    by_uid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_uid
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid_value = row.get("uid")
            height_value = number_or_none(row.get(height_field))
            if uid_value in (None, "") or height_value is None:
                continue
            uid = int(float(uid_value))
            row["_height_value"] = height_value
            by_uid[uid].append(row)
    return by_uid


def summarize_optional(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [number_or_none(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return float(median(vals)) if vals else None


def aggregate(
    buildings_path: Path,
    points_path: Path,
    height_field: str,
    iqr_multiplier: float,
    allow_missing_points: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not points_path.exists() and not allow_missing_points:
        raise FileNotFoundError(points_path)
    buildings = json.loads(buildings_path.read_text(encoding="utf-8"))
    points = read_points(points_path, height_field)
    rows_out = []
    source_counts = defaultdict(int)

    for feature in buildings.get("features", []):
        props = feature.setdefault("properties", {})
        uid = int(props["uid"])
        samples = points.get(uid, [])
        heights = [float(row["_height_value"]) for row in samples]
        result = robust_height(heights, iqr_multiplier)
        prior = number_or_none(props.get("height_prior_m"))
        island_ids = {row.get("island_id") for row in samples if row.get("island_id") not in (None, "")}
        if result["height"] is None:
            final_height = prior
            source = "prior" if prior is not None else "invalid"
            qc = "no_insar_points"
        else:
            final_height = result["height"]
            source = "insar"
            qc = "ok"
            if prior is not None and abs(final_height - prior) > max(20.0, 0.5 * max(prior, 1.0)):
                qc = "prior_mismatch"

        props["height_insar_m"] = result["height"]
        props["height_final_m"] = final_height
        props["height_source"] = source
        props["qc_flag"] = qc
        props["n_points"] = len(samples)
        props["n_points_used"] = result["n_used"]
        props["n_islands"] = len(island_ids)
        props["height_iqr_m"] = result["iqr"]
        props["median_coh"] = summarize_optional(samples, "coh_mean")
        props["median_da"] = summarize_optional(samples, "amplitude_dispersion")
        source_counts[source] += 1
        rows_out.append(
            {
                "uid": uid,
                "height_prior_m": prior,
                "height_insar_m": result["height"],
                "height_final_m": final_height,
                "height_source": source,
                "qc_flag": qc,
                "n_points": len(samples),
                "n_points_used": result["n_used"],
                "n_islands": len(island_ids),
                "height_iqr_m": result["iqr"],
                "median_coh": props["median_coh"],
                "median_da": props["median_da"],
            }
        )

    summary = {
        "buildings": len(buildings.get("features", [])),
        "points_file": str(points_path),
        "points_building_count": len(points),
        "height_field": height_field,
        "iqr_multiplier": iqr_multiplier,
        "source_counts": dict(sorted(source_counts.items())),
    }
    return buildings, rows_out, summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", default="results/geodata/tongji_buildings_normalized.geojson")
    parser.add_argument("--points", default="work/height/height_points.csv")
    parser.add_argument("--height-field", default="height_m")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--allow-missing-points", action="store_true")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_insar.geojson")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_insar.csv")
    parser.add_argument("--summary", default="results/metadata/building_height_aggregation_summary.json")
    args = parser.parse_args()

    buildings, rows, summary = aggregate(
        Path(args.buildings),
        Path(args.points),
        args.height_field,
        args.iqr_multiplier,
        args.allow_missing_points,
    )
    out_geojson = Path(args.output_geojson)
    out_summary = Path(args.summary)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    write_rows(Path(args.output_csv), rows)
    out_geojson.write_text(json.dumps(buildings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
