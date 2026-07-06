#!/usr/bin/env python3
"""Normalize Tongji building GeoJSON attributes for downstream SAR processing.

This is a dependency-light preparatory step. It does not perform geometry
projection or spatial operations; it only creates stable IDs and normalized
height-prior fields while preserving the original WGS84 geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_buildings(config: dict[str, Any], input_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    prior_field = config["height_aggregation"]["prior_height_field"]
    floor_field = config["height_aggregation"]["floor_field"]

    original_ids = []
    valid_prior = 0
    valid_floor = 0
    for uid, feature in enumerate(features, start=1):
        props = feature.setdefault("properties", {})
        original_id = props.get("Id")
        original_ids.append(str(original_id))
        height_prior = number_or_none(props.get(prior_field))
        floor_prior = number_or_none(props.get(floor_field))
        if height_prior is not None:
            valid_prior += 1
        if floor_prior is not None:
            valid_floor += 1
        props["uid"] = uid
        props["original_id"] = original_id
        props["height_prior_m"] = height_prior
        props["floor_prior"] = floor_prior
        props["height_insar_m"] = None
        props["height_final_m"] = height_prior
        props["height_source"] = "prior"
        props["qc_flag"] = "not_processed"
        props["n_points"] = 0
        props["n_islands"] = 0

    id_counts = Counter(original_ids)
    duplicate_ids = {k: v for k, v in id_counts.items() if v > 1}
    summary = {
        "input": str(input_path),
        "feature_count": len(features),
        "uid_start": 1 if features else None,
        "uid_end": len(features) if features else None,
        "original_id_unique_count": len(id_counts),
        "original_id_duplicate_count": len(duplicate_ids),
        "valid_height_prior_count": valid_prior,
        "valid_floor_prior_count": valid_floor,
        "note": "height_final_m is initialized from prior and must be replaced by InSAR aggregation when available.",
    }
    data["name"] = "tongji_buildings_wgs84_normalized_for_insar_height"
    data["normalization_summary"] = summary
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--input", default="", help="Input GeoJSON. Defaults to config paths.buildings_geojson.")
    parser.add_argument(
        "--output",
        default="results/geodata/tongji_buildings_normalized.geojson",
        help="Output normalized GeoJSON.",
    )
    parser.add_argument(
        "--summary",
        default="results/metadata/buildings_normalized_summary.json",
        help="Output summary JSON.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_path = Path(args.input) if args.input else Path(config["paths"]["buildings_geojson"])
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    data, summary = normalize_buildings(config, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
