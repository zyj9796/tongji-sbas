#!/usr/bin/env python3
"""Inventory BC3 SAFE ZIPs and the Tianjin building vector without extracting SLCs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import geopandas as gpd


PAPER_CENTER_LON = 117 + 12 / 60 + 36.10 / 3600
PAPER_CENTER_LAT = 39 + 7 / 60 + 35.90 / 3600


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_text(root: ET.Element, name: str, default: str = "") -> str:
    for elem in root.iter():
        if local_name(elem.tag) == name and elem.text:
            return elem.text.strip()
    return default


def all_text(root: ET.Element, name: str) -> list[str]:
    return [e.text.strip() for e in root.iter() if local_name(e.tag) == name and e.text]


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    for idx, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[idx - 1]
        if (y1 > lat) != (y2 > lat):
            cross_x = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < cross_x:
                inside = not inside
    return inside


def parse_manifest_footprint(text: str) -> list[tuple[float, float]]:
    match = re.search(r"<gml:coordinates>([^<]+)", text)
    if not match:
        raise ValueError("manifest has no gml:coordinates footprint")
    numbers: list[float] = []
    for token in match.group(1).split():
        numbers.extend(float(value) for value in token.split(",") if value)
    if len(numbers) % 2:
        raise ValueError("unexpected odd coordinate count in manifest footprint")
    # The supplied BC3 manifest stores latitude,longitude pairs.
    return [(numbers[i + 1], numbers[i]) for i in range(0, len(numbers), 2)]


def inventory_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest_name = next(name for name in names if name.endswith("manifest.safe"))
        annotation_name = next(
            name
            for name in names
            if "/annotation/" in name and name.endswith(".xml") and "/calibration/" not in name
        )
        calibration_name = next(
            (name for name in names if "/annotation/calibration/calibration-" in name), ""
        )
        measurement_name = next(name for name in names if "/measurement/" in name and name.endswith(".tiff"))
        manifest_text = archive.read(manifest_name).decode("utf-8", "replace")
        annotation_root = ET.fromstring(archive.read(annotation_name))
        measurement_info = archive.getinfo(measurement_name)

    footprint = parse_manifest_footprint(manifest_text)
    bbox = (
        min(point[0] for point in footprint),
        min(point[1] for point in footprint),
        max(point[0] for point in footprint),
        max(point[1] for point in footprint),
    )
    zip_match = re.match(
        r"BC3-SM-SLC-1S(?P<pol>[A-Z]{2})-(?P<date>\d{8})T(?P<time>\d{6})-"
        r"(?P<take>\d+)-(?P<order>\d+)-(?P<tail>[A-Z0-9]+)\.zip",
        path.name,
    )
    if not zip_match:
        raise ValueError(f"unexpected product name: {path.name}")
    groups = zip_match.groupdict()
    return {
        "date": groups["date"],
        "start_hhmmss": groups["time"],
        "mission_data_take": groups["take"],
        "order_id": groups["order"],
        "polarisation": groups["pol"],
        "zip_path": str(path),
        "zip_size_bytes": path.stat().st_size,
        "zip_sha256": "",  # Filled only for selected logical inputs to avoid hashing 70 GiB twice.
        "measurement_member": measurement_name,
        "measurement_size_bytes": measurement_info.file_size,
        "measurement_crc32": f"{measurement_info.CRC:08x}",
        "manifest_member": manifest_name,
        "annotation_member": annotation_name,
        "calibration_member": calibration_name,
        "sensing_start": first_text(annotation_root, "startTime"),
        "sensing_stop": first_text(annotation_root, "stopTime"),
        "pass": first_text(annotation_root, "pass"),
        "mode": first_text(annotation_root, "mode"),
        "swath": first_text(annotation_root, "swath"),
        "platform_heading_deg": float(first_text(annotation_root, "platformHeading")),
        "range_sampling_rate_hz": float(first_text(annotation_root, "rangeSamplingRate")),
        "radar_frequency_hz": float(first_text(annotation_root, "radarFrequency")),
        "range_pixel_spacing_m": float(first_text(annotation_root, "rangePixelSpacing")),
        "azimuth_pixel_spacing_m": float(first_text(annotation_root, "azimuthPixelSpacing")),
        "azimuth_time_interval_s": float(first_text(annotation_root, "azimuthTimeInterval")),
        "number_of_samples": int(first_text(annotation_root, "numberOfSamples")),
        "number_of_lines": int(first_text(annotation_root, "numberOfLines")),
        "mid_swath_incidence_deg": float(first_text(annotation_root, "incidenceAngleMidSwath")),
        "bbox_west": bbox[0],
        "bbox_south": bbox[1],
        "bbox_east": bbox[2],
        "bbox_north": bbox[3],
        "paper_center_intersects": point_in_polygon(PAPER_CENTER_LON, PAPER_CENTER_LAT, footprint),
        "footprint_lonlat": json.dumps(footprint, ensure_ascii=False),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector_summary(vector_path: Path) -> dict[str, Any]:
    frame = gpd.read_file(vector_path)
    invalid = int(((~frame.geometry.is_valid) & frame.geometry.notna()).sum())
    null_geometry = int(frame.geometry.isna().sum())
    floor = frame["Floor"] if "Floor" in frame else None
    return {
        "path": str(vector_path),
        "sha256_components": {
            component.suffix.lower(): sha256(component)
            for component in sorted(vector_path.parent.glob(vector_path.stem + ".*"))
            if component.is_file()
        },
        "crs": str(frame.crs),
        "feature_count": int(len(frame)),
        "geometry_types": dict(Counter(frame.geometry.geom_type.fillna("NULL"))),
        "invalid_geometry_count": invalid,
        "null_geometry_count": null_geometry,
        "bounds_west_south_east_north": [float(value) for value in frame.total_bounds],
        "columns": list(frame.columns),
        "floor_null_count": int(floor.isna().sum()) if floor is not None else None,
        "floor_min": int(floor.min()) if floor is not None else None,
        "floor_median": float(floor.median()) if floor is not None else None,
        "floor_max": int(floor.max()) if floor is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slc-root", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [inventory_zip(path) for path in sorted(args.slc_root.rglob("*.zip"))]
    if not rows:
        raise SystemExit("no ZIP files found")

    # Logical duplicates have the same sensing interval and identical measurement CRC.
    logical_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        logical_groups[(row["sensing_start"], row["sensing_stop"], row["measurement_crc32"])].append(row)
    for group in logical_groups.values():
        preferred = min(group, key=lambda row: (int(row["order_id"]), row["zip_path"]))
        for row in group:
            row["logical_duplicate_count"] = len(group)
            row["logical_preferred"] = row is preferred
            row["selected_center_segment"] = bool(row is preferred and row["paper_center_intersects"])

    selected = [row for row in rows if row["selected_center_segment"]]
    # Hash the 42 selected ZIPs once; this is intentionally deferred until after CRC-based deduplication.
    prior_hashes: dict[str, str] = {}
    prior_selected = args.output_dir / "bc3_selected_center_segments.csv"
    if prior_selected.exists():
        with prior_selected.open(encoding="utf-8", newline="") as stream:
            prior_hashes = {
                prior["zip_path"]: prior["zip_sha256"]
                for prior in csv.DictReader(stream)
                if prior.get("zip_sha256")
            }
    for row in selected:
        row["zip_sha256"] = prior_hashes.get(row["zip_path"]) or sha256(Path(row["zip_path"]))

    output = args.output_dir
    write_csv(output / "bc3_all_products.csv", rows)
    write_csv(output / "bc3_selected_center_segments.csv", selected)
    vector = vector_summary(args.vector)
    (output / "building_vector_summary.json").write_text(
        json.dumps(vector, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    selected_by_date = Counter(row["date"] for row in selected)
    summary = {
        "paper_center_lon": PAPER_CENTER_LON,
        "paper_center_lat": PAPER_CENTER_LAT,
        "zip_count": len(rows),
        "date_count": len(set(row["date"] for row in rows)),
        "logical_product_count": len(logical_groups),
        "duplicate_zip_count": len(rows) - len(logical_groups),
        "selected_center_segment_count": len(selected),
        "selected_segments_per_date": dict(sorted(selected_by_date.items())),
        "all_dates_have_two_selected_segments": bool(
            len(selected_by_date) == 21 and set(selected_by_date.values()) == {2}
        ),
        "total_zip_size_bytes": sum(row["zip_size_bytes"] for row in rows),
        "vector": vector,
    }
    (output / "input_inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
