#!/usr/bin/env python3
"""Inventory local inputs for the Tongji building-height reproduction workflow.

This script intentionally avoids third-party Python dependencies so it can run
before the geospatial/scientific environment is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PAR_KEYS = [
    "title",
    "sensor",
    "date",
    "range_samples",
    "azimuth_lines",
    "image_geometry",
    "range_pixel_spacing",
    "azimuth_pixel_spacing",
    "near_range_slc",
    "center_range_slc",
    "far_range_slc",
    "incidence_angle",
    "radar_frequency",
    "center_latitude",
    "center_longitude",
    "heading",
]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gamma_par(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {"path": str(path)}
    for key in PAR_KEYS:
        match = re.search(rf"^{re.escape(key)}:\s+(.+)$", text, re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).strip()
        value = raw.split()[0] if key != "title" else raw
        if key == "date":
            out[key] = raw
        else:
            out[key] = coerce_value(value)
    return out


def coerce_value(value: str) -> Any:
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", value, re.IGNORECASE):
            return float(value)
    except ValueError:
        pass
    return value


def run_text(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr)


def inventory_rslc(rslc_dir: Path, reference_date: str) -> dict[str, Any]:
    rslc_files = sorted(rslc_dir.glob("*.rslc"))
    par_files = sorted(rslc_dir.glob("*.rslc.par"))
    dates = sorted(p.name.split(".")[0] for p in par_files)
    ref_par = rslc_dir / f"{reference_date}.rslc.par"
    if not ref_par.exists() and par_files:
        ref_par = par_files[0]
    reference = parse_gamma_par(ref_par) if ref_par.exists() else {}
    return {
        "rslc_dir": str(rslc_dir),
        "rslc_count": len(rslc_files),
        "par_count": len(par_files),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "dates": dates,
        "reference_par": reference,
    }


def inventory_vector(path: Path) -> dict[str, Any]:
    code, text = run_text(["ogrinfo", "-al", "-so", str(path)])
    info: dict[str, Any] = {
        "path": str(path),
        "ogrinfo_exit_code": code,
        "ogrinfo_available": code != 127,
    }
    if code == 0:
        feature_match = re.search(r"Feature Count:\s+(\d+)", text)
        extent_match = re.search(r"Extent:\s+\(([^)]+)\) - \(([^)]+)\)", text)
        geom_match = re.search(r"Geometry:\s+(.+)", text)
        fields = []
        for line in text.splitlines():
            field_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", line.strip())
            if field_match and field_match.group(1) not in {"INFO", "Layer", "Metadata", "Geometry", "Extent"}:
                fields.append({"name": field_match.group(1), "type": field_match.group(2)})
        info.update(
            {
                "feature_count": int(feature_match.group(1)) if feature_match else None,
                "geometry": geom_match.group(1).strip() if geom_match else None,
                "extent": extent_match.groups() if extent_match else None,
                "fields": fields,
                "crs_hint": "EPSG:4326" if "ID[\"EPSG\",4326]" in text else None,
            }
        )
    else:
        info["error"] = text.strip()
    return info


def inventory_raster(path: Path) -> dict[str, Any]:
    code, text = run_text(["gdalinfo", str(path)])
    info: dict[str, Any] = {
        "path": str(path),
        "gdalinfo_exit_code": code,
        "gdalinfo_available": code != 127,
    }
    if code == 0:
        size_match = re.search(r"Size is\s+(\d+),\s+(\d+)", text)
        pixel_match = re.search(r"Pixel Size = \(([^,]+),([^)]+)\)", text)
        nodata_match = re.search(r"NoData Value=([^\s]+)", text)
        epsg_codes = [int(x) for x in re.findall(r'ID\["EPSG",(\d+)\]', text)]
        info.update(
            {
                "size": [int(size_match.group(1)), int(size_match.group(2))] if size_match else None,
                "pixel_size": [float(pixel_match.group(1)), float(pixel_match.group(2))] if pixel_match else None,
                "nodata": coerce_value(nodata_match.group(1)) if nodata_match else None,
                "epsg_codes": epsg_codes,
                "epsg_last_seen": epsg_codes[-1] if epsg_codes else None,
            }
        )
    else:
        info["error"] = text.strip()
    return info


def missing_products(config: dict[str, Any]) -> list[dict[str, str]]:
    work = Path(config["paths"]["work_dir"])
    checks = {
        "interferogram_pairs": work / "baselines" / "interferogram_pairs.csv",
        "building_projection_sar": work / "projection",
        "building_mask_sar": work / "masks" / "building_mask_sar.tif",
        "island_label_sar": work / "masks" / "island_label_sar.tif",
        "unwrapped_phase_stack": work / "unwrap",
        "building_height_table": Path(config["paths"]["results_dir"]) / "tables" / "tongji_building_height_insar.csv",
    }
    out = []
    for name, path in checks.items():
        if not path.exists():
            out.append({"product": name, "expected_path": str(path)})
    return out


def build_inventory(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    reference_date = str(config["reference"]["date"])
    return {
        "project": config["project"],
        "reference_date": reference_date,
        "rslc": inventory_rslc(Path(paths["rslc_dir"]), reference_date),
        "buildings": inventory_vector(Path(paths["buildings_shp"])),
        "dsm": inventory_raster(Path(paths["dsm_tif"])),
        "configured_thresholds": {
            "insar": config["insar"],
            "mask": config["mask"],
            "projection": config["projection"],
            "height_aggregation": config["height_aggregation"],
        },
        "missing_downstream_products": missing_products(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--write", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    inventory = build_inventory(config)
    payload = json.dumps(inventory, ensure_ascii=False, indent=2)
    print(payload)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
