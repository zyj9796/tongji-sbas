#!/usr/bin/env python3
"""Prepare a stable-ID building subset on the frozen GAMMA map extent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def gamma_par(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        token = value.strip().split()
        if token:
            try:
                values[key.strip()] = float(token[0])
            except ValueError:
                pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dem-par", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    par = gamma_par(args.dem_par)
    width, nlines = int(par["width"]), int(par["nlines"])
    west = par["corner_lon"]
    north = par["corner_lat"]
    east = west + width * par["post_lon"]
    south = north + nlines * par["post_lat"]
    extent = box(min(west, east), min(south, north), max(west, east), max(south, north))

    source = gpd.read_file(args.input)
    if source.crs is None or source.crs.to_epsg() != 4326:
        raise RuntimeError(f"expected EPSG:4326 building vector, got {source.crs}")
    source = source.reset_index(names="source_fid")
    valid = source.geometry.notna() & ~source.geometry.is_empty & source.geometry.is_valid
    selected = source.loc[valid & source.geometry.intersects(extent)].copy()
    selected["building_uid"] = selected["source_fid"].astype("int64") + 1
    if not selected["building_uid"].is_unique:
        raise RuntimeError("generated building_uid is not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    selected.to_file(args.output, layer="buildings", driver="GPKG")

    floor = selected["Floor"]
    summary = {
        "source": str(args.input),
        "output": str(args.output),
        "source_feature_count": int(len(source)),
        "source_valid_geometry_count": int(valid.sum()),
        "selected_feature_count": int(len(selected)),
        "source_id_unique_count": int(source["Id"].nunique(dropna=False)),
        "source_id_is_usable_unique_key": bool(source["Id"].is_unique),
        "stable_key": "building_uid = source_fid + 1",
        "map_extent_west_south_east_north": [west, south, east, north],
        "floor_min": int(floor.min()),
        "floor_median": float(floor.median()),
        "floor_max": int(floor.max()),
        "policy": "Floor is transferred as a discrete mask attribute only; it is not used to fill, select, constrain, or correct InSAR-derived heights.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
