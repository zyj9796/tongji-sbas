#!/usr/bin/env python3
"""Export a simple LOD1 OBJ model from final Tongji building heights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty]
    return []


def clean_ring(poly: Polygon) -> list[tuple[float, float]]:
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


def write_obj(
    gdf: gpd.GeoDataFrame,
    obj_path: Path,
    mtl_name: str,
    base_z: float,
) -> dict[str, int | float | str]:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = 0
    face_count = 0
    part_count = 0
    skipped_parts = 0
    minx = float(gdf.total_bounds[0])
    miny = float(gdf.total_bounds[1])

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Tongji final building LOD1 model\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl calibrated_height_buildings\n")

        for _, row in gdf.iterrows():
            height = row.get("height_m")
            if height is None or not np.isfinite(height):
                continue
            top_z = base_z + float(height)
            uid = row.get("uid", "unknown")
            for poly in polygon_parts(row.geometry):
                ring = clean_ring(poly)
                if len(ring) < 3:
                    skipped_parts += 1
                    continue
                part_count += 1
                f.write(f"o building_{uid}_part_{part_count}\n")
                bottom_idx = []
                top_idx = []
                for x, y in ring:
                    vertex_count += 1
                    bottom_idx.append(vertex_count)
                    f.write(f"v {x - minx:.3f} {y - miny:.3f} {base_z:.3f}\n")
                for x, y in ring:
                    vertex_count += 1
                    top_idx.append(vertex_count)
                    f.write(f"v {x - minx:.3f} {y - miny:.3f} {top_z:.3f}\n")
                f.write("f " + " ".join(str(i) for i in reversed(bottom_idx)) + "\n")
                f.write("f " + " ".join(str(i) for i in top_idx) + "\n")
                face_count += 2
                n = len(ring)
                for i in range(n):
                    j = (i + 1) % n
                    f.write(f"f {bottom_idx[i]} {bottom_idx[j]} {top_idx[j]} {top_idx[i]}\n")
                    face_count += 1

    return {
        "obj": str(obj_path),
        "crs": str(gdf.crs),
        "coordinate_origin_x": minx,
        "coordinate_origin_y": miny,
        "base_z_m": base_z,
        "buildings": int(len(gdf)),
        "exported_parts": part_count,
        "skipped_parts": skipped_parts,
        "vertices": vertex_count,
        "faces": face_count,
        "note": "OBJ x/y coordinates are projected meters offset from coordinate_origin_x/y; z uses base_z_m plus height_m.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/geodata/tongji_building_height_final.geojson")
    parser.add_argument("--output-obj", default="results/geodata/tongji_building_lod1_final.obj")
    parser.add_argument("--output-mtl", default="results/geodata/tongji_building_lod1_final.mtl")
    parser.add_argument("--summary", default="results/metadata/tongji_building_lod1_final_summary.json")
    parser.add_argument("--target-crs", default="EPSG:32651")
    parser.add_argument("--base-z", type=float, default=4.0)
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    if "height_m" not in gdf.columns:
        raise ValueError("input must contain height_m")
    gdf = gdf.to_crs(args.target_crs)

    mtl_path = Path(args.output_mtl)
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl calibrated_height_buildings",
                "Ka 0.800 0.800 0.800",
                "Kd 0.720 0.760 0.800",
                "Ks 0.080 0.080 0.080",
                "d 1.000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = write_obj(gdf, Path(args.output_obj), mtl_path.name, args.base_z)
    summary["input"] = args.input
    summary["mtl"] = args.output_mtl
    summary["height_min_m"] = float(gdf["height_m"].min())
    summary["height_median_m"] = float(gdf["height_m"].median())
    summary["height_max_m"] = float(gdf["height_m"].max())
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
