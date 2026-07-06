#!/usr/bin/env python3
"""Dissolve adjacent building polygons with identical height attributes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def mode_or_first(values: pd.Series):
    vals = [v for v in values.tolist() if pd.notna(v)]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def connected_components(edges: dict[int, set[int]], nodes: list[int]) -> list[list[int]]:
    seen: set[int] = set()
    comps: list[list[int]] = []
    for node in nodes:
        if node in seen:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in edges.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        comps.append(comp)
    return comps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/shp/tongji_clip_rslc_extent.shp")
    parser.add_argument("--height-field", default="height")
    parser.add_argument("--projected-crs", default="EPSG:32651")
    parser.add_argument("--min-shared-boundary-m", type=float, default=0.20)
    parser.add_argument("--output-dir", default="data/shp/clean_equal_height")
    parser.add_argument("--figure", default="results/pic_all/projection_and_masks/103_equal_height_vector_cleanup.png")
    parser.add_argument("--summary", default="results/metadata/equal_height_vector_cleanup_summary.json")
    args = parser.parse_args()

    src_path = Path(args.input)
    gdf = gpd.read_file(src_path).reset_index(drop=True)
    if args.height_field not in gdf.columns:
        raise ValueError(f"Missing height field: {args.height_field}")
    gdf["source_uid"] = np.arange(1, len(gdf) + 1, dtype=int)
    gdf["geometry"] = gdf.geometry.make_valid()
    work = gdf.to_crs(args.projected_crs)

    edges: dict[int, set[int]] = defaultdict(set)
    shared_records: list[dict[str, object]] = []
    sindex = work.sindex
    for i, geom in enumerate(work.geometry):
        if geom is None or geom.is_empty:
            continue
        for j in sindex.query(geom, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            if work.at[i, args.height_field] != work.at[j, args.height_field]:
                continue
            inter = geom.boundary.intersection(work.geometry.iloc[j].boundary)
            shared_len = float(inter.length) if not inter.is_empty else 0.0
            if shared_len >= args.min_shared_boundary_m:
                edges[i].add(j)
                edges[j].add(i)
                shared_records.append(
                    {
                        "src_i": int(gdf.at[i, "source_uid"]),
                        "src_j": int(gdf.at[j, "source_uid"]),
                        "height": gdf.at[i, args.height_field],
                        "shared_boundary_m": shared_len,
                    }
                )

    comps = connected_components(edges, list(range(len(work))))
    rows = []
    for group_id, comp in enumerate(comps, start=1):
        sub = gdf.iloc[comp]
        geom = unary_union(work.geometry.iloc[comp].tolist())
        geom = gpd.GeoSeries([geom], crs=args.projected_crs).to_crs(gdf.crs).iloc[0]
        height_val = mode_or_first(sub[args.height_field])
        floor_val = mode_or_first(sub["Floor"]) if "Floor" in sub.columns else None
        id_val = mode_or_first(sub["Id"]) if "Id" in sub.columns else None
        rows.append(
            {
                "clean_id": group_id,
                "Id": id_val,
                "Floor": floor_val,
                args.height_field: height_val,
                "source_count": int(len(sub)),
                "source_uids": ";".join(str(int(v)) for v in sub["source_uid"].tolist()),
                "geometry": geom,
            }
        )
    clean = gpd.GeoDataFrame(rows, crs=gdf.crs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_out = out_dir / "tongji_clip_rslc_extent_equal_height_clean.shp"
    geojson_out = out_dir / "tongji_clip_rslc_extent_equal_height_clean.geojson"
    gpkg_out = out_dir / "tongji_clip_rslc_extent_equal_height_clean.gpkg"
    clean.to_file(shp_out)
    clean.to_file(geojson_out, driver="GeoJSON")
    clean.to_file(gpkg_out, driver="GPKG")

    shared_csv = out_dir / "equal_height_shared_edges_removed.csv"
    pd.DataFrame(shared_records).to_csv(shared_csv, index=False)

    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), dpi=220)
    gdf.plot(ax=axes[0], column=args.height_field, cmap="viridis", edgecolor="#222222", linewidth=0.12)
    axes[0].set_title(f"Original vectors: {len(gdf)} polygons")
    clean.plot(ax=axes[1], column=args.height_field, cmap="viridis", edgecolor="#222222", linewidth=0.12)
    axes[1].set_title(f"Equal-height cleaned: {len(clean)} polygons")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("Building vector cleanup: remove internal shared edges only where height is equal")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input": str(src_path),
        "height_field": args.height_field,
        "projected_crs_for_topology": args.projected_crs,
        "min_shared_boundary_m": args.min_shared_boundary_m,
        "original_polygons": int(len(gdf)),
        "cleaned_polygons": int(len(clean)),
        "merged_away_polygons": int(len(gdf) - len(clean)),
        "groups_with_multiple_sources": int((clean["source_count"] > 1).sum()),
        "equal_height_shared_edges_removed": int(len(shared_records)),
        "outputs": {
            "shp": str(shp_out),
            "geojson": str(geojson_out),
            "gpkg": str(gpkg_out),
            "shared_edges_csv": str(shared_csv),
            "figure_png": str(fig_path),
            "figure_svg": str(fig_path.with_suffix(".svg")),
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
