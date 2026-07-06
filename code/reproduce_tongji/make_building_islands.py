#!/usr/bin/env python3
"""Rasterize SAR-coordinate building projections and extract building islands."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import polygon

from inventory_data import parse_gamma_par


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polygon_rings(feature: dict) -> list[np.ndarray]:
    geom = feature.get("geometry", {})
    if geom.get("type") == "Polygon":
        rings = geom.get("coordinates", [])
        return [np.asarray(rings[0], dtype=np.float64)] if rings else []
    if geom.get("type") == "MultiPolygon":
        out = []
        for poly in geom.get("coordinates", []):
            if poly:
                out.append(np.asarray(poly[0], dtype=np.float64))
        return out
    return []


def rasterize_projection(path: Path, rows: int, cols: int, surface: str) -> tuple[np.ndarray, dict[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    uid_mask = np.zeros((rows, cols), dtype=np.int32)
    pixel_counts: Counter[int] = Counter()
    for idx, feature in enumerate(data.get("features", []), start=1):
        props = feature.get("properties", {})
        if surface and props.get("surface") != surface:
            continue
        uid = int(float(props.get("uid", props.get("fid", idx))))
        for ring in polygon_rings(feature):
            if ring.shape[0] < 3:
                continue
            x = ring[:, 0]
            y = ring[:, 1]
            rr, cc = polygon(y, x, shape=uid_mask.shape)
            if rr.size == 0:
                continue
            uid_mask[rr, cc] = uid
            pixel_counts[uid] += int(rr.size)
    return uid_mask, dict(pixel_counts)


def dbscan_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small DBSCAN implementation for 2D pixel coordinates."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    tree = cKDTree(points.astype(np.float64))
    neighbors = tree.query_ball_point(points, eps)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            continue
        labels[i] = cluster_id
        queue: deque[int] = deque(neighbors[i])
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_samples:
                    queue.extend(neighbors[j])
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def extract_islands(uid_mask: np.ndarray, config: dict) -> tuple[np.ndarray, list[dict]]:
    mask_cfg = config["mask"]
    kernel = int(mask_cfg["morphology_opening_kernel"])
    eps = float(mask_cfg["dbscan_eps_pixels"])
    min_samples = int(mask_cfg["dbscan_min_samples"])
    structure = np.ones((kernel, kernel), dtype=bool)
    opened = ndimage.binary_opening(uid_mask > 0, structure=structure)
    labeled, n_components = ndimage.label(opened, structure=np.ones((3, 3), dtype=np.int8))
    island_label = np.zeros(uid_mask.shape, dtype=np.int32)
    rows_out = []
    next_island = 1
    for component_id in range(1, n_components + 1):
        rr, cc = np.nonzero(labeled == component_id)
        if rr.size == 0:
            continue
        points = np.column_stack([rr, cc])
        labels = dbscan_points(points, eps=eps, min_samples=min_samples)
        for local_label in sorted(set(int(x) for x in labels if x >= 0)):
            keep = labels == local_label
            r = rr[keep]
            c = cc[keep]
            if r.size == 0:
                continue
            uids = uid_mask[r, c]
            uids = uids[uids > 0]
            uid_counts = Counter(int(x) for x in uids)
            primary_uid = uid_counts.most_common(1)[0][0] if uid_counts else 0
            island_label[r, c] = next_island
            rows_out.append(
                {
                    "island_id": next_island,
                    "component_id": component_id,
                    "dbscan_label": local_label,
                    "primary_uid": primary_uid,
                    "uid_count": len(uid_counts),
                    "pixel_count": int(r.size),
                    "row_min": int(r.min()),
                    "row_max": int(r.max()),
                    "col_min": int(c.min()),
                    "col_max": int(c.max()),
                }
            )
            next_island += 1
    return island_label, rows_out


def save_preview(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.max() <= 0:
        img = np.zeros(arr.shape, dtype=np.uint8)
    else:
        img = ((arr.astype(np.float64) % 251) / 250.0 * 255).astype(np.uint8)
        img[arr == 0] = 0
    Image.fromarray(img).save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "island_id",
        "component_id",
        "dbscan_label",
        "primary_uid",
        "uid_count",
        "pixel_count",
        "row_min",
        "row_max",
        "col_min",
        "col_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--projection-geojson", required=True)
    parser.add_argument("--surface", default="layover", help="Projection feature surface to rasterize; empty means all.")
    parser.add_argument("--reference-date", default="")
    parser.add_argument("--uid-mask", default="work/masks/building_uid_mask.npy")
    parser.add_argument("--island-label", default="work/masks/island_label.npy")
    parser.add_argument("--islands-csv", default="work/masks/islands.csv")
    parser.add_argument("--uid-preview", default="work/masks/building_uid_mask.png")
    parser.add_argument("--island-preview", default="work/masks/island_label.png")
    parser.add_argument("--summary", default="results/metadata/island_extraction_summary.json")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    date = args.reference_date or str(config["reference"]["date"])
    par = parse_gamma_par(Path(config["paths"]["rslc_dir"]) / f"{date}.rslc.par")
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    uid_mask, projection_pixel_counts = rasterize_projection(Path(args.projection_geojson), rows, cols, args.surface)
    island_label, islands = extract_islands(uid_mask, config)

    for path, arr in [(Path(args.uid_mask), uid_mask), (Path(args.island_label), island_label)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    save_preview(Path(args.uid_preview), uid_mask)
    save_preview(Path(args.island_preview), island_label)
    write_csv(Path(args.islands_csv), islands)
    summary = {
        "projection_geojson": args.projection_geojson,
        "surface": args.surface,
        "reference_date": date,
        "shape_rows_cols": [rows, cols],
        "projected_uid_count": len(projection_pixel_counts),
        "uid_mask_pixels": int(np.sum(uid_mask > 0)),
        "island_count": len(islands),
        "island_pixels": int(np.sum(island_label > 0)),
        "outputs": {
            "uid_mask": args.uid_mask,
            "island_label": args.island_label,
            "islands_csv": args.islands_csv,
            "uid_preview": args.uid_preview,
            "island_preview": args.island_preview,
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
