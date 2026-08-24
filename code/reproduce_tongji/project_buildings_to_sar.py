#!/usr/bin/env python3
"""Project prepared WGS84 building roof polygons into SAR row/column coordinates.

This implements a local zero-Doppler range-Doppler projection using the orbit
state vectors and range timing stored in the GAMMA `.rslc.par` file. It is the
project-local replacement for the non-standalone projection helpers referenced
by `touying_roof_workflow`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy.optimize import brentq, minimize_scalar
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gamma_par(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {"path": str(path)}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        values = rest.strip().split()
        if not values:
            continue
        if key == "title":
            out[key] = rest.strip()
        elif key == "date":
            out[key] = rest.strip()
        else:
            out[key] = coerce(values[0])

    nsv = int(out.get("number_of_state_vectors", 0))
    positions = []
    velocities = []
    for i in range(1, nsv + 1):
        p_match = re.search(
            rf"^state_vector_position_{i}:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text,
            re.MULTILINE,
        )
        v_match = re.search(
            rf"^state_vector_velocity_{i}:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
            text,
            re.MULTILINE,
        )
        if p_match and v_match:
            positions.append([float(x) for x in p_match.groups()])
            velocities.append([float(x) for x in v_match.groups()])
    first_t = float(out["time_of_first_state_vector"])
    interval = float(out["state_vector_interval"])
    out["state_times"] = np.asarray([first_t + i * interval for i in range(len(positions))], dtype=np.float64)
    out["state_positions"] = np.asarray(positions, dtype=np.float64)
    out["state_velocities"] = np.asarray(velocities, dtype=np.float64)
    return out


def coerce(value: str) -> Any:
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


class ZeroDopplerProjector:
    def __init__(self, par: dict[str, Any]):
        self.par = par
        self.times = par["state_times"]
        self.positions = par["state_positions"]
        self.velocities = par["state_velocities"]
        self.start_time = float(par["start_time"])
        self.end_time = float(par["end_time"])
        self.azimuth_line_time = float(par["azimuth_line_time"])
        self.near_range = float(par["near_range_slc"])
        self.range_spacing = float(par["range_pixel_spacing"])
        self.rows = int(par["azimuth_lines"])
        self.cols = int(par["range_samples"])

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([np.interp(t, self.times, self.positions[:, k]) for k in range(3)], dtype=np.float64)
        v = np.array([np.interp(t, self.times, self.velocities[:, k]) for k in range(3)], dtype=np.float64)
        return p, v

    def doppler_zero_function(self, t: float, target: np.ndarray) -> float:
        p, v = self.state(t)
        return float(np.dot(v, target - p))

    def solve_time(self, target: np.ndarray) -> float:
        # Solve over the orbit-state-vector time span, not only the image time
        # span. Targets outside the cropped scene must retain an out-of-frame
        # azimuth coordinate instead of collapsing onto the first/last row.
        a = float(self.times[0])
        b = float(self.times[-1])
        fa = self.doppler_zero_function(a, target)
        fb = self.doppler_zero_function(b, target)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            return float(brentq(lambda t: self.doppler_zero_function(t, target), a, b, xtol=1e-8, maxiter=50))
        result = minimize_scalar(
            lambda t: abs(self.doppler_zero_function(float(t), target)),
            bounds=(a, b),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 80},
        )
        return float(result.x)

    def project_ecef(self, target: np.ndarray) -> tuple[float, float, float]:
        t = self.solve_time(target)
        p, _ = self.state(t)
        slant_range = float(np.linalg.norm(target - p))
        row = (t - self.start_time) / self.azimuth_line_time
        col = (slant_range - self.near_range) / self.range_spacing
        return row, col, slant_range


def sample_dsm_surface_m(geom, dsm_path: Path) -> float | None:
    with rasterio.open(dsm_path) as src:
        geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        window = rasterio.features.geometry_window(src, [geom_proj], pad_x=1, pad_y=1)
        arr = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)
        mask = geometry_mask([geom_proj], out_shape=arr.shape, transform=transform, invert=True)
        values = arr[mask]
        if values.size == 0:
            return None
        values = np.asarray(values.compressed() if hasattr(values, "compressed") else values, dtype=np.float64)
        values = values[np.isfinite(values)]
        values = values[values > -1000]
        if values.size == 0:
            return None
        return float(np.nanmedian(values))


def exterior_lonlat(geom) -> np.ndarray | None:
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
    elif geom.geom_type == "MultiPolygon":
        part = max(list(geom.geoms), key=lambda g: g.area)
        coords = np.asarray(part.exterior.coords, dtype=np.float64)
    else:
        return None
    if coords.shape[0] > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords if coords.shape[0] >= 3 else None


def project_ring(projector: ZeroDopplerProjector, transformer: Transformer, ring: np.ndarray, height_m: float) -> tuple[np.ndarray, list[float]]:
    rows_cols = []
    slant_ranges = []
    for lon, lat in ring:
        x, y, z = transformer.transform(float(lon), float(lat), float(height_m))
        prow, pcol, slant = projector.project_ecef(np.asarray([x, y, z], dtype=np.float64))
        rows_cols.append([pcol, prow])
        slant_ranges.append(slant)
    return np.asarray(rows_cols, dtype=np.float64), slant_ranges


def swept_support(bottom_xy: np.ndarray, roof_xy: np.ndarray):
    """Return the valid 2-D radar support swept from ground to rooftop."""
    if len(bottom_xy) != len(roof_xy) or len(bottom_xy) < 3:
        raise ValueError("Ground and rooftop rings must have matching vertices")
    parts = [Polygon(bottom_xy), Polygon(roof_xy)]
    for idx in range(len(bottom_xy)):
        nxt = (idx + 1) % len(bottom_xy)
        parts.append(Polygon([bottom_xy[idx], bottom_xy[nxt], roof_xy[nxt], roof_xy[idx]]))
    support = unary_union([part.buffer(0) for part in parts if not part.is_empty]).buffer(0)
    if support.is_empty:
        raise ValueError("Height-dependent projected support is empty")
    return support


def feature_metrics(xy: np.ndarray, projector: ZeroDopplerProjector) -> dict[str, Any]:
    finite = np.all(np.isfinite(xy), axis=1)
    in_frame = finite & (xy[:, 0] >= 0) & (xy[:, 0] < projector.cols) & (xy[:, 1] >= 0) & (xy[:, 1] < projector.rows)
    mask_pixels = 0
    if np.sum(finite) >= 3:
        rr, cc = draw_polygon(xy[finite, 1], xy[finite, 0], shape=(projector.rows, projector.cols))
        mask_pixels = int(rr.size)
    return {
        "projected_vertices": int(np.sum(finite)),
        "vertices_in_frame": int(np.sum(in_frame)),
        "mask_pixels": mask_pixels,
        "row_min": float(np.nanmin(xy[:, 1])) if xy.size else None,
        "row_max": float(np.nanmax(xy[:, 1])) if xy.size else None,
        "col_min": float(np.nanmin(xy[:, 0])) if xy.size else None,
        "col_max": float(np.nanmax(xy[:, 0])) if xy.size else None,
    }


def project_buildings(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    date = args.date or str(config["reference"]["date"])
    par_path = Path(config["paths"]["rslc_dir"]) / f"{date}.rslc.par"
    par = parse_gamma_par(par_path)
    projector = ZeroDopplerProjector(par)
    transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    buildings = gpd.read_file(args.buildings)
    buildings = buildings.to_crs("EPSG:4326")

    features = []
    metrics = []
    skipped = []
    for _, row in buildings.iterrows():
        uid_value = row.get("uid", row.get("clean_id"))
        if uid_value is None:
            skipped.append({"uid": None, "reason": "missing_uid_or_clean_id"})
            continue
        uid = int(uid_value)
        ring = exterior_lonlat(row.geometry)
        if ring is None:
            skipped.append({"uid": uid, "reason": "unsupported_geometry"})
            continue
        height_value = row.get("height_prior_m", row.get("height"))
        if height_value is None or not np.isfinite(float(height_value)):
            floor_value = row.get("Floor", row.get("floor"))
            height_value = float(floor_value) * 3.0 if floor_value is not None else 0.0
        height_prior = float(height_value)
        top_h = sample_dsm_surface_m(row.geometry, Path(args.dsm))
        if top_h is None:
            top_h = height_prior
            base_h = 0.0
            height_source = "prior_height_no_dsm"
        else:
            base_h = top_h - height_prior
            height_source = "dsm_surface_minus_prior"

        roof_xy, roof_slant = project_ring(projector, transformer, ring, top_h)
        bottom_xy, bottom_slant = project_ring(projector, transformer, ring, base_h)
        roof_xy[:, 0] += args.col_shift
        roof_xy[:, 1] += args.row_shift
        bottom_xy[:, 0] += args.col_shift
        bottom_xy[:, 1] += args.row_shift
        support_geom = swept_support(bottom_xy, roof_xy)
        support_xy = np.asarray(support_geom.convex_hull.exterior.coords, dtype=np.float64)
        base_props = {
            "uid": uid,
            "height_prior_m": height_prior,
            "base_height_m": float(base_h),
            "top_height_m": float(top_h),
            "height_source": height_source,
            "projection_row_shift": float(args.row_shift),
            "projection_col_shift": float(args.col_shift),
            "mean_slant_range_m": float(np.nanmean(roof_slant + bottom_slant)) if roof_slant or bottom_slant else None,
        }
        for surface, xy in [("bottom", bottom_xy), ("roof", roof_xy), ("layover", support_xy)]:
            props = {**base_props, "surface": surface, **feature_metrics(xy, projector)}
            geometry = mapping(support_geom) if surface == "layover" else {"type": "Polygon", "coordinates": [xy.tolist() + [xy[0].tolist()]]}
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": geometry,
                }
            )
            metrics.append(props)

    payload = {
        "type": "FeatureCollection",
        "name": "tongji_roof_projection_sar_zero_doppler",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "features": features,
    }
    summary = {
        "date": date,
        "par": str(par_path),
        "input_buildings": int(len(buildings)),
        "projected_buildings": len({f["properties"]["uid"] for f in features}),
        "projected_features": len(features),
        "skipped_buildings": len(skipped),
        "shape_rows_cols": [projector.rows, projector.cols],
        "row_shift": float(args.row_shift),
        "col_shift": float(args.col_shift),
        "note": "Zero-Doppler projection from local GAMMA RSLC parameter orbit vectors. Optional row/col shifts are applied in SAR coordinates.",
    }
    return {"geojson": payload, "summary": summary, "skipped": skipped}, metrics


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_overlay(projection_geojson: Path, amp_npy: Path, out_png: Path, max_draw: int) -> None:
    data = json.loads(projection_geojson.read_text(encoding="utf-8"))
    amp = np.load(amp_npy)
    p2, p98 = np.percentile(amp[np.isfinite(amp)], [2, 98])
    bg = np.clip((amp - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=220)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    drawn = 0
    all_xy = []
    for feat in data.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        xy = np.asarray(coords[0], dtype=np.float64)
        all_xy.append(xy)
        if max_draw <= 0 or drawn < max_draw:
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor="#ffb000", linewidth=0.25, alpha=0.7))
            drawn += 1
    if all_xy:
        xy_all = np.vstack(all_xy)
        finite = np.all(np.isfinite(xy_all), axis=1)
        xy_all = xy_all[finite]
        ax.set_xlim(max(0, float(np.min(xy_all[:, 0])) - 60), min(bg.shape[1], float(np.max(xy_all[:, 0])) + 60))
        ax.set_ylim(min(bg.shape[0], float(np.max(xy_all[:, 1])) + 60), max(0, float(np.min(xy_all[:, 1])) - 60))
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Tongji zero-Doppler roof projection")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tongji_reproduction.json")
    parser.add_argument("--date", default="")
    parser.add_argument("--buildings", default="results/geodata/tongji_buildings_prepared.gpkg")
    parser.add_argument("--dsm", default="data/dsm/tongji_real_dsm_1m_rslc_extent.tif")
    parser.add_argument("--row-shift", type=float, default=0.0)
    parser.add_argument("--col-shift", type=float, default=0.0)
    parser.add_argument("--output-geojson", default="work/projection/20200708_roof_projection_sar.geojson")
    parser.add_argument("--metrics-csv", default="work/projection/20200708_roof_projection_sar_metrics.csv")
    parser.add_argument("--summary", default="results/metadata/roof_projection_sar_summary.json")
    parser.add_argument("--overlay-png", default="work/projection/20200708_roof_projection_sar_overlay.png")
    parser.add_argument("--amp-npy", default="work/mli/mean_crop_bmp_amplitude.npy")
    parser.add_argument("--max-draw", type=int, default=0)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    result, metrics = project_buildings(config, args)
    out_geojson = Path(args.output_geojson)
    out_summary = Path(args.summary)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(result["geojson"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics(Path(args.metrics_csv), metrics)
    out_summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if Path(args.amp_npy).exists():
        plot_overlay(out_geojson, Path(args.amp_npy), Path(args.overlay_png), args.max_draw)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
