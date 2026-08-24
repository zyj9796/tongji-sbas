#!/usr/bin/env python3
"""Build an audited prior-height building-body projection with GAMMA.

The clean vector ``height`` field is used only to set the vertical distance
between ground and rooftop proxy vertices.  It is never used as an inversion
result, missing-height fill, calibration target, or projection-quality score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TONGJI_CODE = PROJECT / "code" / "reproduce_tongji"
sys.path.insert(0, str(TONGJI_CODE))

from optimize_projection_shifts import (  # noqa: E402
    edge_map,
    optimize_one,
    rasterize_local_polygon,
    score_shift,
    stretch_amp,
)


PIXEL_RE = re.compile(
    r"^(?:corrected )?SLC/MLI range, azimuth pixel:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--slc-par", default="data/tongji_rslc/20200708.rslc.par")
    p.add_argument("--dem-seg-par", default="work/gamma_sbas/dem/tongji_dsm_1m_seg.dem_par")
    p.add_argument("--lookup", default="work/gamma_sbas/dem/20200708_gc_map.lt")
    p.add_argument("--sim-sar", default="work/gamma_sbas/dem/20200708_sim_sar_demgeo.float")
    p.add_argument("--amplitude", default="work/mli/mean_crop_bmp_amplitude.npy")
    p.add_argument("--bmp-dir", default="data/tongji_rslc")
    p.add_argument("--ground-height-m", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--height-overrides", default="", help="Optional CSV/GeoJSON containing SBAS heights used only to update projection geometry for solved buildings.")
    p.add_argument("--override-id-column", default="clean_id")
    p.add_argument("--override-height-column", default="height_insar_m")
    p.add_argument("--override-damping", type=float, default=1.0, help="Projection-only update fraction: H_next = H_initial + alpha*(H_sbas-H_initial).")
    p.add_argument("--global-row-limit", type=int, default=18)
    p.add_argument("--global-col-limit", type=int, default=120)
    p.add_argument("--global-coarse-step", type=int, default=2)
    p.add_argument("--local-max-shift", type=int, default=4)
    p.add_argument("--local-min-score-gain", type=float, default=2.0)
    p.add_argument("--work-dir", default="work/zjc_original_reproduction/gamma_projection")
    p.add_argument("--output", default="work/zjc_original_reproduction/20200708_all_building_gamma_corrected_projection_sar.geojson")
    p.add_argument("--metrics", default="work/zjc_original_reproduction/20200708_all_building_gamma_projection_metrics.csv")
    p.add_argument("--summary", default="work/zjc_original_reproduction/20200708_all_building_gamma_projection_summary.json")
    return p.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def gamma_env() -> dict[str, str]:
    env = os.environ.copy()
    compat = Path("/tmp/gamma_gdal_compat")
    compat.mkdir(parents=True, exist_ok=True)
    link = compat / "libgdal.so.26"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to("/lib/libgdal.so.30")
    env["LD_LIBRARY_PATH"] = str(compat) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run_gamma(cmd: list[str], log: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.write(proc.stdout)
        if proc.stderr:
            handle.write("\nSTDERR:\n" + proc.stderr)
        handle.write(f"\nRETURN_CODE={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"GAMMA command failed: {' '.join(cmd)}; see {log}")
    return proc.stdout + proc.stderr


def build_gamma_diff(args: argparse.Namespace, work: Path, log: Path, env: dict[str, str]) -> Path:
    amplitude = np.load(resolve(args.amplitude)).astype(">f4")
    amp_bin = work / "20200708_mean_amplitude.float"
    amp_geo = work / "20200708_mean_amplitude_geo.float"
    diff = work / "20200708_projection_refinement.diff_par"
    offs = work / "20200708_projection_refinement.offs"
    ccp = work / "20200708_projection_refinement.ccp"
    offsets = work / "20200708_projection_refinement.offsets"
    coffs = work / "20200708_projection_refinement.coffs"
    coffsets = work / "20200708_projection_refinement.coffsets"
    amplitude.tofile(amp_bin)
    run_gamma(
        ["geocode_back", str(amp_bin), str(amplitude.shape[1]), str(resolve(args.lookup)), str(amp_geo), "1439", "1587", "2", "0", "1", "1", "5", "0"],
        log,
        env,
    )
    if diff.exists():
        diff.unlink()
    run_gamma(["create_diff_par", str(resolve(args.dem_seg_par)), "-", str(diff), "2", "0"], log, env)
    run_gamma(
        ["init_offsetm", str(resolve(args.sim_sar)), str(amp_geo), str(diff), "1", "1", "-", "-", "-", "-", "0.10", "256", "1"],
        log,
        env,
    )
    run_gamma(
        ["offset_pwrm", str(resolve(args.sim_sar)), str(amp_geo), str(diff), str(offs), str(ccp), "128", "128", str(offsets), "2", "12", "12", "0.08", "5", "0.8", "0", "0", "-", "0.01"],
        log,
        env,
    )
    run_gamma(["offset_fitm", str(offs), str(ccp), str(diff), str(coffs), str(coffsets), "0.12", "1", "0"], log, env)
    return diff


def gamma_project_point(task: tuple[float, float, float], slc_par: Path, diff: Path, env: dict[str, str]) -> tuple[tuple[float, float, float], tuple[float, float]]:
    lon, lat, height = task
    proc = subprocess.run(
        ["coord_to_sarpix", str(slc_par), "-", "-", f"{lat:.12f}", f"{lon:.12f}", f"{height:.6f}", str(diff)],
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coord_to_sarpix failed at lon={lon}, lat={lat}, h={height}: {proc.stderr}")
    matches = PIXEL_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        raise RuntimeError(f"Could not parse coord_to_sarpix output at lon={lon}, lat={lat}, h={height}")
    range_col, azimuth_row = map(float, matches[-1])
    return task, (range_col, azimuth_row)


def project_all_vertices(buildings: gpd.GeoDataFrame, ground_h: float, slc_par: Path, diff: Path, workers: int, env: dict[str, str]) -> dict[tuple[float, float, float], tuple[float, float]]:
    tasks: set[tuple[float, float, float]] = set()
    for row in buildings.itertuples(index=False):
        top_h = ground_h + float(row.projection_height_m)
        for lon, lat, *_ in row.geometry.exterior.coords[:-1]:
            tasks.add((float(lon), float(lat), float(ground_h)))
            tasks.add((float(lon), float(lat), float(top_h)))
    result: dict[tuple[float, float, float], tuple[float, float]] = {}
    ordered = sorted(tasks)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {pool.submit(gamma_project_point, task, slc_par, diff, env): task for task in ordered}
        for seq, future in enumerate(as_completed(future_map), start=1):
            task, xy = future.result()
            result[task] = xy
            if seq == 1 or seq % 1000 == 0 or seq == len(ordered):
                print(f"GAMMA顶点投影：{seq}/{len(ordered)}", flush=True)
    return result


def swept_support(bottom_xy: np.ndarray, roof_xy: np.ndarray):
    parts = [Polygon(bottom_xy), Polygon(roof_xy)]
    for idx in range(len(bottom_xy)):
        nxt = (idx + 1) % len(bottom_xy)
        parts.append(Polygon([bottom_xy[idx], bottom_xy[nxt], roof_xy[nxt], roof_xy[idx]]))
    support = unary_union([geom.buffer(0) for geom in parts if not geom.is_empty]).buffer(0)
    if support.is_empty or not support.is_valid:
        raise RuntimeError("Invalid GAMMA-projected building support")
    return support


def initial_geometry_rows(buildings: gpd.GeoDataFrame, projected: dict[tuple[float, float, float], tuple[float, float]], ground_h: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in buildings.itertuples(index=False):
        clean_id = int(row.clean_id)
        height = float(row.projection_height_m)
        lonlat = [(float(x), float(y)) for x, y, *_ in row.geometry.exterior.coords[:-1]]
        bottom = np.asarray([projected[(lon, lat, ground_h)] for lon, lat in lonlat], dtype=float)
        roof = np.asarray([projected[(lon, lat, ground_h + height)] for lon, lat in lonlat], dtype=float)
        rows.append(
            {
                "clean_id": clean_id,
                "height_prior_m": height,
                "projection_height_source": str(row.projection_height_source),
                "floor_prior": int(row.Floor),
                "bottom": Polygon(bottom).buffer(0),
                "roof": Polygon(roof).buffer(0),
                "layover": swept_support(bottom, roof),
            }
        )
    return rows


def extended_union_mask(geometries: list, shape: tuple[int, int], margin: int) -> np.ndarray:
    mask = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin), dtype=bool)
    for geom in geometries:
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            xy = np.asarray(part.exterior.coords)
            rr, cc = draw_polygon(xy[:, 1] + margin, xy[:, 0] + margin, shape=mask.shape)
            mask[rr, cc] = True
    return mask


def global_alignment_score(ext_mask: np.ndarray, amp: np.ndarray, edges: np.ndarray, margin: int, dr: int, dc: int) -> float:
    candidate = ext_mask[margin - dr : margin - dr + amp.shape[0], margin - dc : margin - dc + amp.shape[1]]
    if int(candidate.sum()) < 100:
        return -1e9
    dil = ndimage.binary_dilation(candidate, iterations=2)
    ero = ndimage.binary_erosion(candidate, iterations=1)
    ring = dil & ~candidate
    boundary = dil & ~ero
    inside = float(np.mean(amp[candidate]))
    outside = float(np.mean(amp[ring])) if ring.any() else inside
    edge = float(np.mean(edges[boundary])) if boundary.any() else 0.0
    return 0.95 * inside + 0.55 * (inside - outside) + 0.75 * edge


def estimate_global_shift(roof_geometries: list, amp: np.ndarray, row_limit: int, col_limit: int, coarse: int) -> dict[str, float | int]:
    margin = max(row_limit, col_limit) + 12
    ext = extended_union_mask(roof_geometries, amp.shape, margin)
    amp01 = stretch_amp(np.log1p(np.maximum(amp, 0.0)))
    edges = edge_map(amp01)
    base = global_alignment_score(ext, amp01, edges, margin, 0, 0)
    best = (base, 0, 0)
    for dr in range(-row_limit, row_limit + 1, coarse):
        for dc in range(-col_limit, col_limit + 1, coarse):
            score = global_alignment_score(ext, amp01, edges, margin, dr, dc)
            if score > best[0]:
                best = (score, dr, dc)
    _, br, bc = best
    for dr in range(max(-row_limit, br - coarse), min(row_limit, br + coarse) + 1):
        for dc in range(max(-col_limit, bc - coarse), min(col_limit, bc + coarse) + 1):
            score = global_alignment_score(ext, amp01, edges, margin, dr, dc)
            if score > best[0]:
                best = (score, dr, dc)
    return {"base_score": float(base), "best_score": float(best[0]), "score_gain": float(best[0] - base), "row_shift": int(best[1]), "col_shift": int(best[2])}


def load_split_amplitudes(directory: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    paths = sorted(directory.glob("*.crop.bmp"))
    if len(paths) < 6:
        raise RuntimeError("At least six dated crop BMP images are required for temporal cross-validation")
    train_paths = paths[::2]
    validation_paths = paths[1::2]

    def mean_image(items: list[Path]) -> np.ndarray:
        accum = np.zeros(expected_shape, dtype=np.float64)
        for path in items:
            arr = np.asarray(Image.open(path), dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            if arr.shape != expected_shape:
                raise ValueError(f"Unexpected BMP shape {arr.shape}: {path}")
            accum += arr
        return (accum / len(items)).astype(np.float32)

    return mean_image(train_paths), mean_image(validation_paths), [p.name for p in train_paths], [p.name for p in validation_paths]


def validation_shift_score(amp: np.ndarray, edges: np.ndarray, xy: np.ndarray, dr: int, dc: int, max_shift: int) -> tuple[float, float]:
    rr, cc, rr_ring, cc_ring = rasterize_local_polygon(xy, amp.shape[0], amp.shape[1])
    if rr.size < 8:
        return -1e9, -1e9
    base = score_shift(amp, edges, rr, cc, rr_ring, cc_ring, 0, 0, max_shift)
    candidate = score_shift(amp, edges, rr, cc, rr_ring, cc_ring, dr, dc, max_shift)
    return float(base["score"]), float(candidate["score"])


def calibrate_rows(
    rows: list[dict[str, object]],
    train_amplitude: np.ndarray,
    validation_amplitude: np.ndarray,
    global_shift: dict[str, float | int],
    local_max: int,
    min_gain: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_amp = stretch_amp(np.log1p(np.maximum(train_amplitude, 0.0)))
    validation_amp = stretch_amp(np.log1p(np.maximum(validation_amplitude, 0.0)))
    train_edges = edge_map(train_amp)
    validation_edges = edge_map(validation_amp)
    gdr = int(global_shift["row_shift"])
    gdc = int(global_shift["col_shift"])
    metrics: list[dict[str, object]] = []
    accepted = 0
    for row in rows:
        globally_shifted_roof = affinity.translate(row["roof"], xoff=gdc, yoff=gdr)
        if globally_shifted_roof.geom_type != "Polygon":
            local = {"ok": 0, "row_shift": 0, "col_shift": 0, "score_gain": 0.0, "pixels": 0}
        else:
            xy = np.asarray(globally_shifted_roof.exterior.coords)[:-1]
            local = optimize_one(train_amp, train_edges, xy, local_max, 1)
        if int(local.get("ok", 0)) == 1 and globally_shifted_roof.geom_type == "Polygon":
            xy = np.asarray(globally_shifted_roof.exterior.coords)[:-1]
            validation_opt = optimize_one(validation_amp, validation_edges, xy, local_max, 1)
            validation_base, validation_candidate = validation_shift_score(
                validation_amp,
                validation_edges,
                xy,
                int(local.get("row_shift", 0)),
                int(local.get("col_shift", 0)),
                local_max,
            )
        else:
            validation_opt = {"ok": 0, "row_shift": 0, "col_shift": 0}
            validation_base, validation_candidate = -1e9, -1e9
        validation_gain = validation_candidate - validation_base
        train_dr = int(local.get("row_shift", 0))
        train_dc = int(local.get("col_shift", 0))
        validation_dr = int(validation_opt.get("row_shift", 0))
        validation_dc = int(validation_opt.get("col_shift", 0))
        boundary_hit = abs(train_dr) == local_max or abs(train_dc) == local_max
        temporal_agreement = abs(train_dr - validation_dr) <= 1 and abs(train_dc - validation_dc) <= 1
        accept = bool(
            int(local.get("ok", 0)) == 1
            and int(local.get("pixels", 0)) >= 8
            and float(local.get("score_gain", 0.0)) >= min_gain
            and validation_gain >= min_gain
            and temporal_agreement
            and not boundary_hit
            and (train_dr != 0 or train_dc != 0)
        )
        ldr = train_dr if accept else 0
        ldc = train_dc if accept else 0
        if accept:
            accepted += 1
        for key in ("bottom", "roof", "layover"):
            row[key] = affinity.translate(row[key], xoff=gdc + ldc, yoff=gdr + ldr)
        metrics.append(
            {
                "clean_id": int(row["clean_id"]),
                "height_prior_m_projection_only": float(row["height_prior_m"]),
                "projection_height_source": str(row["projection_height_source"]),
                "gamma_global_row_shift": gdr,
                "gamma_global_col_shift": gdc,
                "local_row_shift": ldr,
                "local_col_shift": ldc,
                "local_shift_accepted": accept,
                "train_score_gain": float(local.get("score_gain", 0.0)),
                "validation_score_gain": float(validation_gain),
                "validation_best_row_shift": validation_dr,
                "validation_best_col_shift": validation_dc,
                "temporal_shift_agreement": temporal_agreement,
                "search_boundary_hit": boundary_hit,
                "roof_pixels": int(local.get("pixels", 0)),
                "height_used_for_inversion_or_fill": False,
            }
        )
    print(f"局部校正通过：{accepted}/{len(rows)}", flush=True)
    return rows, metrics


def write_projection(rows: list[dict[str, object]], output: Path, diff: Path, global_shift: dict[str, float | int]) -> None:
    features = []
    for row in rows:
        common = {
            "clean_id": int(row["clean_id"]),
            "uid": int(row["clean_id"]),
            "height_prior_m": float(row["height_prior_m"]),
            "projection_height_source": str(row["projection_height_source"]),
            "floor_prior": int(row["floor_prior"]),
            "height_role": "projection_geometry_only_not_inversion_or_fill",
            "gamma_diff_par": str(diff.relative_to(PROJECT)),
            "global_row_shift": int(global_shift["row_shift"]),
            "global_col_shift": int(global_shift["col_shift"]),
        }
        for surface in ("bottom", "roof", "layover"):
            features.append({"type": "Feature", "properties": {**common, "surface": surface}, "geometry": mapping(row[surface])})
    payload = {
        "type": "FeatureCollection",
        "name": "tongji_gamma_corrected_prior_height_building_projection",
        "coordinate_system": "SAR image coordinates: x=range column, y=azimuth row",
        "height_policy": "clean vector height is used only to extrude projection geometry; forbidden for inversion result filling",
        "features": features,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    work = resolve(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "gamma_projection.log"
    log.write_text("", encoding="utf-8")
    env = gamma_env()
    buildings = gpd.read_file(resolve(args.buildings)).to_crs("EPSG:4326")
    if not {"clean_id", "height", "Floor"}.issubset(buildings.columns):
        raise ValueError("clean_equal_height input must contain clean_id, height, and Floor")
    if not buildings.geometry.is_valid.all() or not (buildings.geom_type == "Polygon").all():
        raise ValueError("All clean_equal_height geometries must be valid polygons")
    if not np.allclose(pd.to_numeric(buildings["height"]), pd.to_numeric(buildings["Floor"]) * 3.0):
        raise ValueError("Expected clean_equal_height height == Floor * 3 m")
    buildings["projection_height_m"] = pd.to_numeric(buildings["height"], errors="raise").astype(float)
    buildings["projection_height_source"] = "vector_initialization_only"
    override_count = 0
    if args.height_overrides:
        if not 0.0 < args.override_damping <= 1.0:
            raise ValueError("--override-damping must be in (0, 1]")
        override_path = resolve(args.height_overrides)
        overrides = gpd.read_file(override_path) if override_path.suffix.lower() in {".geojson", ".gpkg", ".shp"} else pd.read_csv(override_path)
        required = {args.override_id_column, args.override_height_column}
        if not required.issubset(overrides.columns):
            raise ValueError(f"height override must contain {sorted(required)}")
        override_values = pd.to_numeric(overrides[args.override_height_column], errors="coerce")
        valid_override = np.isfinite(override_values) & (override_values > 0) & (override_values <= 120)
        override_map = dict(zip(overrides.loc[valid_override, args.override_id_column].astype(int), override_values[valid_override].astype(float)))
        mapped = buildings["clean_id"].astype(int).map(override_map)
        use = mapped.notna()
        initial = buildings.loc[use, "projection_height_m"].astype(float)
        buildings.loc[use, "projection_height_m"] = initial + args.override_damping * (mapped[use].astype(float) - initial)
        buildings.loc[use, "projection_height_source"] = "sbas_iteration_update"
        override_count = int(use.sum())

    diff = build_gamma_diff(args, work, log, env)
    projected = project_all_vertices(buildings, args.ground_height_m, resolve(args.slc_par), diff, args.workers, env)
    rows = initial_geometry_rows(buildings, projected, args.ground_height_m)
    amplitude = np.load(resolve(args.amplitude)).astype(np.float32)
    train_amp, validation_amp, train_names, validation_names = load_split_amplitudes(resolve(args.bmp_dir), amplitude.shape)
    train_global = estimate_global_shift([row["roof"] for row in rows], train_amp, args.global_row_limit, args.global_col_limit, args.global_coarse_step)
    validation_global = estimate_global_shift([row["roof"] for row in rows], validation_amp, args.global_row_limit, args.global_col_limit, args.global_coarse_step)
    if abs(int(train_global["row_shift"]) - int(validation_global["row_shift"])) > 2 or abs(int(train_global["col_shift"]) - int(validation_global["col_shift"])) > 2:
        raise RuntimeError(f"Temporal global-shift validation failed: train={train_global}, validation={validation_global}")
    global_shift = {
        "row_shift": int(round((int(train_global["row_shift"]) + int(validation_global["row_shift"])) / 2)),
        "col_shift": int(round((int(train_global["col_shift"]) + int(validation_global["col_shift"])) / 2)),
        "train": train_global,
        "validation": validation_global,
    }
    print("全局残余校正：" + json.dumps(global_shift, ensure_ascii=False), flush=True)
    rows, metrics = calibrate_rows(rows, train_amp, validation_amp, global_shift, args.local_max_shift, args.local_min_score_gain)
    output = resolve(args.output)
    write_projection(rows, output, diff, global_shift)
    metrics_path = resolve(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    projected_gdf = gpd.read_file(output).set_crs(None, allow_override=True)
    supports = projected_gdf[projected_gdf["surface"] == "layover"]
    visible = int(supports.geometry.intersects(box(0, 0, amplitude.shape[1], amplitude.shape[0])).sum())
    accepted = int(sum(bool(row["local_shift_accepted"]) for row in metrics))
    summary = {
        "method": "GAMMA coord_to_sarpix with GAMMA DIFF_par, robust global SAR residual calibration, and conservative local amplitude-edge correction",
        "input_buildings": str(resolve(args.buildings).relative_to(PROJECT)),
        "buildings": int(len(buildings)),
        "projected_features": int(len(projected_gdf)),
        "valid_supports": int(supports.is_valid.sum()),
        "visible_supports": visible,
        "ground_height_m": args.ground_height_m,
        "projection_height_overrides": args.height_overrides or None,
        "sbas_projection_height_override_count": override_count,
        "projection_height_update_damping": args.override_damping,
        "unsolved_projection_geometry_policy": "retain vector height only as geometric initialization; never emit it as an inversion result",
        "height_policy": "height is used only for vertical extrusion of projection geometry; never for inversion, calibration target, QC target, or missing-height fill",
        "gamma_diff_par": str(diff.relative_to(PROJECT)),
        "global_alignment": global_shift,
        "temporal_cross_validation": {
            "training_images": train_names,
            "validation_images": validation_names,
            "local_acceptance_requires_gain_in_both_splits": True,
            "local_acceptance_requires_shift_agreement_within_1_pixel": True,
            "search_boundary_solutions_rejected": True,
        },
        "local_shift_accepted": accepted,
        "local_shift_rejected_or_zero": int(len(buildings) - accepted),
        "output_geojson": str(output.relative_to(PROJECT)),
        "metrics_csv": str(metrics_path.relative_to(PROJECT)),
        "invariants": {
            "all_clean_geometries_valid": bool(buildings.geometry.is_valid.all()),
            "all_projected_supports_valid": bool(supports.is_valid.all()),
            "height_used_for_inversion_or_fill": False,
        },
    }
    summary_path = resolve(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
