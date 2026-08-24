#!/usr/bin/env python3
"""Run vector-isolated, building-wise GAMMA unwrapping + SBAS inversion.

Each building is solved independently using its projected SAR support points
and nearby stable-ground anchors. GAMMA ``multi_def_pt`` performs the default
baseline-time ambiguity resolution (``mcf_pt`` remains an experimental option),
then GAMMA ``mb_pt`` estimates the time series and height correction.

The vector ``height`` attribute is deliberately unavailable during point
selection, unwrapping, inversion, internal quality control, and aggregation.
It is read only after all InSAR-only results have been frozen, for diagnostic
comparison.  Missing or rejected buildings are never filled from the prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import affinity
from scipy.spatial import cKDTree

from benchmark_paper_unwrap import paper_like_unwrap


def gamma_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        "/usr/local/GAMMA/DIFF/bin",
        "/usr/local/GAMMA/DIFF/scripts",
        "/usr/local/GAMMA/ISP/bin",
        "/usr/local/GAMMA/ISP/scripts",
        "/usr/local/GAMMA/IPTA/bin",
        "/usr/local/GAMMA/IPTA/scripts",
    ]
    env["PATH"] = ":".join(paths) + ":" + env.get("PATH", "")
    compat_path = Path("/tmp/gamma_gdal_compat")
    compat_path.mkdir(parents=True, exist_ok=True)
    gdal_compat = compat_path / "libgdal.so.26"
    if not gdal_compat.exists():
        gdal_compat.symlink_to("/lib/libgdal.so.30")
    compat = str(compat_path)
    env["LD_LIBRARY_PATH"] = compat + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run(cmd: list[str], log: Path, cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, env=gamma_env(), text=True, capture_output=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(cmd) + "\n")
        stream.write(proc.stdout)
        if proc.stderr:
            stream.write("\nSTDERR:\n" + proc.stderr)
        stream.write(f"\nRETURN_CODE={proc.returncode}\n\n")
    if proc.returncode:
        raise RuntimeError(f"GAMMA command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", default="work/gamma_native_ipta_sbas")
    p.add_argument(
        "--interferogram-dir",
        default=None,
        help="Optional GAMMA interferogram root when it is shared outside source-dir",
    )
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--point-metadata", default="work/gamma_native_ipta_sbas/ipta/point_metadata.csv")
    p.add_argument("--pcc", default=None, help="Optional existing GAMMA point-coherence stack")
    p.add_argument("--global-punw", default=None, help="Override source IPTA unwrapped phase stack")
    p.add_argument("--support-owner", default=None, help="Optional roof-core clean_id owner .npy used to restrict support points")
    p.add_argument(
        "--spatial-support-projection",
        default=None,
        help=(
            "Optional projection containing per-building layover/search envelopes. "
            "Candidate points are selected spatially for each building and may be "
            "reused during ambiguity search; intended for height_position_coupled."
        ),
    )
    p.add_argument("--support-min-mean-coherence", type=float, default=0.0)
    p.add_argument("--closure-bad-count", default=None, help="Per-IPTA-point non-zero integer triangle-closure count .npy")
    p.add_argument("--max-closure-bad-triangles", type=int, default=-1, help="-1 disables closure screening")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--geometric-reference-par", default="data/tongji_rslc/20200708.rslc.par")
    p.add_argument("--out-dir", default="work/gamma_building_mcf_sbas")
    p.add_argument("--result-csv", default="results/tables/tongji_building_height_gamma_mcf_sbas_insar_only.csv")
    p.add_argument("--result-geojson", default="results/geodata/tongji_building_height_gamma_mcf_sbas_insar_only.geojson")
    p.add_argument("--summary", default="results/metadata/tongji_building_height_gamma_mcf_sbas_summary.json")
    p.add_argument(
        "--unwrap-engine",
        choices=[
            "multi_def_pt", "mcf_pt", "global_multi_def", "global_model_residual_mcf",
            "global_unw_local_multidef",
            "building_uniform_height",
            "height_position_coupled",
            "geometry_wrapped_init",
            "zjc_lgr",
        ],
        default="multi_def_pt",
    )
    p.add_argument("--ground-neighbours", type=int, default=40)
    p.add_argument("--ground-radius", type=float, default=150.0, help="Maximum anchor distance in SAR pixels")
    p.add_argument("--min-ground-points", type=int, default=3)
    p.add_argument(
        "--ground-statistic",
        choices=["point_median", "sector_median"],
        default="point_median",
        help=(
            "How the solved local-ground heights are aggregated. sector_median "
            "first takes one median in each occupied azimuth/range sector and "
            "then gives every sector equal weight, preventing one dense direction "
            "from dominating the roof-minus-ground reference."
        ),
    )
    p.add_argument("--ground-sectors", type=int, default=8)
    p.add_argument("--min-ground-sectors", type=int, default=3)
    p.add_argument("--min-support-points", type=int, default=4)
    p.add_argument("--min-output-points", type=int, default=4)
    p.add_argument("--sbas-gamma", type=float, default=10.0)
    p.add_argument(
        "--mb-sigma-mode",
        choices=["equal", "paper_threshold", "local_roof_coherence", "local_ground_residual"],
        default="equal",
        help=(
            "Pair weighting passed to GAMMA mb_pt. paper_threshold keeps the full "
            "connected network but downweights pairs outside the paper's "
            "30-250 m perpendicular-baseline or 44-day temporal-baseline limits. "
            "local_ground_residual first runs equal-weight mb_pt, estimates each "
            "layer's robust residual sigma on nearby stable ground, and reruns mb_pt."
        ),
    )
    p.add_argument(
        "--paper-pair-sigma-rad",
        type=float,
        default=3.0,
        help="mb_pt phase sigma assigned to a pair outside the paper thresholds",
    )
    p.add_argument("--mb-sigma-min", type=float, default=0.15)
    p.add_argument("--mb-sigma-max", type=float, default=3.0)
    p.add_argument(
        "--mb-coherence-tempering",
        type=float,
        default=1.0,
        help=(
            "Shrink local coherence-derived phase sigmas toward their per-building "
            "geometric mean: 0 gives equal pair weights, 1 gives the full wrapped-normal weights"
        ),
    )
    p.add_argument(
        "--mb-sigma-ratio-limit",
        type=float,
        default=float("inf"),
        help="Robust cap on each coherence-derived phase sigma relative to its per-building centre",
    )
    p.add_argument(
        "--height-sensitivity",
        default="work/gamma_native_ipta_sbas/ipta/phase_height_sensitivity_rad_per_m.npy",
        help="GAMMA-simulated interferometric phase sensitivity stack (pair, point), rad/m",
    )
    p.add_argument("--uniform-height-step", type=float, default=0.25)
    p.add_argument("--uniform-rate-step", type=float, default=0.001)
    p.add_argument(
        "--min-height-search-cost-margin",
        type=float,
        default=0.0,
        help="Minimum circular-cost separation from every solution at least 5 m away",
    )
    p.add_argument(
        "--max-height-search-split-difference",
        type=float,
        default=float("inf"),
        help=(
            "Maximum absolute height difference between two frozen, baseline-stratified "
            "interferogram halves; finite values enable blind split validation"
        ),
    )
    p.add_argument(
        "--projection-geometry",
        default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson",
        help="Bottom/roof GAMMA projection used only to derive per-metre SAR displacement",
    )
    p.add_argument(
        "--wrapped-height-audit",
        default="results/tables/tongji_geometry_constrained_wrapped_height_diagnostic_points.csv",
        help="Per-point wrapped search used only to initialize phase cycles for geometry_wrapped_init",
    )
    p.add_argument(
        "--initialization-height-offset-m",
        type=float,
        default=0.0,
        help=(
            "Blind ambiguity-stability diagnostic: add this constant to every "
            "geometry_wrapped_init height before selecting 2pi cycles. The offset "
            "is never added to the GAMMA output height."
        ),
    )
    p.add_argument(
        "--geometry-ground-initialization",
        choices=["global_multi_def", "local_wrapped_zero"],
        default="global_multi_def",
        help=(
            "Ground phase branch used by geometry_wrapped_init. local_wrapped_zero "
            "unwraps each building's stable-ground points around their local circular "
            "centre and does not inherit global multi_def_pt integer cycles."
        ),
    )
    p.add_argument("--dh-max", type=float, default=180.0)
    p.add_argument("--multi-model", type=int, default=2)
    p.add_argument("--multi-sigma-max", type=float, default=1.2)
    p.add_argument("--multi-sigma-max2", type=float, default=0.75)
    p.add_argument("--building-statistic", choices=["median", "upper_cluster"], default="upper_cluster")
    p.add_argument("--max-median-phase-sigma", type=float, default=1.2)
    p.add_argument("--max-roof-iqr", type=float, default=80.0)
    p.add_argument("--min-physical-height", type=float, default=0.0)
    p.add_argument("--max-physical-height", type=float, default=180.0)
    p.add_argument("--limit-buildings", type=int, default=0, help="0 processes every building")
    p.add_argument("--clean-id", type=int, action="append", default=[], help="Process selected clean_id values")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def parse_spacing(par_path: Path) -> tuple[float, float]:
    values: dict[str, float] = {}
    for line in par_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key.strip() in {"range_pixel_spacing", "azimuth_pixel_spacing", "incidence_angle"}:
            try:
                values[key.strip()] = float(rest.split()[0])
            except (ValueError, IndexError):
                pass
    slant = values.get("range_pixel_spacing", 1.0)
    incidence = math.radians(values.get("incidence_angle", 90.0))
    range_ground = slant / max(math.sin(incidence), 1.0e-3)
    return range_ground, values.get("azimuth_pixel_spacing", 1.0)


def parse_radar_frequency(par_path: Path) -> float:
    for line in par_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("radar_frequency:"):
            return float(line.split(":", 1)[1].split()[0])
    raise ValueError(f"radar_frequency missing from {par_path}")


def finite_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir)
    source_ipta = source / "ipta"
    out = Path(args.out_dir)
    shards = out / "building_shards"
    temp_root = out / "tmp"
    out.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    log = out / "gamma_building_mcf.log"
    if args.no_resume or not log.exists():
        log.write_text("", encoding="utf-8")

    metadata = pd.read_csv(args.point_metadata)
    required = {
        "point_index", "range_pixel", "azimuth_pixel", "point_class", "clean_id",
        "mean_coherence", "amplitude_dispersion",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Point metadata missing columns: {sorted(missing)}")
    metadata = metadata.sort_values("point_index").reset_index(drop=True)
    npoints = len(metadata)
    if not np.array_equal(metadata["point_index"].to_numpy(), np.arange(npoints)):
        raise ValueError("point_index must be contiguous and match GAMMA point-stack order")

    pairs = pd.read_csv(args.pairs_csv, dtype={"master": str, "slave": str}).drop_duplicates(["master", "slave"])
    npairs = len(pairs)
    mb_pair_sigmas: np.ndarray | None = None
    mb_sigma_file: Path | None = None
    if args.mb_sigma_mode == "paper_threshold":
        if "dt_days" in pairs:
            dt_days = pairs["dt_days"].to_numpy(float)
        else:
            dt_days = np.array(
                [abs((pd.Timestamp(slave) - pd.Timestamp(master)).days) for master, slave in pairs[["master", "slave"]].itertuples(index=False)],
                dtype=float,
            )
        if "abs_bperp_m" in pairs:
            abs_bperp = pairs["abs_bperp_m"].to_numpy(float)
        elif "bperp_m" in pairs:
            abs_bperp = np.abs(pairs["bperp_m"].to_numpy(float))
        else:
            raise ValueError("paper_threshold weighting requires abs_bperp_m or bperp_m")
        paper_compliant = (dt_days <= 44.0) & (abs_bperp >= 30.0) & (abs_bperp <= 250.0)
        mb_pair_sigmas = np.where(paper_compliant, 1.0, float(args.paper_pair_sigma_rad))
        mb_sigma_file = out / "mb_pair_sigma_paper_threshold.txt"
        # GAMMA rd_all_sigma parser expects: numeric layer index, mean phase,
        # phase sigma (the continuous-image manual describes the same 3 fields).
        mb_sigma_file.write_text(
            "\n".join(f"{index} 0.00000000 {value:.8f}" for index, value in enumerate(mb_pair_sigmas, start=1))
            + "\n",
            encoding="utf-8",
        )
    plist_path = source_ipta / "tongji.plist"
    pdiff_path = source_ipta / "pdiff"
    pslc_par = source_ipta / "pSLC_par"
    pbase = source_ipta / "pbase"
    itab = source_ipta / "pairs.itab"
    plist_all = np.fromfile(plist_path, dtype=">i4")
    if plist_all.size != 2 * npoints:
        raise ValueError(f"plist size mismatch: {plist_all.size} integers for {npoints} points")
    plist_all = plist_all.reshape(npoints, 2)
    expected_pdiff = npairs * npoints
    if pdiff_path.stat().st_size != expected_pdiff * np.dtype(">c8").itemsize:
        raise ValueError("pdiff size does not match pair and point counts")
    pdiff_all = np.memmap(pdiff_path, dtype=">c8", mode="r", shape=(npairs, npoints))
    global_punw_all = None
    global_pmask_all = None
    global_pres_all = None
    if args.unwrap_engine in {
        "global_multi_def", "global_model_residual_mcf", "global_unw_local_multidef", "building_uniform_height",
        "height_position_coupled", "geometry_wrapped_init",
    }:
        global_punw_path = Path(args.global_punw) if args.global_punw else source_ipta / "punw"
        global_pmask_path = source_ipta / "pmask"
        if global_punw_path.stat().st_size != npairs * npoints * 4:
            raise ValueError("global punw size does not match pair and point counts")
        global_punw_all = np.memmap(global_punw_path, dtype=">f4", mode="r", shape=(npairs, npoints))
        global_pmask_all = np.fromfile(global_pmask_path, dtype=np.uint8)[:npoints] > 0
        if args.unwrap_engine == "global_model_residual_mcf":
            global_pres_path = source_ipta / "pres"
            if global_pres_path.stat().st_size != npairs * npoints * 4:
                raise ValueError("global pres size does not match pair and point counts")
            global_pres_all = np.memmap(global_pres_path, dtype=">f4", mode="r", shape=(npairs, npoints))
    height_sensitivity_all = None
    if args.unwrap_engine in {"building_uniform_height", "height_position_coupled", "geometry_wrapped_init"}:
        height_sensitivity_all = np.load(args.height_sensitivity, mmap_mode="r")
        if height_sensitivity_all.shape != (npairs, npoints):
            raise ValueError(
                f"height sensitivity shape {height_sensitivity_all.shape} does not match {(npairs, npoints)}"
            )
    projection_models: dict[int, tuple[object, float, float]] = {}
    spatial_support_models: dict[int, object] = {}
    if args.unwrap_engine == "height_position_coupled":
        projection = gpd.read_file(args.projection_geometry)
        for clean_id, group in projection.groupby("clean_id"):
            bottoms = group[group["surface"].eq("bottom")]
            roofs = group[group["surface"].eq("roof")]
            if bottoms.empty or roofs.empty:
                continue
            bottom_geom = bottoms.geometry.iloc[0]
            roof_geom = roofs.geometry.iloc[0]
            geometry_height = float(group["height_prior_m"].iloc[0])
            if not np.isfinite(geometry_height) or geometry_height <= 0.0:
                continue
            projection_models[int(clean_id)] = (
                bottom_geom,
                float((roof_geom.centroid.x - bottom_geom.centroid.x) / geometry_height),
                float((roof_geom.centroid.y - bottom_geom.centroid.y) / geometry_height),
            )
    if args.spatial_support_projection:
        spatial_projection = gpd.read_file(args.spatial_support_projection)
        for clean_id, group in spatial_projection.groupby("clean_id"):
            support_rows = group[group["surface"].eq("layover")]
            if not support_rows.empty:
                spatial_support_models[int(clean_id)] = support_rows.geometry.iloc[0].buffer(0.51)

    wrapped_initial: pd.DataFrame | None = None
    if args.unwrap_engine == "geometry_wrapped_init":
        wrapped_initial = pd.read_csv(args.wrapped_height_audit)
        required_wrapped = {
            "clean_id", "point_index", "searched_height_m", "nuisance_rate_m_per_year",
            "selected_for_building_cluster",
        }
        missing_wrapped = required_wrapped - set(wrapped_initial.columns)
        if missing_wrapped:
            raise ValueError(f"Wrapped-height audit missing columns: {sorted(missing_wrapped)}")
        wrapped_initial = wrapped_initial[wrapped_initial["selected_for_building_cluster"].astype(bool)].copy()
        wrapped_initial = wrapped_initial.drop_duplicates("point_index").set_index("point_index", drop=False)

    # Extract the GAMMA ADF coherence rasters at the exact IPTA point locations.
    # DA has already been used for point selection; coherence supplies MCF costs
    # when the explicitly requested mcf_pt experimental branch is used.
    pcc = Path(args.pcc) if args.pcc else out / "pcc"
    if not args.pcc and (args.no_resume or not pcc.exists() or pcc.stat().st_size != npairs * npoints * 4):
        cc_tab = out / "cc.tab"
        interferogram_root = Path(args.interferogram_dir) if args.interferogram_dir else source / "interferograms"
        cc_paths = [interferogram_root / f"{r.master}_{r.slave}" / f"{r.master}_{r.slave}.cc" for r in pairs.itertuples(index=False)]
        absent = [str(path) for path in cc_paths if not path.exists()]
        if absent:
            raise FileNotFoundError(f"Missing GAMMA coherence rasters, first: {absent[0]}")
        cc_tab.write_text("\n".join(str(path) for path in cc_paths) + "\n", encoding="utf-8")
        if pcc.exists():
            pcc.unlink()
        run(["mk_d2pt", str(cc_tab), str(plist_path), "900", "2", "1", "1", str(pcc)], log)
    if not pcc.exists() or pcc.stat().st_size != npairs * npoints * 4:
        raise RuntimeError(f"GAMMA point-coherence stack missing or has wrong size: {pcc}")
    pcc_all = np.memmap(pcc, dtype=">f4", mode="r", shape=(npairs, npoints))

    ground = metadata[metadata["point_class"].eq("ground")].copy()
    if args.unwrap_engine in {
        "global_multi_def", "global_model_residual_mcf", "global_unw_local_multidef", "building_uniform_height",
        "height_position_coupled", "geometry_wrapped_init",
    }:
        # Build the neighbour tree from GAMMA-valid anchors.  Filtering only
        # after querying a fixed number of all ground points can incorrectly
        # report no reference even when a slightly farther valid point exists.
        ground = ground[global_pmask_all[ground["point_index"].to_numpy(np.int64)]].copy()
    if args.spatial_support_projection:
        roof = metadata[metadata["point_class"].eq("roof")].copy()
    elif args.support_owner:
        support_owner = np.load(args.support_owner).astype(np.int64)
        az = metadata["azimuth_pixel"].to_numpy(np.int64)
        rg = metadata["range_pixel"].to_numpy(np.int64)
        if support_owner.ndim != 2 or az.max() >= support_owner.shape[0] or rg.max() >= support_owner.shape[1]:
            raise ValueError("support-owner raster does not cover the IPTA point coordinates")
        owner_at_point = support_owner[az, rg]
        roof = metadata[metadata["point_class"].eq("roof") & (owner_at_point > 0)].copy()
        roof["clean_id"] = owner_at_point[roof.index]
    else:
        roof = metadata[metadata["point_class"].eq("roof") & metadata["clean_id"].gt(0)].copy()
    if args.unwrap_engine == "geometry_wrapped_init":
        assert wrapped_initial is not None
        roof = roof[roof["point_index"].isin(wrapped_initial.index)].copy()
        roof["clean_id"] = roof["point_index"].map(wrapped_initial["clean_id"]).astype(np.int64)
    roof = roof[roof["mean_coherence"].ge(args.support_min_mean_coherence)].copy()
    if args.closure_bad_count and args.max_closure_bad_triangles >= 0:
        closure_bad_count = np.load(args.closure_bad_count)
        if closure_bad_count.shape != (npoints,):
            raise ValueError("closure-bad-count array does not match IPTA point count")
        ground = ground[closure_bad_count[ground["point_index"].to_numpy(np.int64)] <= args.max_closure_bad_triangles].copy()
        roof = roof[closure_bad_count[roof["point_index"].to_numpy(np.int64)] <= args.max_closure_bad_triangles].copy()
    ground_indices = ground["point_index"].to_numpy(np.int64)
    ground_xy = ground[["range_pixel", "azimuth_pixel"]].to_numpy(float)
    ground_tree = cKDTree(ground_xy)
    range_spacing, azimuth_spacing = parse_spacing(Path(args.geometric_reference_par))
    wavelength = 299792458.0 / parse_radar_frequency(Path(args.geometric_reference_par))
    pair_dt_year = (
        pd.to_datetime(pairs["slave"], format="%Y%m%d")
        - pd.to_datetime(pairs["master"], format="%Y%m%d")
    ).dt.days.to_numpy(float) / 365.25
    deformation_phase_per_m_per_year = -4.0 * np.pi * pair_dt_year / wavelength
    # Freeze two approximately equally sensitive pair subsets before looking at
    # any building result.  Alternating the absolute-baseline order prevents one
    # half from receiving only short/weak height baselines.  These masks are
    # used solely for blind ambiguity-stability diagnostics.
    baseline_for_split = (
        pairs["abs_bperp_m"].to_numpy(float)
        if "abs_bperp_m" in pairs
        else np.abs(pairs["bperp_m"].to_numpy(float))
    )
    split_order = np.argsort(baseline_for_split, kind="stable")
    split_a = np.zeros(npairs, dtype=bool)
    split_a[split_order[::2]] = True
    split_b = ~split_a

    clean_ids = (
        sorted(spatial_support_models)
        if args.spatial_support_projection
        else sorted(int(value) for value in roof["clean_id"].unique())
    )
    if args.clean_id:
        requested = set(args.clean_id)
        clean_ids = [clean_id for clean_id in clean_ids if clean_id in requested]
    if args.limit_buildings > 0:
        clean_ids = clean_ids[: args.limit_buildings]

    for ordinal, clean_id in enumerate(clean_ids, start=1):
        shard = shards / f"{clean_id}.json"
        points_shard = shards / f"{clean_id}_points.npz"
        if not args.no_resume and shard.exists():
            continue
        if args.spatial_support_projection:
            support_geometry = spatial_support_models[clean_id]
            support_xy = roof[["range_pixel", "azimuth_pixel"]].to_numpy(float)
            support = roof[
                np.asarray(
                    shapely.contains_xy(support_geometry, support_xy[:, 0], support_xy[:, 1]),
                    dtype=bool,
                )
            ].copy()
            support["clean_id"] = clean_id
        else:
            support = roof[roof["clean_id"].eq(clean_id)].copy()
        support_indices = support["point_index"].to_numpy(np.int64)
        audit: dict[str, object] = {
            "clean_id": clean_id,
            "status": "failed",
            "failure_reason": None,
            "support_points": int(len(support_indices)),
            "ground_points": 0,
            "ground_statistic": args.ground_statistic,
            "ground_occupied_sectors": 0,
            "ground_sector_median_iqr_m": None,
            "ground_sector_leave_one_out_range_m": None,
            "unwrap_valid_support_points": 0,
            "gamma_sbas_output_points": 0,
            "accepted": False,
            "insar_height_m": None,
            "roof_height_iqr_m": None,
            "median_phase_sigma_rad": None,
        }
        if len(support_indices) < args.min_support_points:
            audit["failure_reason"] = "too_few_support_points"
            write_json(shard, audit)
            continue

        centroid = support[["range_pixel", "azimuth_pixel"]].to_numpy(float).mean(axis=0)
        search_k = min(max(args.ground_neighbours * 5, args.ground_neighbours), len(ground_indices))
        distances, neighbours = ground_tree.query(centroid, k=search_k)
        distances = np.atleast_1d(distances)
        neighbours = np.atleast_1d(neighbours)
        inside = np.isfinite(distances) & (distances <= args.ground_radius)
        local_ground_indices = ground_indices[neighbours[inside]]
        if args.unwrap_engine in {
            "global_multi_def", "global_model_residual_mcf", "global_unw_local_multidef", "building_uniform_height",
            "height_position_coupled", "geometry_wrapped_init",
        }:
            local_ground_indices = local_ground_indices[global_pmask_all[local_ground_indices]]
        local_ground_indices = local_ground_indices[: args.ground_neighbours]
        if len(local_ground_indices) < args.min_ground_points:
            audit["failure_reason"] = "too_few_nearby_ground_points"
            audit["nearest_ground_distance_px"] = finite_number(np.nanmin(distances))
            write_json(shard, audit)
            continue
        used_ground_xy = metadata.iloc[local_ground_indices][["range_pixel", "azimuth_pixel"]].to_numpy(float)
        used_ground_distances = np.linalg.norm(used_ground_xy - centroid, axis=1)
        audit["ground_points"] = int(len(local_ground_indices))
        audit["nearest_ground_distance_px"] = finite_number(np.nanmin(used_ground_distances))
        audit["furthest_used_ground_distance_px"] = finite_number(np.nanmax(used_ground_distances))

        local_indices = np.concatenate([local_ground_indices, support_indices])
        local_meta = metadata.iloc[local_indices]
        ground_scores = (
            local_meta.iloc[: len(local_ground_indices)]["mean_coherence"].to_numpy(float)
            - local_meta.iloc[: len(local_ground_indices)]["amplitude_dispersion"].to_numpy(float)
        )
        local_ref = int(np.nanargmax(ground_scores))
        audit["reference_global_point_index"] = int(local_indices[local_ref])

        try:
            with tempfile.TemporaryDirectory(prefix=f"b{clean_id}_", dir=temp_root) as tmp_name:
                tmp = Path(tmp_name)
                local_plist = tmp / "plist"
                local_pdiff = tmp / "pdiff"
                local_pcc = tmp / "pcc"
                local_pmask = tmp / "pmask"
                local_punw = tmp / "punw"
                plist_all[local_indices].astype(">i4", copy=False).tofile(local_plist)
                np.asarray(pdiff_all[:, local_indices]).astype(">c8", copy=False).tofile(local_pdiff)
                weights = np.clip(np.asarray(pcc_all[:, local_indices]), 1.0e-3, 1.0)
                weights.astype(">f4", copy=False).tofile(local_pcc)
                np.ones(len(local_indices), dtype=np.uint8).tofile(local_pmask)
                if args.unwrap_engine == "global_multi_def":
                    np.asarray(global_punw_all[:, local_indices]).astype(">f4", copy=False).tofile(local_punw)
                    global_pmask_all[local_indices].astype(np.uint8).tofile(local_pmask)
                elif args.unwrap_engine == "geometry_wrapped_init":
                    assert wrapped_initial is not None
                    ng = len(local_ground_indices)
                    global_local = np.asarray(global_punw_all[:, local_indices], dtype=np.float64)
                    corrected_local = global_local.copy()
                    if args.geometry_ground_initialization == "local_wrapped_zero":
                        ground_wrapped = np.angle(np.asarray(pdiff_all[:, local_ground_indices]))
                        ground_weights = np.clip(
                            np.asarray(pcc_all[:, local_ground_indices], dtype=np.float64), 1.0e-3, 1.0
                        )
                        ground_vector = np.sum(
                            ground_weights * np.exp(1j * ground_wrapped), axis=1
                        ) / np.maximum(np.sum(ground_weights, axis=1), 1.0e-9)
                        ground_centre = np.angle(ground_vector)
                        corrected_local[:, :ng] = ground_wrapped + 2.0 * np.pi * np.rint(
                            (ground_centre[:, None] - ground_wrapped) / (2.0 * np.pi)
                        )
                        ground_center_unw = np.nanmedian(corrected_local[:, :ng], axis=1)
                    else:
                        ground_center_unw = np.nanmedian(global_local[:, :ng], axis=1)
                    initialization = wrapped_initial.loc[support_indices]
                    initial_height_unshifted = initialization["searched_height_m"].to_numpy(float)
                    initial_height = initial_height_unshifted + args.initialization_height_offset_m
                    initial_rate = initialization["nuisance_rate_m_per_year"].to_numpy(float)
                    sensitivity = np.asarray(height_sensitivity_all[:, support_indices], dtype=np.float64)
                    roof_target = (
                        ground_center_unw[:, None]
                        + sensitivity * initial_height[None, :]
                        + deformation_phase_per_m_per_year[:, None] * initial_rate[None, :]
                    )
                    roof_wrapped = np.angle(np.asarray(pdiff_all[:, support_indices]))
                    corrected_local[:, ng:] = roof_wrapped + 2.0 * np.pi * np.rint(
                        (roof_target - roof_wrapped) / (2.0 * np.pi)
                    )
                    corrected_local.astype(">f4").tofile(local_punw)
                    if args.geometry_ground_initialization == "local_wrapped_zero":
                        np.ones(len(local_indices), dtype=np.uint8).tofile(local_pmask)
                    else:
                        global_pmask_all[local_indices].astype(np.uint8).tofile(local_pmask)
                    audit.update(
                        {
                            "wrapped_initial_height_median": float(np.nanmedian(initial_height)),
                            "wrapped_initial_height_unshifted_median": float(
                                np.nanmedian(initial_height_unshifted)
                            ),
                            "initialization_height_offset_m": args.initialization_height_offset_m,
                            "wrapped_initial_rate_median": float(np.nanmedian(initial_rate)),
                            "wrapped_initial_points": int(len(initial_height)),
                            "initialization_policy": (
                                "R-D interval constrains phase-cycle initialization only; "
                                "GAMMA mb_pt estimates final height"
                            ),
                            "ground_initialization_policy": args.geometry_ground_initialization,
                        }
                    )
                elif args.unwrap_engine in {"building_uniform_height", "height_position_coupled"}:
                    # Estimate one common roof height directly from wrapped
                    # roof-minus-local-ground phases.  The search uses only
                    # GAMMA orbit geometry, interferometric phase, coherence,
                    # and the flat-roof constraint; no vector height is read.
                    local_complex = np.asarray(pdiff_all[:, local_indices])
                    local_coherence = np.asarray(pcc_all[:, local_indices], dtype=np.float64)
                    ng = len(local_ground_indices)
                    ground_complex = local_complex[:, :ng]
                    roof_complex = local_complex[:, ng:]
                    point_da = local_meta["amplitude_dispersion"].to_numpy(np.float64)
                    point_da_weight = np.exp(-np.square(point_da / 0.40))
                    ground_weight = (
                        np.clip(local_coherence[:, :ng], 1.0e-3, 1.0)
                        * point_da_weight[None, :ng]
                    )
                    roof_weight = (
                        np.clip(local_coherence[:, ng:], 1.0e-3, 1.0)
                        * point_da_weight[None, ng:]
                    )
                    # Average unit phase vectors.  The interferogram magnitude
                    # is not a paper weight and must not be counted a second
                    # time on top of coherence and amplitude dispersion.
                    ground_unit = np.exp(1j * np.angle(ground_complex))
                    roof_unit = np.exp(1j * np.angle(roof_complex))
                    ground_mean = np.sum(ground_weight * ground_unit, axis=1) / np.maximum(
                        np.sum(ground_weight, axis=1), 1.0e-9
                    )
                    sensitivity = np.asarray(height_sensitivity_all[:, local_indices], dtype=np.float64)
                    height_grid = np.arange(
                        args.min_physical_height,
                        args.max_physical_height + 0.5 * args.uniform_height_step,
                        args.uniform_height_step,
                    )
                    rate_grid = np.arange(
                        -0.01, 0.01 + 0.5 * args.uniform_rate_step, args.uniform_rate_step
                    )
                    circular_cost = np.full((len(height_grid), len(rate_grid)), np.nan, dtype=np.float64)
                    circular_cost_a = np.full_like(circular_cost, np.nan)
                    circular_cost_b = np.full_like(circular_cost, np.nan)
                    candidate_masks: list[np.ndarray | None] = [None] * len(height_grid)
                    pair_weights: list[np.ndarray | None] = [None] * len(height_grid)
                    if args.unwrap_engine == "height_position_coupled":
                        if clean_id not in projection_models:
                            raise RuntimeError("missing bottom/roof projection geometry")
                        bottom_geom, dx_per_m, dy_per_m = projection_models[clean_id]
                        support_x = local_meta.iloc[ng:]["range_pixel"].to_numpy(float)
                        support_y = local_meta.iloc[ng:]["azimuth_pixel"].to_numpy(float)
                    for hi, candidate_height in enumerate(height_grid):
                        if args.unwrap_engine == "height_position_coupled":
                            candidate_geom = affinity.translate(
                                bottom_geom, xoff=dx_per_m * candidate_height, yoff=dy_per_m * candidate_height
                            ).buffer(0.51)
                            candidate_keep = np.asarray(
                                shapely.contains_xy(candidate_geom, support_x, support_y), dtype=bool
                            )
                        else:
                            candidate_keep = np.ones(roof_complex.shape[1], dtype=bool)
                        if int(candidate_keep.sum()) < args.min_output_points:
                            continue
                        candidate_masks[hi] = candidate_keep
                        candidate_roof_weight = roof_weight[:, candidate_keep]
                        roof_mean = np.sum(
                            candidate_roof_weight * roof_unit[:, candidate_keep], axis=1
                        ) / np.maximum(np.sum(candidate_roof_weight, axis=1), 1.0e-9)
                        relative_wrapped = np.angle(roof_mean * np.conj(ground_mean))
                        # Circular concentration measures within-class phase
                        # consistency; median point quality prevents a coherent
                        # but uniformly poor pair from receiving high weight.
                        pair_weight = np.sqrt(np.abs(ground_mean) * np.abs(roof_mean))
                        pair_weight *= np.sqrt(
                            np.nanmedian(ground_weight, axis=1)
                            * np.nanmedian(candidate_roof_weight, axis=1)
                        )
                        pair_weight = np.where(
                            np.isfinite(pair_weight), np.clip(pair_weight, 0.0, 1.0), 0.0
                        )
                        pair_weights[hi] = pair_weight
                        roof_sensitivity = np.nanmedian(sensitivity[:, ng:][:, candidate_keep], axis=1)
                        model = (
                            roof_sensitivity[:, None] * candidate_height
                            + deformation_phase_per_m_per_year[:, None] * rate_grid[None, :]
                        )
                        circular_residual = 1.0 - np.cos(relative_wrapped[:, None] - model)
                        circular_cost[hi] = np.sum(
                            pair_weight[:, None] * circular_residual, axis=0
                        ) / max(float(pair_weight.sum()), 1.0e-6)
                        for split_mask, split_cost in (
                            (split_a, circular_cost_a), (split_b, circular_cost_b)
                        ):
                            split_weight = pair_weight[split_mask]
                            split_cost[hi] = np.sum(
                                split_weight[:, None] * circular_residual[split_mask], axis=0
                            ) / max(float(split_weight.sum()), 1.0e-6)
                    if not np.any(np.isfinite(circular_cost)):
                        raise RuntimeError("no candidate height has enough projected roof support points")
                    best_flat = int(np.nanargmin(circular_cost))
                    best_h_index, best_rate_index = np.unravel_index(best_flat, circular_cost.shape)
                    common_height = float(height_grid[best_h_index])
                    common_rate = float(rate_grid[best_rate_index])
                    common_cost = float(circular_cost[best_h_index, best_rate_index])
                    best_candidate_keep = candidate_masks[best_h_index]
                    if best_candidate_keep is None:
                        raise RuntimeError("best height candidate has no support mask")
                    best_pair_weight = pair_weights[best_h_index]
                    height_profile = np.nanmin(circular_cost, axis=1)
                    outside = np.abs(height_grid - common_height) >= 5.0
                    second_cost = float(np.nanmin(height_profile[outside])) if np.any(outside) else float("nan")
                    split_a_flat = int(np.nanargmin(circular_cost_a))
                    split_b_flat = int(np.nanargmin(circular_cost_b))
                    split_a_h_index, split_a_rate_index = np.unravel_index(
                        split_a_flat, circular_cost_a.shape
                    )
                    split_b_h_index, split_b_rate_index = np.unravel_index(
                        split_b_flat, circular_cost_b.shape
                    )
                    split_a_height = float(height_grid[split_a_h_index])
                    split_b_height = float(height_grid[split_b_h_index])
                    split_height_difference = abs(split_a_height - split_b_height)
                    best_candidate_sensitivity = np.nanmedian(
                        sensitivity[:, ng:][:, best_candidate_keep], axis=1
                    )
                    search_span = float(args.max_physical_height - args.min_physical_height)
                    coarse_pair_mask = (
                        np.abs(best_candidate_sensitivity) * search_span <= 2.0 * np.pi
                    ) & (best_pair_weight > 0.0)
                    coarse_height = float("nan")
                    coarse_cost_margin = float("nan")
                    if int(coarse_pair_mask.sum()) >= 3:
                        coarse_cost = np.full_like(circular_cost, np.nan)
                        for hi, candidate_height in enumerate(height_grid):
                            candidate_keep = candidate_masks[hi]
                            candidate_pair_weight = pair_weights[hi]
                            if candidate_keep is None or candidate_pair_weight is None:
                                continue
                            candidate_sensitivity = np.nanmedian(
                                sensitivity[:, ng:][:, candidate_keep], axis=1
                            )
                            coarse_model = (
                                candidate_sensitivity[coarse_pair_mask, None] * candidate_height
                                + deformation_phase_per_m_per_year[coarse_pair_mask, None]
                                * rate_grid[None, :]
                            )
                            candidate_roof_weight = roof_weight[:, candidate_keep]
                            candidate_roof_mean = np.sum(
                                candidate_roof_weight * roof_unit[:, candidate_keep], axis=1
                            ) / np.maximum(np.sum(candidate_roof_weight, axis=1), 1.0e-9)
                            candidate_relative = np.angle(candidate_roof_mean * np.conj(ground_mean))
                            candidate_residual = 1.0 - np.cos(
                                candidate_relative[coarse_pair_mask, None] - coarse_model
                            )
                            cw = candidate_pair_weight[coarse_pair_mask]
                            coarse_cost[hi] = np.sum(
                                cw[:, None] * candidate_residual, axis=0
                            ) / max(float(cw.sum()), 1.0e-6)
                        if np.any(np.isfinite(coarse_cost)):
                            coarse_flat = int(np.nanargmin(coarse_cost))
                            coarse_hi, _ = np.unravel_index(coarse_flat, coarse_cost.shape)
                            coarse_height = float(height_grid[coarse_hi])
                            coarse_profile = np.nanmin(coarse_cost, axis=1)
                            coarse_outside = np.abs(height_grid - coarse_height) >= 5.0
                            if np.any(coarse_outside):
                                coarse_cost_margin = float(
                                    np.nanmin(coarse_profile[coarse_outside]) - coarse_profile[coarse_hi]
                                )
                    audit.update(
                        {
                            "uniform_search_height_m": common_height,
                            "uniform_search_rate_m_per_year": common_rate,
                            "uniform_search_circular_cost": common_cost,
                            "uniform_search_second_cost_5m_away": second_cost,
                            "uniform_search_cost_margin": second_cost - common_cost,
                            "uniform_search_pair_weight_sum": float(best_pair_weight.sum()),
                            "height_position_support_points": int(best_candidate_keep.sum()),
                            "height_search_split_a_m": split_a_height,
                            "height_search_split_b_m": split_b_height,
                            "height_search_split_difference_m": split_height_difference,
                            "height_search_split_a_rate_m_per_year": float(rate_grid[split_a_rate_index]),
                            "height_search_split_b_rate_m_per_year": float(rate_grid[split_b_rate_index]),
                            "coarse_unambiguous_pair_count": int(coarse_pair_mask.sum()),
                            "coarse_height_search_m": finite_number(coarse_height),
                            "coarse_height_cost_margin": finite_number(coarse_cost_margin),
                            "phase_average_policy": "unit phasor weighted by coherence and amplitude dispersion",
                        }
                    )

                    global_local = np.asarray(global_punw_all[:, local_indices], dtype=np.float64)
                    corrected_local = global_local.copy()
                    if args.geometry_ground_initialization == "local_wrapped_zero":
                        ground_wrapped = np.angle(ground_complex)
                        ground_vector = np.sum(
                            ground_weight * np.exp(1j * ground_wrapped), axis=1
                        ) / np.maximum(np.sum(ground_weight, axis=1), 1.0e-9)
                        ground_centre = np.angle(ground_vector)
                        corrected_local[:, :ng] = ground_wrapped + 2.0 * np.pi * np.rint(
                            (ground_centre[:, None] - ground_wrapped) / (2.0 * np.pi)
                        )
                        ground_center_unw = np.nanmedian(corrected_local[:, :ng], axis=1)
                    else:
                        ground_center_unw = np.nanmedian(global_local[:, :ng], axis=1)
                    roof_target = (
                        ground_center_unw[:, None]
                        + sensitivity[:, ng:] * common_height
                        + deformation_phase_per_m_per_year[:, None] * common_rate
                    )
                    roof_wrapped = np.angle(roof_complex)
                    roof_unwrapped = roof_wrapped + 2.0 * np.pi * np.rint(
                        (roof_target - roof_wrapped) / (2.0 * np.pi)
                    )
                    corrected_local[:, ng:] = roof_unwrapped
                    corrected_local.astype(">f4").tofile(local_punw)
                    coupled_mask = (
                        np.ones(len(local_indices), dtype=bool)
                        if args.geometry_ground_initialization == "local_wrapped_zero"
                        else global_pmask_all[local_indices].copy()
                    )
                    if args.unwrap_engine == "height_position_coupled":
                        coupled_mask[ng:] &= best_candidate_keep
                    coupled_mask.astype(np.uint8).tofile(local_pmask)
                elif args.unwrap_engine == "global_unw_local_multidef":
                    global_unwrapped = np.asarray(global_punw_all[:, local_indices]).astype(">f4")
                    global_unwrapped.tofile(local_pdiff)
                    local_pmask_in = tmp / "pmask_in"
                    global_pmask_all[local_indices].astype(np.uint8).tofile(local_pmask_in)
                    pres = tmp / "pres"
                    pdh_initial = tmp / "pdh_initial"
                    pdef_initial = tmp / "pdef_initial"
                    psigma_initial = tmp / "psigma_initial"
                    run(
                        [
                            "multi_def_pt", str(local_plist), str(local_pmask_in), str(pslc_par), "-", str(itab),
                            str(pbase), "0", str(local_pdiff), "0", str(local_ref), str(pres), str(pdh_initial),
                            str(pdef_initial), str(local_punw), str(psigma_initial), str(local_pmask),
                            str(args.dh_max), "-0.01", "0.01", "100", str(args.multi_sigma_max),
                            str(args.multi_sigma_max2), str(args.multi_model), "1", "1", "-1", "-1",
                            str(args.ground_radius),
                        ],
                        log,
                    )
                elif args.unwrap_engine == "global_model_residual_mcf":
                    local_global_mask = global_pmask_all[local_indices]
                    local_global_mask.astype(np.uint8).tofile(local_pmask)
                    global_unwrapped = np.asarray(global_punw_all[:, local_indices]).astype(np.float64)
                    global_residual = np.asarray(global_pres_all[:, local_indices]).astype(np.float64)
                    global_model = global_unwrapped - global_residual
                    residual_complex = np.exp(1j * global_residual).astype(">c8")
                    residual_complex.tofile(local_pdiff)
                    residual_unw_path = tmp / "residual_unw"
                    run(
                        [
                            "mcf_pt", str(local_plist), str(local_pmask), str(local_pdiff), "-", str(local_pcc), "-",
                            str(residual_unw_path), f"{range_spacing:.8f}", f"{azimuth_spacing:.8f}", str(local_ref), "0",
                            "1", "1", "-", "-", "0",
                        ],
                        log,
                    )
                    residual_unwrapped = np.fromfile(residual_unw_path, dtype=">f4")
                    if residual_unwrapped.size != npairs * len(local_indices):
                        raise RuntimeError("residual mcf_pt output size mismatch")
                    residual_unwrapped = residual_unwrapped.reshape(npairs, len(local_indices))
                    corrected_unwrapped = global_model + residual_unwrapped
                    corrected_unwrapped.astype(">f4").tofile(local_punw)
                elif args.unwrap_engine == "zjc_lgr":
                    local_complex = np.asarray(pdiff_all[:, local_indices])
                    local_range = local_meta["range_pixel"].to_numpy(np.int64)
                    local_azimuth = local_meta["azimuth_pixel"].to_numpy(np.int64)
                    r0, r1 = int(local_range.min()), int(local_range.max()) + 1
                    a0, a1 = int(local_azimuth.min()), int(local_azimuth.max()) + 1
                    local_rows = local_azimuth - a0
                    local_cols = local_range - r0
                    valid_patch = np.zeros((a1 - a0, r1 - r0), dtype=bool)
                    valid_patch[local_rows, local_cols] = True
                    zjc_unwrapped = np.full((npairs, len(local_indices)), np.nan, dtype=np.float32)
                    zjc_status: dict[str, int] = {}
                    for record in range(npairs):
                        phase_patch = np.full(valid_patch.shape, np.nan, dtype=np.float32)
                        phase_patch[local_rows, local_cols] = np.angle(local_complex[record]).astype(np.float32)
                        coherence_patch = np.full(valid_patch.shape, np.nan, dtype=np.float32)
                        coherence_patch[local_rows, local_cols] = weights[record].astype(np.float32)
                        da_patch = np.full(valid_patch.shape, np.nan, dtype=np.float32)
                        da_patch[local_rows, local_cols] = local_meta["amplitude_dispersion"].to_numpy(np.float32)
                        unwrapped_patch, info = paper_like_unwrap(
                            phase_patch,
                            valid_patch,
                            wavelength,
                            coherence_patch=coherence_patch,
                            amplitude_dispersion_patch=da_patch,
                        )
                        status = str(info.get("status", "unknown"))
                        zjc_status[status] = zjc_status.get(status, 0) + 1
                        values = unwrapped_patch[local_rows, local_cols]
                        if np.isfinite(values[local_ref]):
                            values = values - values[local_ref] + np.angle(local_complex[record, local_ref])
                        zjc_unwrapped[record] = values
                    audit["zjc_unwrap_status_counts"] = zjc_status
                    zjc_unwrapped.astype(">f4").tofile(local_punw)
                elif args.unwrap_engine == "mcf_pt":
                    run(
                        [
                            "mcf_pt", str(local_plist), str(local_pmask), str(local_pdiff), "-", str(local_pcc), "-",
                            str(local_punw), f"{range_spacing:.8f}", f"{azimuth_spacing:.8f}", str(local_ref), "0",
                            "1", "1", "-", "-", "0",
                        ],
                        log,
                    )
                else:
                    pres = tmp / "pres"
                    pdh_initial = tmp / "pdh_initial"
                    pdef_initial = tmp / "pdef_initial"
                    psigma_initial = tmp / "psigma_initial"
                    run(
                        [
                            "multi_def_pt", str(local_plist), "-", str(pslc_par), "-", str(itab), str(pbase), "0",
                            str(local_pdiff), "1", str(local_ref), str(pres), str(pdh_initial), str(pdef_initial),
                            str(local_punw), str(psigma_initial), str(local_pmask), str(args.dh_max), "-0.01", "0.01",
                            "100", str(args.multi_sigma_max), str(args.multi_sigma_max2), str(args.multi_model),
                            "1", "1", "-1", "-1", str(args.ground_radius),
                        ],
                        log,
                    )
                expected_unw = npairs * len(local_indices)
                local_unw = np.fromfile(local_punw, dtype=">f4")
                if local_unw.size != expected_unw:
                    raise RuntimeError(f"mcf_pt output size {local_unw.size}, expected {expected_unw}")
                local_unw = local_unw.reshape(npairs, len(local_indices))
                unwrap_valid = np.all(np.isfinite(local_unw) & (local_unw != 0.0), axis=0)
                if args.unwrap_engine in {
                    "multi_def_pt", "global_multi_def", "global_model_residual_mcf", "global_unw_local_multidef",
                    "building_uniform_height",
                    "height_position_coupled", "geometry_wrapped_init",
                }:
                    unwrap_valid &= np.fromfile(local_pmask, dtype=np.uint8)[: len(local_indices)] > 0
                unwrap_valid[local_ref] = True
                audit["unwrap_valid_support_points"] = int(unwrap_valid[len(local_ground_indices):].sum())
                audit["unwrap_engine"] = args.unwrap_engine
                if int(unwrap_valid[len(local_ground_indices):].sum()) < args.min_output_points:
                    raise RuntimeError(f"too few support points valid in every {args.unwrap_engine} record")
                unwrap_mask = unwrap_valid.astype(np.uint8)
                unwrap_mask.tofile(local_pmask)

                itab_ts = tmp / "pairs_ts.itab"
                pdiff_ts = tmp / "pdiff_ts"
                pdiff_sim = tmp / "pdiff_sim"
                psigma_ts = tmp / "psigma_ts"
                phgt = tmp / "phgt"
                prate = tmp / "prate"
                pconst = tmp / "pconst"
                psigma_fit = tmp / "psigma_fit"
                local_mb_sigma: Path | None = mb_sigma_file
                if args.mb_sigma_mode == "local_roof_coherence":
                    roof_pair_coherence = np.nanmedian(
                        np.asarray(pcc_all[:, support_indices], dtype=float), axis=1
                    )
                    roof_pair_coherence = np.clip(roof_pair_coherence, 1.0e-3, 0.999999)
                    # Wrapped-normal coherence relation: gamma=exp(-sigma_phi^2/2).
                    # This gives a physically interpretable phase standard deviation
                    # for mb_pt without consulting vector or reference heights.
                    pair_sigma = np.sqrt(-2.0 * np.log(roof_pair_coherence))
                    pair_sigma = np.clip(pair_sigma, args.mb_sigma_min, args.mb_sigma_max)
                    tempering = float(np.clip(args.mb_coherence_tempering, 0.0, 1.0))
                    sigma_centre = float(np.exp(np.nanmean(np.log(pair_sigma))))
                    pair_sigma = sigma_centre * np.power(pair_sigma / sigma_centre, tempering)
                    sigma_ratio_limit = max(float(args.mb_sigma_ratio_limit), 1.0)
                    if np.isfinite(sigma_ratio_limit):
                        pair_sigma = np.clip(
                            pair_sigma,
                            sigma_centre / sigma_ratio_limit,
                            sigma_centre * sigma_ratio_limit,
                        )
                    local_mb_sigma = tmp / "mb_pair_sigma_local_roof_coherence.txt"
                    local_mb_sigma.write_text(
                        "\n".join(
                            f"{index} 0.00000000 {value:.8f}"
                            for index, value in enumerate(pair_sigma, start=1)
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    audit["mb_pair_sigma_median_rad"] = float(np.nanmedian(pair_sigma))
                    audit["mb_coherence_tempering"] = tempering
                    audit["mb_sigma_ratio_limit"] = finite_number(sigma_ratio_limit)
                elif args.mb_sigma_mode == "local_ground_residual":
                    # GAMMA recommends using the simulated-minus-observed residuals
                    # to iteratively improve the inversion.  Estimate layer noise
                    # only from the nearby stable-ground block so roof height does
                    # not tune its own weights and no vector height enters P.
                    initial_itab_ts = tmp / "pairs_ts_equal_initial.itab"
                    initial_pdiff_ts = tmp / "pdiff_ts_equal_initial"
                    initial_pdiff_sim = tmp / "pdiff_sim_equal_initial"
                    initial_psigma_ts = tmp / "psigma_ts_equal_initial"
                    initial_phgt = tmp / "phgt_equal_initial"
                    initial_prate = tmp / "prate_equal_initial"
                    initial_pconst = tmp / "pconst_equal_initial"
                    initial_psigma_fit = tmp / "psigma_fit_equal_initial"
                    run(
                        [
                            "mb_pt", str(local_plist), str(local_pmask), str(pslc_par), str(itab), str(local_punw),
                            str(local_ref), "-", str(initial_itab_ts), str(initial_pdiff_ts),
                            str(initial_pdiff_sim), str(initial_psigma_ts), "1", str(initial_phgt),
                            str(args.sbas_gamma), str(initial_prate), str(initial_pconst),
                            str(initial_psigma_fit), str(Path(args.geometric_reference_par)),
                        ],
                        log,
                    )
                    initial_sim = np.fromfile(initial_pdiff_sim, dtype=">f4")
                    if initial_sim.size != expected_unw:
                        raise RuntimeError(
                            f"mb_pt initial simulated-phase size {initial_sim.size}, expected {expected_unw}"
                        )
                    initial_sim = initial_sim.reshape(npairs, len(local_indices)).astype(float)
                    ground_valid_for_sigma = unwrap_valid[: len(local_ground_indices)].copy()
                    # The reference point has a forced zero residual and would
                    # otherwise make small local samples look over-confident.
                    if local_ref < len(local_ground_indices):
                        ground_valid_for_sigma[local_ref] = False
                    if int(ground_valid_for_sigma.sum()) < 3:
                        raise RuntimeError("too few stable-ground residuals for mb_pt layer weighting")
                    ground_residual = (
                        local_unw[:, : len(local_ground_indices)][:, ground_valid_for_sigma]
                        - initial_sim[:, : len(local_ground_indices)][:, ground_valid_for_sigma]
                    )
                    pair_centre = np.nanmedian(ground_residual, axis=1)
                    pair_mad = np.nanmedian(
                        np.abs(ground_residual - pair_centre[:, None]), axis=1
                    )
                    pair_sigma = 1.4826 * pair_mad
                    finite_positive = np.isfinite(pair_sigma) & (pair_sigma > 0.0)
                    fallback_sigma = (
                        float(np.nanmedian(pair_sigma[finite_positive]))
                        if np.any(finite_positive)
                        else 1.0
                    )
                    pair_sigma = np.where(finite_positive, pair_sigma, fallback_sigma)
                    pair_sigma = np.clip(pair_sigma, args.mb_sigma_min, args.mb_sigma_max)
                    sigma_centre = float(np.exp(np.nanmean(np.log(pair_sigma))))
                    sigma_ratio_limit = max(float(args.mb_sigma_ratio_limit), 1.0)
                    if np.isfinite(sigma_ratio_limit):
                        pair_sigma = np.clip(
                            pair_sigma,
                            sigma_centre / sigma_ratio_limit,
                            sigma_centre * sigma_ratio_limit,
                        )
                    local_mb_sigma = tmp / "mb_pair_sigma_local_ground_residual.txt"
                    local_mb_sigma.write_text(
                        "\n".join(
                            f"{index} {centre:.8f} {value:.8f}"
                            for index, (centre, value) in enumerate(
                                zip(pair_centre, pair_sigma, strict=True), start=1
                            )
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    audit["mb_pair_sigma_median_rad"] = float(np.nanmedian(pair_sigma))
                    audit["mb_pair_sigma_p05_rad"] = float(np.nanpercentile(pair_sigma, 5))
                    audit["mb_pair_sigma_p95_rad"] = float(np.nanpercentile(pair_sigma, 95))
                    audit["mb_sigma_ground_points"] = int(ground_valid_for_sigma.sum())
                    audit["mb_sigma_ratio_limit"] = finite_number(sigma_ratio_limit)
                run(
                    [
                        "mb_pt", str(local_plist), str(local_pmask), str(pslc_par), str(itab), str(local_punw),
                        str(local_ref), str(local_mb_sigma) if local_mb_sigma else "-", str(itab_ts), str(pdiff_ts), str(pdiff_sim), str(psigma_ts), "1",
                        str(phgt), str(args.sbas_gamma), str(prate), str(pconst), str(psigma_fit),
                        str(Path(args.geometric_reference_par)),
                    ],
                    log,
                )
                height = np.fromfile(phgt, dtype=">f4")[: len(local_indices)].astype(float)
                sigma = np.fromfile(psigma_fit, dtype=">f4")[: len(local_indices)].astype(float)
                solved = unwrap_valid & np.isfinite(height) & np.isfinite(sigma) & (sigma > 0.0)
                ground_solved = solved[: len(local_ground_indices)]
                support_solved = solved[len(local_ground_indices):]
                audit["gamma_sbas_output_points"] = int(support_solved.sum())
                if int(support_solved.sum()) < args.min_output_points:
                    raise RuntimeError("too few finite GAMMA SBAS support-point solutions")
                if int(ground_solved.sum()) < 1:
                    raise RuntimeError("no finite local ground solution")

                solved_ground_height = height[: len(local_ground_indices)][ground_solved]
                solved_ground_xy = used_ground_xy[ground_solved]
                occupied_sector_medians: list[float] = []
                if args.ground_statistic == "sector_median" and args.ground_sectors >= 2:
                    relative_xy = solved_ground_xy - centroid[None, :]
                    angles = np.mod(np.arctan2(relative_xy[:, 1], relative_xy[:, 0]), 2.0 * np.pi)
                    sector_index = np.floor(
                        angles / (2.0 * np.pi / float(args.ground_sectors))
                    ).astype(np.int64)
                    occupied_sector_medians = [
                        float(np.nanmedian(solved_ground_height[sector_index == sector]))
                        for sector in range(args.ground_sectors)
                        if np.any(sector_index == sector)
                    ]
                audit["ground_occupied_sectors"] = int(len(occupied_sector_medians))
                if len(occupied_sector_medians) >= args.min_ground_sectors:
                    sector_values = np.asarray(occupied_sector_medians, dtype=float)
                    local_ground_height = float(np.nanmedian(sector_values))
                    q1_sector, q3_sector = np.nanpercentile(sector_values, [25, 75])
                    audit["ground_sector_median_iqr_m"] = float(q3_sector - q1_sector)
                    if len(sector_values) >= 4:
                        leave_one_out = np.asarray(
                            [np.nanmedian(np.delete(sector_values, index)) for index in range(len(sector_values))],
                            dtype=float,
                        )
                        audit["ground_sector_leave_one_out_range_m"] = float(
                            np.nanmax(leave_one_out) - np.nanmin(leave_one_out)
                        )
                else:
                    local_ground_height = float(np.nanmedian(solved_ground_height))
                support_height = height[len(local_ground_indices):][support_solved] - local_ground_height
                support_sigma = sigma[len(local_ground_indices):][support_solved]
                support_point_ids = support_indices[support_solved]
                if args.building_statistic == "upper_cluster" and len(support_height) >= 4:
                    upper_threshold = float(np.nanpercentile(support_height, 75))
                    statistic_keep = support_height >= upper_threshold
                else:
                    upper_threshold = None
                    statistic_keep = np.ones(len(support_height), dtype=bool)
                statistic_height = support_height[statistic_keep]
                statistic_sigma = support_sigma[statistic_keep]
                statistic_point_ids = support_point_ids[statistic_keep]
                q1, q3 = np.nanpercentile(statistic_height, [25, 75])
                iqr = float(q3 - q1)
                iqr_keep = (statistic_height >= q1 - 1.5 * iqr) & (statistic_height <= q3 + 1.5 * iqr)
                cleaned_height = statistic_height[iqr_keep]
                cleaned_sigma = statistic_sigma[iqr_keep]
                cleaned_point_ids = statistic_point_ids[iqr_keep]
                building_height = float(np.nanmedian(cleaned_height))
                median_sigma = float(np.nanmedian(cleaned_sigma))
                audit.update(
                    {
                        "status": "solved",
                        "failure_reason": None,
                        "local_ground_delta_height_m": local_ground_height,
                        "aggregation": args.building_statistic,
                        "upper_cluster_threshold_m": upper_threshold,
                        "points_after_statistic": int(len(statistic_height)),
                        "points_after_iqr": int(len(cleaned_height)),
                        "insar_height_m": building_height,
                        "roof_height_iqr_m": iqr,
                        "median_phase_sigma_rad": median_sigma,
                        "median_mean_coherence": float(np.nanmedian(metadata.iloc[cleaned_point_ids]["mean_coherence"])),
                    }
                )
                rejection_reasons: list[str] = []
                if len(cleaned_height) < args.min_output_points:
                    rejection_reasons.append("too_few_points_after_aggregation")
                if median_sigma > args.max_median_phase_sigma:
                    rejection_reasons.append("phase_sigma_above_limit")
                if iqr > args.max_roof_iqr:
                    rejection_reasons.append("roof_iqr_above_limit")
                if not (args.min_physical_height <= building_height <= args.max_physical_height):
                    rejection_reasons.append("outside_physical_height_range")
                if args.unwrap_engine in {"building_uniform_height", "height_position_coupled"}:
                    if float(audit.get("uniform_search_cost_margin", float("-inf"))) < args.min_height_search_cost_margin:
                        rejection_reasons.append("height_search_cost_margin_below_limit")
                    if float(audit.get("height_search_split_difference_m", float("inf"))) > args.max_height_search_split_difference:
                        rejection_reasons.append("height_search_pair_split_unstable")
                audit["accepted"] = not rejection_reasons
                audit["rejection_reasons"] = rejection_reasons
                np.savez_compressed(
                    points_shard,
                    point_index=support_point_ids,
                    height_m=support_height.astype(np.float32),
                    phase_sigma_rad=support_sigma.astype(np.float32),
                    statistic_keep=statistic_keep,
                )
        except Exception as exc:  # retain a per-building failure record and continue
            audit["status"] = "failed"
            audit["failure_reason"] = str(exc)
            audit["accepted"] = False
        write_json(shard, audit)
        if ordinal % 25 == 0 or ordinal == len(clean_ids):
            print(f"processed {ordinal}/{len(clean_ids)} buildings", flush=True)

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(shards.glob("[0-9]*.json"), key=lambda p: int(p.stem))]
    audit_table = pd.DataFrame(rows)
    audit_csv = out / "building_audit_insar_only.csv"
    audit_table.to_csv(audit_csv, index=False)
    accepted = audit_table[audit_table["accepted"].eq(True)].copy()
    result_columns = [
        "clean_id", "insar_height_m", "support_points", "ground_points", "unwrap_valid_support_points",
        "gamma_sbas_output_points", "points_after_iqr", "roof_height_iqr_m", "median_phase_sigma_rad",
        "median_mean_coherence",
    ]
    result = accepted[[column for column in result_columns if column in accepted]].copy()
    result["solution_source"] = f"GAMMA_building_isolated_{args.unwrap_engine}_mb_pt_local_ground"
    # Use integer flags so GeoJSON/GDAL round-trips cannot turn the strings
    # "False"/"True" into truthy values in downstream Python code.
    result["filled_from_prior"] = np.int8(0)
    result["insar_only"] = np.int8(args.unwrap_engine != "geometry_wrapped_init")

    # The vector height column is first read here for comparison.  The
    # geometry_wrapped_init branch has already consumed an external R-D
    # interval audit derived from prior-height projection, so it is explicitly
    # classified as hybrid rather than InSAR-only.
    buildings = gpd.read_file(args.buildings)
    prior = (
        buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"})
        if "height" in buildings
        else pd.DataFrame({"clean_id": buildings["clean_id"]})
    )
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result["insar_height_m"] - result.get("prior_height_m", np.nan)
    Path(args.result_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.result_csv, index=False)
    result_geo = buildings.merge(result, on="clean_id", how="left", suffixes=("_vector", ""))
    Path(args.result_geojson).parent.mkdir(parents=True, exist_ok=True)
    result_geo.to_file(args.result_geojson, driver="GeoJSON")

    prior_pair = result[["insar_height_m", "prior_height_m"]].dropna() if "prior_height_m" in result else pd.DataFrame()
    prior_policy = (
        "no value copied or filled; the external prior-derived R-D roof audit is used only to initialize 2pi phase cycles, "
        "then GAMMA mb_pt re-estimates height; comparison to the same prior is non-independent"
        if args.unwrap_engine == "geometry_wrapped_init"
        else "prior loaded only after InSAR-only solve and internal acceptance; comparison-only; never used for selection, unwrapping, inversion, QC, aggregation, or filling"
    )
    summary = {
        "method": f"per-building SAR support + nearby stable ground -> {args.unwrap_engine} -> GAMMA mb_pt -> local-ground subtraction -> IQR/{args.building_statistic} aggregation",
        "unwrap_engine": args.unwrap_engine,
        "global_punw": args.global_punw,
        "wrapped_height_audit": args.wrapped_height_audit if args.unwrap_engine == "geometry_wrapped_init" else None,
        "initialization_height_offset_m": (
            args.initialization_height_offset_m
            if args.unwrap_engine == "geometry_wrapped_init"
            else None
        ),
        "geometry_ground_initialization": (
            args.geometry_ground_initialization
            if args.unwrap_engine == "geometry_wrapped_init"
            else None
        ),
        "product_class": "hybrid_GAMMA_SBAS_with_RD_cycle_initialization" if args.unwrap_engine == "geometry_wrapped_init" else "InSAR_only",
        "prior_policy": prior_policy,
        "pairs": npairs,
        "mb_sigma_mode": args.mb_sigma_mode,
        "mb_coherence_tempering": (
            float(np.clip(args.mb_coherence_tempering, 0.0, 1.0))
            if args.mb_sigma_mode == "local_roof_coherence"
            else None
        ),
        "mb_sigma_ratio_limit": (
            finite_number(args.mb_sigma_ratio_limit)
            if args.mb_sigma_mode in {"local_roof_coherence", "local_ground_residual"}
            else None
        ),
        "paper_pair_sigma_rad": args.paper_pair_sigma_rad if args.mb_sigma_mode == "paper_threshold" else None,
        "paper_compliant_pairs": int(np.sum(mb_pair_sigmas == 1.0)) if mb_pair_sigmas is not None else None,
        "paper_downweighted_pairs": int(np.sum(mb_pair_sigmas != 1.0)) if mb_pair_sigmas is not None else None,
        "interferogram_dir": args.interferogram_dir,
        "support_owner": args.support_owner,
        "spatial_support_projection": args.spatial_support_projection,
        "support_min_mean_coherence": args.support_min_mean_coherence,
        "closure_bad_count": args.closure_bad_count,
        "max_closure_bad_triangles": args.max_closure_bad_triangles,
        "candidate_buildings": len(clean_ids),
        "processed_buildings": int(len(audit_table)),
        "gamma_solved_buildings": int(audit_table["status"].eq("solved").sum()),
        "accepted_buildings": int(len(result)),
        "height_median_m": finite_number(result["insar_height_m"].median()) if len(result) else None,
        "height_p05_p95_m": [
            finite_number(result["insar_height_m"].quantile(0.05)),
            finite_number(result["insar_height_m"].quantile(0.95)),
        ] if len(result) else None,
        "phase_sigma_median_rad": finite_number(result["median_phase_sigma_rad"].median()) if len(result) else None,
        "failure_counts": audit_table["failure_reason"].fillna("none").value_counts().to_dict(),
        "rejection_policy": {
            "min_output_points": args.min_output_points,
            "max_median_phase_sigma": args.max_median_phase_sigma,
            "max_roof_iqr_m": args.max_roof_iqr,
            "physical_height_range_m": [args.min_physical_height, args.max_physical_height],
            "min_height_search_cost_margin": args.min_height_search_cost_margin,
            "max_height_search_split_difference_m": finite_number(args.max_height_search_split_difference),
        },
        "prior_comparison_after_acceptance_non_independent": {
            "count": int(len(prior_pair)),
            "median_difference_m": finite_number((prior_pair["insar_height_m"] - prior_pair["prior_height_m"]).median()) if len(prior_pair) else None,
            "mae_m": finite_number((prior_pair["insar_height_m"] - prior_pair["prior_height_m"]).abs().mean()) if len(prior_pair) else None,
            "correlation": finite_number(prior_pair.corr().iloc[0, 1]) if len(prior_pair) > 1 else None,
        },
        "artifacts": {
            "audit_csv": str(audit_csv),
            "result_csv": args.result_csv,
            "result_geojson": args.result_geojson,
            "gamma_log": str(log),
            "building_shards": str(shards),
        },
    }
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
