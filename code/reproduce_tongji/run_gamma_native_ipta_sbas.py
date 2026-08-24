#!/usr/bin/env python3
"""Run a GAMMA-native differential-InSAR/IPTA SBAS building-height chain.

The building ``height`` attribute is never used to form an interferogram, select
or reject a point, estimate a parameter, or fill a missing result.  It is read
only after the GAMMA solution has been frozen, to produce diagnostic comparison
columns.  The DEM used for differential interferometry is a zero/bare reference
surface; final building height is roof elevation minus a locally estimated
stable-ground elevation from the same GAMMA ``multi_def_pt`` solution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_polygon


ROWS = 630
COLS = 900


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
    compat = "/tmp/gamma_gdal_compat"
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
        raise RuntimeError(f"GAMMA command failed ({proc.returncode}): {' '.join(cmd)}; see {log}")


def read_be_float(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    data = np.fromfile(path, dtype=">f4").astype(np.float32)
    return data.reshape(shape) if shape else data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--rslc-dir", default="data/tongji_rslc")
    p.add_argument("--reference-date", default="20200708")
    p.add_argument("--reference-hgt", default="work/gamma_native_ipta_sbas/reference_flat_4m.hgt")
    p.add_argument("--roof-owner", default="work/roof_sbas_optimized/roof_core_clean_id_mask.npy")
    p.add_argument("--projection", default="work/zjc_original_reproduction/20200708_all_building_sar_feature_refined_projection_sar.geojson")
    p.add_argument("--point-domain", choices=["roof", "layover", "search_union"], default="roof")
    p.add_argument("--building-statistic", choices=["median", "upper_cluster"], default="median")
    p.add_argument("--ground-mask", default="work/roof_sbas_optimized/stable_ground_reference_mask.npy")
    p.add_argument("--amplitude-dispersion", default="work/mli/amplitude_dispersion_crop_bmp.npy")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--out-dir", default="work/gamma_native_ipta_sbas")
    p.add_argument("--interferogram-source-dir", default=None, help="Reuse an existing interferograms directory")
    p.add_argument("--result-csv", default="results/tables/tongji_building_height_gamma_native_ipta_sbas.csv")
    p.add_argument("--result-geojson", default="results/geodata/tongji_building_height_gamma_native_ipta_sbas.geojson")
    p.add_argument("--summary", default="results/metadata/tongji_building_height_gamma_native_ipta_sbas_summary.json")
    p.add_argument("--adf-alpha", type=float, default=0.4)
    p.add_argument("--adf-nfft", type=int, default=32)
    p.add_argument("--max-da", type=float, default=0.40)
    p.add_argument("--min-mean-coherence", type=float, default=0.65)
    p.add_argument("--min-coherent-pairs", type=int, default=12)
    p.add_argument("--pair-coherence-threshold", type=float, default=0.55)
    p.add_argument("--dh-max", type=float, default=180.0)
    p.add_argument("--sigma-max", type=float, default=1.2)
    p.add_argument("--sigma-max2", type=float, default=0.75)
    p.add_argument("--model", type=int, default=2)
    p.add_argument("--sbas-engine", choices=["mb_pt", "multi_def_pt"], default="mb_pt")
    p.add_argument("--sbas-gamma", type=float, default=1.0)
    p.add_argument("--mb-sigma-mode", choices=["equal", "stable_ground_mad"], default="equal")
    p.add_argument("--mb-sigma-min", type=float, default=0.15)
    p.add_argument("--mb-sigma-max", type=float, default=3.0)
    p.add_argument("--ipta-phase-source", choices=["direct", "raw", "adf"], default="adf")
    p.add_argument("--no-baseline-refinement", action="store_true")
    p.add_argument("--local-ground-neighbours", type=int, default=30)
    p.add_argument("--local-ground-radius", type=float, default=150.0)
    p.add_argument("--skip-interferograms", action="store_true")
    return p.parse_args()


def rasterize_polygon(geometry) -> np.ndarray:
    mask = np.zeros((ROWS, COLS), dtype=bool)
    parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for part in parts:
        if part.is_empty:
            continue
        xy = np.asarray(part.exterior.coords)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape=mask.shape)
        mask[rr, cc] = True
        for ring in part.interiors:
            hole = np.asarray(ring.coords)
            rr, cc = draw_polygon(hole[:, 1], hole[:, 0], shape=mask.shape)
            mask[rr, cc] = False
    return mask


def layover_owner_mask(projection_path: str) -> tuple[np.ndarray, int]:
    projection = gpd.read_file(projection_path)
    owner = np.zeros((ROWS, COLS), dtype=np.int32)
    conflicts = np.zeros((ROWS, COLS), dtype=bool)
    for row in projection[projection["surface"].eq("layover")].sort_values("clean_id").itertuples(index=False):
        clean_id = int(row.clean_id)
        mask = rasterize_polygon(row.geometry)
        conflicts |= mask & (owner > 0) & (owner != clean_id)
        owner[mask & (owner == 0)] = clean_id
    owner[conflicts] = 0
    return owner, int(conflicts.sum())


def layover_union_mask(projection_path: str) -> tuple[np.ndarray, int]:
    """Return one shared candidate domain; ownership is resolved per building later."""
    projection = gpd.read_file(projection_path)
    owner = np.zeros((ROWS, COLS), dtype=np.int32)
    for geometry in projection[projection["surface"].eq("layover")].geometry:
        owner[rasterize_polygon(geometry)] = 1
    return owner, 0


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    intf_root = Path(args.interferogram_source_dir) if args.interferogram_source_dir else out / "interferograms"
    ipta = out / "ipta"
    intf_root.mkdir(parents=True, exist_ok=True)
    ipta.mkdir(parents=True, exist_ok=True)
    log = out / "gamma_native.log"
    log.write_text("", encoding="utf-8")

    pairs = pd.read_csv(args.pairs_csv, dtype={"master": str, "slave": str})
    pairs = pairs.drop_duplicates(["master", "slave"]).reset_index(drop=True)
    rslc_dir = Path(args.rslc_dir)
    geometric_reference_par = rslc_dir / f"{args.reference_date}.rslc.par"
    reference_hgt = Path(args.reference_hgt)
    reference_hgt.parent.mkdir(parents=True, exist_ok=True)
    if not reference_hgt.exists():
        # GAMMA treats an all-zero simulated phase raster as NULL in sub_phase.
        # A spatially constant 4 m surface is therefore used only as a non-NULL
        # phase datum.  It cancels in roof-minus-local-ground height and is not
        # an assumed final ground elevation.
        np.full((ROWS, COLS), 4.0, dtype=">f4").tofile(reference_hgt)
    diff_files: list[Path] = []
    raw_diff_files: list[Path] = []
    cc_files: list[Path] = []

    for pair_row in pairs.itertuples(index=False):
        master, slave = str(pair_row.master), str(pair_row.slave)
        pair = f"{master}_{slave}"
        pair_dir = intf_root / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        off = pair_dir / f"{pair}.off"
        sim = pair_dir / f"{pair}.sim_unw"
        raw_int = pair_dir / f"{pair}.int"
        diff_par = pair_dir / f"{pair}.diff_par"
        raw_diff = pair_dir / f"{pair}.diff"
        filt = pair_dir / f"{pair}.adf.diff"
        cc = pair_dir / f"{pair}.cc"
        diff_files.append(filt)
        raw_diff_files.append(raw_diff)
        cc_files.append(cc)
        if args.skip_interferograms and filt.exists() and cc.exists():
            continue
        m_slc = rslc_dir / f"{master}.rslc"
        s_slc = rslc_dir / f"{slave}.rslc"
        m_par = rslc_dir / f"{master}.rslc.par"
        s_par = rslc_dir / f"{slave}.rslc.par"
        run(["create_offset", str(m_par), str(s_par), str(off), "1", "1", "1", "0"], log)
        run(["phase_sim_orb", str(m_par), str(s_par), str(off), str(reference_hgt), str(sim), str(geometric_reference_par), "-", "-", "1", "1"], log)
        # The installed GAMMA 2021 SLC_diff_intf writes zeros for this TSX
        # SCOMPLEX crop.  The documented equivalent is used: form the complex
        # interferogram, subtract simulated phase, then filter.
        run(["SLC_intf", str(m_slc), str(s_slc), str(m_par), str(s_par), str(off), str(raw_int), "1", "1", "-", "-", "0", "0"], log)
        run(["create_diff_par", str(off), "-", str(diff_par), "0", "0"], log)
        run(["sub_phase", str(raw_int), str(sim), str(diff_par), str(raw_diff), "1", "0", "0"], log)
        run(["adf", str(raw_diff), str(filt), str(cc), str(COLS), str(args.adf_alpha), str(args.adf_nfft), "5"], log)

    coherence_count = np.zeros((ROWS, COLS), dtype=np.int16)
    coherence_valid_count = np.zeros((ROWS, COLS), dtype=np.int16)
    coherence_sum = np.zeros((ROWS, COLS), dtype=np.float64)
    for cc_file in cc_files:
        cc = read_be_float(cc_file, (ROWS, COLS))
        finite = np.isfinite(cc)
        coherent = finite & (cc >= args.pair_coherence_threshold)
        coherence_count += coherent.astype(np.int16)
        coherence_valid_count += finite.astype(np.int16)
        coherence_sum += np.where(finite, cc, 0.0)
    coherence_mean = np.divide(
        coherence_sum,
        coherence_valid_count,
        out=np.full((ROWS, COLS), np.nan, dtype=np.float64),
        where=coherence_valid_count > 0,
    )
    da = np.load(args.amplitude_dispersion).astype(np.float64)
    if args.point_domain == "roof":
        roof_owner = np.load(args.roof_owner).astype(np.int32)
        domain_conflicts = 0
    elif args.point_domain == "search_union":
        roof_owner, domain_conflicts = layover_union_mask(args.projection)
    else:
        roof_owner, domain_conflicts = layover_owner_mask(args.projection)
    ground_base = np.load(args.ground_mask).astype(bool)
    quality = (
        np.isfinite(da)
        & (da <= args.max_da)
        & np.isfinite(coherence_mean)
        & (coherence_mean >= args.min_mean_coherence)
        & (coherence_count >= args.min_coherent_pairs)
    )
    roof = (roof_owner > 0) & quality
    ground = ground_base & quality
    if not ground.any() or not roof.any():
        raise RuntimeError(f"No GAMMA IPTA candidates: roof={int(roof.sum())}, ground={int(ground.sum())}")
    combined = roof | ground
    candidate = np.where(combined, 1.0, 0.0).astype(">f4")
    candidate_file = ipta / "candidate.float"
    candidate.tofile(candidate_file)
    plist = ipta / "tongji.plist"
    run(["thres_im_pt", str(candidate_file), str(COLS), str(plist), "0.5", "1.5", "1", "1"], log)

    # thres_im_pt enumerates the image in line-major order.  Keep this exact
    # ordering for the point metadata and the GAMMA output records.
    az, rg = np.nonzero(combined)
    point_table = pd.DataFrame(
        {
            "point_index": np.arange(az.size, dtype=np.int64),
            "range_pixel": rg,
            "azimuth_pixel": az,
            "point_class": np.where(ground[az, rg], "ground", "roof"),
            "clean_id": roof_owner[az, rg],
            "amplitude_dispersion": da[az, rg],
            "mean_coherence": coherence_mean[az, rg],
            "coherent_pair_count": coherence_count[az, rg],
        }
    )
    ground_indices = point_table.index[point_table["point_class"].eq("ground")].to_numpy()
    ground_scores = point_table.loc[ground_indices, "mean_coherence"].to_numpy() - point_table.loc[ground_indices, "amplitude_dispersion"].to_numpy()
    np_ref = int(ground_indices[int(np.argmax(ground_scores))])
    point_table["is_phase_reference"] = point_table["point_index"].eq(np_ref)
    point_table.to_csv(ipta / "point_metadata.csv", index=False)

    dates = sorted(set(pairs["master"]) | set(pairs["slave"]))
    date_to_rec = {date: idx + 1 for idx, date in enumerate(dates)}
    slc_par_tab = ipta / "slc_par.tab"
    slc_par_tab.write_text("\n".join(str(rslc_dir / f"{date}.rslc.par") for date in dates) + "\n", encoding="utf-8")
    slc_tab = ipta / "slc.tab"
    slc_tab.write_text(
        "\n".join(f"{rslc_dir / (date + '.rslc')} {rslc_dir / (date + '.rslc.par')}" for date in dates) + "\n",
        encoding="utf-8",
    )
    itab = ipta / "pairs.itab"
    itab.write_text(
        "\n".join(
            f"{date_to_rec[row.master]} {date_to_rec[row.slave]} {idx + 1} 1"
            for idx, row in enumerate(pairs.itertuples(index=False))
        )
        + "\n",
        encoding="utf-8",
    )
    diff_tab = ipta / "diff.tab"
    inversion_diff_files = raw_diff_files if args.ipta_phase_source == "raw" else diff_files
    diff_tab.write_text("\n".join(str(path) for path in inversion_diff_files) + "\n", encoding="utf-8")

    pslc_par = ipta / "pSLC_par"
    pbase = ipta / "pbase"
    pdiff = ipta / "pdiff"
    for target in [pslc_par, pbase, pdiff]:
        if target.exists():
            target.unlink()
    run(["SLC_par2pt", str(slc_par_tab), str(pslc_par), "-"], log)
    run(["base_orbit_pt", str(pslc_par), str(itab), "-", str(pbase)], log)
    if args.ipta_phase_source == "direct":
        pslc = ipta / "pSLC"
        pint = ipta / "pint"
        phgt_ref = ipta / "reference_flat_4m.phgt"
        psim = ipta / "psim_unw"
        for target in [pslc, pint, phgt_ref, psim, pdiff]:
            if target.exists():
                target.unlink()
        np.full(len(point_table), 4.0, dtype=">f4").tofile(phgt_ref)
        run(["SLC2pt", str(slc_tab), str(plist), "-", "-", str(pslc), "-"], log)
        run(["intf_pt", str(plist), "-", str(itab), "-", str(pslc), str(pint), "1"], log)
        run(
            ["phase_sim_orb_pt", str(plist), "-", str(pslc_par), "-", str(itab), "-", str(phgt_ref), str(psim), str(geometric_reference_par), "-", "0"],
            log,
        )
        run(["sub_phase_pt", str(plist), "-", str(pint), "-", str(psim), str(pdiff), "1", "0", "1"], log)
    else:
        run(["mk_d2pt", str(diff_tab), str(plist), str(COLS), "0", "1", "1", str(pdiff)], log)

    pres = ipta / "pres"
    pdh = ipta / "pdh"
    pdef = ipta / "pdef"
    punw = ipta / "punw"
    psigma = ipta / "psigma"
    pmask = ipta / "pmask"
    for target in [pres, pdh, pdef, punw, psigma, pmask]:
        if target.exists():
            target.unlink()
    run(
        [
            "multi_def_pt", str(plist), "-", str(pslc_par), "-", str(itab), str(pbase), "0", str(pdiff), "1", str(np_ref),
            str(pres), str(pdh), str(pdef), str(punw), str(psigma), str(pmask), str(args.dh_max), "-0.01", "0.01", "100",
            str(args.sigma_max), str(args.sigma_max2), str(args.model), "1", "1", "-1", "-1", str(args.local_ground_radius),
        ],
        log,
    )

    baseline_refined = not args.no_baseline_refinement
    if baseline_refined:
        initial_mask = np.fromfile(pmask, dtype=np.uint8)[: len(point_table)] > 0
        initial_sigma = read_be_float(psigma)[: len(point_table)]
        ground_fit = (
            point_table["point_class"].eq("ground").to_numpy()
            & initial_mask
            & np.isfinite(initial_sigma)
            & (initial_sigma <= args.sigma_max2)
        )
        if int(ground_fit.sum()) < 100:
            raise RuntimeError(f"Too few stable-ground points for GAMMA base_ls_pt: {int(ground_fit.sum())}")
        ground_fit_mask = ipta / "ground_fit.pmask"
        ground_fit.astype(np.uint8).tofile(ground_fit_mask)
        ground_fit_hgt = ipta / "ground_fit.phgt"
        np.full(len(point_table), 4.0, dtype=">f4").tofile(ground_fit_hgt)
        pbase_precision = ipta / "pbase_precision"
        shutil.copyfile(pbase, pbase_precision)
        run(
            [
                "base_ls_pt", str(plist), str(ground_fit_mask), str(pslc_par), "-", str(itab), "-", str(punw),
                str(ground_fit_hgt), str(pbase_precision), "0", "1", "1", "1", "0", "30",
            ],
            log,
        )
        for target in [pres, pdh, pdef, punw, psigma, pmask]:
            if target.exists():
                target.unlink()
        run(
            [
                "multi_def_pt", str(plist), "-", str(pslc_par), "-", str(itab), str(pbase_precision), "1", str(pdiff), "1", str(np_ref),
                str(pres), str(pdh), str(pdef), str(punw), str(psigma), str(pmask), str(args.dh_max), "-0.01", "0.01", "100",
                str(args.sigma_max), str(args.sigma_max2), str(args.model), "1", "1", "-1", "-1", str(args.local_ground_radius),
            ],
            log,
        )

    height_solution = pdh
    sigma_solution = psigma
    mb_sigma_file: Path | None = None
    mb_pair_sigmas: list[float] | None = None
    if args.sbas_engine == "mb_pt":
        itab_ts = ipta / "pairs_ts.itab"
        pdiff_ts = ipta / "pdiff_ts"
        pdiff_sim = ipta / "pdiff_sim"
        psigma_ts = ipta / "psigma_ts"
        phgt_mb = ipta / "phgt_mb"
        prate_mb = ipta / "prate_mb"
        pconst_mb = ipta / "pconst_mb"
        psigma_fit_mb = ipta / "psigma_fit_mb"
        for target in [itab_ts, pdiff_ts, pdiff_sim, psigma_ts, phgt_mb, prate_mb, pconst_mb, psigma_fit_mb]:
            if target.exists():
                target.unlink()
        mb_sigma_arg = "-"
        if args.mb_sigma_mode == "stable_ground_mad":
            accepted_initial = np.fromfile(pmask, dtype=np.uint8)[: len(point_table)] > 0
            residual_stack = read_be_float(pres).reshape(len(pairs), len(point_table))
            ground_for_sigma = point_table["point_class"].eq("ground").to_numpy() & accepted_initial
            mb_pair_sigmas = []
            for record in residual_stack:
                values = record[ground_for_sigma]
                values = values[np.isfinite(values) & (np.abs(values) > 1.0e-30)]
                if len(values) < 20:
                    values = record[accepted_initial]
                    values = values[np.isfinite(values) & (np.abs(values) > 1.0e-30)]
                if len(values):
                    center = float(np.nanmedian(values))
                    robust_sigma = 1.4826 * float(np.nanmedian(np.abs(values - center)))
                else:
                    robust_sigma = 1.0
                mb_pair_sigmas.append(float(np.clip(robust_sigma, args.mb_sigma_min, args.mb_sigma_max)))
            mb_sigma_file = ipta / "mb_pair_sigma.txt"
            # GAMMA 2021 rd_all_sigma expects: record index followed by two
            # floating-point sigma fields.  Use the same robust estimate in
            # both fields; single-column input is silently skipped.
            mb_sigma_file.write_text(
                "\n".join(f"{index} {value:.8f} {value:.8f}" for index, value in enumerate(mb_pair_sigmas, start=1))
                + "\n",
                encoding="utf-8",
            )
            mb_sigma_arg = str(mb_sigma_file)
        run(
            [
                "mb_pt", str(plist), str(pmask), str(pslc_par), str(itab), str(punw), str(np_ref), mb_sigma_arg, str(itab_ts),
                str(pdiff_ts), str(pdiff_sim), str(psigma_ts), "1", str(phgt_mb), str(args.sbas_gamma), str(prate_mb),
                str(pconst_mb), str(psigma_fit_mb), str(geometric_reference_par),
            ],
            log,
        )
        height_solution = phgt_mb
        sigma_solution = psigma_fit_mb

    npoints = len(point_table)
    point_table["gamma_delta_height_m"] = read_be_float(height_solution)[:npoints]
    point_table["gamma_deformation_m_per_year"] = read_be_float(pdef)[:npoints]
    point_table["gamma_phase_sigma_rad"] = read_be_float(sigma_solution)[:npoints]
    point_table["gamma_accepted"] = np.fromfile(pmask, dtype=np.uint8)[:npoints] > 0

    valid_ground = point_table[
        point_table["point_class"].eq("ground")
        & point_table["gamma_accepted"]
        & np.isfinite(point_table["gamma_delta_height_m"])
    ].copy()
    valid_roof = point_table[
        point_table["point_class"].eq("roof")
        & point_table["gamma_accepted"]
        & np.isfinite(point_table["gamma_delta_height_m"])
    ].copy()
    if valid_ground.empty:
        raise RuntimeError("GAMMA multi_def_pt accepted no stable-ground points")
    ground_tree = cKDTree(valid_ground[["range_pixel", "azimuth_pixel"]].to_numpy(float))
    k = min(args.local_ground_neighbours, len(valid_ground))
    distances, neighbours = ground_tree.query(valid_roof[["range_pixel", "azimuth_pixel"]].to_numpy(float), k=k)
    if k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    ground_values = valid_ground["gamma_delta_height_m"].to_numpy()[neighbours]
    ground_values = np.where(distances <= args.local_ground_radius, ground_values, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        local_ground = np.nanmedian(ground_values, axis=1)
    fallback_ground = float(np.nanmedian(valid_ground["gamma_delta_height_m"]))
    local_ground = np.where(np.isfinite(local_ground), local_ground, fallback_ground)
    valid_roof["local_ground_height_m"] = local_ground
    valid_roof["insar_building_height_m"] = valid_roof["gamma_delta_height_m"].to_numpy() - local_ground
    point_table = point_table.merge(
        valid_roof[["point_index", "local_ground_height_m", "insar_building_height_m"]], on="point_index", how="left"
    )
    point_table.to_csv(ipta / "gamma_point_solution.csv", index=False)

    building_rows: list[dict[str, object]] = []
    for clean_id, group in valid_roof.groupby("clean_id"):
        values = group["insar_building_height_m"].to_numpy(float)
        if args.building_statistic == "upper_cluster" and len(values) >= 4:
            values = values[values >= np.nanpercentile(values, 75)]
            group = group.loc[group["insar_building_height_m"] >= np.nanpercentile(group["insar_building_height_m"], 75)].copy()
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        keep = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
        cleaned = values[keep]
        building_rows.append(
            {
                "clean_id": int(clean_id),
                "insar_height_m": float(np.nanmedian(cleaned)),
                "roof_points": int(len(values)),
                "roof_points_after_iqr": int(len(cleaned)),
                "roof_height_iqr_m": float(np.nanpercentile(cleaned, 75) - np.nanpercentile(cleaned, 25)),
                "median_phase_sigma_rad": float(np.nanmedian(group.loc[keep, "gamma_phase_sigma_rad"])),
                "median_mean_coherence": float(np.nanmedian(group.loc[keep, "mean_coherence"])),
                "solution_source": (
                    "GAMMA_SLC_intf_phase_sim_orb_sub_phase_adf_"
                    f"multi_def_pt_{args.sbas_engine}_local_ground"
                ),
                "filled_from_prior": False,
            }
        )
    result = pd.DataFrame(building_rows)
    buildings = gpd.read_file(args.buildings)
    if "clean_id" not in buildings:
        raise ValueError("building data must contain clean_id")
    prior = buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"}) if "height" in buildings else pd.DataFrame({"clean_id": buildings["clean_id"]})
    # This merge occurs only after all GAMMA selection and estimation is done.
    result = result.merge(prior, on="clean_id", how="left")
    result["difference_to_prior_m"] = result["insar_height_m"] - result.get("prior_height_m", np.nan)
    Path(args.result_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.result_csv, index=False)
    result_geo = buildings.merge(result, on="clean_id", how="left", suffixes=("_vector", ""))
    Path(args.result_geojson).parent.mkdir(parents=True, exist_ok=True)
    result_geo.to_file(args.result_geojson, driver="GeoJSON")

    solved = result[np.isfinite(result["insar_height_m"])]
    summary = {
        "method": (
            f"GAMMA-native SLC_intf -> phase_sim_orb/sub_phase -> adf -> IPTA multi_def_pt(model {args.model})"
            + (f" -> mb_pt(gamma={args.sbas_gamma:g})" if args.sbas_engine == "mb_pt" else "")
            + " -> local stable-ground subtraction -> per-building IQR and median"
        ),
        "paper_alignment": {"minimum_pairs": 12, "amplitude_dispersion_max": args.max_da, "building_aggregation": "IQR 1.5 then median"},
        "prior_policy": "height is merged only after the GAMMA solution is frozen; it is comparison-only and never used for interferograms, point selection, inversion, rejection, calibration, or filling",
        "reference_surface": str(reference_hgt),
        "reference_surface_role": "constant non-NULL GAMMA phase datum only; cancels in roof-minus-local-ground height",
        "geometric_coregistration_reference": str(geometric_reference_par),
        "precision_baseline_refinement": baseline_refined,
        "ipta_phase_source": args.ipta_phase_source,
        "sbas_engine": args.sbas_engine,
        "sbas_gamma": args.sbas_gamma if args.sbas_engine == "mb_pt" else None,
        "mb_sigma_mode": args.mb_sigma_mode if args.sbas_engine == "mb_pt" else None,
        "mb_pair_sigma_rad": mb_pair_sigmas,
        "point_domain": args.point_domain,
        "building_statistic": args.building_statistic,
        "domain_conflict_pixels_removed": domain_conflicts,
        "pairs": int(len(pairs)),
        "dates": int(len(dates)),
        "candidate_roof_points": int(roof.sum()),
        "candidate_ground_points": int(ground.sum()),
        "gamma_accepted_roof_points": int(len(valid_roof)),
        "gamma_accepted_ground_points": int(len(valid_ground)),
        "phase_reference_point_index": np_ref,
        "buildings_solved": int(len(solved)),
        "height_median_m": float(solved["insar_height_m"].median()) if len(solved) else None,
        "height_p05_p95_m": [float(solved["insar_height_m"].quantile(0.05)), float(solved["insar_height_m"].quantile(0.95))] if len(solved) else None,
        "internal_phase_sigma_median_rad": float(solved["median_phase_sigma_rad"].median()) if len(solved) else None,
        "prior_comparison_after_selection": {
            "count": int(solved["prior_height_m"].notna().sum()) if "prior_height_m" in solved else 0,
            "median_difference_m": float(solved["difference_to_prior_m"].median()) if len(solved) else None,
            "mae_m": float(solved["difference_to_prior_m"].abs().mean()) if len(solved) else None,
            "correlation": float(solved[["insar_height_m", "prior_height_m"]].corr().iloc[0, 1]) if len(solved) > 1 and "prior_height_m" in solved else None,
        },
        "artifacts": {
            "point_solution": str(ipta / "gamma_point_solution.csv"),
            "result_csv": args.result_csv,
            "result_geojson": args.result_geojson,
            "gamma_log": str(log),
        },
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
