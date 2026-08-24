#!/usr/bin/env python3
"""Search wrapped-phase height inside each PS's R-D admissible interval.

This is a hybrid ambiguity diagnostic, not a final SBAS product. It uses the
GAMMA differential phase and GAMMA orbital height sensitivity, constrains each
stable PS to heights geometrically compatible with its SAR position, and
marginalizes a linear time-dependent nuisance term. Repeated point clusters are
required at building level. No prior value is copied into the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def clusters(table: pd.DataFrame, tolerance: float) -> list[pd.DataFrame]:
    if table.empty:
        return []
    table = table.sort_values("searched_height_m").reset_index(drop=True)
    label = np.cumsum(np.r_[True, np.diff(table["searched_height_m"]) > tolerance])
    return [part.copy() for _, part in table.groupby(label)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-audit", default="", help="Optional point-specific R-D interval table; empty uses a common prior-independent interval")
    p.add_argument("--ipta", default="work/gamma_native_ipta_sbas/ipta")
    p.add_argument("--pcc", default="work/gamma_building_roofcore_closurecorrected_gamma100_full/pcc")
    p.add_argument("--pairs", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--buildings", default="data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg")
    p.add_argument("--height-step", type=float, default=0.5)
    p.add_argument("--common-height-min", type=float, default=0.0)
    p.add_argument("--common-height-max", type=float, default=120.0)
    p.add_argument("--sensitivity-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    p.add_argument("--rate-limit", type=float, default=0.03)
    p.add_argument("--rate-step", type=float, default=0.001)
    p.add_argument("--max-circular-cost", type=float, default=0.90)
    p.add_argument("--maximum-split-height-difference", type=float, default=8.0)
    p.add_argument("--minimum-distant-cost-margin", type=float, default=0.015)
    p.add_argument("--cluster-tolerance", type=float, default=5.0)
    p.add_argument("--min-cluster-points", type=int, default=2)
    p.add_argument("--cluster-selection", choices=["largest", "highest"], default="largest")
    p.add_argument("--limit-buildings", type=int, default=0, help="0 searches all candidate buildings")
    p.add_argument("--max-points-per-building", type=int, default=12)
    p.add_argument("--output", default="results/tables/tongji_geometry_constrained_wrapped_height_diagnostic.csv")
    p.add_argument("--summary", default="results/metadata/tongji_geometry_constrained_wrapped_height_diagnostic_summary.json")
    args = p.parse_args()

    ipta = Path(args.ipta)
    metadata = pd.read_csv(ipta / "point_metadata.csv").sort_values("point_index").reset_index(drop=True)
    pairs = pd.read_csv(args.pairs, dtype={"master": str, "slave": str}).drop_duplicates(["master", "slave"])
    if args.geometry_audit:
        geom = pd.read_csv(args.geometry_audit)
        geom = geom[np.isfinite(geom["geometry_height_min_m"]) & np.isfinite(geom["geometry_height_max_m"])].copy()
        interval_policy = "point-specific R-D interval"
    else:
        geom = metadata.loc[metadata["point_class"].eq("roof"), ["clean_id", "point_index"]].copy()
        geom["geometry_height_min_m"] = float(args.common_height_min)
        geom["geometry_height_max_m"] = float(args.common_height_max)
        interval_policy = "common prior-independent height interval"
    npoints, npairs = len(metadata), len(pairs)
    pdiff = np.memmap(ipta / "pdiff", dtype=">c8", mode="r", shape=(npairs, npoints))
    pcc = np.memmap(args.pcc, dtype=">f4", mode="r", shape=(npairs, npoints))
    sensitivity = np.load(ipta / "phase_height_sensitivity_rad_per_m.npy", mmap_mode="r")
    pmask = np.fromfile(ipta / "pmask", dtype=np.uint8)[:npoints] > 0
    ground = metadata[metadata["point_class"].eq("ground") & pmask].copy()
    ground_ids = ground["point_index"].to_numpy(int)
    ground_tree = cKDTree(ground[["range_pixel", "azimuth_pixel"]].to_numpy(float))
    wavelength = 299792458.0 / 5.4050005e9
    q = -4.0 * np.pi * (pairs["dt_days"].to_numpy(float) / 365.25) / wavelength
    rates = np.arange(-args.rate_limit, args.rate_limit + 0.5 * args.rate_step, args.rate_step)
    baseline = np.abs(pairs["bperp_m"].to_numpy(float))
    order = np.argsort(baseline, kind="stable")
    split_a = np.zeros(npairs, dtype=bool); split_a[order[::2]] = True
    split_b = ~split_a

    point_rows: list[dict[str, object]] = []
    building_rows: list[dict[str, object]] = []
    grouped = list(geom.groupby("clean_id"))
    if args.limit_buildings:
        grouped = grouped[: args.limit_buildings]
    for clean_id, group in grouped:
        if len(group) > args.max_points_per_building:
            quality_order = metadata.set_index("point_index").loc[group["point_index"].to_numpy(int), "mean_coherence"].sort_values(ascending=False)
            group = group[group["point_index"].isin(quality_order.index[: args.max_points_per_building])].copy()
        point_ids = group["point_index"].to_numpy(int)
        xy = metadata.iloc[point_ids][["range_pixel", "azimuth_pixel"]].to_numpy(float)
        centroid = xy.mean(axis=0)
        _, near = ground_tree.query(centroid, k=min(40, len(ground_ids)))
        local_ground = ground_ids[np.atleast_1d(near)]
        ground_weight = np.clip(np.asarray(pcc[:, local_ground], dtype=float), 1.0e-3, 1.0)
        ground_complex = np.asarray(pdiff[:, local_ground])
        ground_mean = np.sum(ground_weight * ground_complex, axis=1) / np.sum(ground_weight, axis=1)
        ground_coh = np.nanmedian(ground_weight, axis=1)
        accepted_points = []
        for row in group.itertuples(index=False):
            point_id = int(row.point_index)
            hs = np.arange(
                float(row.geometry_height_min_m),
                float(row.geometry_height_max_m) + 0.5 * args.height_step,
                args.height_step,
            )
            relative = np.angle(np.asarray(pdiff[:, point_id]) * np.conj(ground_mean))
            weight = np.sqrt(np.clip(np.asarray(pcc[:, point_id], dtype=float), 0, 1) * ground_coh)
            model = (
                args.sensitivity_sign * np.asarray(sensitivity[:, point_id], dtype=float)[:, None, None] * hs[None, :, None]
                + q[:, None, None] * rates[None, None, :]
            )
            cost = np.sum(
                weight[:, None, None] * (1.0 - np.cos(relative[:, None, None] - model)), axis=0
            ) / max(float(weight.sum()), 1.0e-6)
            hi, ri = np.unravel_index(int(np.nanargmin(cost)), cost.shape)
            split_heights = []
            for split in (split_a, split_b):
                split_cost = np.sum(
                    weight[split, None, None] * (1.0 - np.cos(relative[split, None, None] - model[split])), axis=0
                ) / max(float(weight[split].sum()), 1.0e-6)
                split_hi, _split_ri = np.unravel_index(int(np.nanargmin(split_cost)), split_cost.shape)
                split_heights.append(float(hs[split_hi]))
            split_height_difference = abs(split_heights[0] - split_heights[1])
            distant = np.abs(hs - float(hs[hi])) >= 10.0
            distant_best = float(np.nanmin(cost[distant])) if np.any(distant) else float("inf")
            distant_margin = distant_best - float(cost[hi, ri])
            result = {
                "clean_id": int(clean_id),
                "point_index": point_id,
                "searched_height_m": float(hs[hi]),
                "nuisance_rate_m_per_year": float(rates[ri]),
                "rate_at_boundary": bool(ri == 0 or ri == len(rates) - 1),
                "circular_cost": float(cost[hi, ri]),
                "split_a_height_m": split_heights[0],
                "split_b_height_m": split_heights[1],
                "split_height_difference_m": float(split_height_difference),
                "distant_cost_margin": float(distant_margin),
                "geometry_height_min_m": float(row.geometry_height_min_m),
                "geometry_height_max_m": float(row.geometry_height_max_m),
                "accepted_cost": bool(
                    cost[hi, ri] <= args.max_circular_cost
                    and split_height_difference <= args.maximum_split_height_difference
                    and distant_margin >= args.minimum_distant_cost_margin
                ),
                "selected_for_building_cluster": False,
            }
            point_rows.append(result)
            if result["accepted_cost"]:
                accepted_points.append(result)
        accepted_table = pd.DataFrame(accepted_points)
        eligible = [part for part in clusters(accepted_table, args.cluster_tolerance) if len(part) >= args.min_cluster_points]
        if not eligible:
            continue
        if args.cluster_selection == "highest":
            winner = max(eligible, key=lambda part: float(part["searched_height_m"].median()))
        else:
            winner = min(eligible, key=lambda part: (-len(part), float(part["circular_cost"].median())))
        winner_ids = set(winner["point_index"].astype(int))
        for result in accepted_points:
            if int(result["point_index"]) in winner_ids:
                result["selected_for_building_cluster"] = True
        building_rows.append(
            {
                "clean_id": int(clean_id),
                "height_m": float(winner["searched_height_m"].median()),
                "cluster_points": int(len(winner)),
                "cluster_iqr_m": float(winner["searched_height_m"].quantile(.75) - winner["searched_height_m"].quantile(.25)),
                "median_circular_cost": float(winner["circular_cost"].median()),
                "rate_boundary_fraction": float(winner["rate_at_boundary"].mean()),
                "solution_source": "HYBRID_wrapped_phase_RD_geometry_search_not_final_GAMMA_SBAS",
                "filled_from_prior": False,
                "insar_only": False,
            }
        )

    point_output = Path(args.output).with_name(Path(args.output).stem + "_points.csv")
    point_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(point_output, index=False)
    result = pd.DataFrame(building_rows)
    buildings = gpd.read_file(args.buildings)
    result = result.merge(buildings[["clean_id", "height"]].rename(columns={"height": "prior_height_m"}), on="clean_id", how="left")
    result["difference_to_prior_m"] = result["height_m"] - result["prior_height_m"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    compared = result[["height_m", "prior_height_m"]].dropna()
    summary = {
        "method": f"wrapped phase search within {interval_policy}; {args.cluster_selection} repeated cluster",
        "product_class": (
            "prior-independent ambiguity diagnostic; must be rerun through GAMMA mb_pt before any product use"
            if not args.geometry_audit
            else "hybrid ambiguity diagnostic; must be rerun through GAMMA mb_pt before any product use"
        ),
        "prior_policy": (
            "no height read, copied, fitted, selected, or filled; common interval is prior-independent; prior is loaded only after the result is frozen for reporting"
            if not args.geometry_audit
            else "no height copied or filled; point-specific projection geometry participates, so comparison to the same prior is not independent"
        ),
        "thresholds": {"rate_limit_m_per_year": args.rate_limit, "max_circular_cost": args.max_circular_cost, "maximum_split_height_difference_m": args.maximum_split_height_difference, "minimum_distant_cost_margin": args.minimum_distant_cost_margin, "cluster_tolerance_m": args.cluster_tolerance, "minimum_cluster_points": args.min_cluster_points, "cluster_selection": args.cluster_selection},
        "points_searched": int(len(point_rows)),
        "points_accepted_cost": int(sum(row["accepted_cost"] for row in point_rows)),
        "accepted_points_rate_boundary_fraction": float(np.mean([row["rate_at_boundary"] for row in point_rows if row["accepted_cost"]])) if any(row["accepted_cost"] for row in point_rows) else None,
        "buildings_with_repeated_cluster": int(len(result)),
        "height_median_m": float(result["height_m"].median()) if len(result) else None,
        "height_p05_p95_m": [float(result["height_m"].quantile(.05)), float(result["height_m"].quantile(.95))] if len(result) else None,
        "median_cluster_cost": float(result["median_circular_cost"].median()) if len(result) else None,
        "median_rate_boundary_fraction": float(result["rate_boundary_fraction"].median()) if len(result) else None,
        "post_comparison_not_independent": {"count": int(len(compared)), "mae_m": float((compared.height_m-compared.prior_height_m).abs().mean()) if len(compared) else None, "correlation": float(compared.corr().iloc[0,1]) if len(compared)>1 else None},
        "artifacts": {"building_csv": str(output), "point_csv": str(point_output)},
    }
    summary_path = Path(args.summary); summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
