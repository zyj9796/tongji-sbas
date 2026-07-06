#!/usr/bin/env python3
"""Create a prior-calibrated height product using shp height as calibration target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return 1.0, 0.0
    # Clip extreme residual leverage but keep the fit deterministic and transparent.
    lo, hi = np.percentile(x, [2, 98])
    keep = (x >= lo) & (x <= hi)
    coef = np.polyfit(x[keep], y[keep], 1)
    return float(coef[0]), float(coef[1])


def quantile_map(values: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    valid = np.isfinite(src) & np.isfinite(dst)
    src = np.sort(src[valid])
    dst = np.sort(dst[valid])
    out = np.full_like(values, np.nan, dtype=np.float64)
    good = np.isfinite(values)
    if len(src) < 2:
        out[good] = values[good]
        return out
    q = np.linspace(0.0, 1.0, len(src))
    src_q = np.quantile(src, q)
    dst_q = np.quantile(dst, q)
    # Ensure strictly increasing x positions for interpolation.
    src_unique, idx = np.unique(src_q, return_index=True)
    dst_unique = dst_q[idx]
    out[good] = np.interp(values[good], src_unique, dst_unique, left=dst_unique[0], right=dst_unique[-1])
    return out


def metrics(df: pd.DataFrame, col: str) -> dict[str, float]:
    valid = np.isfinite(df[col]) & np.isfinite(df["height_prior_m"])
    err = df.loc[valid, col] - df.loc[valid, "height_prior_m"]
    if len(err) == 0:
        return {"mae_m": np.nan, "rmse_m": np.nan, "median_abs_m": np.nan, "bias_m": np.nan}
    return {
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "median_abs_m": float(np.median(np.abs(err))),
        "bias_m": float(np.mean(err)),
    }


def plot_calibration(df: pd.DataFrame, out: Path) -> None:
    insar = df[df["height_source"] == "insar"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=220)
    axes[0, 0].scatter(insar["height_prior_m"], insar["height_insar_m"], s=8, alpha=0.45, color="#4477aa", label="raw")
    axes[0, 0].scatter(insar["height_prior_m"], insar["height_calibrated_m"], s=8, alpha=0.45, color="#117733", label="calibrated")
    lim = float(np.nanmax([insar["height_prior_m"].max(), insar["height_calibrated_m"].max(), 1.0]))
    axes[0, 0].plot([0, lim], [0, lim], color="#333333", linewidth=0.8)
    axes[0, 0].set_title("Prior vs raw/calibrated InSAR")
    axes[0, 0].set_xlabel("height_prior_m")
    axes[0, 0].set_ylabel("height m")
    axes[0, 0].legend(frameon=False)

    raw_err = insar["height_insar_m"] - insar["height_prior_m"]
    cal_err = insar["height_calibrated_m"] - insar["height_prior_m"]
    axes[0, 1].hist(raw_err.dropna(), bins=45, alpha=0.65, color="#cc6677", label="raw-prior")
    axes[0, 1].hist(cal_err.dropna(), bins=45, alpha=0.65, color="#117733", label="cal-prior")
    axes[0, 1].axvline(0, color="#333333", linewidth=0.8)
    axes[0, 1].set_title("Residual to shp height")
    axes[0, 1].set_xlabel("m")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].hist(insar["height_calibrated_m"].dropna(), bins=45, color="#44aa99", edgecolor="white")
    axes[1, 0].set_title("Calibrated heights")
    axes[1, 0].set_xlabel("m")

    df.plot(ax=axes[1, 1], color="#eeeeee", linewidth=0.03, edgecolor="#aaaaaa")
    insar.plot(ax=axes[1, 1], column="height_calibrated_m", cmap="viridis", linewidth=0.04, edgecolor="#333333", legend=True)
    axes[1, 1].set_title("Calibrated InSAR-covered buildings")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/geodata/tongji_building_height_qc.geojson")
    parser.add_argument("--method", choices=["quantile", "linear", "prior"], default="quantile")
    parser.add_argument("--output-csv", default="results/tables/tongji_building_height_calibrated_to_prior.csv")
    parser.add_argument("--output-geojson", default="results/geodata/tongji_building_height_calibrated_to_prior.geojson")
    parser.add_argument("--summary", default="results/metadata/building_height_calibrated_to_prior_summary.json")
    parser.add_argument("--figure", default="results/pic_all/28_building_height_calibrated_to_prior.png")
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    raw = gdf["height_insar_m"].to_numpy(dtype=np.float64)
    prior = gdf["height_prior_m"].to_numpy(dtype=np.float64)
    insar_mask = gdf["height_source"].eq("insar").to_numpy()

    calibrated = np.full(len(gdf), np.nan, dtype=np.float64)
    if args.method == "prior":
        calibrated[insar_mask] = prior[insar_mask]
        model = {"type": "identity_to_prior"}
    elif args.method == "linear":
        slope, intercept = fit_linear(raw[insar_mask], prior[insar_mask])
        calibrated[insar_mask] = slope * raw[insar_mask] + intercept
        model = {"type": "linear", "slope": slope, "intercept": intercept}
    else:
        calibrated[insar_mask] = quantile_map(raw[insar_mask], raw[insar_mask], prior[insar_mask])
        model = {"type": "quantile_mapping"}

    # This product is explicitly calibrated to shp height; clamp only after calibration.
    calibrated = np.where(np.isfinite(calibrated), np.clip(calibrated, 0.0, None), np.nan)
    gdf["height_raw_insar_m"] = gdf["height_insar_m"]
    gdf["height_calibrated_m"] = calibrated
    gdf["height_final_m"] = np.where(gdf["height_source"].eq("insar"), calibrated, gdf["height_prior_m"])
    gdf["height_insar_m"] = np.where(gdf["height_source"].eq("insar"), calibrated, np.nan)
    gdf["height_calibration_method"] = np.where(gdf["height_source"].eq("insar"), args.method, "prior_fallback")
    gdf["height_abs_diff_calibrated_m"] = np.where(
        gdf["height_source"].eq("insar"),
        (gdf["height_calibrated_m"] - gdf["height_prior_m"]).abs(),
        np.nan,
    )

    out_csv = Path(args.output_csv)
    out_geojson = Path(args.output_geojson)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gdf.drop(columns="geometry")).to_csv(out_csv, index=False)
    gdf.to_file(out_geojson, driver="GeoJSON")
    plot_calibration(gdf, Path(args.figure))

    ins = gdf[gdf["height_source"] == "insar"].copy()
    summary = {
        "input": args.input,
        "method": args.method,
        "model": model,
        "buildings": int(len(gdf)),
        "insar_calibrated_buildings": int(len(ins)),
        "prior_fallback_buildings": int((gdf["height_source"] != "insar").sum()),
        "raw_vs_prior": metrics(ins.rename(columns={"height_raw_insar_m": "raw"}), "raw"),
        "calibrated_vs_prior": metrics(ins, "height_calibrated_m"),
        "output_csv": args.output_csv,
        "output_geojson": args.output_geojson,
        "figure": args.figure,
        "note": "This is a shp-height-calibrated product requested for agreement with the height field, not an independent InSAR-only validation product.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
