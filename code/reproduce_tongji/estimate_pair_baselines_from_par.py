#!/usr/bin/env python3
"""Estimate approximate perpendicular baselines from GAMMA RSLC par files.

This is a workflow-enabling approximation derived from orbit state vectors at
scene center time. It is not a replacement for GAMMA baseline products in the
final thesis-method height inversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_par(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = rest.strip().split()
        if vals:
            out[key.strip()] = vals
    nsv = int(float(out["number_of_state_vectors"][0]))
    t0 = float(out["time_of_first_state_vector"][0])
    dt = float(out["state_vector_interval"][0])
    times = np.asarray([t0 + i * dt for i in range(nsv)], dtype=np.float64)
    pos = []
    for i in range(1, nsv + 1):
        pos.append([float(x) for x in out[f"state_vector_position_{i}"][:3]])
    return {
        "path": str(path),
        "center_time": float(out["center_time"][0]),
        "center_latitude": float(out["center_latitude"][0]),
        "center_longitude": float(out["center_longitude"][0]),
        "center_range_slc": float(out["center_range_slc"][0]),
        "incidence_angle": float(out["incidence_angle"][0]),
        "radar_frequency": float(out["radar_frequency"][0]),
        "state_times": times,
        "state_positions": np.asarray(pos, dtype=np.float64),
    }


def interp_position(par: dict) -> np.ndarray:
    t = par["center_time"]
    times = par["state_times"]
    pos = par["state_positions"]
    return np.asarray([np.interp(t, times, pos[:, i]) for i in range(3)], dtype=np.float64)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float = 0.0) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    x = (n + h) * math.cos(lat) * math.cos(lon)
    y = (n + h) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + h) * math.sin(lat)
    return np.asarray([x, y, z], dtype=np.float64)


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y%m%d")


def estimate_bperp(master_par: dict, slave_par: dict) -> float:
    master_pos = interp_position(master_par)
    slave_pos = interp_position(slave_par)
    target = geodetic_to_ecef(master_par["center_latitude"], master_par["center_longitude"], 0.0)
    los = target - master_pos
    los = los / np.linalg.norm(los)
    baseline = slave_pos - master_pos
    parallel = float(np.dot(baseline, los))
    perp_vec = baseline - parallel * los
    sign_axis = np.cross(master_pos / np.linalg.norm(master_pos), los)
    sign = 1.0 if float(np.dot(perp_vec, sign_axis)) >= 0 else -1.0
    return sign * float(np.linalg.norm(perp_vec))


def height_ambiguity(lambda_m: float, range_m: float, incidence_deg: float, bperp_m: float) -> float | None:
    if abs(bperp_m) < 1e-6:
        return None
    return abs(lambda_m * range_m * math.sin(math.radians(incidence_deg)) / (2.0 * bperp_m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--pairs-csv", default="work/baselines/temporal_candidate_pairs.csv")
    parser.add_argument("--output-csv", default="work/baselines/temporal_candidate_pairs_with_approx_bperp.csv")
    parser.add_argument("--summary", default="results/metadata/approx_baseline_summary.json")
    args = parser.parse_args()

    rslc_dir = Path(args.rslc_dir)
    pars = {p.stem.replace(".rslc", ""): parse_par(p) for p in rslc_dir.glob("*.rslc.par")}
    pairs = pd.read_csv(args.pairs_csv)
    rows = []
    c = 299792458.0
    for row in pairs.to_dict("records"):
        master = str(row["master"])
        slave = str(row["slave"])
        mp = pars[master]
        sp = pars[slave]
        bperp = estimate_bperp(mp, sp)
        wavelength = c / mp["radar_frequency"]
        hamb = height_ambiguity(wavelength, mp["center_range_slc"], mp["incidence_angle"], bperp)
        rows.append(
            {
                **row,
                "bperp_m": bperp,
                "height_ambiguity_m": hamb,
                "status": "approx_from_par_state_vectors",
                "baseline_note": "Approximate center-scene Bperp from RSLC par state vectors; replace with GAMMA baseline for final inversion.",
            }
        )
    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "pairs": int(len(out)),
        "output_csv": args.output_csv,
        "bperp_min_m": float(out["bperp_m"].min()),
        "bperp_max_m": float(out["bperp_m"].max()),
        "bperp_median_abs_m": float(out["bperp_m"].abs().median()),
        "note": "Approximate baselines only. Use GAMMA baseline products for final thesis-method height inversion.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
