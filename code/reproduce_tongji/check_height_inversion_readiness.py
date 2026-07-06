#!/usr/bin/env python3
"""Check readiness for thesis-method final height inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def file_status(path: str) -> dict:
    p = Path(path)
    return {"path": path, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", default="work/projection/20200708_blue_aligned_bottom_touying.geojson")
    parser.add_argument("--island-label", default="work/masks/island_label_touying_blue_bottom.npy")
    parser.add_argument("--fid-map", default="results/tables/touying_fid_uid_map.csv")
    parser.add_argument("--approx-baselines", default="work/baselines/temporal_candidate_pairs_with_approx_bperp.csv")
    parser.add_argument("--gamma-baselines", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--raw-observations", default="work/height/raw_island_phase_observations.csv")
    parser.add_argument("--differential-observations", default="work/height/unwrapped_differential_island_phase_observations.csv")
    parser.add_argument("--output", default="results/metadata/height_inversion_readiness.json")
    args = parser.parse_args()

    checks = {
        "projection": file_status(args.projection),
        "island_label": file_status(args.island_label),
        "fid_map": file_status(args.fid_map),
        "approx_baselines": file_status(args.approx_baselines),
        "gamma_baselines": file_status(args.gamma_baselines),
        "raw_observations": file_status(args.raw_observations),
        "differential_observations": file_status(args.differential_observations),
    }
    blockers = []
    if not checks["projection"]["exists"]:
        blockers.append("Missing Touying blue-aligned projection.")
    if not checks["island_label"]["exists"]:
        blockers.append("Missing island label raster.")
    if not checks["fid_map"]["exists"]:
        blockers.append("Missing touying_fid to local uid map.")
    if not checks["gamma_baselines"]["exists"]:
        blockers.append("Missing precise GAMMA perpendicular baseline table.")
    if not checks["differential_observations"]["exists"]:
        blockers.append("Missing differential island phase observations.")

    if checks["approx_baselines"]["exists"]:
        approx = pd.read_csv(args.approx_baselines)
        checks["approx_baselines"]["pairs"] = int(len(approx))
        checks["approx_baselines"]["status_values"] = sorted(str(x) for x in approx["status"].dropna().unique()) if "status" in approx else []
    if checks["raw_observations"]["exists"]:
        raw = pd.read_csv(args.raw_observations)
        checks["raw_observations"]["rows"] = int(len(raw))
        checks["raw_observations"]["islands"] = int(raw["island_id"].nunique()) if "island_id" in raw else 0

    ready = len(blockers) == 0
    summary = {
        "ready_for_final_height_inversion": ready,
        "checks": checks,
        "blockers": blockers,
        "next_required_inputs": []
        if ready
        else [
            "GAMMA precise Bperp table at work/baselines/temporal_candidate_pairs_gamma_bperp.csv",
            "Spatially unwrapped flat-earth/DEM-removed differential island phase observations at work/height/unwrapped_differential_island_phase_observations.csv",
        ],
        "note": "Raw preview phases and approximate par-derived baselines are diagnostic only.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
