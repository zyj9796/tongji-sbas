#!/usr/bin/env python3
"""Validate the frozen strict-reproduction deliverables without using priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def check(name: str, passed: bool, detail: object) -> dict:
    return {"check": name, "pass": bool(passed), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checks = []

    rslc = sorted((root / "work/rslc").glob("*.rslc"))
    reference = root / "work/slc_crop/20231007.slc"
    stack_count = len(rslc) + int(reference.exists())
    checks.append(check("21景配准复数栈（1主影像+20辅影像）", stack_count == 21,
                        {"reference": int(reference.exists()), "secondary_rslc": len(rslc)}))
    pair_obs = sorted((root / "work/pair_observations_independent_expanded_48").glob("*.npz"))
    checks.append(check("48对差分干涉观测", len(pair_obs) == 48, len(pair_obs)))
    unwrap = sorted((root / "work/unwrapped_independent_expanded_48").glob("*.npz"))
    checks.append(check("48对GAMMA-MCF解缠", len(unwrap) == 48, len(unwrap)))

    mcf_summary = json.loads((root / "inventory/gamma_mcf_independent_expanded_48_all_summary.json").read_text())
    failures = sum(row["failed_building_count"] for row in mcf_summary)
    checks.append(check("独立建筑MCF无失败", len(mcf_summary) == 48 and failures == 0,
                        {"pair_count": len(mcf_summary), "failure_sum": failures}))

    csv_result = pd.read_csv(root / "results/paper_strict/building_height_final.csv")
    gdf = gpd.read_file(root / "results/paper_strict/building_height_final.gpkg", layer="building_height")
    solved = gdf["recommended_building_height_m"].notna()
    checks.append(check("候选建筑有解数", len(csv_result) == 198 and int(solved.sum()) == 198,
                        {"csv": len(csv_result), "gpkg": int(solved.sum())}))
    checks.append(check("无解保持空值", int((~solved).sum()) == len(gdf) - 198,
                        {"all_buildings": len(gdf), "null_height": int((~solved).sum())}))
    exact = np.allclose(
        csv_result["recommended_building_height_m"],
        csv_result["robust_q95_q05_height_m"], equal_nan=True,
    )
    checks.append(check("正式高度未被先验替换", exact,
                        "recommended_building_height_m == robust_q95_q05_height_m"))

    figures = sorted((root / "results/figures").glob("*.svg"))
    svg_ok = len(figures) == 6 and all("<svg" in p.read_text(encoding="utf-8", errors="ignore")[:1000] for p in figures)
    checks.append(check("仅输出六张独立SVG", svg_ok, [p.name for p in figures]))

    report = {
        "all_pass": all(row["pass"] for row in checks),
        "checks": checks,
        "prior_audit_policy": "Floor appears only in floor_prior_posthoc_audit.csv after inversion is frozen",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
