#!/usr/bin/env python3
"""Run GAMMA `base_calc` for the Tongji RSLC stack and normalize its outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


DATE_RE = re.compile(r"(\d{8})")


def build_slc_tab(rslc_dir: Path, out: Path) -> list[str]:
    dates = sorted(p.name.split(".")[0] for p in rslc_dir.glob("*.rslc"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for date in dates:
            f.write(f"{rslc_dir / (date + '.rslc')} {rslc_dir / (date + '.rslc.par')}\n")
    return dates


def parse_gamma_bperp(path: Path, dates: list[str], itab_path: Path | None) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and DATE_RE.fullmatch(fields[1] or "") and DATE_RE.fullmatch(fields[2] or ""):
            # bperp_file format:
            # pair_index date1 date2 Bperp delta_T MJD1 MJD2 Bperp1 Bperp2
            rows.append(
                {
                    "master": fields[1],
                    "slave": fields[2],
                    "bperp_m": float(fields[3]),
                    "dt_days": int(round(float(fields[4]))),
                }
            )
    if rows:
        return pd.DataFrame(rows).drop_duplicates(subset=["master", "slave"])

    if itab_path is None or not itab_path.exists():
        return pd.DataFrame(columns=["master", "slave", "bperp_m", "dt_days"])
    # Fallback: at least normalize selected pair list from itab.
    itab_rows = []
    with itab_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            vals = line.split()
            if len(vals) < 4:
                continue
            try:
                i1, i2, _, flag = [int(float(x)) for x in vals[:4]]
            except ValueError:
                continue
            if flag != 1:
                continue
            if 1 <= i1 <= len(dates) and 1 <= i2 <= len(dates):
                itab_rows.append({"master": dates[i1 - 1], "slave": dates[i2 - 1], "bperp_m": None, "dt_days": None})
    return pd.DataFrame(itab_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--reference-date", default="20200708")
    parser.add_argument("--work-dir", default="work/gamma_sbas")
    parser.add_argument("--dt-max", type=int, default=44)
    parser.add_argument("--bperp-max", type=float, default=800.0)
    parser.add_argument("--output-csv", default="work/baselines/temporal_candidate_pairs_gamma_bperp.csv")
    parser.add_argument("--summary", default="results/metadata/gamma_baseline_summary.json")
    args = parser.parse_args()

    rslc_dir = Path(args.rslc_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    slc_tab = work_dir / "tongji_slc_tab.txt"
    bperp_file = work_dir / "tongji_base_calc_bperp.txt"
    itab = work_dir / "tongji_sbas.itab"
    log = work_dir / "base_calc.log"
    dates = build_slc_tab(rslc_dir, slc_tab)
    ref_par = rslc_dir / f"{args.reference_date}.rslc.par"
    cmd = [
        "base_calc",
        str(slc_tab),
        str(ref_par),
        str(bperp_file),
        str(itab),
        "1",
        "0",
        "-",
        str(args.bperp_max),
        "0",
        str(args.dt_max),
        "-",
    ]
    env = os.environ.copy()
    compat_lib = "/tmp/gamma_gdal_compat"
    env["LD_LIBRARY_PATH"] = compat_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, env=env)
    log.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"base_calc failed with code {proc.returncode}; see {log}")
    df = parse_gamma_bperp(bperp_file, dates, itab)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary = {
        "dates": len(dates),
        "pairs": int(len(df)),
        "slc_tab": str(slc_tab),
        "bperp_file": str(bperp_file),
        "itab": str(itab),
        "output_csv": args.output_csv,
        "log": str(log),
        "base_calc_command": " ".join(cmd),
        "ld_library_path_prefix": compat_lib,
        "returncode": proc.returncode,
    }
    if "bperp_m" in df and df["bperp_m"].notna().any():
        summary.update(
            {
                "bperp_min_m": float(df["bperp_m"].min()),
                "bperp_max_m": float(df["bperp_m"].max()),
                "bperp_median_abs_m": float(df["bperp_m"].abs().median()),
            }
        )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
