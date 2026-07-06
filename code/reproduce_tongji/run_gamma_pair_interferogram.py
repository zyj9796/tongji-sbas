#!/usr/bin/env python3
"""Run a minimal GAMMA interferogram chain for one co-registered RSLC pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from inventory_data import parse_gamma_par


def gamma_env() -> dict:
    env = os.environ.copy()
    compat = "/tmp/gamma_gdal_compat"
    env["LD_LIBRARY_PATH"] = compat + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run_cmd(cmd: list[str], log_path: Path) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=gamma_env())
    with log_path.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\nSTDERR:\n" + proc.stderr)
        f.write(f"\nRETURN_CODE={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="20200525")
    parser.add_argument("--slave", default="20200616")
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--out-root", default="work/gamma_sbas/intf")
    parser.add_argument("--rlks", type=int, default=1)
    parser.add_argument("--azlks", type=int, default=1)
    parser.add_argument("--adf-alpha", default="0.4")
    parser.add_argument("--adf-nfft", default="32")
    parser.add_argument("--make-diff", action="store_true")
    parser.add_argument(
        "--hgt-rdc",
        default="-",
        help="Radar-coordinate height map for phase_sim_orb. Use '-' for no DEM/HGT reference.",
    )
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    rslc_dir = Path(args.rslc_dir)
    pair = f"{args.master}_{args.slave}"
    out_dir = Path(args.out_root) / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "gamma_pair.log"
    if log.exists():
        log.unlink()
    m_slc = rslc_dir / f"{args.master}.rslc"
    s_slc = rslc_dir / f"{args.slave}.rslc"
    m_par = rslc_dir / f"{args.master}.rslc.par"
    s_par = rslc_dir / f"{args.slave}.rslc.par"
    off = out_dir / f"{pair}.off"
    intf = out_dir / f"{pair}.int"
    filt = out_dir / f"{pair}.adf"
    cc = out_dir / f"{pair}.cc"
    diff_par = out_dir / f"{pair}.diff_par"
    sim_orb = out_dir / f"{pair}.sim_orb"
    diff = out_dir / f"{pair}.diff"

    par = parse_gamma_par(m_par)
    width = int(par["range_samples"]) // args.rlks
    rows = int(par["azimuth_lines"]) // args.azlks
    run_cmd(["create_offset", str(m_par), str(s_par), str(off), "1", str(args.rlks), str(args.azlks), "0"], log)
    run_cmd(["SLC_intf", str(m_slc), str(s_slc), str(m_par), str(s_par), str(off), str(intf), str(args.rlks), str(args.azlks), "-", "-", "0", "0"], log)
    run_cmd(["adf", str(intf), str(filt), str(cc), str(width), args.adf_alpha, args.adf_nfft, "5"], log)
    run_cmd(["create_diff_par", str(off), "-", str(diff_par), "0", "0"], log)
    if args.make_diff:
        run_cmd(["phase_sim_orb", str(m_par), str(s_par), str(off), str(args.hgt_rdc), str(sim_orb), "-", "-", "-", "1", "1"], log)
        run_cmd(["sub_phase", str(filt), str(sim_orb), str(diff_par), str(diff), "1", "0", "0"], log)

    summary = {
        "pair": pair,
        "master": args.master,
        "slave": args.slave,
        "width": width,
        "rows": rows,
        "rlks": args.rlks,
        "azlks": args.azlks,
        "outputs": {
            "off": str(off),
            "interferogram": str(intf),
            "filtered_interferogram": str(filt),
            "coherence": str(cc),
            "diff_par": str(diff_par),
            "sim_orb": str(sim_orb) if args.make_diff else None,
            "differential_interferogram": str(diff) if args.make_diff else None,
            "hgt_rdc": str(args.hgt_rdc) if args.make_diff else None,
            "log": str(log),
        },
    }
    summary_path = Path(args.summary) if args.summary else out_dir / f"{pair}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
