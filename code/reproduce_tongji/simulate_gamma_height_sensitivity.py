#!/usr/bin/env python3
"""Simulate the GAMMA differential-phase sensitivity to one metre of height.

The calculation uses ``phase_sim_orb_pt`` twice at 4 m and 104 m, subtracts
the resulting unwrapped orbital phase stacks, and divides by 100 m.  A wide
finite-difference interval plus ``ph_flag=1`` avoids cancellation of large
ellipsoidal phase terms in GAMMA's FLOAT output.  No building height attribute
is read.  The output has shape ``(interferogram, IPTA point)`` in rad/m.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def gamma_env() -> dict[str, str]:
    env = os.environ.copy()
    bins = [
        "/usr/local/GAMMA/IPTA/bin",
        "/usr/local/GAMMA/IPTA/scripts",
        "/usr/local/GAMMA/DIFF/bin",
        "/usr/local/GAMMA/ISP/bin",
    ]
    env["PATH"] = ":".join(bins) + ":" + env.get("PATH", "")
    compat = "/tmp/gamma_gdal_compat"
    env["LD_LIBRARY_PATH"] = compat + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ipta-dir", default="work/gamma_native_ipta_sbas/ipta")
    p.add_argument("--pairs-csv", default="work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv")
    p.add_argument("--reference-par", default="data/tongji_rslc/20200708.rslc.par")
    p.add_argument("--output", default="work/gamma_native_ipta_sbas/ipta/phase_height_sensitivity_rad_per_m.npy")
    p.add_argument("--summary", default="results/metadata/gamma_phase_height_sensitivity_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ipta = Path(args.ipta_dir)
    metadata = pd.read_csv(ipta / "point_metadata.csv")
    pairs = pd.read_csv(args.pairs_csv).drop_duplicates(["master", "slave"])
    npoints, npairs = len(metadata), len(pairs)
    h4 = ipta / "height_sensitivity_4m.phgt"
    h5 = ipta / "height_sensitivity_104m.phgt"
    sim4 = ipta / "height_sensitivity_4m.psim"
    sim5 = ipta / "height_sensitivity_104m.psim"
    np.full(npoints, 4.0, dtype=">f4").tofile(h4)
    np.full(npoints, 104.0, dtype=">f4").tofile(h5)
    base = [
        "phase_sim_orb_pt", str(ipta / "tongji.plist"), "-", str(ipta / "pSLC_par"), "-",
        str(ipta / "pairs.itab"), "-",
    ]
    for height, output in [(h4, sim4), (h5, sim5)]:
        proc = subprocess.run(
            base + [str(height), str(output), str(Path(args.reference_par)), "-", "1"],
            env=gamma_env(), text=True, capture_output=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stdout + "\n" + proc.stderr)
    a = np.fromfile(sim4, dtype=">f4")
    b = np.fromfile(sim5, dtype=">f4")
    expected = npairs * npoints
    if a.size != expected or b.size != expected:
        raise RuntimeError(f"phase simulation size mismatch: {a.size}, {b.size}, expected {expected}")
    sensitivity = ((b.reshape(npairs, npoints) - a.reshape(npairs, npoints)) / 100.0).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, sensitivity)
    payload = {
        "method": "[GAMMA phase_sim_orb_pt(104 m) - phase_sim_orb_pt(4 m)] / 100 m; ph_flag=1",
        "prior_height_used": False,
        "pairs": npairs,
        "points": npoints,
        "median_abs_sensitivity_rad_per_m": float(np.nanmedian(np.abs(sensitivity))),
        "p05_p95_sensitivity_rad_per_m": [
            float(np.nanpercentile(sensitivity, 5)), float(np.nanpercentile(sensitivity, 95))
        ],
        "output": str(output),
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
