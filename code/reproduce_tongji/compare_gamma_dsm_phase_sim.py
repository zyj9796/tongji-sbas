#!/usr/bin/env python3
"""Compare zero-height and DSM-height GAMMA phase simulation for one pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inventory_data import parse_gamma_par


def gamma_env() -> dict[str, str]:
    env = os.environ.copy()
    compat = "/tmp/gamma_gdal_compat"
    env["LD_LIBRARY_PATH"] = compat + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def run(cmd: list[str], log: Path) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=gamma_env())
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\nSTDERR:\n" + proc.stderr)
        f.write(f"\nRETURN_CODE={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}; see {log}")


def read_fcomplex(path: Path, rows: int, width: int) -> np.ndarray:
    return np.fromfile(path, dtype=">c8").reshape(rows, width)


def read_float(path: Path, rows: int, width: int) -> np.ndarray:
    return np.fromfile(path, dtype=">f4").reshape(rows, width)


def circ_std(phase: np.ndarray) -> float:
    z = np.exp(1j * phase[np.isfinite(phase)])
    if z.size == 0:
        return float("nan")
    r = np.abs(np.mean(z))
    return float(np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="20200525")
    parser.add_argument("--slave", default="20200616")
    parser.add_argument("--rslc-dir", default="data/tongji_rslc")
    parser.add_argument("--intf-root", default="work/gamma_sbas/intf")
    parser.add_argument("--hgt", default="work/gamma_sbas/dem/20200708_dsm_rdc.hgt")
    parser.add_argument("--out-dir", default="work/gamma_sbas/dem_phase_diagnostic")
    parser.add_argument("--figure", default="results/pic_all/24_gamma_dsm_phase_sim_comparison.png")
    parser.add_argument("--summary", default="results/metadata/gamma_dsm_phase_sim_comparison_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pair = f"{args.master}_{args.slave}"
    pair_dir = Path(args.intf_root) / pair
    out_dir = Path(args.out_dir) / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{pair}_dsm_phase_sim.log"
    if args.overwrite and log.exists():
        log.unlink()

    m_par = Path(args.rslc_dir) / f"{args.master}.rslc.par"
    s_par = Path(args.rslc_dir) / f"{args.slave}.rslc.par"
    off = pair_dir / f"{pair}.off"
    adf = pair_dir / f"{pair}.adf"
    diff_par = pair_dir / f"{pair}.diff_par"
    zero_diff = pair_dir / f"{pair}.diff"
    dsm_sim = out_dir / f"{pair}.dsm.sim_orb"
    dsm_diff = out_dir / f"{pair}.dsm.diff"
    if args.overwrite or not dsm_sim.exists():
        run(["phase_sim_orb", str(m_par), str(s_par), str(off), args.hgt, str(dsm_sim), "-", "-", "-", "1", "1"], log)
    if args.overwrite or not dsm_diff.exists():
        run(["sub_phase", str(adf), str(dsm_sim), str(diff_par), str(dsm_diff), "1", "0", "0"], log)

    par = parse_gamma_par(m_par)
    width = int(par["range_samples"])
    rows = int(par["azimuth_lines"])
    zero = read_fcomplex(zero_diff, rows, width)
    dsm = read_fcomplex(dsm_diff, rows, width)
    coh = read_float(pair_dir / f"{pair}.cc", rows, width)
    hgt = read_float(Path(args.hgt), rows, width)
    valid = np.isfinite(coh) & (coh > 0.25) & np.isfinite(hgt)
    zero_phase = np.angle(zero)
    dsm_phase = np.angle(dsm)
    delta = np.angle(np.exp(1j * (dsm_phase - zero_phase)))

    summary = {
        "pair": pair,
        "hgt": args.hgt,
        "dsm_sim_orb": str(dsm_sim),
        "dsm_diff": str(dsm_diff),
        "valid_pixels_coh_gt_0_25": int(valid.sum()),
        "zero_diff_phase_circ_std_rad": circ_std(zero_phase[valid]),
        "dsm_diff_phase_circ_std_rad": circ_std(dsm_phase[valid]),
        "dsm_minus_zero_phase_circ_std_rad": circ_std(delta[valid]),
        "median_coherence_valid": float(np.median(coh[valid])),
        "median_hgt_valid_m": float(np.median(hgt[valid])),
        "note": "Diagnostic only. DSM includes building heights, so this product should not replace the current height inversion without validating the height reference model.",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=220)
    im0 = axes[0, 0].imshow(zero_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title("Zero-height diff phase")
    im1 = axes[0, 1].imshow(dsm_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("DSM-height diff phase")
    im2 = axes[1, 0].imshow(delta, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title("DSM minus zero phase")
    axes[1, 1].hist(delta[valid], bins=80, color="#4477aa", edgecolor="white")
    axes[1, 1].set_title("Phase difference histogram, coh > 0.25")
    axes[1, 1].set_xlabel("rad")
    for ax in axes.ravel()[:3]:
        ax.set_xlabel("range column")
        ax.set_ylabel("azimuth row")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    plt.close(fig)

    summary["figure"] = args.figure
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
