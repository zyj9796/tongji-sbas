#!/usr/bin/env python3
"""Build a radar-coordinate DSM/HGT product for GAMMA phase simulation."""

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


def write_figure(hgt_path: Path, rows: int, width: int, figure: Path) -> dict[str, float | int]:
    hgt = np.fromfile(hgt_path, dtype=">f4").reshape(rows, width)
    valid = np.isfinite(hgt) & (hgt > -1000.0) & (hgt < 10000.0)
    vals = hgt[valid]
    stats = {
        "valid_pixels": int(valid.sum()),
        "width": int(width),
        "rows": int(rows),
        "min_m": float(np.min(vals)),
        "p02_m": float(np.quantile(vals, 0.02)),
        "median_m": float(np.median(vals)),
        "p98_m": float(np.quantile(vals, 0.98)),
        "max_m": float(np.max(vals)),
    }
    show = np.where(valid, hgt, np.nan)
    vmax = max(stats["p98_m"], stats["median_m"] + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    im = axes[0].imshow(show, cmap="terrain", vmin=stats["p02_m"], vmax=vmax)
    axes[0].set_title("DSM in SAR/RDC geometry")
    axes[0].set_xlabel("range column")
    axes[0].set_ylabel("azimuth row")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="m")
    axes[1].hist(vals, bins=80, color="#4477aa", edgecolor="white")
    axes[1].set_title("RDC DSM height distribution")
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("pixels")
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsm", default="data/dsm/tongji_real_dsm_1m_rslc_extent.tif")
    parser.add_argument("--slc-par", default="data/tongji_rslc/20200708.rslc.par")
    parser.add_argument("--out-dir", default="work/gamma_sbas/dem")
    parser.add_argument("--scene", default="20200708")
    parser.add_argument("--summary", default="results/metadata/gamma_rdc_dsm_summary.json")
    parser.add_argument("--figure", default="results/pic_all/23_gamma_rdc_dsm.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{args.scene}_rdc_dsm.log"
    if args.overwrite and log.exists():
        log.unlink()

    dem = out_dir / "tongji_dsm_1m.dem"
    dem_par = out_dir / "tongji_dsm_1m.dem_par"
    dem_seg = out_dir / "tongji_dsm_1m_seg.dem"
    dem_seg_par = out_dir / "tongji_dsm_1m_seg.dem_par"
    lookup = out_dir / f"{args.scene}_gc_map.lt"
    sim_sar = out_dir / f"{args.scene}_sim_sar_demgeo.float"
    hgt_rdc = out_dir / f"{args.scene}_dsm_rdc.hgt"

    if args.overwrite or not dem.exists() or not dem_par.exists():
        run(
            [
                "dem_import",
                args.dsm,
                str(dem),
                str(dem_par),
                "0",
                "1",
                "-",
                "-",
                "-",
                "-",
                "-",
                "0",
                "-9999",
            ],
            log,
        )
    if args.overwrite or not lookup.exists() or not dem_seg.exists() or not dem_seg_par.exists():
        run(
            [
                "gc_map",
                args.slc_par,
                "-",
                str(dem_par),
                str(dem),
                str(dem_seg_par),
                str(dem_seg),
                str(lookup),
                "1",
                "1",
                str(sim_sar),
                out_dir / f"{args.scene}_u.float",
                out_dir / f"{args.scene}_v.float",
                out_dir / f"{args.scene}_inc.float",
                out_dir / f"{args.scene}_psi.float",
                out_dir / f"{args.scene}_pix.float",
                out_dir / f"{args.scene}_ls_map.float",
                "8",
                "2",
                "2",
            ],
            log,
        )

    slc_par = parse_gamma_par(Path(args.slc_par))
    width = int(slc_par["range_samples"])
    rows = int(slc_par["azimuth_lines"])
    dem_seg_width = 1439
    for line in dem_seg_par.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("width:"):
            dem_seg_width = int(line.split()[1])
            break
    if args.overwrite or not hgt_rdc.exists():
        run(
            [
                "geocode",
                str(lookup),
                str(dem_seg),
                str(dem_seg_width),
                str(hgt_rdc),
                str(width),
                str(rows),
                "2",
                "0",
                "1",
                "1",
                "2",
                "16",
                "4",
            ],
            log,
        )

    stats = write_figure(hgt_rdc, rows, width, Path(args.figure))
    summary = {
        "dsm": args.dsm,
        "slc_par": args.slc_par,
        "dem": str(dem),
        "dem_par": str(dem_par),
        "dem_seg": str(dem_seg),
        "dem_seg_par": str(dem_seg_par),
        "lookup_table": str(lookup),
        "rdc_hgt": str(hgt_rdc),
        "figure": args.figure,
        "note": "DSM was resampled to reference SLC range-Doppler coordinates. Use with care: the DSM includes buildings.",
        **stats,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
