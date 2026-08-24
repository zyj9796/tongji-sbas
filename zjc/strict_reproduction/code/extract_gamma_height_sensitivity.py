#!/usr/bin/env python3
"""Extract per-island dphase/dheight from GAMMA dh_map_orb on a 2x6 grid."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


MLI_WIDTH = 5000
MLI_LINES = 1166


def gamma_environment() -> dict[str, str]:
    env = os.environ.copy()
    compat = Path("/tmp/gamma_gdal_compat")
    compat.mkdir(parents=True, exist_ok=True)
    link = compat / "libgdal.so.26"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to("/lib/libgdal.so.30")
    env["LD_LIBRARY_PATH"] = str(compat) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def run(command: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
        stream.write(f"\nRETURN_CODE={result.returncode}\n\n")
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")


def paths(date: str, reference: str, crop: Path, rslc: Path) -> tuple[Path, Path]:
    if date == reference:
        return crop / f"{date}.slc", crop / f"{date}.slc.par"
    return rslc / f"{date}.rslc", rslc / f"{date}.rslc.par"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--height-2x6", type=Path, required=True)
    parser.add_argument("--quality-metrics", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    args = parser.parse_args()

    quality = np.load(args.quality_metrics)
    row = quality["row"].astype(np.int32)
    col = quality["col"].astype(np.int32)
    mli_row = np.minimum(row // 6, MLI_LINES - 1)
    mli_col = np.minimum(col // 2, MLI_WIDTH - 1)
    flat = mli_row.astype(np.int64) * MLI_WIDTH + mli_col
    env = gamma_environment()
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.summary.exists():
        results = {item["pair_name"]: item for item in json.loads(args.summary.read_text())}

    lines = [line.split() for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    for sequence, fields in enumerate(lines, start=1):
        pair_index, primary, secondary = int(fields[0]), fields[1], fields[2]
        name = f"{primary}_{secondary}"
        output = args.output_root / f"{name}.npz"
        print(f"[{sequence}/{len(lines)}] {name}", flush=True)
        if output.exists():
            with np.load(output) as current:
                if current["phase_sensitivity_rad_per_m"].shape == row.shape:
                    print("existing_valid", flush=True)
                    continue
        work = args.work_root / name
        work.mkdir(parents=True, exist_ok=True)
        log = work / "dh_map_orb.log"
        log.write_text("", encoding="utf-8")
        _, par1 = paths(primary, args.reference, args.crop_root, args.rslc_root)
        _, par2 = paths(secondary, args.reference, args.crop_root, args.rslc_root)
        off = work / "dpdh_2x6.off"
        dpdh = work / "dpdh_2x6"
        run(["create_offset", str(par1), str(par2), str(off), "1", "2", "6", "0"], log, env)
        run(
            ["dh_map_orb", str(par1), str(par2), str(off), str(args.height_2x6), "-", str(dpdh), "-", str(args.crop_root / f"{args.reference}.slc.par"), "1"],
            log,
            env,
        )
        expected = MLI_WIDTH * MLI_LINES * 4
        if dpdh.stat().st_size != expected:
            raise RuntimeError(f"invalid dpdh raster size for {name}")
        raster = np.memmap(dpdh, dtype=">f4", mode="r", shape=(MLI_WIDTH * MLI_LINES,))
        sensitivity = raster[flat].astype(np.float32)
        np.savez_compressed(
            output,
            row=row,
            col=col,
            phase_sensitivity_rad_per_m=sensitivity,
        )
        item = {
            "pair_index": pair_index,
            "pair_name": name,
            "finite_pixel_count": int(np.isfinite(sensitivity).sum()),
            "sensitivity_min_rad_per_m": float(np.nanmin(sensitivity)),
            "sensitivity_median_rad_per_m": float(np.nanmedian(sensitivity)),
            "sensitivity_max_rad_per_m": float(np.nanmax(sensitivity)),
            "gamma_program": "dh_map_orb",
            "looks": [2, 6],
        }
        results[name] = item
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([results[key] for key in sorted(results)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dpdh.unlink()
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
