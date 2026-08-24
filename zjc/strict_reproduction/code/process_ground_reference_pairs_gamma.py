#!/usr/bin/env python3
"""Generate 2x6 GAMMA differential phases at stable ground-reference points."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


WIDTH = 5000
LINES = 1166


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


def slc_paths(date: str, reference: str, crop: Path, rslc: Path) -> tuple[Path, Path]:
    if date == reference:
        return crop / f"{date}.slc", crop / f"{date}.slc.par"
    return rslc / f"{date}.rslc", rslc / f"{date}.rslc.par"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--height-2x6", type=Path, required=True)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    args = parser.parse_args()

    ground = np.load(args.ground_points)
    row, col = ground["mli_row"].astype(np.int32), ground["mli_col"].astype(np.int32)
    flat = row.astype(np.int64) * WIDTH + col
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = gamma_environment()
    results = {}
    if args.summary.exists():
        results = {item["pair_name"]: item for item in json.loads(args.summary.read_text())}
    pairs = [line.split() for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]

    for sequence, fields in enumerate(pairs, start=1):
        index, primary, secondary = int(fields[0]), fields[1], fields[2]
        name = f"{primary}_{secondary}"
        output = args.output_root / f"{name}.npz"
        print(f"[{sequence}/{len(pairs)}] {name}", flush=True)
        if output.exists():
            with np.load(output) as current:
                if current["wrapped_phase_rad"].shape == row.shape:
                    print("existing_valid", flush=True)
                    continue
        work = args.work_root / name
        work.mkdir(parents=True, exist_ok=True)
        log = work / "gamma.log"
        log.write_text("", encoding="utf-8")
        slc1, par1 = slc_paths(primary, args.reference, args.crop_root, args.rslc_root)
        slc2, par2 = slc_paths(secondary, args.reference, args.crop_root, args.rslc_root)
        off, sim, diff = work / "pair_2x6.off", work / "sim_2x6", work / "diff_2x6"
        run(["create_offset", str(par1), str(par2), str(off), "1", "2", "6", "0"], log, env)
        run(
            ["phase_sim_orb", str(par1), str(par2), str(off), str(args.height_2x6), str(sim), str(args.crop_root / f"{args.reference}.slc.par"), "-", "-", "1", "1"],
            log,
            env,
        )
        run(
            ["SLC_diff_intf", str(slc1), str(slc2), str(par1), str(par2), str(off), str(sim), str(diff), "2", "6"],
            log,
            env,
        )
        expected = WIDTH * LINES * 8
        if diff.stat().st_size != expected:
            raise RuntimeError(f"invalid 2x6 differential interferogram for {name}")
        values = np.memmap(diff, dtype=">c8", mode="r", shape=(WIDTH * LINES,))[flat]
        phase = np.angle(values).astype(np.float32)
        np.savez_compressed(output, mli_row=row, mli_col=col, wrapped_phase_rad=phase)
        item = {
            "pair_index": index,
            "pair_name": name,
            "ground_reference_count": int(len(row)),
            "finite_phase_count": int(np.isfinite(phase).sum()),
            "looks": [2, 6],
        }
        results[name] = item
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([results[key] for key in sorted(results)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sim.unlink()
        diff.unlink()
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
