#!/usr/bin/env python3
"""Stream GAMMA differential interferograms and retain only building-island observations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


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
        process = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True
        )
        stream.write(f"\nRETURN_CODE={process.returncode}\n\n")
    if process.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}; see {log}")


def slc_paths(date: str, reference: str, crop_root: Path, rslc_root: Path) -> tuple[Path, Path]:
    if date == reference:
        return crop_root / f"{date}.slc", crop_root / f"{date}.slc.par"
    return rslc_root / f"{date}.rslc", rslc_root / f"{date}.rslc.par"


def read_pairs(path: Path) -> list[dict[str, object]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        pairs.append(
            {
                "pair_index": int(fields[0]),
                "primary": fields[1],
                "secondary": fields[2],
                "perpendicular_baseline_m": float(fields[3]),
                "temporal_baseline_days": float(fields[4]),
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--height-rdc", type=Path, required=True)
    parser.add_argument("--island-points", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--pair-indices", nargs="*", type=int, default=[])
    parser.add_argument("--width", type=int, default=10_000)
    parser.add_argument("--lines", type=int, default=7_000)
    parser.add_argument("--global-range-offset", type=int, default=0)
    parser.add_argument("--global-azimuth-offset", type=int, default=0)
    args = parser.parse_args()

    points = np.load(args.island_points)
    keep = points["label"] > 0
    row = points["row"][keep].astype(np.int32)
    col = points["col"][keep].astype(np.int32)
    local_row = row - args.global_azimuth_offset
    local_col = col - args.global_range_offset
    flat = local_row.astype(np.int64) * args.width + local_col
    if (
        np.any(local_row < 0) or np.any(local_row >= args.lines)
        or np.any(local_col < 0) or np.any(local_col >= args.width)
    ):
        raise RuntimeError("island point outside selected radar processing window")

    pairs = read_pairs(args.pairs)
    if args.pair_indices:
        selected = set(args.pair_indices)
        pairs = [pair for pair in pairs if pair["pair_index"] in selected]
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = gamma_environment()
    prior = {}
    if args.summary.exists():
        prior = {
            item["pair_name"]: item
            for item in json.loads(args.summary.read_text(encoding="utf-8"))
        }

    for sequence, pair in enumerate(pairs, start=1):
        primary = str(pair["primary"])
        secondary = str(pair["secondary"])
        name = f"{primary}_{secondary}"
        output = args.output_root / f"{name}.npz"
        print(f"[{sequence}/{len(pairs)}] {name}", flush=True)
        if output.exists():
            with np.load(output) as existing:
                if existing["filtered_wrapped_phase_rad"].shape == row.shape:
                    prior[name] = {**pair, "pair_name": name, "status": "existing_valid"}
                    print("existing_valid", flush=True)
                    continue

        pair_work = args.work_root / name
        pair_work.mkdir(parents=True, exist_ok=True)
        log = pair_work / "gamma.log"
        log.write_text("", encoding="utf-8")
        slc1, par1 = slc_paths(primary, args.reference, args.crop_root, args.rslc_root)
        slc2, par2 = slc_paths(secondary, args.reference, args.crop_root, args.rslc_root)
        off = pair_work / f"{name}.off"
        sim = pair_work / f"{name}.sim_unw"
        diff = pair_work / f"{name}.diff"
        coherence = pair_work / f"{name}.cc"
        filtered = pair_work / f"{name}.adf"
        filtered_cc = pair_work / f"{name}.adf_cc"

        run(["create_offset", str(par1), str(par2), str(off), "1", "1", "1", "0"], log, env)
        run(
            ["phase_sim_orb", str(par1), str(par2), str(off), str(args.height_rdc), str(sim), str(par1), "-", "-", "1", "1"],
            log,
            env,
        )
        run(
            ["SLC_diff_intf", str(slc1), str(slc2), str(par1), str(par2), str(off), str(sim), str(diff), "1", "1"],
            log,
            env,
        )
        run(["cc_wave", str(diff), "-", "-", str(coherence), str(args.width), "5", "5", "3"], log, env)
        run(["adf", str(diff), str(filtered), str(filtered_cc), str(args.width), "0.40", "32", "5"], log, env)

        expected_complex = args.width * args.lines * 8
        expected_float = args.width * args.lines * 4
        if diff.stat().st_size != expected_complex or filtered.stat().st_size != expected_complex:
            raise RuntimeError(f"invalid complex raster size for {name}")
        if coherence.stat().st_size != expected_float:
            raise RuntimeError(f"invalid coherence raster size for {name}")
        raw_values = np.memmap(diff, dtype=">c8", mode="r", shape=(args.width * args.lines,))[flat]
        filtered_values = np.memmap(filtered, dtype=">c8", mode="r", shape=(args.width * args.lines,))[flat]
        coherence_values = np.memmap(coherence, dtype=">f4", mode="r", shape=(args.width * args.lines,))[flat]
        np.savez_compressed(
            output,
            row=row,
            col=col,
            label=points["label"][keep].astype(np.int32),
            building_uid=points["building_uid"][keep].astype(np.int32),
            floor_audit_only=points["floor"][keep].astype(np.int16),
            wrapped_phase_rad=np.angle(raw_values).astype(np.float32),
            filtered_wrapped_phase_rad=np.angle(filtered_values).astype(np.float32),
            coherence=coherence_values.astype(np.float32),
        )
        metrics = {
            **pair,
            "pair_name": name,
            "status": "processed",
            "building_island_pixel_count": int(len(flat)),
            "finite_coherence_count": int(np.isfinite(coherence_values).sum()),
            "mean_building_island_coherence": float(np.nanmean(coherence_values)),
            "filter": {"name": "GAMMA_adf", "alpha": 0.4, "nfft": 32, "cc_window": 5},
            "coherence": {"name": "GAMMA_cc_wave", "window": [5, 5], "mode": 3},
            "processing_window": {
                "global_range_offset": args.global_range_offset,
                "global_azimuth_offset": args.global_azimuth_offset,
                "width": args.width,
                "lines": args.lines,
            },
        }
        prior[name] = metrics
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([prior[key] for key in sorted(prior)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for temporary in (sim, diff, coherence, filtered, filtered_cc):
            if temporary.exists():
                temporary.unlink()
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
