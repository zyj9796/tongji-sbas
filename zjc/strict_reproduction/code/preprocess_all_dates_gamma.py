#!/usr/bin/env python3
"""Import, concatenate, and crop every BC3 date to the frozen paper window."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_CROP_BYTES = 10_000 * 7_000 * 4


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


def run(command: list[str], log: Path, env: dict[str, str]) -> str:
    process = subprocess.run(command, env=env, text=True, capture_output=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.write(process.stdout)
        if process.stderr:
            stream.write("\nSTDERR:\n" + process.stderr)
        stream.write(f"\nRETURN_CODE={process.returncode}\n\n")
    if process.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}; see {log}")
    return process.stdout + process.stderr


def read_value(path: Path, key: str) -> str:
    prefix = key + ":"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip().split()[0]
    raise KeyError(f"{key} not found in {path}")


def valid_crop(slc: Path, parameter: Path) -> bool:
    if not slc.exists() or not parameter.exists() or slc.stat().st_size != EXPECTED_CROP_BYTES:
        return False
    return read_value(parameter, "range_samples") == "10000" and read_value(
        parameter, "azimuth_lines"
    ) == "7000"


def process_date(
    date: str,
    inventory: Path,
    work_root: Path,
    crop_root: Path,
    keep_intermediate: bool,
    range_offset: int,
    azimuth_offset: int,
    force: bool,
) -> dict[str, object]:
    output_slc = crop_root / f"{date}.slc"
    output_par = crop_root / f"{date}.slc.par"
    if not force and valid_crop(output_slc, output_par):
        return {"date": date, "status": "existing_valid", "crop_bytes": output_slc.stat().st_size}

    date_root = work_root / date
    import_dir = date_root / "segments"
    mosaic_dir = date_root / "mosaic"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)
    log = date_root / "pipeline.log"
    log.write_text("", encoding="utf-8")
    env = gamma_environment()

    run(
        [
            str(Path(os.sys.executable)),
            str(SCRIPT_DIR / "import_bc3_segments_gamma.py"),
            "--inventory",
            str(inventory),
            "--date",
            date,
            "--output-dir",
            str(import_dir),
        ],
        log,
        env,
    )

    seg1 = import_dir / f"{date}_seg1.slc"
    seg2 = import_dir / f"{date}_seg2.slc"
    par1 = import_dir / f"{date}_seg1.slc.par"
    par2 = import_dir / f"{date}_seg2.slc.par"
    tab1 = date_root / "seg1.tab"
    tab2 = date_root / "seg2.tab"
    tab1.write_text(f"{seg1} {par1}\n", encoding="utf-8")
    tab2.write_text(f"{seg2} {par2}\n", encoding="utf-8")
    mosaic_tab = date_root / "mosaic.tab"

    base = ["SLC_cat_all", str(tab1), str(tab2), str(mosaic_dir), str(mosaic_tab)]
    run(base + ["0"], log, env)
    run(base + ["1"], log, env)
    off = mosaic_dir / f"{date}_seg1_{date}_seg2.off"
    initial_azimuth = float(read_value(off, "azimuth_offset_polynomial"))
    azimuth_patch_center = int(math.ceil(-initial_azimuth)) + 400
    run(base + ["2", "1", "-", str(azimuth_patch_center)], log, env)
    run(base + ["3", "1", "-", str(azimuth_patch_center), "3", "1"], log, env)
    fitted_range = float(read_value(off, "range_offset_polynomial"))
    fitted_azimuth = float(read_value(off, "azimuth_offset_polynomial"))
    run(base + ["4", "1", "-", str(azimuth_patch_center), "3", "1"], log, env)

    mosaic_slc = mosaic_dir / f"{date}_seg1.slc"
    mosaic_par = mosaic_dir / f"{date}_seg1.slc.par"
    run(
        [
            "SLC_copy",
            str(mosaic_slc),
            str(mosaic_par),
            str(output_slc),
            str(output_par),
            "4",
            "1.0",
            str(range_offset),
            "10000",
            str(azimuth_offset),
            "7000",
        ],
        log,
        env,
    )
    if not valid_crop(output_slc, output_par):
        raise RuntimeError(f"invalid final crop for {date}")

    corner_text = run(["SLC_corners", str(output_par)], log, env)
    footprint = re.search(
        r"min\. latitude \(deg\.\):\s+([-+0-9.]+)\s+max\. latitude \(deg\.\):\s+([-+0-9.]+).*?"
        r"min\. longitude \(deg\.\):\s+([-+0-9.]+)\s+max\. longitude \(deg\.\):\s+([-+0-9.]+)",
        corner_text,
        re.S,
    )
    if not footprint:
        raise RuntimeError(f"could not parse cropped footprint for {date}")

    if not keep_intermediate:
        for generated in (seg1, seg2, mosaic_slc):
            if generated.exists():
                generated.unlink()

    return {
        "date": date,
        "status": "processed",
        "crop_bytes": output_slc.stat().st_size,
        "crop_range_offset_px": range_offset,
        "crop_azimuth_offset_px": azimuth_offset,
        "mosaic_range_offset_px": fitted_range,
        "mosaic_azimuth_offset_px": fitted_azimuth,
        "azimuth_patch_center_px": azimuth_patch_center,
        "crop_bounds_minlat_maxlat_minlon_maxlon": [float(value) for value in footprint.groups()],
        "intermediate_slcs_retained": keep_intermediate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dates", nargs="*", default=[])
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--crop-offsets-json",
        type=Path,
        help="Optional per-date offsets produced by estimate_adaptive_crop_offsets.py",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    crop_offsets: dict[str, tuple[int, int]] = {}
    if args.crop_offsets_json:
        records = json.loads(args.crop_offsets_json.read_text(encoding="utf-8"))
        crop_offsets = {
            str(item["date"]): (int(item["range_offset_px"]), int(item["azimuth_offset_px"]))
            for item in records
        }

    if args.dates:
        dates = args.dates
    else:
        import csv

        with args.inventory.open(encoding="utf-8", newline="") as stream:
            dates = sorted({row["date"] for row in csv.DictReader(stream)})

    prior: dict[str, dict[str, object]] = {}
    if args.summary.exists():
        prior = {item["date"]: item for item in json.loads(args.summary.read_text(encoding="utf-8"))}
    results = prior
    for index, date in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] {date}", flush=True)
        result = process_date(
            date,
            args.inventory,
            args.work_root,
            args.crop_root,
            args.keep_intermediate,
            *crop_offsets.get(date, (2400, 10000)),
            args.force,
        )
        results[date] = result
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([results[key] for key in sorted(results)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
