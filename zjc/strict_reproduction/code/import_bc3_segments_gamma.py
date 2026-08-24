#!/usr/bin/env python3
"""Import deduplicated BC3 stripmap SAFE segments with GAMMA par_S1_SLC."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import zipfile
from pathlib import Path


def gamma_environment() -> dict[str, str]:
    env = os.environ.copy()
    compat = Path("/tmp/gamma_gdal_compat")
    compat.mkdir(parents=True, exist_ok=True)
    link = compat / "libgdal.so.26"
    if link.is_symlink() or link.exists():
        link.unlink()
    # The installed 2021 GAMMA build requests GDAL 3.0's soname 26. Ubuntu 22.04
    # provides ABI-compatible GDAL 3.4 as soname 30. Every imported parameter and
    # raster dimension is checked against the source XML/TIFF after this shim.
    link.symlink_to("/lib/libgdal.so.30")
    env["LD_LIBRARY_PATH"] = str(compat) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def parse_gamma_parameter(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().split()[0] if value.strip() else ""
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameter-only", action="store_true")
    args = parser.parse_args()

    with args.inventory.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["date"] == args.date]
    if len(rows) != 2:
        raise SystemExit(f"expected two selected segments for {args.date}, found {len(rows)}")

    output = args.output_dir
    xml_dir = output / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "gamma_import.log"
    log_path.write_text("", encoding="utf-8")
    env = gamma_environment()

    for index, row in enumerate(sorted(rows, key=lambda item: item["sensing_start"]), start=1):
        zip_path = Path(row["zip_path"])
        segment = f"{args.date}_seg{index}"
        with zipfile.ZipFile(zip_path) as archive:
            annotation = xml_dir / f"{segment}_annotation.xml"
            calibration = xml_dir / f"{segment}_calibration.xml"
            annotation.write_bytes(archive.read(row["annotation_member"]))
            calibration.write_bytes(archive.read(row["calibration_member"]))

        virtual_tiff = f"/vsizip/{zip_path}/{row['measurement_member']}"
        parameter = output / f"{segment}.slc.par"
        slc = "-" if args.parameter_only else str(output / f"{segment}.slc")
        command = [
            "par_S1_SLC",
            virtual_tiff,
            str(annotation),
            str(calibration),
            "-",
            str(parameter),
            slc,
            "-",  # Stripmap: no TOPS parameter file.
            "1",  # SCOMPLEX, preserves the compact complex-int16 representation.
        ]
        process = subprocess.run(command, env=env, text=True, capture_output=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.write(process.stdout)
            if process.stderr:
                log.write("\nSTDERR:\n" + process.stderr)
            log.write(f"\nRETURN_CODE={process.returncode}\n\n")
        if process.returncode:
            raise SystemExit(f"GAMMA import failed for {segment}; see {log_path}")

        # par_S1_SLC hard-codes Sentinel-1's right-looking geometry. BC3 in this
        # data set is explicitly left-looking. GAMMA encodes look side through
        # azimuth_angle (+90 right, -90 left), so correct it before any R-D work.
        correction = subprocess.run(
            ["set_value", str(parameter), str(parameter), "azimuth_angle", "-90.0"],
            env=env,
            text=True,
            capture_output=True,
        )
        if correction.returncode:
            raise SystemExit(f"failed to set left-looking geometry for {segment}")
        corners = subprocess.run(
            ["SLC_corners", str(parameter)], env=env, text=True, capture_output=True
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ set_value {parameter} {parameter} azimuth_angle -90.0\n")
            log.write(correction.stdout + correction.stderr + "\n")
            log.write(f"$ SLC_corners {parameter}\n")
            log.write(corners.stdout + corners.stderr + "\n")
        if corners.returncode:
            raise SystemExit(f"SLC_corners failed for {segment}")
        center_match = re.search(
            r"center latitude \(deg\.\):\s+([-+0-9.]+)\s+center longitude \(deg\.\):\s+([-+0-9.]+)",
            corners.stdout,
        )
        bounds_match = re.search(
            r"min\. latitude \(deg\.\):\s+([-+0-9.]+)\s+max\. latitude \(deg\.\):\s+([-+0-9.]+).*?"
            r"min\. longitude \(deg\.\):\s+([-+0-9.]+)\s+max\. longitude \(deg\.\):\s+([-+0-9.]+)",
            corners.stdout,
            re.S,
        )
        if not center_match or not bounds_match:
            raise SystemExit(f"could not parse left-looking corners for {segment}")
        center_lat, center_lon = center_match.groups()
        for keyword, value in (
            ("center_latitude", center_lat),
            ("center_longitude", center_lon),
        ):
            update = subprocess.run(
                ["set_value", str(parameter), str(parameter), keyword, value],
                env=env,
                text=True,
                capture_output=True,
            )
            if update.returncode:
                raise SystemExit(f"failed to update {keyword} for {segment}")
        min_lat, max_lat, min_lon, max_lon = map(float, bounds_match.groups())
        source_bounds = tuple(
            float(row[key])
            for key in ("bbox_west", "bbox_south", "bbox_east", "bbox_north")
        )
        gamma_bounds = (min_lon, min_lat, max_lon, max_lat)
        max_bound_difference = max(abs(a - b) for a, b in zip(source_bounds, gamma_bounds))
        if max_bound_difference > 0.01:
            raise SystemExit(
                f"left-looking GAMMA footprint differs from manifest by {max_bound_difference:.6f} deg"
            )

        values = parse_gamma_parameter(parameter)
        checks = {
            "range_samples": (int(values["range_samples"]), int(row["number_of_samples"])),
            "azimuth_lines": (int(values["azimuth_lines"]), int(row["number_of_lines"])),
        }
        failed = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
        if failed:
            raise SystemExit(f"GAMMA/XML dimension mismatch for {segment}: {failed}")
        if not args.parameter_only:
            expected_bytes = int(row["number_of_samples"]) * int(row["number_of_lines"]) * 4
            actual_bytes = (output / f"{segment}.slc").stat().st_size
            if actual_bytes != expected_bytes:
                raise SystemExit(
                    f"unexpected SCOMPLEX size for {segment}: {actual_bytes} != {expected_bytes}"
                )
        print(
            f"{segment}: {values['range_samples']} x {values['azimuth_lines']}, "
            f"format={values.get('image_format')}, start={row['sensing_start']}, "
            f"left-look footprint max_delta={max_bound_difference:.6f} deg"
        )


if __name__ == "__main__":
    main()
