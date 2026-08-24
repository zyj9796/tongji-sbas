#!/usr/bin/env python3
"""Coregister the cropped BC3 stack to 20231007 and enforce the paper QA gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


EXPECTED_BYTES = 10_000 * 7_000 * 4


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


def parse_quality(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    accepted = re.search(r"final solution:\s+(\d+) offset estimates accepted out of (\d+)", text)
    sigma = re.search(
        r"final model fit std\. dev\. \(samples\) range:\s+([-+0-9.eE]+)\s+azimuth:\s+([-+0-9.eE]+)",
        text,
    )
    range_poly = re.search(r"final range offset poly\. coeff\.:\s+([-+0-9.eE]+)", text)
    az_poly = re.search(r"final azimuth offset poly\. coeff\.:\s+([-+0-9.eE]+)", text)
    if not all((accepted, sigma, range_poly, az_poly)):
        raise RuntimeError(f"incomplete coregistration quality report: {path}")
    return {
        "accepted_offsets": int(accepted.group(1)),
        "tested_offsets": int(accepted.group(2)),
        "final_range_offset_intercept_px": float(range_poly.group(1)),
        "final_azimuth_offset_intercept_px": float(az_poly.group(1)),
        "final_range_fit_sigma_px": float(sigma.group(1)),
        "final_azimuth_fit_sigma_px": float(sigma.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--rslc-root", type=Path, required=True)
    parser.add_argument("--height-2x6", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--dates", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.rslc_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    env = gamma_environment()
    dates = args.dates or sorted(path.stem for path in args.crop_root.glob("*.slc"))
    ref_slc = args.crop_root / f"{args.reference}.slc"
    ref_par = args.crop_root / f"{args.reference}.slc.par"
    results: dict[str, dict[str, object]] = {}
    if args.summary.exists():
        results = {item["date"]: item for item in json.loads(args.summary.read_text(encoding="utf-8"))}

    for index, date in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] {date}", flush=True)
        if date == args.reference:
            results[date] = {"date": date, "status": "reference", "qa_pass": True}
        else:
            slc = args.crop_root / f"{date}.slc"
            par = args.crop_root / f"{date}.slc.par"
            rslc = args.rslc_root / f"{date}.rslc"
            rpar = args.rslc_root / f"{date}.rslc.par"
            quality = Path(str(rslc) + ".coreg_quality")
            reusable = (
                not args.force
                and rslc.exists()
                and rslc.stat().st_size == EXPECTED_BYTES
                and rpar.exists()
                and quality.exists()
            )
            if reusable:
                try:
                    metrics = parse_quality(quality)
                except RuntimeError:
                    reusable = False
            if reusable:
                status = "existing_valid"
            else:
                log = args.log_root / f"{date}.log"
                with log.open("w", encoding="utf-8") as stream:
                    proc = subprocess.run(
                        [
                            "SLC_coreg", str(slc), str(par), str(rslc), str(rpar), "-", "-",
                            str(ref_slc), str(ref_par), str(args.height_2x6), "2", "6",
                        ],
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if proc.returncode != 0:
                    raise RuntimeError(f"SLC_coreg failed for {date}; see {log}")
                if not rslc.exists() or rslc.stat().st_size != EXPECTED_BYTES:
                    raise RuntimeError(f"invalid RSLC size for {date}")
                metrics = parse_quality(quality)
                status = "processed"
                for intermediate in (Path(str(slc) + ".mli"), Path(str(slc) + ".mli.par")):
                    if intermediate.exists():
                        intermediate.unlink()

            qa_pass = (
                int(metrics["accepted_offsets"]) >= 100
                and int(metrics["accepted_offsets"]) / int(metrics["tested_offsets"]) >= 0.1
                and
                abs(float(metrics["final_range_offset_intercept_px"])) < 0.1
                and abs(float(metrics["final_azimuth_offset_intercept_px"])) < 0.1
                and float(metrics["final_range_fit_sigma_px"]) < 0.1
                and float(metrics["final_azimuth_fit_sigma_px"]) < 0.1
            )
            if not qa_pass:
                raise RuntimeError(f"paper <0.1 px coregistration gate failed for {date}: {metrics}")
            results[date] = {"date": date, "status": status, "qa_pass": qa_pass, **metrics}

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([results[key] for key in sorted(results)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(results[date], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
