#!/usr/bin/env python3
"""Recover per-date crop origins by feature matching BC3 quick-look images."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform


def quicklook(row: dict[str, str], step: int) -> tuple[np.ndarray, tuple[int, int]]:
    with zipfile.ZipFile(row["zip_path"]) as archive:
        member = next(name for name in archive.namelist() if name.endswith("preview/quick-look.png"))
        with Image.open(io.BytesIO(archive.read(member))) as image:
            native = image.size
            image = image.resize(
                (max(1, image.width // step), max(1, image.height // step)),
                Image.Resampling.BILINEAR,
            )
            data = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(data, [1.0, 99.5])
    normalized = np.clip((data - low) / max(high - low, 1e-6), 0.0, 1.0)
    return gaussian_filter(normalized, 0.7), native


def features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    detector = ORB(n_keypoints=3000, fast_threshold=0.02)
    detector.detect_and_extract(image)
    return detector.keypoints, detector.descriptors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="20231007")
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--reference-range-offset", type=float, default=2400.0)
    parser.add_argument("--reference-azimuth-offset", type=float, default=10000.0)
    args = parser.parse_args()

    with args.inventory.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    first = {}
    for row in sorted(rows, key=lambda item: item["sensing_start"]):
        first.setdefault(row["date"], row)
    reference_image, reference_native = quicklook(first[args.reference], args.step)
    reference_keypoints, reference_descriptors = features(reference_image)

    results = []
    for date in sorted(first):
        if date == args.reference:
            matrix = np.eye(3)
            matches = inliers = len(reference_keypoints)
            native = reference_native
        else:
            image, native = quicklook(first[date], args.step)
            keypoints, descriptors = features(image)
            pairs = match_descriptors(
                reference_descriptors, descriptors, cross_check=True, max_ratio=0.85
            )
            source = reference_keypoints[pairs[:, 0]][:, ::-1]
            destination = keypoints[pairs[:, 1]][:, ::-1]
            model, inlier_mask = ransac(
                (source, destination),
                AffineTransform,
                min_samples=3,
                residual_threshold=2.0,
                max_trials=5000,
                rng=0,
            )
            if model is None:
                raise RuntimeError(f"quick-look feature registration failed for {date}")
            matrix = model.params
            matches = len(pairs)
            inliers = int(inlier_mask.sum())
            if inliers < 20 or inliers / max(matches, 1) < 0.5:
                raise RuntimeError(
                    f"weak quick-look registration for {date}: {inliers}/{matches} inliers"
                )

        reference_point = np.array(
            [args.reference_range_offset / args.step, args.reference_azimuth_offset / args.step, 1.0]
        )
        mapped = matrix @ reference_point
        range_offset = int(round(mapped[0] * args.step))
        azimuth_offset = int(round(mapped[1] * args.step))
        results.append(
            {
                "date": date,
                "quicklook_native_width_height": list(native),
                "feature_matches": int(matches),
                "ransac_inliers": int(inliers),
                "inlier_fraction": float(inliers / max(matches, 1)),
                "affine_reference_to_date_downsampled": matrix.tolist(),
                "range_offset_px": range_offset,
                "azimuth_offset_px": azimuth_offset,
                "range_delta_from_reference_px": range_offset - int(args.reference_range_offset),
                "azimuth_delta_from_reference_px": azimuth_offset - int(args.reference_azimuth_offset),
            }
        )
        print(date, range_offset, azimuth_offset, f"inliers={inliers}/{matches}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
