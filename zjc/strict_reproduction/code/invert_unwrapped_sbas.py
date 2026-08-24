#!/usr/bin/env python3
"""Paper-model SBAS inversion of GAMMA-MCF unwrapped building phases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


C = 299_792_458.0


def qdict(values: np.ndarray) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    return {
        str(q): (float(np.quantile(values, q)) if len(values) else None)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def solve_pixel(
    observation: np.ndarray,
    sensitivity: np.ndarray,
    velocity_coefficient: np.ndarray,
    weight: np.ndarray,
    minimum_pairs: int,
    bisquare_c: float,
) -> tuple[float, float, float, int, float]:
    valid = np.isfinite(observation + sensitivity + velocity_coefficient + weight) & (weight > 0)
    if valid.sum() < minimum_pairs:
        return np.nan, np.nan, np.nan, int(valid.sum()), np.nan
    y = observation[valid]
    design = np.column_stack((velocity_coefficient[valid], sensitivity[valid]))
    base = weight[valid]
    robust = np.ones_like(y)
    estimate = np.zeros(2, dtype=np.float64)
    for _ in range(12):
        total = base * robust
        normal = design.T @ (total[:, None] * design)
        rhs = design.T @ (total * y)
        if np.linalg.cond(normal) > 1.0e12:
            return np.nan, np.nan, np.nan, int(valid.sum()), np.nan
        updated = np.linalg.solve(normal, rhs)
        residual = y - design @ updated
        center = np.median(residual)
        scale = max(float(1.4826 * np.median(np.abs(residual - center))), 0.05)
        u = residual / (bisquare_c * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if np.linalg.norm(updated - estimate) < 1.0e-6:
            estimate = updated
            break
        estimate = updated
    residual = y - design @ estimate
    effective = int(np.sum(robust > 0.05))
    rms = float(np.sqrt(np.average(residual**2, weights=base)))
    total = base * robust
    normal = design.T @ (total[:, None] * design)
    try:
        covariance = np.linalg.inv(normal) * max(float(np.average(residual**2, weights=total)), 1.0e-8)
        height_sigma = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        height_sigma = np.nan
    return float(estimate[1]), float(estimate[0]), rms, effective, height_sigma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-observations", type=Path, required=True)
    parser.add_argument("--unwrapped-root", type=Path, required=True)
    parser.add_argument("--pair-sensitivity", type=Path, required=True)
    parser.add_argument("--quality-metrics", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--pixel-output", type=Path, required=True)
    parser.add_argument("--building-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--zero-mode", choices=("fixed-far-ground", "far-ground", "original-first-pixel"),
        default="fixed-far-ground"
    )
    parser.add_argument("--radar-frequency-hz", type=float, default=5.4000001e9)
    parser.add_argument("--minimum-pairs", type=int, default=12)
    parser.add_argument("--minimum-pair-coherence", type=float, default=0.30)
    parser.add_argument("--bisquare-c", type=float, default=4.685)
    args = parser.parse_args()

    baseline = [line.split() for line in args.baseline_file.read_text().splitlines() if line.strip()]
    dt = {f"{fields[1]}_{fields[2]}": float(fields[4]) for fields in baseline}
    observation_paths = [
        path for path in sorted(args.pair_observations.glob("*.npz")) if path.stem in dt
    ]
    names = [path.stem for path in observation_paths]
    if set(names) != set(dt):
        raise RuntimeError("baseline network and available pair observations do not match")
    quality = np.load(args.quality_metrics)
    selected = quality["paper_quality_selected"].astype(bool)
    row = quality["row"][selected]
    col = quality["col"][selected]
    label = quality["label"][selected]
    uid = quality["building_uid"][selected]
    da = quality["amplitude_dispersion"][selected]
    mean_cc = quality["mean_coherence"][selected]
    phase_key = {
        "fixed-far-ground": "unwrapped_phase_fixed_far_ground_zero_rad",
        "far-ground": "unwrapped_phase_far_ground_zero_rad",
        "original-first-pixel": "unwrapped_phase_original_first_pixel_zero_rad",
    }[args.zero_mode]
    phase = np.stack(
        [np.load(args.unwrapped_root / f"{name}.npz")[phase_key][selected] for name in names]
    ).astype(np.float64)
    coherence = np.stack([np.load(path)["coherence"][selected] for path in observation_paths]).astype(np.float64)
    sensitivity = np.stack(
        [np.load(args.pair_sensitivity / f"{name}.npz")["phase_sensitivity_rad_per_m"][selected] for name in names]
    ).astype(np.float64)

    wavelength = C / args.radar_frequency_hz
    velocity_coefficient = (4.0 * np.pi / wavelength) * np.array([dt[name] for name in names]) / 365.25
    valid = np.isfinite(phase + coherence + sensitivity) & (coherence >= args.minimum_pair_coherence)
    weight = np.where(valid, coherence**2, 0.0)

    height = np.full(len(row), np.nan, dtype=np.float64)
    velocity = np.full(len(row), np.nan, dtype=np.float64)
    residual_rms = np.full(len(row), np.nan, dtype=np.float64)
    effective_pairs = np.zeros(len(row), dtype=np.int16)
    sigma_height = np.full(len(row), np.nan, dtype=np.float64)
    for pixel in range(len(row)):
        result = solve_pixel(
            phase[:, pixel], sensitivity[:, pixel], velocity_coefficient,
            weight[:, pixel], args.minimum_pairs, args.bisquare_c,
        )
        height[pixel], velocity[pixel], residual_rms[pixel], effective_pairs[pixel], sigma_height[pixel] = result

    args.pixel_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pixel_output, row=row, col=col, label=label, building_uid=uid,
        amplitude_dispersion=da, mean_coherence=mean_cc,
        dem_error_or_height_above_anchor_m=height.astype(np.float32),
        velocity_m_per_year=velocity.astype(np.float32),
        phase_residual_rms_rad=residual_rms.astype(np.float32),
        effective_pair_count=effective_pairs, height_sigma_m=sigma_height.astype(np.float32),
    )

    building_rows: list[dict[str, float | int]] = []
    for building in np.unique(uid):
        member = (uid == building) & np.isfinite(height) & (effective_pairs >= args.minimum_pairs)
        values = height[member]
        if not len(values):
            continue
        q1, q3 = np.quantile(values, (0.25, 0.75))
        iqr = q3 - q1
        retained = member & (height >= q1 - 1.5 * iqr) & (height <= q3 + 1.5 * iqr)
        retained_values = height[retained]
        building_rows.append({
            "building_uid": int(building),
            "island_label": int(np.median(label[member])),
            "selected_pixel_count": int(member.sum()),
            "iqr_retained_pixel_count": int(retained.sum()),
            "insar_height_above_anchor_m": float(np.median(retained_values)),
            "insar_height_iqr_m": float(np.quantile(retained_values, 0.75) - np.quantile(retained_values, 0.25)),
            "median_velocity_m_per_year": float(np.nanmedian(velocity[retained])),
            "median_phase_residual_rms_rad": float(np.nanmedian(residual_rms[retained])),
            "median_height_sigma_m": float(np.nanmedian(sigma_height[retained])),
        })
    args.building_output.parent.mkdir(parents=True, exist_ok=True)
    with args.building_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(building_rows[0]) if building_rows else ["building_uid"])
        writer.writeheader()
        writer.writerows(building_rows)

    result = {
        "method": "GAMMA Delaunay MCF independent unwrapping + two-parameter weighted SVD-equivalent WLS",
        "zero_mode": args.zero_mode,
        "paper_selected_pixel_count": int(len(row)),
        "interferogram_pair_count": len(names),
        "solved_pixel_count": int(np.isfinite(height).sum()),
        "solved_building_count": len(building_rows),
        "height_above_anchor_m_quantiles": qdict(height),
        "velocity_m_per_year_quantiles": qdict(velocity),
        "phase_residual_rms_rad_quantiles": qdict(residual_rms),
        "height_sigma_m_quantiles": qdict(sigma_height),
        "building_height_above_anchor_m_quantiles": qdict(
            np.array([item["insar_height_above_anchor_m"] for item in building_rows])
        ),
        "minimum_pairs": args.minimum_pairs,
        "bisquare_tuning_constant": args.bisquare_c,
        "height_prior_used_in_inversion": False,
        "floor_use": "search-mask geometry only; never used as observation, fill, bound, or correction",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
