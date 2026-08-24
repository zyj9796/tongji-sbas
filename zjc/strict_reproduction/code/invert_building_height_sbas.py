#!/usr/bin/env python3
"""Invert building height and linear LOS velocity from wrapped SBAS phases.

No Floor or other building-height prior is read by the inversion.  A broad,
fixed circular grid search resolves the integer ambiguity, followed by
coherence-weighted iterative WLS with Tukey Bisquare residual weights (4.685).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SPEED_OF_LIGHT = 299_792_458.0


def wrap(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def qdict(value: np.ndarray) -> dict[str, float | None]:
    finite = value[np.isfinite(value)]
    if not len(finite):
        return {str(q): None for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {str(q): float(np.quantile(finite, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def circular_grid_search(
    observation: np.ndarray,
    coefficient: np.ndarray,
    weight: np.ndarray,
    candidates: np.ndarray,
    fixed_phase: np.ndarray | None = None,
    chunk_size: int = 128,
) -> np.ndarray:
    """Maximize weighted circular agreement independently for every pixel."""
    count = observation.shape[1]
    answer = np.full(count, np.nan, dtype=np.float64)
    if fixed_phase is None:
        fixed_phase = np.zeros_like(observation)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        obs = observation[:, start:stop, None]
        coef = coefficient[:, start:stop, None]
        fixed = fixed_phase[:, start:stop, None]
        current_weight = weight[:, start:stop, None]
        score = np.sum(current_weight * np.cos(obs - fixed - coef * candidates[None, None, :]), axis=0)
        answer[start:stop] = candidates[np.argmax(score, axis=1)]
    return answer


def robust_integer_wls(
    phase: np.ndarray,
    height_coefficient: np.ndarray,
    velocity_coefficient: np.ndarray,
    base_weight: np.ndarray,
    initial_height: float,
    initial_velocity: float,
    bisquare_c: float,
) -> tuple[float, float, float, int, float]:
    valid = np.isfinite(phase + height_coefficient + velocity_coefficient + base_weight) & (base_weight > 0)
    if valid.sum() < 12:
        return np.nan, np.nan, np.nan, int(valid.sum()), np.nan
    y = phase[valid]
    design = np.column_stack((height_coefficient[valid], velocity_coefficient[valid]))
    base = base_weight[valid]
    estimate = np.array([initial_height, initial_velocity], dtype=np.float64)
    robust = np.ones_like(y)
    for _ in range(12):
        predicted = design @ estimate
        unwrapped = y + 2.0 * np.pi * np.rint((predicted - y) / (2.0 * np.pi))
        total = base * robust
        normal = design.T @ (total[:, None] * design)
        rhs = design.T @ (total * unwrapped)
        if np.linalg.cond(normal) > 1.0e12:
            return np.nan, np.nan, np.nan, int(valid.sum()), np.nan
        updated = np.linalg.solve(normal, rhs)
        residual = unwrapped - design @ updated
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        scale = max(float(scale), 0.05)
        u = residual / (bisquare_c * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if np.linalg.norm(updated - estimate) < 1.0e-5:
            estimate = updated
            break
        estimate = updated
    final_residual = wrap(y - design @ estimate)
    effective = int(np.sum(robust > 0.05))
    rms = float(np.sqrt(np.average(final_residual**2, weights=base)))
    total = base * robust
    normal = design.T @ (total[:, None] * design)
    try:
        covariance = np.linalg.inv(normal) * max(float(np.average(final_residual**2, weights=total)), 1.0e-8)
        height_sigma = float(np.sqrt(max(covariance[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        height_sigma = np.nan
    return float(estimate[0]), float(estimate[1]), rms, effective, height_sigma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-observations", type=Path, required=True)
    parser.add_argument("--pair-sensitivity", type=Path, required=True)
    parser.add_argument("--referenced-phases", type=Path)
    parser.add_argument(
        "--phase-source", choices=("paper-differential", "ground-referenced"),
        default="paper-differential",
    )
    parser.add_argument("--quality-metrics", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--pixel-output", type=Path, required=True)
    parser.add_argument("--building-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--radar-frequency-hz", type=float, default=5.4000001e9)
    parser.add_argument("--height-min-m", type=float, default=-100.0)
    parser.add_argument("--height-max-m", type=float, default=400.0)
    parser.add_argument("--height-step-m", type=float, default=0.5)
    parser.add_argument("--velocity-min-m-per-year", type=float, default=-0.20)
    parser.add_argument("--velocity-max-m-per-year", type=float, default=0.20)
    parser.add_argument("--velocity-step-m-per-year", type=float, default=0.002)
    parser.add_argument("--minimum-pairs", type=int, default=12)
    parser.add_argument("--minimum-pair-coherence", type=float, default=0.30)
    parser.add_argument("--minimum-reference-resultant", type=float, default=0.20)
    parser.add_argument("--bisquare-c", type=float, default=4.685)
    args = parser.parse_args()

    observation_paths = sorted(args.pair_observations.glob("*.npz"))
    names = [path.stem for path in observation_paths]
    quality = np.load(args.quality_metrics)
    selected = quality["paper_quality_selected"].astype(bool)
    rows = quality["row"][selected]
    cols = quality["col"][selected]
    labels = quality["label"][selected]
    building_uid = quality["building_uid"][selected]
    amplitude_dispersion = quality["amplitude_dispersion"][selected]
    mean_coherence = quality["mean_coherence"][selected]

    if args.phase_source == "ground-referenced":
        if args.referenced_phases is None:
            raise ValueError("--referenced-phases is required for ground-referenced phase source")
        phase = np.stack(
            [np.load(args.referenced_phases / f"{name}.npz")["ground_referenced_filtered_wrapped_phase_rad"][selected] for name in names]
        ).astype(np.float64)
        reference_resultant = np.stack(
            [np.load(args.referenced_phases / f"{name}.npz")["local_ground_reference_resultant"][selected] for name in names]
        ).astype(np.float64)
    else:
        phase = np.stack(
            [np.load(path)["filtered_wrapped_phase_rad"][selected] for path in observation_paths]
        ).astype(np.float64)
        reference_resultant = np.ones_like(phase)
    coherence = np.stack([np.load(path)["coherence"][selected] for path in observation_paths]).astype(np.float64)
    sensitivity = np.stack(
        [np.load(args.pair_sensitivity / f"{name}.npz")["phase_sensitivity_rad_per_m"][selected] for name in names]
    ).astype(np.float64)

    baseline_rows = [line.split() for line in args.baseline_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_by_name = {f"{fields[1]}_{fields[2]}": float(fields[4]) for fields in baseline_rows}
    temporal_days = np.array([baseline_by_name[name] for name in names], dtype=np.float64)
    wavelength = SPEED_OF_LIGHT / args.radar_frequency_hz
    velocity_coefficient_1d = (4.0 * np.pi / wavelength) * (temporal_days / 365.25)
    velocity_coefficient = np.broadcast_to(velocity_coefficient_1d[:, None], phase.shape)

    valid = (
        np.isfinite(phase + coherence + sensitivity + reference_resultant)
        & (coherence >= args.minimum_pair_coherence)
        & (reference_resultant >= args.minimum_reference_resultant)
    )
    weight = np.where(valid, coherence**2 * reference_resultant**2, 0.0)
    enough = np.sum(valid, axis=0) >= args.minimum_pairs

    heights = np.arange(args.height_min_m, args.height_max_m + 0.5 * args.height_step_m, args.height_step_m)
    velocities = np.arange(
        args.velocity_min_m_per_year,
        args.velocity_max_m_per_year + 0.5 * args.velocity_step_m_per_year,
        args.velocity_step_m_per_year,
    )
    initial_height = circular_grid_search(phase, sensitivity, weight, heights)
    fixed_height_phase = sensitivity * initial_height[None, :]
    initial_velocity = circular_grid_search(
        phase, velocity_coefficient, weight, velocities, fixed_phase=fixed_height_phase
    )
    fixed_velocity_phase = velocity_coefficient * initial_velocity[None, :]
    initial_height = circular_grid_search(
        phase, sensitivity, weight, heights, fixed_phase=fixed_velocity_phase
    )

    height = np.full(len(rows), np.nan, dtype=np.float64)
    velocity = np.full(len(rows), np.nan, dtype=np.float64)
    phase_rms = np.full(len(rows), np.nan, dtype=np.float64)
    effective_pairs = np.zeros(len(rows), dtype=np.int16)
    height_sigma = np.full(len(rows), np.nan, dtype=np.float64)
    for pixel in np.flatnonzero(enough):
        result = robust_integer_wls(
            phase[:, pixel], sensitivity[:, pixel], velocity_coefficient[:, pixel], weight[:, pixel],
            initial_height[pixel], initial_velocity[pixel], args.bisquare_c,
        )
        height[pixel], velocity[pixel], phase_rms[pixel], effective_pairs[pixel], height_sigma[pixel] = result

    args.pixel_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pixel_output,
        row=rows, col=cols, label=labels, building_uid=building_uid,
        amplitude_dispersion=amplitude_dispersion, mean_coherence=mean_coherence,
        initial_height_m=initial_height.astype(np.float32),
        initial_velocity_m_per_year=initial_velocity.astype(np.float32),
        height_m=height.astype(np.float32), velocity_m_per_year=velocity.astype(np.float32),
        wrapped_phase_rms_rad=phase_rms.astype(np.float32), effective_pair_count=effective_pairs,
        height_sigma_m=height_sigma.astype(np.float32),
    )

    # Paper-prescribed building-level 1.5-IQR cleaning followed by the median.
    building_rows: list[dict[str, float | int]] = []
    for uid in np.unique(building_uid):
        member = (building_uid == uid) & np.isfinite(height) & (effective_pairs >= args.minimum_pairs)
        values = height[member]
        if not len(values):
            continue
        q1, q3 = np.quantile(values, (0.25, 0.75))
        iqr = q3 - q1
        retained = member & (height >= q1 - 1.5 * iqr) & (height <= q3 + 1.5 * iqr)
        retained_values = height[retained]
        building_rows.append(
            {
                "building_uid": int(uid),
                "island_label": int(np.median(labels[member])),
                "selected_pixel_count": int(member.sum()),
                "iqr_retained_pixel_count": int(retained.sum()),
                "insar_height_m": float(np.median(retained_values)),
                "insar_height_iqr_m": float(np.subtract(*np.quantile(retained_values, (0.75, 0.25)))),
                "median_velocity_m_per_year": float(np.nanmedian(velocity[retained])),
                "median_phase_rms_rad": float(np.nanmedian(phase_rms[retained])),
                "median_height_sigma_m": float(np.nanmedian(height_sigma[retained])),
            }
        )
    args.building_output.parent.mkdir(parents=True, exist_ok=True)
    with args.building_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(building_rows[0]) if building_rows else ["building_uid"])
        writer.writeheader()
        writer.writerows(building_rows)

    finite_height = height[np.isfinite(height)]
    summary = {
        "method": "wrapped circular initialization plus coherence-weighted Bisquare WLS SBAS",
        "phase_source": args.phase_source,
        "paper_quality_selected_pixel_count": int(len(rows)),
        "minimum_valid_pair_count": args.minimum_pairs,
        "pixels_with_minimum_pairs": int(enough.sum()),
        "solved_pixel_count": int(np.isfinite(height).sum()),
        "solved_building_count": len(building_rows),
        "height_search_m": [args.height_min_m, args.height_max_m, args.height_step_m],
        "velocity_search_m_per_year": [
            args.velocity_min_m_per_year, args.velocity_max_m_per_year, args.velocity_step_m_per_year
        ],
        "height_m_quantiles": qdict(finite_height),
        "velocity_m_per_year_quantiles": qdict(velocity),
        "wrapped_phase_rms_rad_quantiles": qdict(phase_rms),
        "height_sigma_m_quantiles": qdict(height_sigma),
        "building_height_m_quantiles": qdict(np.array([row["insar_height_m"] for row in building_rows])),
        "bisquare_tuning_constant": args.bisquare_c,
        "height_prior_used": False,
        "floor_attribute_read_by_inversion": False,
        "ground_reference_use": (
            "not used (paper mainline)" if args.phase_source == "paper-differential"
            else "phase zero only; not part of building aggregation"
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
