#!/usr/bin/env python3
"""Independently unwrap every building search region with GAMMA Delaunay MCF."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--amplitude-metrics", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pair-indices", nargs="*", type=int, default=[])
    parser.add_argument("--reuse-unwrapped-root", type=Path)
    parser.add_argument("--reuse-points", type=Path)
    args = parser.parse_args()

    amplitude = np.load(args.amplitude_metrics)
    row = amplitude["row"].astype(np.int32)
    col = amplitude["col"].astype(np.int32)
    uid = amplitude["building_uid"].astype(np.int32)
    dispersion = amplitude["amplitude_dispersion"].astype(np.float64)
    pair_paths = sorted(args.pair_root.glob("*.npz"))
    if args.pair_indices:
        wanted = set(args.pair_indices)
        pair_paths = [path for index, path in enumerate(pair_paths, start=1) if index in wanted]
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = gamma_environment()
    prior: dict[str, dict[str, object]] = {}
    if args.summary.exists():
        prior = {item["pair_name"]: item for item in json.loads(args.summary.read_text())}

    unique_uid = np.unique(uid)
    reusable: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if args.reuse_unwrapped_root or args.reuse_points:
        if not (args.reuse_unwrapped_root and args.reuse_points):
            raise ValueError("reuse requires both --reuse-unwrapped-root and --reuse-points")
        reuse_points = np.load(args.reuse_points)
        reuse_row = reuse_points["row"].astype(np.int32)
        reuse_col = reuse_points["col"].astype(np.int32)
        reuse_uid = reuse_points["building_uid"].astype(np.int32)
        for building in unique_uid:
            target_index = np.flatnonzero(uid == building)
            source_index = np.flatnonzero(reuse_uid == building)
            if len(target_index) != len(source_index) or not len(target_index):
                continue
            target_flat = row[target_index].astype(np.int64) * 10_000 + col[target_index]
            source_flat = reuse_row[source_index].astype(np.int64) * 10_000 + reuse_col[source_index]
            source_order = np.argsort(source_flat)
            source_sorted = source_flat[source_order]
            location = np.searchsorted(source_sorted, target_flat)
            if np.all(location < len(source_sorted)) and np.array_equal(source_sorted[location], target_flat):
                reusable[int(building)] = (target_index, source_index[source_order][location])
    for pair_sequence, path in enumerate(pair_paths, start=1):
        name = path.stem
        output = args.output_root / f"{name}.npz"
        print(f"[{pair_sequence}/{len(pair_paths)}] {name}", flush=True)
        if output.exists():
            with np.load(output) as existing:
                if existing["unwrapped_phase_far_ground_zero_rad"].shape == row.shape:
                    print("existing_valid", flush=True)
                    continue
        pair = np.load(path)
        if not np.array_equal(pair["row"], row) or not np.array_equal(pair["col"], col):
            raise RuntimeError(f"point ordering mismatch: {path}")
        phase = pair["filtered_wrapped_phase_rad"].astype(np.float64)
        coherence = pair["coherence"].astype(np.float64)
        amplitude_quality = np.clip(1.0 - dispersion / 0.4, 0.0, 1.0)
        point_weight = np.clip(coherence, 0.0, 1.0) * (0.05 + 0.95 * amplitude_quality)
        unwrapped_far = np.full(len(row), np.nan, dtype=np.float32)
        unwrapped_first = np.full(len(row), np.nan, dtype=np.float32)
        reuse_pair = None
        if args.reuse_unwrapped_root:
            reuse_path = args.reuse_unwrapped_root / path.name
            if reuse_path.exists():
                reuse_pair = np.load(reuse_path)
        pair_work = args.work_root / name
        pair_work.mkdir(parents=True, exist_ok=True)
        log = pair_work / "gamma_mcf.log"
        log.write_text("", encoding="utf-8")
        failed = []
        reused_count = 0
        for building_sequence, building in enumerate(unique_uid, start=1):
            member_index = np.flatnonzero(uid == building)
            if reuse_pair is not None and int(building) in reusable:
                target_index, source_index = reusable[int(building)]
                unwrapped_far[target_index] = reuse_pair[
                    "unwrapped_phase_far_ground_zero_rad"
                ][source_index]
                unwrapped_first[target_index] = reuse_pair[
                    "unwrapped_phase_original_first_pixel_zero_rad"
                ][source_index]
                reused_count += 1
                continue
            rr, cc = row[member_index], col[member_index]
            r0, r1 = int(rr.min()), int(rr.max()) + 1
            c0, c1 = int(cc.min()), int(cc.max()) + 1
            height, width = r1 - r0, c1 - c0
            local_r, local_c = rr - r0, cc - c0
            local_interferogram = np.zeros((height, width), dtype=">c8")
            local_weight = np.zeros((height, width), dtype=">f4")
            local_mask = np.zeros((height, width), dtype=np.uint8)
            local_interferogram[local_r, local_c] = np.exp(1j * phase[member_index]).astype(">c8")
            local_weight[local_r, local_c] = point_weight[member_index].astype(">f4")
            local_mask[local_r, local_c] = 255
            # Far-range edge is the physical ground-side anchor of the
            # directional layover strip.  Among its outer 5%, choose the most
            # reliable observed point.
            far_threshold = np.quantile(local_c, 0.95)
            candidates = np.flatnonzero(local_c >= far_threshold)
            root_member = candidates[np.argmax(point_weight[member_index[candidates]])]
            root_c, root_r = int(local_c[root_member]), int(local_r[root_member])

            stem = pair_work / f"building_{int(building)}"
            int_path = stem.with_suffix(".int")
            weight_path = stem.with_suffix(".wgt")
            mask_path = stem.with_suffix(".tif")
            unw_path = stem.with_suffix(".unw")
            local_interferogram.tofile(int_path)
            local_weight.tofile(weight_path)
            Image.fromarray(local_mask, mode="L").save(mask_path, compression="tiff_lzw")
            command = [
                "mcf", str(int_path), str(weight_path), str(mask_path), str(unw_path), str(width),
                "1", "0", "0", str(width), str(height), "1", "1", "-", str(root_c), str(root_r), "1",
            ]
            with log.open("a", encoding="utf-8") as stream:
                stream.write("$ " + " ".join(command) + "\n")
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
                stream.write(f"RETURN_CODE={result.returncode}\n")
            if result.returncode == 0 and unw_path.exists() and unw_path.stat().st_size == width * height * 4:
                local_unwrapped = np.memmap(unw_path, dtype=">f4", mode="r", shape=(height, width))
                values = np.asarray(local_unwrapped[local_r, local_c], dtype=np.float64)
                unwrapped_far[member_index] = values.astype(np.float32)
                unwrapped_first[member_index] = (values - values[0]).astype(np.float32)
                del local_unwrapped
            else:
                failed.append(int(building))
            for temporary in (int_path, weight_path, mask_path, unw_path):
                if temporary.exists():
                    temporary.unlink()
            if building_sequence % 25 == 0:
                print(f"  building {building_sequence}/{len(unique_uid)}", flush=True)

        np.savez_compressed(
            output, row=row, col=col, building_uid=uid,
            unwrapped_phase_far_ground_zero_rad=unwrapped_far,
            unwrapped_phase_original_first_pixel_zero_rad=unwrapped_first,
        )
        item = {
            "pair_name": name,
            "building_count": int(len(unique_uid)),
            "failed_building_count": len(failed),
            "failed_building_uid": failed,
            "reused_unchanged_geometry_building_count": reused_count,
            "recomputed_building_count": int(len(unique_uid) - reused_count),
            "finite_unwrapped_pixel_count": int(np.isfinite(unwrapped_far).sum()),
            "gamma_program": "mcf",
            "triangulation_mode": 1,
            "independent_partition_key": "building_uid",
            "weight": "pair coherence * (0.05 + 0.95*clip(1-DA/0.4,0,1))",
            "primary_zero": "highest-weight point in far-range 5% edge, phase forced to zero",
            "audit_zero": "first point in row-major order forced to zero by constant subtraction",
        }
        prior[name] = item
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps([prior[key] for key in sorted(prior)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
