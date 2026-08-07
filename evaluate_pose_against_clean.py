"""Evaluate unposed LongSplat camera trajectories against a clean reference run.

This evaluator is for pose comparison between weather conditions. The reference
may be a clean LongSplat trajectory or a standard 3DGS ``cameras.json`` export
of a clean COLMAP reconstruction. For each candidate run, camera centres are
matched by image stem and aligned to the reference centres by a 7-DoF similarity
transform (Sim(3)).

The exported ``cameras_all_train.json`` stores ``R`` in the codebase's
camera-to-world convention and ``T`` as world-to-camera translation. Therefore
the camera centre is ``-R @ T`` and the camera-to-world rotation is ``R``.

Example
-------
python evaluate_pose_against_clean.py \
  --reference /path/to/clean/cameras_all_train.json \
  --run og=/path/to/og/cameras_all_train.json \
  --run v3=/path/to/v3/cameras_all_train.json \
  --output_csv /tmp/grass_pose.csv \
  --per_frame_csv /tmp/grass_pose_per_frame.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def project_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation to its closest proper SO(3) matrix."""
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def load_poses(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load image-stem keyed c2w centres and rotations from supported JSON files.

    Supported formats are:
    - LongSplat/HGSplat ``cameras_all_train.json``: ``image_name``, ``R``, ``T``.
    - Standard 3DGS ``cameras.json``: ``img_name``, ``position``, ``rotation``.
      The latter is the camera export of the COLMAP poses loaded by standard 3DGS.
    """
    with path.open() as handle:
        raw: list[dict[str, Any]] = json.load(handle)

    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera in raw:
        if "image_name" in camera:
            name = Path(str(camera["image_name"])).stem
            rotation_c2w_raw = np.asarray(camera["R"], dtype=np.float64)
            translation_w2c = np.asarray(camera["T"], dtype=np.float64)
            if rotation_c2w_raw.shape != (3, 3) or translation_w2c.shape != (3,):
                raise ValueError(f"Invalid LongSplat pose for {name} in {path}")
            if not np.isfinite(rotation_c2w_raw).all() or not np.isfinite(translation_w2c).all():
                continue
            rotation_c2w = project_to_so3(rotation_c2w_raw)
            center = -rotation_c2w @ translation_w2c
        elif "img_name" in camera:
            name = Path(str(camera["img_name"])).stem
            center = np.asarray(camera["position"], dtype=np.float64)
            rotation_c2w = np.asarray(camera["rotation"], dtype=np.float64)
            if center.shape != (3,) or rotation_c2w.shape != (3, 3):
                raise ValueError(f"Invalid standard-3DGS pose for {name} in {path}")
            if not np.isfinite(center).all() or not np.isfinite(rotation_c2w).all():
                continue
            rotation_c2w = project_to_so3(rotation_c2w)
        else:
            raise ValueError(f"Unsupported camera JSON format in {path}")
        poses[name] = (center, rotation_c2w)
    return poses


def sim3_align(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source centres to target centres."""
    if len(source) < 3:
        raise ValueError("Sim(3) alignment needs at least three matched frames.")

    src_mean, tgt_mean = source.mean(axis=0), target.mean(axis=0)
    src_zero, tgt_zero = source - src_mean, target - tgt_mean
    covariance = (tgt_zero.T @ src_zero) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src_zero**2, axis=1))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate predicted camera-centre trajectory.")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = tgt_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_run(
    label: str,
    candidate_path: Path,
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = load_poses(candidate_path)
    names = sorted(set(reference) & set(candidate))
    if len(names) < 3:
        raise ValueError(f"{label}: only {len(names)} matched frames")

    reference_centres = np.stack([reference[name][0] for name in names])
    candidate_centres = np.stack([candidate[name][0] for name in names])
    scale, global_rotation, translation = sim3_align(candidate_centres, reference_centres)
    aligned_centres = (scale * (global_rotation @ candidate_centres.T)).T + translation

    ate_errors = np.linalg.norm(aligned_centres - reference_centres, axis=1)
    absolute_rotation_errors = np.asarray([
        rotation_angle_degrees(reference[name][1].T @ global_rotation @ candidate[name][1])
        for name in names
    ])

    rpe_translation_errors: list[float] = []
    rpe_rotation_errors: list[float] = []
    for index in range(len(names) - 1):
        ref_delta = reference_centres[index + 1] - reference_centres[index]
        pred_delta = aligned_centres[index + 1] - aligned_centres[index]
        rpe_translation_errors.append(float(np.linalg.norm(pred_delta - ref_delta)))

        ref_relative_rotation = reference[names[index]][1].T @ reference[names[index + 1]][1]
        pred_relative_rotation = candidate[names[index]][1].T @ candidate[names[index + 1]][1]
        rpe_rotation_errors.append(rotation_angle_degrees(ref_relative_rotation.T @ pred_relative_rotation))

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        rows.append({
            "run": label,
            "image_name": name,
            "ate_error_sim3_aligned": ate_errors[index],
            "absolute_rotation_error_deg_sim3_aligned": absolute_rotation_errors[index],
            "rpe_translation_to_next_sim3_aligned": (
                rpe_translation_errors[index] if index < len(rpe_translation_errors) else ""
            ),
            "rpe_rotation_to_next_deg": (
                rpe_rotation_errors[index] if index < len(rpe_rotation_errors) else ""
            ),
        })

    summary = {
        "run": label,
        "candidate_json": str(candidate_path),
        "matched_frames": len(names),
        "reference_frames": len(reference),
        "candidate_frames": len(candidate),
        "sim3_scale": scale,
        "ate_mean_sim3_aligned": float(np.mean(ate_errors)),
        "absolute_rotation_mean_deg_sim3_aligned": float(np.mean(absolute_rotation_errors)),
        "rpe_translation_mean_sim3_aligned": float(np.mean(rpe_translation_errors)),
        "rpe_rotation_mean_deg": float(np.mean(rpe_rotation_errors)),
    }
    return summary, rows


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=/path/to/cameras_all_train.json")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must use LABEL=/path/to/cameras_all_train.json")
    return label, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path,
                        help="Clean LongSplat cameras_all_train.json used as a reference trajectory.")
    parser.add_argument("--run", required=True, action="append", type=parse_run,
                        help="Candidate run as LABEL=/path/to/cameras_all_train.json. Repeat per run.")
    parser.add_argument("--output_csv", required=True, type=Path, help="Per-run summary CSV.")
    parser.add_argument("--per_frame_csv", type=Path, help="Optional frame-level error CSV.")
    args = parser.parse_args()

    reference = load_poses(args.reference)
    if len(reference) < 3:
        raise ValueError("Reference trajectory has fewer than three valid poses.")

    summaries: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    for label, path in args.run:
        summary, rows = evaluate_run(label, path, reference)
        summaries.append(summary)
        per_frame_rows.extend(rows)

    write_csv(args.output_csv, summaries)
    if args.per_frame_csv:
        write_csv(args.per_frame_csv, per_frame_rows)

    print(f"Reference: {args.reference} ({len(reference)} valid frames)")
    for summary in summaries:
        print(
            f"{summary['run']}: n={summary['matched_frames']}  "
            f"ATE-mean={summary['ate_mean_sim3_aligned']:.6f}  "
            f"RPE-t={summary['rpe_translation_mean_sim3_aligned']:.6f}  "
            f"RPE-r={summary['rpe_rotation_mean_deg']:.4f} deg"
        )
    print(f"Wrote summary: {args.output_csv}")
    if args.per_frame_csv:
        print(f"Wrote per-frame errors: {args.per_frame_csv}")


if __name__ == "__main__":
    main()
