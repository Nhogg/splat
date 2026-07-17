#!/usr/bin/env python3
"""Create a native-resolution Nerfstudio dataset from calibrated ZED stereo pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-confidence", type=int, default=25)
    args = parser.parse_args()

    calibration = json.loads((args.capture / "calibration.json").read_text())
    poses = json.loads((args.capture / "zed-poses.json").read_text())["frames"]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("left", "right"):
        link = args.output / name
        if not link.exists():
            link.symlink_to(
                os.path.relpath((args.capture / name).resolve(), args.output.resolve()),
                target_is_directory=True,
            )
    cloud = args.output / "sparse_pc.ply"
    if not cloud.exists():
        cloud.symlink_to(
            os.path.relpath((args.capture.parent.parent / "datasets/capture2-zed/sparse_pc.ply").resolve(), args.output.resolve())
        )

    baseline = np.eye(4)
    baseline[:3, 3] = np.asarray(calibration["right_from_left_translation_mm"]) / 1000.0
    frames = []
    for pose in poses:
        if pose["tracking_state"] != "OK" or pose["pose_confidence"] < args.min_confidence:
            continue
        name = f"{pose['source_index']:06d}.png"
        left_c2w = np.asarray(pose["camera_to_world"], dtype=np.float64)
        right_c2w = left_c2w @ baseline
        common = {"pose_confidence": pose["pose_confidence"], "source_index": pose["source_index"]}
        frames.append({"file_path": f"left/{name}", "transform_matrix": left_c2w.tolist(), **common})
        right = calibration["right"]
        frames.append(
            {
                "file_path": f"right/{name}",
                "transform_matrix": right_c2w.tolist(),
                "fl_x": right["fx"], "fl_y": right["fy"],
                "cx": right["cx"], "cy": right["cy"],
                **common,
            }
        )

    left = calibration["left"]
    result = {
        "camera_model": "OPENCV", "fl_x": left["fx"], "fl_y": left["fy"],
        "cx": left["cx"], "cy": left["cy"], "w": calibration["width"],
        "h": calibration["height"], "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "ply_file_path": "sparse_pc.ply", "frames": frames,
    }
    (args.output / "transforms.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(frames)} calibrated stereo views")


if __name__ == "__main__":
    main()
