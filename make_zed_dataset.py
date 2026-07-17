#!/usr/bin/env python3
"""Convert extracted ZED metric poses into a Nerfstudio transforms dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-confidence", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads((args.capture / "calibration.json").read_text())
    pose_data = json.loads((args.capture / "zed-poses.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    image_link = args.output / "images"
    if not image_link.exists():
        image_link.symlink_to(
            os.path.relpath((args.capture / "left").resolve(), args.output.resolve()),
            target_is_directory=True,
        )

    frames = []
    rejected = []
    for frame in pose_data["frames"]:
        source = args.capture / frame["file_path"]
        if (
            frame["tracking_state"] != "OK"
            or frame["pose_confidence"] < args.min_confidence
            or not source.exists()
        ):
            rejected.append(frame["source_index"])
            continue
        frames.append(
            {
                "file_path": f"images/{source.name}",
                "transform_matrix": frame["camera_to_world"],
                "pose_confidence": frame["pose_confidence"],
                "source_index": frame["source_index"],
            }
        )

    left = calibration["left"]
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": left["fx"],
        "fl_y": left["fy"],
        "cx": left["cx"],
        "cy": left["cy"],
        "w": calibration["width"],
        "h": calibration["height"],
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    (args.output / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n"
    )
    print(f"Wrote {len(frames)} frames to {args.output / 'transforms.json'}")
    print(f"Rejected {len(rejected)} frames: {rejected}")


if __name__ == "__main__":
    main()
