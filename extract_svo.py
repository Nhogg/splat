#!/usr/bin/env python3
"""Export rectified stereo frames and calibration from a ZED SVO recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pyzed.sl as sl


def camera_dict(camera: sl.CameraParameters) -> dict[str, float | list[float]]:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "distortion": [float(value) for value in camera.disto],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("svo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo.resolve()))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open {args.svo}: {status}")

    info = zed.get_camera_information()
    config = info.camera_configuration
    calibration = config.calibration_parameters
    source_fps = float(config.fps)
    stride = max(1, round(source_fps / args.sample_fps))

    left_dir = args.output / "left"
    right_dir = args.output / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(args.svo.resolve()),
        "camera_model": str(info.camera_model),
        "serial_number": int(info.serial_number),
        "width": int(config.resolution.width),
        "height": int(config.resolution.height),
        "source_fps": source_fps,
        "sample_fps": source_fps / stride,
        "stride": stride,
        "frame_count": int(zed.get_svo_number_of_frames()),
        "left": camera_dict(calibration.left_cam),
        "right": camera_dict(calibration.right_cam),
        # PyZED reports this calibration translation in millimeters.
        "right_from_left_translation_mm": [
            float(value)
            for value in calibration.stereo_transform.get_translation().get()
        ],
    }
    (args.output / "calibration.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    left = sl.Mat()
    right = sl.Mat()
    runtime = sl.RuntimeParameters()
    exported = 0
    source_index = 0

    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if source_index % stride == 0:
            zed.retrieve_image(left, sl.VIEW.LEFT)
            zed.retrieve_image(right, sl.VIEW.RIGHT)
            name = f"{source_index:06d}.png"
            left_bgr = cv2.cvtColor(left.get_data(), cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(right.get_data(), cv2.COLOR_BGRA2BGR)
            if not cv2.imwrite(str(left_dir / name), left_bgr):
                raise RuntimeError(f"Failed to write {left_dir / name}")
            if not cv2.imwrite(str(right_dir / name), right_bgr):
                raise RuntimeError(f"Failed to write {right_dir / name}")
            exported += 1
            if args.limit is not None and exported >= args.limit:
                break
        source_index += 1

    zed.close()
    print(f"Exported {exported} stereo pairs to {args.output}")


if __name__ == "__main__":
    main()
