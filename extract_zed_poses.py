#!/usr/bin/env python3
"""Extract metric left-camera poses and tracking states from a ZED SVO."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("svo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo.resolve()))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open SVO: {status}")

    tracking = sl.PositionalTrackingParameters()
    tracking.set_as_static = False
    tracking.set_floor_as_origin = False
    status = zed.enable_positional_tracking(tracking)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise SystemExit(f"Could not enable positional tracking: {status}")

    runtime = sl.RuntimeParameters()
    pose = sl.Pose()
    frames: list[dict] = []
    states: Counter[str] = Counter()
    source_index = 0

    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        state_name = str(state)
        states[state_name] += 1
        if source_index % args.stride == 0:
            matrix = pose.pose_data(sl.Transform()).m
            frames.append(
                {
                    "source_index": source_index,
                    "file_path": f"left/{source_index:06d}.png",
                    "tracking_state": state_name,
                    "pose_confidence": int(pose.pose_confidence),
                    "timestamp_ns": int(pose.timestamp.get_nanoseconds()),
                    "camera_to_world": [
                        [float(matrix[row][col]) for col in range(4)]
                        for row in range(4)
                    ],
                }
            )
            if args.limit is not None and len(frames) >= args.limit:
                break
        source_index += 1

    zed.disable_positional_tracking()
    zed.close()

    result = {
        "source": str(args.svo.resolve()),
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "units": "METER",
        "stride": args.stride,
        "tracking_state_counts_all_frames": dict(states),
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    sampled = Counter(frame["tracking_state"] for frame in frames)
    print(f"Wrote {len(frames)} sampled poses to {args.output}")
    print(f"Sampled tracking states: {dict(sampled)}")
    if frames:
        confidences = [frame["pose_confidence"] for frame in frames]
        print(
            "Pose confidence: "
            f"min={min(confidences)} mean={sum(confidences) / len(confidences):.1f} "
            f"max={max(confidences)}"
        )


if __name__ == "__main__":
    main()
