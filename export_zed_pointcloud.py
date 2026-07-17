#!/usr/bin/env python3
"""Build a voxelized metric RGB point cloud from sampled ZED depth maps."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("svo", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--pixel-step", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=6.0)
    return parser.parse_args()


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    vertex = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["r"], vertex["g"], vertex["b"] = rgb.T
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(vertex.tobytes())


def main() -> None:
    args = parse_args()
    calibration = json.loads((args.capture / "calibration.json").read_text())
    poses = json.loads((args.capture / "zed-poses.json").read_text())["frames"]
    poses = [p for p in poses if p["pose_confidence"] >= 25][:: args.frame_step]

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo.resolve()))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open SVO: {status}")

    depth_mat, image_mat = sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    left = calibration["left"]
    fx, fy, cx, cy = left["fx"], left["fy"], left["cx"], left["cy"]
    chunks_xyz, chunks_rgb = [], []

    for number, frame in enumerate(poses, 1):
        zed.set_svo_position(frame["source_index"])
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        depth = np.asarray(depth_mat.get_data())
        bgra = np.asarray(image_mat.get_data())
        rows = np.arange(0, depth.shape[0], args.pixel_step)
        cols = np.arange(0, depth.shape[1], args.pixel_step)
        uu, vv = np.meshgrid(cols, rows)
        d = depth[vv, uu]
        valid = np.isfinite(d) & (d >= args.min_depth) & (d <= args.max_depth)
        d, uu, vv = d[valid], uu[valid], vv[valid]
        # RIGHT_HANDED_Y_UP matches Nerfstudio/OpenGL: +X right, +Y up,
        # and the camera views along -Z.
        local = np.column_stack(
            ((uu - cx) * d / fx, -(vv - cy) * d / fy, -d, np.ones_like(d))
        )
        c2w = np.asarray(frame["camera_to_world"], dtype=np.float64)
        chunks_xyz.append((local @ c2w.T)[:, :3].astype(np.float32))
        chunks_rgb.append(bgra[vv, uu, :3][:, ::-1].copy())
        if number % 25 == 0:
            print(f"Processed {number}/{len(poses)} depth frames", flush=True)
    zed.close()

    xyz = np.concatenate(chunks_xyz)
    rgb = np.concatenate(chunks_rgb)
    voxel = np.floor(xyz / args.voxel_size).astype(np.int32)
    _, keep = np.unique(voxel, axis=0, return_index=True)
    xyz, rgb = xyz[keep], rgb[keep]
    write_binary_ply(args.output, xyz, rgb)
    print(f"Wrote {len(xyz)} points to {args.output}")


if __name__ == "__main__":
    main()
