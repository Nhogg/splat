#!/usr/bin/env python3
"""Create a temporally uniform registered-camera subset of a COLMAP model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")

    reconstruction = pycolmap.Reconstruction(args.model)
    registered = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: image.name,
    )
    keep_ids = {image.image_id for image in registered[:: args.stride]}
    keep_ids.add(registered[-1].image_id)

    remove_frame_ids = {
        image.frame_id
        for image in registered
        if image.image_id not in keep_ids
    }
    for frame_id in remove_frame_ids:
        reconstruction.deregister_frame(frame_id)

    args.output.mkdir(parents=True)
    reconstruction.write(args.output)
    print(
        f"Wrote {reconstruction.num_reg_images()} registered images and "
        f"{reconstruction.num_points3D()} points to {args.output}"
    )


if __name__ == "__main__":
    main()
