#!/usr/bin/env python3
"""Retry COLMAP mapping with thresholds suitable for a smooth video trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.min_model_size = 4
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.init_min_tri_angle = 2.0
    options.mapper.init_max_forward_motion = 0.99
    options.mapper.init_min_num_inliers = 50
    options.mapper.abs_pose_min_num_inliers = 20
    options.mapper.filter_min_tri_angle = 0.25
    options.triangulation.min_angle = 0.25
    options.triangulation.ignore_two_view_tracks = False

    models = pycolmap.incremental_mapping(
        database_path=args.database,
        image_path=args.images,
        output_path=args.output,
        options=options,
    )
    summaries = []
    for model_id, reconstruction in models.items():
        summaries.append(
            {
                "model_id": int(model_id),
                "registered_images": int(reconstruction.num_reg_images()),
                "points3D": int(reconstruction.num_points3D()),
                "mean_reprojection_error": float(
                    reconstruction.compute_mean_reprojection_error()
                ),
            }
        )
    summary_path = args.output.parent / f"{args.output.name}-summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
