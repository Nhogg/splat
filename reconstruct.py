#!/usr/bin/env python3
"""Build a calibrated COLMAP sparse reconstruction from exported ZED frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera", choices=("left", "right"), default="left")
    parser.add_argument("--overlap", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = (args.capture / args.camera).resolve()
    calibration_path = args.capture / "calibration.json"
    database_path = args.output / "database.db"
    sparse_path = args.output / "sparse"

    if database_path.exists() or sparse_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing reconstruction in {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir()

    calibration = json.loads(calibration_path.read_text())
    camera = calibration[args.camera]
    camera_params = ",".join(
        str(camera[key]) for key in ("fx", "fy", "cx", "cy")
    )

    reader = pycolmap.ImageReaderOptions(
        camera_model="PINHOLE",
        camera_params=camera_params,
    )
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192

    print("Extracting SIFT features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
    )

    print("Matching neighboring frames...")
    pairing = pycolmap.SequentialPairingOptions(
        overlap=args.overlap,
        quadratic_overlap=True,
    )
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pairing,
    )

    print("Running incremental mapping...")
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_path,
        options=options,
    )
    if not reconstructions:
        raise SystemExit("COLMAP did not produce a model")

    model_id, reconstruction = max(
        reconstructions.items(), key=lambda item: item[1].num_reg_images()
    )
    summary = {
        "model_id": int(model_id),
        "registered_images": int(reconstruction.num_reg_images()),
        "total_images": len(list(image_path.glob("*.png"))),
        "points3D": int(reconstruction.num_points3D()),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
