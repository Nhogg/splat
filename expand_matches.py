#!/usr/bin/env python3
"""Add a dense temporal window of matches to an existing COLMAP database."""

from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--overlap", type=int, default=12)
    args = parser.parse_args()

    pairing = pycolmap.SequentialPairingOptions(
        overlap=args.overlap,
        quadratic_overlap=False,
    )
    pycolmap.match_sequential(
        database_path=args.database,
        pairing_options=pairing,
    )


if __name__ == "__main__":
    main()
