#!/usr/bin/env python3
"""Utility script to reshape a Roboflow export into the EcoGrow layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecogrow.data.plant_data import roboflow_format  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formatta un export Roboflow nella struttura dataset di EcoGrow."
    )
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "roboflow_data"),
        help="Percorso della directory Roboflow da cui leggere i dati (default: ./roboflow_data).",
    )
    parser.add_argument(
        "--dest",
        default=str(PROJECT_ROOT / "datasets"),
        help="Percorso della directory di destinazione (default: ./datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive eventuali file esistenti nella destinazione.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    roboflow_format(
        init_root=args.source,
        final_root=args.dest,
        overwrite=args.overwrite,
    )

    print(f"Dataset formattato in '{args.dest}'.")


if __name__ == "__main__":
    main()

