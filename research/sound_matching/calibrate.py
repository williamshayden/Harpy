"""Create deterministic memberships or run the bounded calibration-only pilot.

Run with the source checkout installed (or PYTHONPATH=src). No final-test actor
evaluation is available through this script. Outputs are create-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harpy.tasks.sound_match.calibration import run_pilot, write_calibration
from harpy.tasks.sound_match.datasets import build_manifests, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("manifests", "pilot", "fixtures"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-directory", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    if args.operation == "manifests":
        manifests = build_manifests(
            args.output, counts={"calibration": 96} if args.calibration_only else None
        )
        result = {
            manifest.split: {"count": len(manifest.entries), "sha256": manifest.manifest_sha256}
            for manifest in manifests
        }
    elif args.operation == "fixtures":
        result = write_calibration(args.output)
    else:
        result = run_pilot(
            load_manifest("calibration", directory=args.manifest_directory), args.output
        )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
