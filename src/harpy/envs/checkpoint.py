"""Machine-readable command for the Milestone C sine-pitch checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from harpy.envs.baselines import evaluate_all_baselines


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the fixed checkpoint matrix and write one JSON document."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    summaries = evaluate_all_baselines(
        episodes=arguments.episodes,
        seed=arguments.seed,
    )
    payload = {
        "schema_version": 1,
        "checkpoint_id": "harpy-milestone-c-sine-pitch-v0",
        "config_id": "fixed-default-sine-v0",
        "episodes": arguments.episodes,
        "seed": arguments.seed,
        "results": [summary.to_dict() for summary in summaries],
    }
    document = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write(f"{document}\n")
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harpy-sine-gym")
    parser.add_argument("--episodes", type=_positive_integer, default=10)
    parser.add_argument("--seed", type=_nonnegative_integer, default=0)
    return parser


def _positive_integer(value: str) -> int:
    parsed = _decimal_integer(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = _decimal_integer(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _decimal_integer(value: str) -> int:
    try:
        return int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error


if __name__ == "__main__":
    raise SystemExit(main())
