"""Isolated warm analysis/encoding timing, not whole-application throughput."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np

from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS, RobustnessEvidenceCache
from harpy.envs.spectrum import encode_log_spectrum
from harpy.learning.artifacts import write_new_bytes
from research.spectrum_preservation import encoders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new CSV file")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = Path(encoders.__file__).resolve()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    controls = np.zeros(3, dtype=np.int16)
    methods = {
        "legacy-point": encode_log_spectrum,
        "cell-max": encoders.encode_cell_max,
        "quadratic-fft": lambda samples: encoders.encode_fft_reference(samples, controls),
    }
    rows = []
    cache = RobustnessEvidenceCache(max_bytes=0)
    capture_index = 0
    for source in (4800, 6000, 6857, 7200):
        for condition in (CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[6]):
            evidence = cache.evidence(source, condition, 17)
            for method in methods.values():
                method(evidence.waveform)  # One warm-up per method/capture, excluded.
            names = tuple(methods)
            for repeat in range(5):
                shift = (capture_index + repeat) % len(names)
                order = names[shift:] + names[:shift]
                for position, name in enumerate(order):
                    start = time.perf_counter_ns()
                    methods[name](evidence.waveform)
                    elapsed_ns = time.perf_counter_ns() - start
                    rows.append(
                        {
                            "method": name,
                            "source_cents": source,
                            "condition": condition.id,
                            "nuisance_seed": 17,
                            "waveform_sha256": evidence.waveform_sha256,
                            "repeat": repeat,
                            "position": position,
                            "elapsed_ms": elapsed_ns / 1_000_000,
                            "encoder_source_sha256": source_hash,
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                        }
                    )
            capture_index += 1
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("encoder source changed during timing")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new_bytes(args.output, stream.getvalue().encode("utf-8"))
    for name in methods:
        values = np.array([row["elapsed_ms"] for row in rows if row["method"] == name])
        print(
            json.dumps(
                {
                    "method": name,
                    "calls": len(values),
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "scope": "warm analysis plus encoding; excludes synthesis, control and imports",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
