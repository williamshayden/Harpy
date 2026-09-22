"""Compare a retained preflight prefix with a completed study, ignoring only time."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.durable import iter_records


def scientific_bytes(payload: dict) -> bytes:
    document = copy.deepcopy(payload)
    if document["record"] is not None:
        del document["record"]["elapsed_seconds"]
    return canonical_json_bytes(document)


def exported_records(path: Path, receipt: dict):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            count += 1
            digest.update(line)
            yield decode_json_bytes(line)
    if count != receipt["committed_records"] or digest.hexdigest() != receipt["export_sha256"]:
        raise ValueError("neutral prefix export differs from its strict-reader receipt")


def compare(before: Path, after: Path, output: Path) -> dict:
    if before.is_dir():
        original = iter_records(before, allow_incomplete=True)
        before_manifest = sha256_file(before / "manifest.json")
        export_receipt = None
    else:
        export_receipt = decode_json_bytes(Path(str(before) + ".receipt.json").read_bytes())
        if export_receipt["schema_id"] != "harpy.sound-match.verified-prefix-export/1":
            raise ValueError("unsupported strict-reader prefix receipt")
        if before.stat().st_size != export_receipt["export_size_bytes"]:
            raise ValueError("neutral prefix export size differs")
        original = exported_records(before, export_receipt)
        before_manifest = export_receipt["bundle_manifest_sha256"]
    sentinel = object()
    expected = next(original, sentinel)
    before_digest, after_digest = hashlib.sha256(), hashlib.sha256()
    compared = total = 0
    for total, actual in enumerate(iter_records(after), start=1):
        if expected is sentinel:
            continue
        left, right = scientific_bytes(expected), scientific_bytes(actual)
        if left != right:
            raise ValueError(
                f"scientific prefix differs at committed record {total}: {actual['key']}"
            )
        before_digest.update(left)
        after_digest.update(right)
        compared += 1
        expected = next(original, sentinel)
    if expected is not sentinel or not compared:
        raise ValueError("completed study does not contain the entire nonempty preflight prefix")
    document = {
        "schema_id": "harpy.sound-match.exact-prefix-comparison/1",
        "before_manifest_sha256": before_manifest,
        "before_export_receipt": export_receipt,
        "after_manifest_sha256": sha256_file(after / "manifest.json"),
        "compared_records": compared,
        "after_total_records_strictly_read": total,
        "ignored_fields": ["record.elapsed_seconds"],
        "hash_input": "newline-separated canonical payloads with only ignored fields removed",
        "before_scientific_sha256": before_digest.hexdigest(),
        "after_scientific_sha256": after_digest.hexdigest(),
        "exact_match": True,
        "preserved": [
            "patches",
            "audio hashes",
            "objective components",
            "decisions",
            "statuses",
            "model calls",
            "evaluation counts",
            "DSP renders",
            "waveform cache counts",
        ],
    }
    write_new_bytes(output, canonical_json_bytes(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json_bytes(compare(args.before, args.after, args.output)).decode())


if __name__ == "__main__":
    main()
