"""Export strict committed envelopes using a retained bundle's own source reader."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from harpy._journal import source_identity
from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.durable import iter_records


def export(bundle: Path, output: Path) -> dict:
    manifest = decode_json_bytes((bundle / "manifest.json").read_bytes())
    reader_source = source_identity()
    if reader_source != manifest["identity"]["source"]:
        raise ValueError("use the exact archived evaluator source to export this prefix")
    count = 0
    digest = hashlib.sha256()
    with output.open("xb") as stream:
        for payload in iter_records(bundle, allow_incomplete=True):
            count += 1
            encoded = canonical_json_bytes(payload) + b"\n"
            stream.write(encoded)
            digest.update(encoded)
    if source_identity() != reader_source:
        raise ValueError("archived reader source changed during export")
    receipt = {
        "schema_id": "harpy.sound-match.verified-prefix-export/1",
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "verified_reader_source_sha256": reader_source["sha256"],
        "committed_records": count,
        "export_sha256": digest.hexdigest(),
        "export_size_bytes": output.stat().st_size,
        "transformation": "canonical committed payload JSON lines; all fields preserved",
    }
    write_new_bytes(Path(str(output) + ".receipt.json"), canonical_json_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json_bytes(export(args.bundle, args.output)).decode())


if __name__ == "__main__":
    main()
