"""Prepare a portable, checksum-bound copy of retained three-seed v1 evidence.

Originals are never changed. Only artifact inventory paths in the two results
are relabeled; all records, metrics, source identities and model bytes remain.
This utility validates artifacts without importing Torch or loading weights.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.experiments.artifacts import load_artifact
from harpy.experiments.qualification import qualify_reference
from harpy.experiments.results import load_result, load_result_document


def prepare(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    frozen_status = decode_json_bytes((source / "qualification-status.json").read_bytes())
    protocol = decode_json_bytes((source / "qualification-protocol.json").read_bytes())
    for seed in (0, 1, 2):
        artifact = load_artifact(source / "models" / f"pitch-{seed}")
        if artifact.manifest_sha256 != protocol["models"][str(seed)]["manifest_sha256"]:
            raise ValueError("retained model differs from frozen declaration")
        shutil.copytree(artifact.root, output / "models" / f"pitch-{seed}")
    transformations = []
    results = []
    for name in ("robustness.json", "confirmation.json"):
        original = source / name
        if sha256_file(original) != frozen_status["result_sha256"][name]:
            raise ValueError("retained result differs from frozen status")
        result = load_result(original)
        document = result.to_document()
        changed = []
        for actor in document["actors"]:
            for entry in actor["artifact_files"]:
                path = Path(entry["path"])
                relative = Path("models") / actor["name"] / path.name
                copied = output / relative
                if (
                    copied.stat().st_size != entry["size_bytes"]
                    or sha256_file(copied) != entry["sha256"]
                ):
                    raise ValueError("result inventory differs from copied model")
                entry["path"] = relative.as_posix()
                changed.append(relative.as_posix())
        portable = load_result_document(document)
        content = canonical_json_bytes(portable.to_document())
        if b"/home/" in content or b"C:\\" in content:
            raise ValueError("portable result still contains a machine root")
        write_new_bytes(output / name, content)
        transformations.append(
            {
                "result": name,
                "original_sha256": sha256_file(original),
                "portable_sha256": hashlib.sha256(content).hexdigest(),
                "inventory_paths_replaced": changed,
                "rule": (
                    "actors[*].artifact_files[*].path becomes models/<actor-name>/<basename>; "
                    "canonical JSON encoding"
                ),
                "scientific_records_changed": False,
            }
        )
        results.append(portable)
    qualification = qualify_reference(*results)
    if any(qualification[key] != frozen_status[key] for key in qualification):
        raise ValueError("recomputed qualification differs from frozen result")
    for name in ("qualification-protocol.json", "qualification-status.json"):
        shutil.copyfile(source / name, output / name)
    write_new_bytes(output / "recomputed-qualification.json", canonical_json_bytes(qualification))
    readme = """# Portable v1 three-seed qualification evidence

This local archive contains the exact three model artifacts and portable copies
of all 23,800 retained actor-episode records. It has not been published. The two
original result hashes and exact inventory-label transformation are recorded in
archive-manifest.json; original source evidence was left unchanged. The original
qualification-status.json result hashes refer to the original files, while the
archive manifest identifies the portable copies. No source or training digest
has been invented where the historical evidence records an unknown value.

After extracting, verify every manifest file's SHA-256 and size. With harpy
installed, recompute without Torch:

```python
from pathlib import Path
from harpy.experiments.results import load_result
from harpy.experiments.qualification import qualify_reference
print(qualify_reference(load_result(Path('robustness.json')),
                        load_result(Path('confirmation.json'))))
```

All three artifacts passed metadata/payload hash validation during preparation.
Fresh model deserialization is a separate optional-dependency check; preparing
this archive does not assert a new fit or a new 23,800-episode evaluation.
The archived records retain their original evaluator source and runtime identities.
"""
    write_new_bytes(output / "README.md", readme.encode())
    manifest = {
        "schema_id": "harpy.v1.portable-qualification-archive/1",
        "publication_status": "local-prepared-not-published",
        "transformations": transformations,
        "qualification_recomputed": True,
        "model_payload_validation": "all-three-manifests-metadata-and-checksums",
        "model_deserialization": "not-performed-by-archive-preparation",
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
    }
    write_new_bytes(output / "archive-manifest.json", canonical_json_bytes(manifest))
    archive = output.with_suffix(".tar.gz")
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w|") as tar,
    ):
        for path in sorted(output.rglob("*")):
            if path.is_file():
                content = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(output).as_posix())
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                tar.addfile(info, io.BytesIO(content))
    # Read every compressed member back, proving archive bytes match the directory.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if tar.extractfile(member).read() != (output / member.name).read_bytes():
                raise ValueError("compressed member differs from prepared evidence")
    digest = sha256_file(archive)
    write_new_bytes(
        archive.with_suffix(archive.suffix + ".sha256"), f"{digest}  {archive.name}\n".encode()
    )
    return {
        "archive": archive.name,
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
        "files": len(manifest["files"]) + 1,
        "actor_episodes": sum(len(r.records) for r in results),
        "qualification": qualification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json_bytes(prepare(args.source, args.output)).decode())


if __name__ == "__main__":
    main()
