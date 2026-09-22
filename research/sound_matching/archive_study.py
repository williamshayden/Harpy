"""Package a completed frozen study with bounded memory and portable checksums."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path, PurePosixPath

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.inverse_artifacts import load_inverse_artifact


def portable_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("archive path must be a portable relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("archive path must be a portable relative path")
    return value


def prepare(
    study: Path, models: Path, evaluator: Path, attachments: list[Path], output: Path
) -> dict:
    """Archive existing evidence only: no inference, training or test selection."""
    plan = decode_json_bytes((study / "plan.json").read_bytes())
    study_report = decode_json_bytes((study / "report.json").read_bytes())
    if (
        study_report["study_plan_sha256"] != sha256_file(study / "plan.json")
        or not study_report["final_tests_complete"]
    ):
        raise ValueError("final report differs from frozen study")
    reports = {item["split"]: item for item in study_report["splits"]}
    if (
        len(study_report["splits"]) != 3
        or len(reports) != 3
        or set(reports) != {"validation", "iid-test", "compositional-test"}
    ):
        raise ValueError("final report must describe all three splits exactly once")
    driver = Path(__file__).with_name("study.py")
    if sha256_file(driver) != plan["driver_sha256"]:
        raise ValueError("archive report reader differs from frozen study driver")
    module_spec = importlib.util.spec_from_file_location("_harpy_frozen_study_reader", driver)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
    verified_reports = []
    paths = {}
    for split in ("validation", "iid-test", "compositional-test"):
        report = reader.split_report(study / split, plan, sha256_file(study / "plan.json"), split)
        if not report["report"]["complete"] or report["report"]["completed"] != 192 * 11:
            raise ValueError("archive requires every complete frozen study split")
        if canonical_json_bytes(report) != canonical_json_bytes(reports[split]):
            raise ValueError("retained report metrics differ from strict bundle recomputation")
        verified_reports.append(report)
        for path in sorted((study / split).rglob("*")):
            if path.is_file():
                paths["study/" + path.relative_to(study).as_posix()] = path
    for name in ("plan.json", "report.json", "listening-index.json"):
        paths["study/" + name] = study / name
    expected_report = {
        "schema_id": "harpy.sound-match.matched-study-report/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "splits": verified_reports,
        "reference_quality": reader.reference_quality(verified_reports[0]["actors"]),
        "final_tests_complete": True,
        "perceptual_success_cutoff": None,
        "listening_review": "not-performed",
    }
    if canonical_json_bytes(expected_report) != canonical_json_bytes(study_report):
        raise ValueError("study qualification or reporting contract differs from recomputation")
    selector = Path(__file__).with_name("listening.py")
    if sha256_file(selector) != sha256_file(evaluator / "research/sound_matching/listening.py"):
        raise ValueError("listening selector differs from frozen evaluator snapshot")
    selector_spec = importlib.util.spec_from_file_location(
        "_harpy_frozen_listening_reader", selector
    )
    selector_module = importlib.util.module_from_spec(selector_spec)
    selector_spec.loader.exec_module(selector_module)
    recorded_index = decode_json_bytes((study / "listening-index.json").read_bytes())
    if canonical_json_bytes(selector_module.build_index(study)) != canonical_json_bytes(
        recorded_index
    ):
        raise ValueError("listening selections differ from frozen rule and retained records")
    for path in sorted(study.glob("*-time.txt")):
        paths["study/" + path.name] = path
    for path in sorted(study.glob("*.log")):
        paths["study/" + path.name] = path
    for seed in range(3):
        artifact = load_inverse_artifact(models / f"seed{seed}")
        if artifact.manifest_sha256 != plan["models"][seed]["manifest_sha256"]:
            raise ValueError("model differs from study freeze")
        for path in artifact.root.iterdir():
            paths[f"models/seed{seed}/" + path.name] = path
    snapshot = decode_json_bytes((evaluator / "snapshot-manifest.json").read_bytes())
    source = evaluator / "source/harpy"
    if {path.relative_to(source).as_posix() for path in source.rglob("*.py")} != {
        row["path"] for row in plan["source"]["files"]
    }:
        raise ValueError("evaluator Python membership differs from study freeze")
    for row in plan["source"]["files"]:
        path = source / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("evaluator Python source differs from study freeze")
    if sha256_file(evaluator / "research/sound_matching/study.py") != plan["driver_sha256"]:
        raise ValueError("evaluator driver differs from study freeze")
    if (
        sha256_file(source / "tasks/sound_match/data/catalog.json")
        != plan["dataset_catalog_sha256"]
    ):
        raise ValueError("evaluator dataset catalog differs from study freeze")
    snapshot_names = [portable_path(row["path"]) for row in snapshot["files"]]
    required = {"source/harpy/" + row["path"] for row in plan["source"]["files"]}
    required |= {
        "research/sound_matching/study.py",
        "research/sound_matching/listening.py",
        "source/harpy/tasks/sound_match/data/catalog.json",
    }
    required |= {f"source/harpy/tasks/sound_match/data/{split}.json" for split in plan["splits"]}
    if len(set(snapshot_names)) != len(snapshot_names) or not required <= set(snapshot_names):
        raise ValueError("evaluator inventory duplicates or omits frozen source inputs")
    for row in snapshot["files"]:
        path = evaluator / portable_path(row["path"])
        if not path.resolve().is_relative_to(evaluator.resolve()):
            raise ValueError("evaluator inventory must stay inside its snapshot")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("evaluator snapshot differs from its retained identity")
        paths["evaluator/" + row["path"]] = path
    paths["evaluator/snapshot-manifest.json"] = evaluator / "snapshot-manifest.json"
    for path in attachments:
        name = "attachments/" + path.name
        if name in paths:
            raise ValueError("attachment basenames must be distinct")
        paths[name] = path
    files = []
    for name, path in sorted(paths.items()):
        portable_path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("archive inputs must be regular files")
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    document = {
        "schema_id": "harpy.sound-match.complete-study-archive/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "publication_status": "local-prepared-not-published",
        "listening_review": "not-performed",
        "files": files,
    }
    manifest_bytes = canonical_json_bytes(document)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w|") as archive,
    ):
        for row in files:
            info = tarfile.TarInfo(row["path"])
            info.size, info.mode, info.mtime = row["size_bytes"], 0o644, 0
            with paths[row["path"]].open("rb") as stream:
                archive.addfile(info, stream)
        info = tarfile.TarInfo("archive-manifest.json")
        info.size, info.mode, info.mtime = len(manifest_bytes), 0o644, 0
        archive.addfile(info, io.BytesIO(manifest_bytes))
    expected = {row["path"]: row for row in files}
    expected["archive-manifest.json"] = {
        "size_bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    with tarfile.open(output, "r|gz") as archive:
        for member in archive:
            row = expected.pop(member.name)
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if member.size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("compressed evidence differs from prepared inventory")
    if expected:
        raise ValueError("compressed archive omitted files")
    digest = sha256_file(output)
    write_new_bytes(Path(str(output) + ".sha256"), f"{digest}  {output.name}\n".encode())
    return {
        "schema_id": "harpy.sound-match.archive-verification/1",
        "archive": output.name,
        "sha256": digest,
        "size_bytes": output.stat().st_size,
        "files": len(files) + 1,
        "all_compressed_members_verified": True,
        "actor_episodes": 6336,
        "publication_status": "local-prepared-not-published",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "models", "evaluator", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--attach", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        canonical_json_bytes(
            prepare(args.study, args.models, args.evaluator, args.attach, args.output)
        ).decode()
    )


if __name__ == "__main__":
    main()
