"""Package a completed frozen study with bounded memory and portable checksums."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import shutil
import tarfile
import tempfile
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


def verify_resource_receipts(study: Path) -> Path:
    """Bind exported measurements to retained successful local execution evidence."""
    collector = Path(__file__).with_name("resource_receipts.py")
    spec = importlib.util.spec_from_file_location("_harpy_resource_receipt_reader", collector)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    recorded = decode_json_bytes((study / "resource-receipts.json").read_bytes())
    if canonical_json_bytes(reader.build_receipts(study)) != canonical_json_bytes(recorded):
        raise ValueError("resource receipts differ from frozen execution evidence")
    return collector


def expand_distribution(distribution: Path, destination: Path, source_identity: dict) -> dict:
    """Copy regular sdist members into an isolated tree, preserving their exact bytes."""
    digest = sha256_file(distribution)
    members = []
    prefixes = set()
    seen = set()
    with tarfile.open(distribution, "r|gz") as archive:
        for member in archive:
            name = portable_path(member.name.rstrip("/") if member.isdir() else member.name)
            parts = PurePosixPath(name).parts
            prefixes.add(parts[0])
            if len(prefixes) != 1 or name in seen:
                raise ValueError("source distribution must have one unique root and member paths")
            seen.add(name)
            if member.isdir():
                continue
            if not member.isfile() or len(parts) < 2:
                raise ValueError("source distribution must contain regular files below its root")
            relative = PurePosixPath(*parts[1:]).as_posix()
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as payload, path.open("xb") as target:
                shutil.copyfileobj(payload, target, length=1024 * 1024)
            if path.stat().st_size != member.size:
                raise ValueError("source distribution member size differs after expansion")
            members.append(
                {"path": relative, "size_bytes": member.size, "sha256": sha256_file(path)}
            )
    required = {
        "pyproject.toml",
        "PKG-INFO",
        "docs/v2-acceptance.md",
        "docs/v2-calibration.md",
        "research/sound_matching/README.md",
    }
    if not required <= {row["path"] for row in members}:
        raise ValueError("source distribution omits the package or final qualification documents")
    source = destination / "src/harpy"
    if {path.relative_to(source).as_posix() for path in source.rglob("*.py")} != {
        row["path"] for row in source_identity["files"]
    }:
        raise ValueError("qualification source membership differs from frozen evaluator")
    for row in source_identity["files"]:
        path = source / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("qualification source differs from frozen evaluator")
    if sha256_file(distribution) != digest:
        raise ValueError("source distribution changed while expanding")
    return {
        "distribution": "attachments/" + distribution.name,
        "distribution_sha256": digest,
        "member_prefix": next(iter(prefixes)),
        "archive_prefix": "qualification/",
        "members": sorted(members, key=lambda item: item["path"]),
    }


def prepare(
    study: Path,
    models: Path,
    evaluator: Path,
    attachments: list[Path],
    output: Path,
    distribution: Path | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="harpy-study-qualification-") as temporary:
        return _prepare(
            study, models, evaluator, attachments, output, distribution, Path(temporary)
        )


def _prepare(
    study: Path,
    models: Path,
    evaluator: Path,
    attachments: list[Path],
    output: Path,
    distribution: Path | None,
    qualification: Path,
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
    if distribution is None:
        raise ValueError("archive requires the exact final source distribution")
    qualification_identity = expand_distribution(distribution, qualification, plan["source"])
    qualification_data = qualification / "src/harpy/tasks/sound_match/data"
    if sha256_file(qualification_data / "catalog.json") != plan["dataset_catalog_sha256"] or any(
        sha256_file(qualification_data / f"{split}.json") != identity["sha256"]
        for split, identity in plan["splits"].items()
    ):
        raise ValueError("qualification dataset files differ from frozen evaluator")
    resource_collector = verify_resource_receipts(study)
    driver = Path(__file__).with_name("study.py")
    if sha256_file(driver) != plan["driver_sha256"]:
        raise ValueError("archive report reader differs from frozen study driver")
    module_spec = importlib.util.spec_from_file_location("_harpy_frozen_study_reader", driver)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
    verified_reports = []
    readme = (
        (qualification / "research/sound_matching/README.md")
        .read_text()
        .replace("(../../docs/", "(qualification/docs/")
        .encode()
    )
    archive_readme = study / "archive-README.md"
    if archive_readme.exists():
        if archive_readme.read_bytes() != readme:
            raise ValueError("prepared archive README differs from current portable guide")
    else:
        write_new_bytes(archive_readme, readme)
    paths = {
        "README.md": archive_readme,
        "verification/resource_receipts.py": resource_collector,
        "verification/scheduling.py": resource_collector.with_name("scheduling.py"),
        "attachments/" + distribution.name: distribution,
    }
    for row in qualification_identity["members"]:
        paths["qualification/" + row["path"]] = qualification / row["path"]
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
    for name in (
        "plan.json",
        "report.json",
        "listening-index.json",
        "resource-receipts.json",
        "scheduling-receipt.json",
        "scheduling-observations.json",
    ):
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
            if paths[name].resolve() == path.resolve():
                continue
            raise ValueError("attachment basenames must be distinct")
        paths[name] = path
    files = []
    for name, path in sorted(paths.items()):
        portable_path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("archive inputs must be regular files")
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if (
        next(
            row["sha256"] for row in files if row["path"] == qualification_identity["distribution"]
        )
        != qualification_identity["distribution_sha256"]
    ):
        raise ValueError("attached distribution differs from its expanded qualification tree")
    document = {
        "schema_id": "harpy.sound-match.complete-study-archive/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "publication_status": "local-prepared-not-published",
        "readme_transformation": (
            "packaging-only relocation of source guide documentation links into qualification/docs"
        ),
        "qualification_distribution": qualification_identity,
        "resource_receipt_verification": {
            "method": "exact recomputation from retained local GNU time and progress logs",
            "collector": "verification/resource_receipts.py",
            "collector_sha256": sha256_file(resource_collector),
            "scheduling_validator_sha256": sha256_file(
                resource_collector.with_name("scheduling.py")
            ),
            "raw_logs_included": False,
        },
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
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--attach", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        canonical_json_bytes(
            prepare(
                args.study, args.models, args.evaluator, args.attach, args.output, args.distribution
            )
        ).decode()
    )


if __name__ == "__main__":
    main()
