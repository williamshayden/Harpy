"""Qualify an installed wheel or sdist from a clean directory outside the checkout.

Run this script with the installed environment's Python. The default base lane
forbids optional model/execution imports. Explicit inverse/pitch/ppo lanes perform
real small CPU training jobs; they never download dependencies or models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import io
import json
import platform
import subprocess
import sys
import tarfile
import time
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path, PurePosixPath
from unittest.mock import patch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict:
    versions = {}
    for name in ("numpy", "gymnasium", "torch", "stable-baselines3"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def save_json(path: Path, value: dict) -> None:
    from harpy._serialization import canonical_json_bytes, write_new_bytes

    write_new_bytes(path, canonical_json_bytes(value))


class BlockModels(importlib.abc.MetaPathFinder):
    allow_historical_artifacts = False

    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname.split(".")[0] in {"torch", "stable_baselines3"}
            or fullname == "harpy.learning.workflows"
            or (fullname == "harpy.learning.artifacts" and not self.allow_historical_artifacts)
        ):
            raise AssertionError(
                f"Installed base operation imported model/execution code: {fullname}"
            )


class Commands:
    def __init__(self, root: Path):
        self.root = root
        self.rows = []

    def __call__(self, arguments: list[str]):
        from harpy.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        started = time.perf_counter()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(arguments)
            except SystemExit as error:
                code = error.code
        index = len(self.rows) + 1
        (self.root / f"command-{index:02d}.stdout.txt").write_text(
            stdout.getvalue(), encoding="utf-8"
        )
        (self.root / f"command-{index:02d}.stderr.txt").write_text(
            stderr.getvalue(), encoding="utf-8"
        )
        self.rows.append(
            {
                "arguments": arguments,
                "returncode": code,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        require(code == 0, f"CLI command failed: {arguments!r}\n{stderr.getvalue()}")
        return stdout.getvalue()


def archive_readers(archive_path: Path, root: Path, commands: Commands) -> dict:
    """Verify the prepared v1 archive and recompute its gates, without model loading."""
    from harpy.experiments.qualification import qualify_reference
    from harpy.experiments.results import load_result

    expected_reports = {"robustness.json", "confirmation.json", "recomputed-qualification.json"}
    copied = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "Archive repeats a file name")
        for member in members:
            path = PurePosixPath(member.name)
            require(
                member.isfile() and not path.is_absolute() and ".." not in path.parts,
                "Archive contains a nonregular or unsafe member",
            )
        stream = archive.extractfile("archive-manifest.json")
        require(stream is not None, "Archive has no inventory")
        manifest = json.load(stream)
        require(
            manifest["schema_id"] == "harpy.v1.portable-qualification-archive/1",
            "Unsupported historical archive",
        )
        inventory = {row["path"]: row for row in manifest["files"]}
        require(len(inventory) == len(manifest["files"]), "Archive inventory repeats a file")
        require(set(names) == {"archive-manifest.json", *inventory}, "Archive inventory differs")
        for member in members:
            if member.name == "archive-manifest.json":
                continue
            record = inventory[member.name]
            require(member.size == record["size_bytes"], "Archive member size differs")
            stream = archive.extractfile(member)
            require(stream is not None, "Missing archive payload")
            digest = hashlib.sha256()
            chunks = [] if member.name in expected_reports else None
            if chunks is not None:
                require(member.size <= 64 * 1024 * 1024, "Historical report exceeds check limit")
            for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            require(digest.hexdigest() == record["sha256"], "Archive payload hash differs")
            if chunks is not None:
                copied[member.name] = root / member.name
                copied[member.name].write_bytes(b"".join(chunks))
    require(set(copied) == expected_reports, "Archive lacks qualification evidence")
    # Each ordinary result is parsed by its strict codec and its aggregates are
    # recomputed by the installed implementation. Models remain unopened.
    for name in ("robustness.json", "confirmation.json"):
        commands(["summarize", str(copied[name]), "--format", "markdown"])
    recomputed = qualify_reference(
        load_result(copied["robustness.json"]), load_result(copied["confirmation.json"])
    )
    require(
        recomputed == json.loads(copied["recomputed-qualification.json"].read_bytes()),
        "Installed qualification differs from archived recomputation",
    )
    return {
        "archive_sha256": file_hash(archive_path),
        "verified_payloads": len(inventory),
        "reports_read": ["robustness.json", "confirmation.json"],
        "performance_qualified": recomputed["performance_qualified"],
        "model_deserialization": "not-performed",
    }


def base_check(root: Path, archive: Path | None, historical_reports: list[Path]) -> dict:
    blocker = BlockModels()
    require(
        not any(name in sys.modules for name in ("torch", "stable_baselines3")),
        "Run base qualification in a fresh interpreter",
    )
    sys.meta_path.insert(0, blocker)
    try:
        from harpy.experiments import durable as pitch
        from harpy.experiments.results import load_result
        from harpy.tasks.sound_match import durable as sound
        from harpy.tasks.sound_match.actors import builtin_actor_spec
        from harpy.tasks.sound_match.datasets import SPLIT_COUNTS, load_manifest, validate_disjoint
        from harpy.tasks.sound_match.patch import Patch
        from harpy.tasks.sound_match.protocols import smoke_episodes
        from harpy.tasks.sound_match.renderer import render
        from harpy.tasks.sound_match.results import _wav_bytes, audio_hash, save_result
        from harpy.tasks.sound_match.runner import run

        commands = Commands(root)
        commands(["--help"])
        console = Path(sys.executable).with_name("harpy")
        require(console.is_file(), "Installed console entry point is missing")
        subprocess.run([str(console), "--help"], check=True, capture_output=True, timeout=30)
        ordinary = root / "pitch-legacy.json"
        commands(["run", "--actor", "spectrum-peak", "--output", str(ordinary)])
        legacy = load_result(ordinary)
        require(
            len(legacy.records) == 1 and legacy.records[0].terminal.submitted_success,
            "Installed v1 ordinary episode failed",
        )
        commands(["summarize", str(ordinary), "--format", "csv"])
        pitch_bundle = root / "pitch-durable"
        commands(
            [
                "evaluate",
                "--bundle",
                "--suite",
                "smoke",
                "--actor",
                "spectrum-peak",
                "--output",
                str(pitch_bundle),
            ]
        )
        committed = (pitch_bundle / "records.jsonl").read_bytes()
        commands(["evaluate", "--resume", "--output", str(pitch_bundle)])
        require(
            (pitch_bundle / "records.jsonl").read_bytes() == committed,
            "Completed pitch resume repeated members",
        )
        commands(["summarize", str(pitch_bundle), "--verify-replay"])
        sound_bundle = root / "sound-durable"
        commands(
            [
                "run",
                "--task",
                "sound-match-v1",
                "--suite",
                "chord-smoke",
                "--actor",
                "fixed",
                "--output",
                str(sound_bundle),
            ]
        )
        moved = root / "relocated-sound"
        sound_bundle.rename(moved)
        commands(["run", "--resume", "--output", str(moved)])
        for fmt in ("text", "markdown", "csv"):
            commands(["summarize", str(moved), "--format", fmt, "--verify-replay"])
        record = next(sound.iter_records(moved))
        exports = {item["role"]: item for item in record["artifacts"]}
        patch_document = json.loads((moved / exports["submitted_patch"]["path"]).read_bytes())
        audio = render(Patch.from_dict(patch_document["patch"]))
        require(
            audio_hash(audio) == record["record"]["initial"]["audio_sha256"],
            "Installed patch renderer changed submitted audio",
        )
        wav = _wav_bytes(audio)
        require(
            wav == (moved / exports["submitted_audio"]["path"]).read_bytes(),
            "Saved patch did not reproduce exact exported WAV bytes",
        )
        (root / "rerendered-submitted.wav").write_bytes(wav)
        old_sound = root / "sound-m1"
        save_result(run(smoke_episodes("tone-smoke")[0], builtin_actor_spec("fixed")), old_sound)
        commands(["summarize", str(old_sound), "--verify-replay"])
        manifests = tuple(load_manifest(split) for split in SPLIT_COUNTS)
        validate_disjoint(manifests)
        require(
            all(len(item.entries) == SPLIT_COUNTS[item.split] for item in manifests),
            "Installed frozen dataset membership is incomplete",
        )
        calibration = load_manifest("calibration")
        commands(
            [
                "run",
                "--task",
                "sound-match-v1",
                "--suite",
                "calibration",
                "--episode",
                calibration.episodes[0].episode_id,
                "--actor",
                "fixed",
                "--output",
                str(root / "calibration-member"),
            ]
        )
        archived = archive_readers(archive, root, commands) if archive else None
        # Retained historical codecs may use the legacy pure-Python artifact
        # schema namespace. Active operations above are independently isolated.
        blocker.allow_historical_artifacts = True
        historical = []
        for index, source in enumerate(historical_reports):
            target = root / f"historical-{index:02d}.json"
            target.write_bytes(source.read_bytes())
            for fmt in ("text", "markdown"):
                commands(["summarize", str(target), "--format", fmt])
            historical.append({"sha256": file_hash(target), "name": source.name})
        require(
            not any(
                name in sys.modules
                for name in (
                    "torch",
                    "stable_baselines3",
                    "harpy.learning.workflows",
                )
            ),
            "Installed base lane crossed the optional execution boundary",
        )
        return {
            "commands": commands.rows,
            "optional_models_imported": False,
            "historical_artifact_codec_imported": "harpy.learning.artifacts" in sys.modules,
            "pitch_committed": pitch.read_bundle(pitch_bundle).completed,
            "sound_committed": sound.read_bundle(moved).completed,
            "saved_patch_exact_wav": True,
            "datasets": {item.split: item.manifest_sha256 for item in manifests},
            "archive": archived,
            "historical_reports": historical,
        }
    finally:
        sys.meta_path.remove(blocker)


def inverse_check(root: Path) -> dict:
    import torch

    from harpy.tasks.sound_match.durable import evaluate_bundle, verify_replay
    from harpy.tasks.sound_match.inverse import (
        inverse_actor_spec,
        load_inverse_artifact,
        train_inverse,
    )
    from harpy.tasks.sound_match.inverse_artifacts import state_sha256
    from harpy.tasks.sound_match.protocols import smoke_episodes

    torch.set_num_threads(1)
    artifact = train_inverse(root / "model", seed=0, profile="quick", device="cpu")
    summary = artifact.summary
    require(
        summary["initial_state_sha256"] != summary["selected_state_sha256"],
        "Inverse quick training did not update parameters",
    )
    first, second = artifact.load_model(), load_inverse_artifact(artifact.root).load_model()
    require(
        state_sha256(first.state_dict())
        == state_sha256(second.state_dict())
        == summary["selected_state_sha256"],
        "Inverse reload changed learned parameters",
    )
    episodes = smoke_episodes("tone-smoke")[:1]
    bundle = root / "evaluation"
    report = evaluate_bundle(
        episodes,
        [inverse_actor_spec(artifact.root), inverse_actor_spec(artifact.root, refine=True)],
        bundle,
    )
    require(report.complete and report.completed == 2, "Inverse actor evaluation incomplete")
    verify_replay(bundle)
    return {
        "training_summary": summary,
        "evaluation": report.to_document(),
        "parameter_updates_verified": True,
        "exact_reload_verified": True,
    }


def v1_learning_check(root: Path, trainer: str) -> dict:
    import torch

    from harpy.experiments.actors import pitch_actor_spec, ppo_actor_spec
    from harpy.experiments.artifacts import load_artifact
    from harpy.experiments.durable import evaluate_bundle, verify_replay
    from harpy.experiments.protocols import smoke_episodes
    from harpy.experiments.training import train
    from harpy.tasks.sound_match.inverse_artifacts import state_sha256

    torch.set_num_threads(1)
    initial = {}
    module = importlib.import_module(f"harpy.learning.{trainer}")
    method = "_train_one_epoch" if trainer == "pitch" else "make_ppo_model"
    original = getattr(module, method)

    def capture_initial(*args, **kwargs):
        if trainer == "pitch":
            if not initial:
                initial.update(
                    {name: value.detach().clone() for name, value in args[0].state_dict().items()}
                )
            return original(*args, **kwargs)
        model = original(*args, **kwargs)
        initial.update(
            {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        )
        return model

    with patch.object(module, method, capture_initial):
        artifact = train(
            trainer=trainer, profile="smoke", seed=3, device="cpu", output=root / "model"
        )
    first, second = artifact.load_model(), load_artifact(artifact.root).load_model()
    state = (first if trainer == "pitch" else first.policy).state_dict()
    other = (second if trainer == "pitch" else second.policy).state_dict()
    require(
        initial and any(not torch.equal(initial[name], value) for name, value in state.items()),
        f"{trainer} smoke training did not update parameters",
    )
    require(
        all(torch.isfinite(value).all() for value in state.values()), "Nonfinite model parameters"
    )
    require(state_sha256(state) == state_sha256(other), "v1 reload changed learned parameters")
    actor = (pitch_actor_spec if trainer == "pitch" else ppo_actor_spec)(artifact.root)
    bundle = root / "evaluation"
    report = evaluate_bundle(smoke_episodes(), [actor], bundle, trace=True)
    require(
        report.complete and all(group["error"] == 0 for group in report.metrics["groups"]),
        "v1 learned actor evaluation did not complete",
    )
    verify_replay(bundle)
    metadata = json.loads((artifact.root / "metadata.json").read_bytes())
    return {
        "training_summary": metadata["summary"],
        "evaluation": report.to_document(),
        "initial_state_sha256": state_sha256(initial),
        "selected_state_sha256": state_sha256(state),
        "parameter_updates_verified": True,
        "exact_reload_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", action="append", choices=("base", "inverse", "pitch", "ppo"))
    parser.add_argument("--distribution", type=Path, help="Wheel or sdist being qualified")
    parser.add_argument("--archive", type=Path, help="Prepared v1 qualification tar.gz")
    parser.add_argument("--historical-report", type=Path, action="append", default=[])
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    import harpy
    from harpy._journal import source_identity

    prefix, package = Path(sys.prefix).resolve(), Path(harpy.__file__).resolve()
    require(
        sys.prefix != sys.base_prefix and package.is_relative_to(prefix),
        "Qualification must import an installed package from this virtual environment",
    )
    installed = distribution("harpy-audio")
    direct = installed.read_text("direct_url.json")
    require(
        not direct or not json.loads(direct).get("dir_info", {}).get("editable"),
        "An editable checkout is not an installed distribution",
    )
    source = source_identity()
    if args.expected_source_sha256:
        require(
            source["sha256"] == args.expected_source_sha256,
            "Installed package source does not match the expected snapshot",
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    lanes = args.lane or ["base"]
    require(len(lanes) == len(set(lanes)), "A lane may be requested only once")
    report = {
        "schema_id": "harpy.installed-qualification/1",
        "package_version": harpy.__version__,
        "package_path": str(package),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "gymnasium": version("gymnasium"),
        "dependency_versions": dependency_versions(),
        "source": source,
        "distribution_sha256": file_hash(args.distribution) if args.distribution else None,
        "working_directory": str(Path.cwd()),
        "lanes": {},
    }
    for lane in lanes:
        root = args.output / lane
        root.mkdir()
        if lane == "base":
            result = base_check(root, args.archive, args.historical_report)
        elif lane == "inverse":
            result = inverse_check(root)
        else:
            result = v1_learning_check(root, lane)
        report["lanes"][lane] = result
        save_json(root / "passed.json", result)
        print(f"Installed {lane} lane passed", flush=True)
    require(source_identity() == source, "Installed evaluator source changed during qualification")
    report["elapsed_seconds"] = time.perf_counter() - started
    report["optional_module_paths"] = {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in ("torch", "stable_baselines3")
        if name in sys.modules
    }
    try:
        import resource

        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        report["peak_rss_mib"] = None
    save_json(args.output / "report.json", report)
    print(f"Saved installed qualification: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
