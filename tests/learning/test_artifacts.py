from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
import threading
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest

from harpy.learning import artifacts
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCompletion,
    ArtifactManifest,
    ArtifactStatus,
    ArtifactWriter,
    BCTrainingCounts,
    CriterionStatus,
    FileRecord,
    LoadedArtifact,
    PendingArtifactView,
    PPOTrainingCounts,
    RuntimeStatus,
    SourceStatus,
    TrainingConfigDocument,
    TrainingSummaryDocument,
    canonical_json_bytes,
    capture_source_status,
    decode_json_bytes,
    load_artifact,
    read_training_config,
    read_training_summary,
    required_payload_names,
    write_new_bytes,
)
from harpy.learning.models import (
    ARCHITECTURE_SCHEMA_ID,
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    PREPROCESSING_SCHEMA_ID,
    PROFILE_CONFIGS,
    SPECTRUM_GRID_ID,
    BCEpochMetrics,
    BCTrainingSummary,
    DeviceName,
    EvaluationSuiteId,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)
from harpy.learning.suites import TRAIN_DISTRIBUTION_ID, fixed_evaluation_suite

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _run_git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _write_required_source_inputs(repo: Path) -> None:
    (repo / "src/harpy/learning").mkdir(parents=True)
    (repo / "src/harpy/learning/models.py").write_text("MODEL = 1\n", encoding="utf-8")
    (repo / "src/harpy/learning/suites.py").write_text("SUITE = 1\n", encoding="utf-8")
    (repo / "uv.lock").write_bytes(b"version = 1\n")


def _git_repo(path: Path, *, commit_required_inputs: bool = True) -> Path:
    path.mkdir()
    _run_git(path, "init", "-q")
    _run_git(path, "config", "user.email", "tests@example.invalid")
    _run_git(path, "config", "user.name", "Harpy Tests")
    _write_required_source_inputs(path)
    if commit_required_inputs:
        _run_git(
            path, "add", "src/harpy/learning/models.py", "src/harpy/learning/suites.py", "uv.lock"
        )
    else:
        (path / "README.md").write_text("tracked\n", encoding="utf-8")
        _run_git(path, "add", "README.md")
    _run_git(path, "commit", "-q", "-m", "fixture")
    return path


def _suite_records(profile: ProfileName) -> tuple[tuple[EvaluationSuiteId, str], ...]:
    return tuple(
        (suite_id, fixed_evaluation_suite(suite_id).digest_sha256)
        for suite_id in PROFILE_CONFIGS[profile].evaluation_suites
    )


def _config_document(
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    device: DeviceName = DeviceName.CPU,
) -> TrainingConfigDocument:
    config = PROFILE_CONFIGS[profile]
    return TrainingConfigDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=trainer,
        profile=profile,
        seed=seed,
        device=device,
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=_suite_records(profile),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        profile_config=config.bc if trainer is TrainerKind.BC else config.ppo,
        bc_training_digest_sha256=_DIGEST_A if trainer is TrainerKind.BC else None,
        bc_validation_digest_sha256=_DIGEST_B if trainer is TrainerKind.BC else None,
    )


def _summary_document(
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
) -> TrainingSummaryDocument:
    if trainer is TrainerKind.BC:
        profile_config = PROFILE_CONFIGS[profile].bc
        summary = BCTrainingSummary(
            history=(
                BCEpochMetrics(
                    epoch=1,
                    training_loss=0.75,
                    validation_loss=0.5,
                    validation_accuracy=0.625,
                ),
            ),
            selected_epoch=1,
            training_examples=profile_config.train_episodes + 100,
            validation_examples=profile_config.validation_episodes + 50,
            training_wall_time_seconds=1.25,
        )
    else:
        steps = PROFILE_CONFIGS[profile].ppo.total_timesteps
        summary = PPOTrainingSummary(
            requested_environment_steps=steps,
            completed_environment_steps=steps,
            training_wall_time_seconds=2.5,
        )
    return TrainingSummaryDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=trainer,
        profile=profile,
        seed=seed,
        summary=summary,
    )


def _source_status(*, clean: bool = False, committed: bool = True) -> SourceStatus:
    return SourceStatus(
        commit="1" * 40,
        dirty_tree=not clean,
        tracked_diff_sha256=_EMPTY_SHA256 if clean else _DIGEST_A,
        dependency_lock_sha256=_DIGEST_B,
        required_inputs_committed=committed,
    )


def _runtime_status(device: DeviceName = DeviceName.CPU) -> RuntimeStatus:
    return RuntimeStatus(
        python_version="3.12.11",
        platform="Linux-6.8-x86_64",
        processor="x86_64",
        numpy_version="2.3.2",
        gymnasium_version="1.3.0",
        torch_version="2.8.0",
        stable_baselines3_version="2.9.0",
        device=device,
        device_description="CPU" if device is DeviceName.CPU else "NVIDIA Test GPU",
        cuda_runtime_version=None if device is DeviceName.CPU else "12.8",
        cuda_driver_version=None if device is DeviceName.CPU else "570.1",
    )


def _incomplete_manifest(
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    *,
    source: SourceStatus | None = None,
    device: DeviceName = DeviceName.CPU,
) -> ArtifactManifest:
    profile_config = PROFILE_CONFIGS[profile]
    counts: BCTrainingCounts | PPOTrainingCounts
    if trainer is TrainerKind.BC:
        counts = BCTrainingCounts(
            configured_training_episodes=profile_config.bc.train_episodes,
            configured_validation_episodes=profile_config.bc.validation_episodes,
            training_examples=None,
            validation_examples=None,
        )
    else:
        counts = PPOTrainingCounts(
            requested_environment_steps=profile_config.ppo.total_timesteps,
            completed_environment_steps=None,
        )
    return ArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        status=ArtifactStatus.INCOMPLETE,
        trainer=trainer,
        profile=profile,
        seed=seed,
        created_at_utc="2026-08-10T12:00:00Z",
        completed_at_utc=None,
        source=_source_status() if source is None else source,
        runtime=_runtime_status(device),
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=tuple(
            (suite_id.value, digest) for suite_id, digest in _suite_records(profile)
        ),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        parameter_count=57_113,
        training_counts=counts,
        evaluation_device=None,
        criterion_eligible=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        files=(),
    )


def _completion(
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    *,
    evaluation_device: DeviceName = DeviceName.CPU,
    bc_criterion_met: bool | None = None,
) -> ArtifactCompletion:
    summary = _summary_document(trainer, profile).summary
    if isinstance(summary, BCTrainingSummary):
        counts: BCTrainingCounts | PPOTrainingCounts = BCTrainingCounts(
            configured_training_episodes=PROFILE_CONFIGS[profile].bc.train_episodes,
            configured_validation_episodes=PROFILE_CONFIGS[profile].bc.validation_episodes,
            training_examples=summary.training_examples,
            validation_examples=summary.validation_examples,
        )
    else:
        counts = PPOTrainingCounts(
            requested_environment_steps=summary.requested_environment_steps,
            completed_environment_steps=summary.completed_environment_steps,
        )
    return ArtifactCompletion(
        completed_at_utc="2026-08-10T12:01:00Z",
        training_counts=counts,
        evaluation_device=evaluation_device,
        bc_criterion_met=bc_criterion_met,
    )


def _publish_payloads(
    writer: ArtifactWriter,
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    *,
    device: DeviceName = DeviceName.CPU,
) -> None:
    writer.publish_json(
        "training-config.json",
        _config_document(trainer, profile, seed, device).to_document(),
    )
    writer.publish_json(
        "training-summary.json",
        _summary_document(trainer, profile, seed).to_document(),
    )
    model_name = "model.pt" if trainer is TrainerKind.BC else "model.zip"
    writer.publish_model(model_name, lambda path: path.write_bytes(b"persisted-model\n"))
    for filename in required_payload_names(trainer, profile):
        if filename.startswith("evaluation-"):
            writer.publish_json(filename, {"schema_version": 1, "filename": filename})


def _complete_artifact(
    root: Path,
    trainer: TrainerKind = TrainerKind.BC,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    *,
    source: SourceStatus | None = None,
    training_device: DeviceName = DeviceName.CPU,
    evaluation_device: DeviceName = DeviceName.CPU,
    bc_criterion_met: bool | None = None,
) -> LoadedArtifact:
    writer = ArtifactWriter.begin(
        root,
        _incomplete_manifest(
            trainer,
            profile,
            seed,
            source=source,
            device=training_device,
        ),
    )
    _publish_payloads(writer, trainer, profile, seed, device=training_device)
    return writer.complete(
        _completion(
            trainer,
            profile,
            evaluation_device=evaluation_device,
            bc_criterion_met=bc_criterion_met,
        )
    )


def test_canonical_json_is_compact_sorted_finite_utf8_with_one_newline() -> None:
    document = {"z": [True, None, 2.5], "a": "caf\u00e9"}

    encoded = canonical_json_bytes(document)

    assert encoded == b'{"a":"caf\\u00e9","z":[true,null,2.5]}\n'
    assert decode_json_bytes(encoded) == document


@pytest.mark.parametrize(
    "content, message",
    [
        (b'{"key":1,"key":2}\n', "duplicate"),
        (b'{"nested":{"key":1,"key":2}}\n', "duplicate"),
        (b'{"value":NaN}\n', "finite"),
        (b'{"value":Infinity}\n', "finite"),
        (b'{"value":-Infinity}\n', "finite"),
        (b"{broken}\n", "JSON"),
        (b"[]\n", "object"),
        (b'\xff{"value":1}\n', "UTF-8"),
    ],
)
def test_json_decoder_rejects_ambiguous_or_nonfinite_documents(
    content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_json_bytes(content)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_values_at_any_depth(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"nested": [value]})


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "../model.pt", "/tmp/model.pt", "nested/../model.pt", "nested\\model.pt"],
)
def test_file_records_reject_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative_path"):
        FileRecord(relative_path=path, size_bytes=1, sha256=_DIGEST_A)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"size_bytes": True}, "size_bytes"),
        ({"size_bytes": 0}, "size_bytes"),
        ({"sha256": "A" * 64}, "SHA-256"),
        ({"sha256": "a" * 63}, "SHA-256"),
    ],
)
def test_file_records_reject_wrong_sizes_and_hashes(
    changes: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "relative_path": "model.pt",
        "size_bytes": 1,
        "sha256": _DIGEST_A,
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        FileRecord(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("trainer", list(TrainerKind))
@pytest.mark.parametrize("profile", list(ProfileName))
def test_training_documents_round_trip_exact_schema(
    trainer: TrainerKind, profile: ProfileName
) -> None:
    config = _config_document(trainer, profile)
    summary = _summary_document(trainer, profile)

    assert TrainingConfigDocument.from_document(config.to_document()) == config
    assert TrainingSummaryDocument.from_document(summary.to_document()) == summary
    assert decode_json_bytes(canonical_json_bytes(config.to_document())) == config.to_document()
    assert decode_json_bytes(canonical_json_bytes(summary.to_document())) == summary.to_document()


@pytest.mark.parametrize("document_factory", [_config_document, _summary_document])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_training_document_codecs_reject_missing_and_extra_keys(
    document_factory: object, mutation: str
) -> None:
    model = document_factory()  # type: ignore[operator]
    document = model.to_document()
    if mutation == "missing":
        document.pop("seed")
    else:
        document["unexpected"] = True

    with pytest.raises(ValueError, match="fields must be exactly"):
        type(model).from_document(document)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("seed", True),
        ("trainer", "unknown"),
        ("profile", "unknown"),
        ("device", "metal"),
    ],
)
def test_training_config_codec_rejects_wrong_scalar_types_and_enums(
    field: str, value: object
) -> None:
    document = _config_document().to_document()
    document[field] = value  # type: ignore[assignment]

    with pytest.raises(ValueError, match=field):
        TrainingConfigDocument.from_document(document)


def test_training_config_rejects_wrong_profile_variant_and_digest_ownership() -> None:
    bc_document = _config_document().to_document()
    bc_document["profile_config"] = _config_document(TrainerKind.PPO).to_document()[
        "profile_config"
    ]
    with pytest.raises(ValueError, match="profile_config"):
        TrainingConfigDocument.from_document(bc_document)

    ppo_document = _config_document(TrainerKind.PPO).to_document()
    ppo_document["bc_training_digest_sha256"] = _DIGEST_A
    with pytest.raises(ValueError, match="digest"):
        TrainingConfigDocument.from_document(ppo_document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_training_summary_codec_rejects_nonfinite_history_and_timing(value: float) -> None:
    history_document = _summary_document().to_document()
    history_document["summary"]["history"][0]["training_loss"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        TrainingSummaryDocument.from_document(history_document)

    timing_document = _summary_document(TrainerKind.PPO).to_document()
    timing_document["summary"]["training_wall_time_seconds"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        TrainingSummaryDocument.from_document(timing_document)


def test_manifest_round_trips_exact_nested_schema() -> None:
    manifest = _incomplete_manifest()

    document = manifest.to_document()

    assert ArtifactManifest.from_document(document) == manifest
    assert (
        ArtifactManifest.from_document(decode_json_bytes(canonical_json_bytes(document)))
        == manifest
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_codec_rejects_missing_and_extra_keys(mutation: str) -> None:
    document = _incomplete_manifest().to_document()
    if mutation == "missing":
        document.pop("source")
    else:
        document["unexpected"] = None

    with pytest.raises(ValueError, match="fields must be exactly"):
        ArtifactManifest.from_document(document)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"seed": True}, "seed"),
        ({"trainer": "unknown"}, "trainer"),
        ({"profile": "unknown"}, "profile"),
        ({"created_at_utc": "2026-08-10"}, "created_at_utc"),
        ({"created_at_utc": "2026-08-10T12:00:00+01:00"}, "created_at_utc"),
    ],
)
def test_manifest_codec_rejects_wrong_version_enums_integer_and_timestamp(
    changes: dict[str, object], message: str
) -> None:
    document = _incomplete_manifest().to_document()
    document.update(changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        ArtifactManifest.from_document(document)


def test_incomplete_manifest_requires_pinned_profile_counts_and_no_completion_claims() -> None:
    manifest = _incomplete_manifest()
    assert isinstance(manifest.training_counts, BCTrainingCounts)

    with pytest.raises(ValueError, match="configured_training_episodes"):
        replace(
            manifest,
            training_counts=replace(
                manifest.training_counts,
                configured_training_episodes=1,
            ),
        )
    with pytest.raises(ValueError, match="incomplete"):
        replace(manifest, completed_at_utc="2026-08-10T12:01:00Z")
    with pytest.raises(ValueError, match="incomplete"):
        replace(manifest, files=(FileRecord("model.pt", 1, _DIGEST_A),))
    with pytest.raises(ValueError, match="training_counts"):
        replace(
            manifest,
            training_counts=PPOTrainingCounts(
                requested_environment_steps=2_048,
                completed_environment_steps=None,
            ),
        )


@pytest.mark.parametrize("state", ["clean", "tracked-dirty", "staged", "untracked"])
def test_source_capture_uses_real_git_status_and_hashes(tmp_path: Path, state: str) -> None:
    repo = _git_repo(tmp_path / "source")
    if state == "tracked-dirty":
        (repo / "src/harpy/learning/models.py").write_text("MODEL = 2\n", encoding="utf-8")
    elif state == "staged":
        (repo / "src/harpy/learning/models.py").write_text("MODEL = 2\n", encoding="utf-8")
        _run_git(repo, "add", "src/harpy/learning/models.py")
    elif state == "untracked":
        (repo / "scratch.txt").write_text("dirty\n", encoding="utf-8")

    source = capture_source_status(repo / "src/harpy/learning/models.py")

    expected_diff = _run_git(repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    assert source == SourceStatus(
        commit=_run_git(repo, "rev-parse", "HEAD").decode().strip(),
        dirty_tree=state != "clean",
        tracked_diff_sha256=hashlib.sha256(expected_diff).hexdigest(),
        dependency_lock_sha256=hashlib.sha256(b"version = 1\n").hexdigest(),
        required_inputs_committed=True,
    )


def test_source_capture_requires_all_inputs_tracked_at_head(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "source", commit_required_inputs=False)

    source = capture_source_status(repo / "README.md")

    assert source.required_inputs_committed is False
    assert source.dependency_lock_sha256 == hashlib.sha256(b"version = 1\n").hexdigest()


def test_source_capture_stays_bound_to_linked_source_worktree_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = _git_repo(tmp_path / "primary")
    linked = tmp_path / "linked"
    _run_git(primary, "worktree", "add", "-q", "-b", "linked-test", str(linked), "HEAD")
    (primary / "src/harpy/learning/models.py").write_text("DIRTY = 1\n", encoding="utf-8")
    monkeypatch.chdir(primary)

    source = capture_source_status(linked / "src/harpy/learning/models.py")

    assert source.commit == _run_git(linked, "rev-parse", "HEAD").decode().strip()
    assert source.dirty_tree is False
    assert source.tracked_diff_sha256 == _EMPTY_SHA256


def test_required_payload_names_are_the_closed_schema_v1_inventories() -> None:
    assert {
        (trainer, profile): required_payload_names(trainer, profile)
        for trainer in TrainerKind
        for profile in ProfileName
    } == {
        (TrainerKind.BC, ProfileName.SMOKE): (
            "training-config.json",
            "training-summary.json",
            "model.pt",
            "evaluation-smoke.json",
        ),
        (TrainerKind.PPO, ProfileName.SMOKE): (
            "training-config.json",
            "training-summary.json",
            "model.zip",
            "evaluation-smoke.json",
        ),
        (TrainerKind.BC, ProfileName.CHECKPOINT): (
            "training-config.json",
            "training-summary.json",
            "model.pt",
            "evaluation-iid.json",
            "evaluation-ood.json",
        ),
        (TrainerKind.PPO, ProfileName.CHECKPOINT): (
            "training-config.json",
            "training-summary.json",
            "model.zip",
            "evaluation-iid.json",
            "evaluation-ood.json",
        ),
    }


def test_writer_bootstrap_publishes_canonical_incomplete_manifest_first(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "artifact"
    manifest = _incomplete_manifest()

    writer = ArtifactWriter.begin(output, manifest)

    assert writer.file_records == ()
    assert output.resolve() == writer.root
    assert (output / "manifest.json").read_bytes() == canonical_json_bytes(manifest.to_document())
    assert set(path.name for path in output.iterdir()) == {"manifest.json"}
    with pytest.raises(ValueError, match="incomplete"):
        load_artifact(output)


@pytest.mark.parametrize("existing_kind", ["file", "directory"])
def test_writer_bootstrap_never_touches_an_existing_output(
    tmp_path: Path, existing_kind: str
) -> None:
    output = tmp_path / "artifact"
    if existing_kind == "file":
        output.write_bytes(b"keep-file\n")
    else:
        output.mkdir()
        (output / "keep.txt").write_bytes(b"keep-directory\n")
    before = (
        output.read_bytes()
        if output.is_file()
        else {path.name: path.read_bytes() for path in output.iterdir()}
    )

    with pytest.raises(FileExistsError):
        ArtifactWriter.begin(output, _incomplete_manifest())

    after = (
        output.read_bytes()
        if output.is_file()
        else {path.name: path.read_bytes() for path in output.iterdir()}
    )
    assert after == before


def test_parent_creation_failure_never_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "missing-parent" / "artifact"
    original_mkdir = Path.mkdir

    def fail_parent(path: Path, *args: object, **kwargs: object) -> None:
        if path == output.parent:
            raise OSError("injected parent creation failure")
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fail_parent)

    with pytest.raises(OSError, match="parent creation"):
        ArtifactWriter.begin(output, _incomplete_manifest())

    assert not output.exists()


def test_exclusive_directory_creation_failure_never_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "parent" / "artifact"
    original_mkdir = Path.mkdir

    def fail_output(path: Path, *args: object, **kwargs: object) -> None:
        if path == output:
            raise OSError("injected exclusive directory failure")
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fail_output)

    with pytest.raises(OSError, match="exclusive directory"):
        ArtifactWriter.begin(output, _incomplete_manifest())

    assert output.parent.is_dir()
    assert not output.exists()


@pytest.mark.parametrize("failure", ["temp-open", "flush", "rename", "keyboard-before"])
def test_handled_pre_manifest_failures_remove_empty_or_temp_only_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output = tmp_path / "artifact"
    if failure in {"temp-open", "keyboard-before"}:
        original_open = Path.open

        def fail_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if path.parent == output and path.name.startswith(".manifest.json."):
                if failure == "keyboard-before":
                    raise KeyboardInterrupt
                raise OSError("injected temporary open failure")
            return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "open", fail_open)
    elif failure == "flush":
        monkeypatch.setattr(
            artifacts.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(OSError("injected flush failure")),
        )
    else:
        monkeypatch.setattr(
            artifacts.os,
            "replace",
            lambda _source, _target: (_ for _ in ()).throw(OSError("injected rename failure")),
        )

    expected_error = KeyboardInterrupt if failure == "keyboard-before" else OSError
    with pytest.raises(expected_error):
        ArtifactWriter.begin(output, _incomplete_manifest())

    assert not output.exists()


def test_interrupt_after_first_manifest_leaves_valid_incomplete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    original_replace = os.replace

    def replace_then_interrupt(source: Path, target: Path) -> None:
        original_replace(source, target)
        if Path(target).name == "manifest.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(artifacts.os, "replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        ArtifactWriter.begin(output, _incomplete_manifest())

    assert set(path.name for path in output.iterdir()) == {"manifest.json"}
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_json_and_model_publication_are_atomic_and_model_temp_keeps_suffix(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    saver_paths: list[Path] = []

    config_record = writer.publish_json("training-config.json", _config_document().to_document())

    def save_model(path: Path) -> None:
        saver_paths.append(path)
        path.write_bytes(b"model\n")

    model_record = writer.publish_model("model.pt", save_model)

    assert saver_paths[0].suffix == ".pt"
    assert saver_paths[0].name != "model.pt"
    assert not saver_paths[0].exists()
    assert config_record.relative_path == "training-config.json"
    assert model_record == FileRecord(
        relative_path="model.pt",
        size_bytes=6,
        sha256=hashlib.sha256(b"model\n").hexdigest(),
    )
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        "training-config.json",
        "model.pt",
    }


def test_publication_failures_leave_incomplete_manifest_and_real_filesystem_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())

    with pytest.raises(ValueError, match="finite"):
        writer.publish_json("training-config.json", {"value": float("nan")})
    assert set(path.name for path in output.iterdir()) == {"manifest.json"}

    def fail_saver(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("injected saver failure")

    with pytest.raises(RuntimeError, match="saver"):
        writer.publish_model("model.pt", fail_saver)
    assert set(path.name for path in output.iterdir()) == {"manifest.json"}

    original_link = os.link

    def fail_payload_link(source: Path, target: Path) -> None:
        if Path(target).name == "training-config.json":
            raise OSError("injected payload link failure")
        original_link(source, target)

    monkeypatch.setattr(artifacts.os, "link", fail_payload_link)
    with pytest.raises(OSError, match="payload link"):
        writer.publish_json("training-config.json", _config_document().to_document())
    assert set(path.name for path in output.iterdir()) == {"manifest.json"}
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_hash_failure_keeps_published_payload_but_never_records_or_completes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    monkeypatch.setattr(
        artifacts,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(OSError("injected hash failure")),
    )

    with pytest.raises(OSError, match="hash failure"):
        writer.publish_json("training-config.json", _config_document().to_document())

    assert (output / "training-config.json").is_file()
    assert writer.file_records == ()
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_pending_view_is_writer_owned_hash_checked_and_public_loader_stays_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    writer.publish_json("training-config.json", _config_document().to_document())
    writer.publish_json("training-summary.json", _summary_document().to_document())
    writer.publish_model("model.pt", lambda path: path.write_bytes(b"model\n"))

    pending = writer.pending_view(("training-config.json", "training-summary.json", "model.pt"))

    assert isinstance(pending, PendingArtifactView)
    assert pending.manifest.status is ArtifactStatus.INCOMPLETE
    assert pending.file("model.pt").read_bytes() == b"model\n"
    assert read_training_config(pending) == _config_document()
    assert read_training_summary(pending) == _summary_document()
    with pytest.raises(ValueError, match="published"):
        writer.pending_view(("evaluation-smoke.json",))
    with pytest.raises(ValueError, match="incomplete"):
        load_artifact(output)

    (output / "model.pt").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match=r"size|hash"):
        writer.pending_view(("model.pt",))


def test_pending_view_accessors_reject_a_view_minted_before_completion(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    pending = writer.pending_view(("training-config.json", "model.pt"))

    loaded = writer.complete(_completion())

    assert loaded.file("model.pt").is_file()
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.file("model.pt")
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")


def test_pending_view_accessors_reject_after_owning_writer_is_destroyed(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    writer.publish_json("training-config.json", _config_document().to_document())
    pending = writer.pending_view(("training-config.json",))
    del writer
    gc.collect()

    assert (output / "training-config.json").is_file()
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")


@pytest.mark.parametrize("trainer", list(TrainerKind))
@pytest.mark.parametrize("profile", list(ProfileName))
def test_completion_derives_exact_inventory_and_preserves_bootstrap_identity(
    tmp_path: Path, trainer: TrainerKind, profile: ProfileName
) -> None:
    output = tmp_path / f"{trainer.value}-{profile.value}"
    bootstrap = _incomplete_manifest(trainer, profile)
    writer = ArtifactWriter.begin(output, bootstrap)
    _publish_payloads(writer, trainer, profile)

    loaded = writer.complete(_completion(trainer, profile))

    assert loaded == load_artifact(output)
    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    assert tuple(
        record.relative_path for record in loaded.manifest.files
    ) == required_payload_names(trainer, profile)
    for field in (
        "schema_version",
        "trainer",
        "profile",
        "seed",
        "created_at_utc",
        "source",
        "runtime",
        "environment_id",
        "environment_contract_id",
        "train_distribution_id",
        "evaluation_suites",
        "spectrum_grid_id",
        "preprocessing_schema_id",
        "architecture_schema_id",
        "parameter_count",
    ):
        assert getattr(loaded.manifest, field) == getattr(bootstrap, field)
    assert read_training_config(loaded) == _config_document(trainer, profile)
    assert read_training_summary(loaded) == _summary_document(trainer, profile)
    with pytest.raises(RuntimeError, match="complete"):
        writer.complete(_completion(trainer, profile))
    with pytest.raises(RuntimeError, match="complete"):
        writer.pending_view(("model.pt" if trainer is TrainerKind.BC else "model.zip",))


@pytest.mark.parametrize(
    (
        "trainer",
        "profile",
        "seed",
        "source",
        "training_device",
        "evaluation_device",
        "bc_result",
        "eligible",
        "status",
        "criterion_met",
    ),
    [
        (
            TrainerKind.BC,
            ProfileName.CHECKPOINT,
            0,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CPU,
            True,
            True,
            CriterionStatus.CRITERION_MET,
            True,
        ),
        (
            TrainerKind.BC,
            ProfileName.CHECKPOINT,
            0,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CPU,
            False,
            True,
            CriterionStatus.CRITERION_NOT_MET,
            False,
        ),
        (
            TrainerKind.PPO,
            ProfileName.CHECKPOINT,
            4,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CPU,
            None,
            True,
            CriterionStatus.ELIGIBLE_FOR_AGGREGATE,
            None,
        ),
        (
            TrainerKind.PPO,
            ProfileName.CHECKPOINT,
            5,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CPU,
            None,
            False,
            CriterionStatus.INELIGIBLE,
            None,
        ),
        (
            TrainerKind.BC,
            ProfileName.SMOKE,
            0,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CPU,
            True,
            False,
            CriterionStatus.INELIGIBLE,
            None,
        ),
        (
            TrainerKind.BC,
            ProfileName.CHECKPOINT,
            0,
            _source_status(clean=False),
            DeviceName.CPU,
            DeviceName.CPU,
            True,
            False,
            CriterionStatus.INELIGIBLE,
            None,
        ),
        (
            TrainerKind.BC,
            ProfileName.CHECKPOINT,
            0,
            _source_status(clean=True),
            DeviceName.CUDA,
            DeviceName.CPU,
            True,
            False,
            CriterionStatus.INELIGIBLE,
            None,
        ),
        (
            TrainerKind.BC,
            ProfileName.CHECKPOINT,
            0,
            _source_status(clean=True),
            DeviceName.CPU,
            DeviceName.CUDA,
            True,
            False,
            CriterionStatus.INELIGIBLE,
            None,
        ),
    ],
)
def test_completion_derives_closed_scientific_eligibility(
    tmp_path: Path,
    trainer: TrainerKind,
    profile: ProfileName,
    seed: int,
    source: SourceStatus,
    training_device: DeviceName,
    evaluation_device: DeviceName,
    bc_result: bool | None,
    eligible: bool,
    status: CriterionStatus,
    criterion_met: bool | None,
) -> None:
    loaded = _complete_artifact(
        tmp_path / "artifact",
        trainer,
        profile,
        seed,
        source=source,
        training_device=training_device,
        evaluation_device=evaluation_device,
        bc_criterion_met=bc_result,
    )

    assert loaded.manifest.criterion_eligible is eligible
    assert loaded.manifest.criterion_status is status
    assert loaded.manifest.criterion_met is criterion_met


def test_completion_cross_validates_documents_and_actual_counts_before_publication(
    tmp_path: Path,
) -> None:
    wrong_config_output = tmp_path / "wrong-config"
    wrong_config_writer = ArtifactWriter.begin(wrong_config_output, _incomplete_manifest())
    wrong_config_writer.publish_json("training-config.json", _config_document(seed=1).to_document())
    wrong_config_writer.publish_json("training-summary.json", _summary_document().to_document())
    wrong_config_writer.publish_model("model.pt", lambda path: path.write_bytes(b"model\n"))
    wrong_config_writer.publish_json("evaluation-smoke.json", {"schema_version": 1, "valid": True})

    with pytest.raises(ValueError, match=r"config.*manifest|identity"):
        wrong_config_writer.complete(_completion())
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((wrong_config_output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE

    wrong_counts_output = tmp_path / "wrong-counts"
    wrong_counts_writer = ArtifactWriter.begin(wrong_counts_output, _incomplete_manifest())
    _publish_payloads(wrong_counts_writer)
    valid_completion = _completion()
    assert isinstance(valid_completion.training_counts, BCTrainingCounts)
    wrong_counts = replace(
        valid_completion,
        training_counts=replace(
            valid_completion.training_counts,
            training_examples=valid_completion.training_counts.training_examples + 1,  # type: ignore[operator]
        ),
    )

    with pytest.raises(ValueError, match="training_examples"):
        wrong_counts_writer.complete(wrong_counts)
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((wrong_counts_output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


@pytest.mark.parametrize("fault", ["missing", "unexpected", "temporary", "tampered"])
def test_completion_rejects_nonclosed_or_unhashed_filesystem_inventory(
    tmp_path: Path, fault: str
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    if fault == "missing":
        (output / "evaluation-smoke.json").unlink()
    elif fault == "unexpected":
        (output / "notes.txt").write_text("unexpected\n", encoding="utf-8")
    elif fault == "temporary":
        (output / ".model.tmp.pt").write_bytes(b"temporary")
    else:
        (output / "model.pt").write_bytes(b"tampered-model\n")

    with pytest.raises(ValueError, match=r"inventory|missing|unexpected|size|hash"):
        writer.complete(_completion())

    persisted = ArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_final_manifest_publication_failure_leaves_atomic_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    original_replace = os.replace

    def fail_final_manifest(source: Path, target: Path) -> None:
        if Path(target).name == "manifest.json":
            raise OSError("injected final manifest publication failure")
        original_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_final_manifest)

    with pytest.raises(OSError, match="final manifest"):
        writer.complete(_completion())

    assert not any(path.name.startswith(".manifest.json.") for path in output.iterdir())
    persisted = ArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_interrupt_after_final_manifest_rename_returns_the_committed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    pending = writer.pending_view(("training-config.json", "model.pt"))
    original_replace = os.replace

    def replace_then_interrupt(source: Path, target: Path) -> None:
        original_replace(source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(artifacts.os, "replace", replace_then_interrupt)

    try:
        loaded = writer.complete(_completion())
    except KeyboardInterrupt:
        persisted = ArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        )
        pytest.fail(
            f"committed {persisted.status.value} manifest propagated KeyboardInterrupt",
            pytrace=False,
        )

    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    assert load_artifact(output) == loaded
    assert not any(path.name.startswith(".manifest.json.") for path in output.iterdir())
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.file("model.pt")
    with pytest.raises(RuntimeError, match="complete"):
        writer.pending_view(("model.pt",))


def test_completion_has_no_fallible_directory_fsync_after_final_manifest_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    pending = writer.pending_view(("training-config.json", "model.pt"))
    original_fsync_directory = artifacts._fsync_directory

    def reject_post_commit_fsync(path: Path) -> None:
        persisted = ArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        )
        if persisted.status is ArtifactStatus.COMPLETE:
            raise OSError("injected post-replace directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(artifacts, "_fsync_directory", reject_post_commit_fsync)

    loaded = writer.complete(_completion())

    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    assert load_artifact(output) == loaded
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.file("model.pt")
    with pytest.raises(RuntimeError, match="complete"):
        writer.pending_view(("model.pt",))


def test_completion_does_not_reload_or_run_fallible_io_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    pending = writer.pending_view(("training-config.json",))

    def forbidden_post_publication_load(_root: Path) -> LoadedArtifact:
        raise OSError("injected post-publication load failure")

    monkeypatch.setattr(artifacts, "load_artifact", forbidden_post_publication_load)

    loaded = writer.complete(_completion())

    assert loaded.root == output.resolve()
    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    assert (
        ArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        ).status
        is ArtifactStatus.COMPLETE
    )
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")


def test_write_new_bytes_is_atomic_create_only_and_never_overwrites(tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    write_new_bytes(report, b"first\n")
    with pytest.raises(FileExistsError):
        write_new_bytes(report, b"second\n")

    assert report.read_bytes() == b"first\n"
    assert set(path.name for path in tmp_path.iterdir()) == {"report.json"}


def test_write_new_bytes_link_failure_leaves_no_final_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        artifacts.os,
        "link",
        lambda _source, _target: (_ for _ in ()).throw(OSError("injected link failure")),
    )

    with pytest.raises(OSError, match="link failure"):
        write_new_bytes(report, b"content\n")

    assert list(tmp_path.iterdir()) == []


def test_json_publication_never_overwrites_file_raced_in_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    final = output / "training-config.json"
    raced_content = b"raced-json-content\n"
    original_write_fsynced_file = artifacts._write_fsynced_file

    def stage_then_race(path: Path, content: bytes) -> None:
        original_write_fsynced_file(path, content)
        final.write_bytes(raced_content)

    monkeypatch.setattr(artifacts, "_write_fsynced_file", stage_then_race)

    with pytest.raises(FileExistsError):
        writer.publish_json("training-config.json", _config_document().to_document())

    assert final.read_bytes() == raced_content
    assert writer.file_records == ()
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        "training-config.json",
    }


def test_model_publication_never_overwrites_file_raced_in_by_saver(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    final = output / "model.pt"
    raced_content = b"raced-model-content\n"

    def save_and_race(path: Path) -> None:
        path.write_bytes(b"staged-model\n")
        final.write_bytes(raced_content)

    with pytest.raises(FileExistsError):
        writer.publish_model("model.pt", save_and_race)

    assert final.read_bytes() == raced_content
    assert writer.file_records == ()
    assert set(path.name for path in output.iterdir()) == {"manifest.json", "model.pt"}


def _rewrite_payload_and_manifest(root: Path, filename: str, content: bytes) -> None:
    (root / filename).write_bytes(content)
    manifest_path = root / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    files = document["files"]
    assert isinstance(files, list)
    for value in files:
        assert isinstance(value, dict)
        if value["relative_path"] == filename:
            value["size_bytes"] = len(content)
            value["sha256"] = hashlib.sha256(content).hexdigest()
            break
    else:
        raise AssertionError(f"missing file record for {filename}")
    manifest_path.write_bytes(canonical_json_bytes(document))


def test_dangling_existing_output_symlink_is_not_followed_or_touched(tmp_path: Path) -> None:
    missing_target = tmp_path / "must-stay-missing"
    output = tmp_path / "artifact"
    output.symlink_to(missing_target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        ArtifactWriter.begin(output, _incomplete_manifest())

    assert output.is_symlink()
    assert os.readlink(output) == str(missing_target)
    assert not missing_target.exists()


@pytest.mark.parametrize(
    ("trainer", "profile", "filename"),
    [
        (TrainerKind.BC, ProfileName.SMOKE, "training-config.json"),
        (TrainerKind.PPO, ProfileName.SMOKE, "training-summary.json"),
        (TrainerKind.BC, ProfileName.SMOKE, "evaluation-smoke.json"),
        (TrainerKind.BC, ProfileName.CHECKPOINT, "evaluation-iid.json"),
        (TrainerKind.PPO, ProfileName.CHECKPOINT, "evaluation-ood.json"),
    ],
)
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"schema_version":1,"schema_version":1}\n', "duplicate"),
        (b'{"schema_version":\n', "JSON"),
        (b'{"value":NaN}\n', "finite"),
        (b'{"value":Infinity}\n', "finite"),
    ],
)
def test_loader_decodes_every_json_payload_after_integrity_checks(
    tmp_path: Path,
    trainer: TrainerKind,
    profile: ProfileName,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    output = tmp_path / "artifact"
    _complete_artifact(output, trainer, profile)
    _rewrite_payload_and_manifest(output, filename, content)

    with pytest.raises(ValueError, match=message):
        load_artifact(output)


@pytest.mark.parametrize(
    "fault",
    ["missing", "unexpected", "temporary", "wrong-size", "wrong-hash", "directory", "symlink"],
)
def test_public_loader_rejects_every_nonclosed_or_nonintegral_inventory(
    tmp_path: Path, fault: str
) -> None:
    output = tmp_path / "artifact"
    _complete_artifact(output)
    if fault == "missing":
        (output / "model.pt").unlink()
    elif fault == "unexpected":
        (output / "notes.txt").write_text("unexpected\n", encoding="utf-8")
    elif fault == "temporary":
        (output / ".model.partial.pt").write_bytes(b"partial")
    elif fault == "wrong-size":
        (output / "model.pt").write_bytes(b"different-size-model\n")
    elif fault == "wrong-hash":
        original = (output / "model.pt").read_bytes()
        (output / "model.pt").write_bytes(b"X" + original[1:])
    elif fault == "directory":
        (output / "model.pt").unlink()
        (output / "model.pt").mkdir()
    else:
        outside = tmp_path / "outside-model.pt"
        outside.write_bytes((output / "model.pt").read_bytes())
        (output / "model.pt").unlink()
        (output / "model.pt").symlink_to(outside)

    with pytest.raises(
        ValueError,
        match=r"inventory|missing|unexpected|size|hash|regular|symbolic",
    ):
        load_artifact(output)


def test_loader_checks_closed_inventory_before_training_document_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    _complete_artifact(output)
    (output / "unexpected.bin").write_bytes(b"unexpected")

    def forbidden_constructor(
        cls: type[TrainingConfigDocument], document: object
    ) -> TrainingConfigDocument:
        raise AssertionError(f"constructor ran for {cls.__name__}: {document!r}")

    monkeypatch.setattr(TrainingConfigDocument, "from_document", classmethod(forbidden_constructor))

    with pytest.raises(ValueError, match="inventory"):
        load_artifact(output)


def test_loader_decodes_evaluations_before_training_document_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    _complete_artifact(output)
    _rewrite_payload_and_manifest(output, "evaluation-smoke.json", b'{"value":NaN}\n')

    def forbidden_constructor(
        cls: type[TrainingConfigDocument], document: object
    ) -> TrainingConfigDocument:
        raise AssertionError(f"constructor ran for {cls.__name__}: {document!r}")

    monkeypatch.setattr(TrainingConfigDocument, "from_document", classmethod(forbidden_constructor))

    with pytest.raises(ValueError, match="finite"):
        load_artifact(output)


@pytest.mark.parametrize("residue", ["empty", "manifest-temp", "payload-temp"])
def test_loader_identifies_only_empty_or_temp_only_bootstrap_residue_as_safe_to_remove(
    tmp_path: Path, residue: str
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    if residue == "manifest-temp":
        (output / f".manifest.json.{'a' * 32}.tmp").write_bytes(b"partial")
    elif residue == "payload-temp":
        (output / f".model.{'b' * 32}.tmp.pt").write_bytes(b"partial")

    with pytest.raises(ValueError, match="safe to remove"):
        load_artifact(output)


@pytest.mark.parametrize("residue", ["deceptive-file", "exact-temp-directory"])
def test_loader_does_not_call_deceptive_or_nonregular_temp_residue_safe_to_remove(
    tmp_path: Path, residue: str
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    if residue == "deceptive-file":
        kept = output / ".important.tmp.backup"
        kept.write_bytes(b"keep\n")
    else:
        kept = output / f".manifest.json.{'c' * 32}.tmp"
        kept.mkdir()

    with pytest.raises(ValueError, match="missing or unsafe") as raised:
        load_artifact(output)

    assert "safe to remove" not in str(raised.value)
    assert kept.exists()


@pytest.mark.parametrize("manifest_kind", ["directory", "fifo"])
def test_loader_rejects_nonregular_manifest_without_reading_or_blocking(
    tmp_path: Path, manifest_kind: str
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    manifest_path = output / "manifest.json"
    fifo_descriptor: int | None = None
    release_fifo: threading.Timer | None = None
    if manifest_kind == "directory":
        manifest_path.mkdir()
    else:
        os.mkfifo(manifest_path)
        fifo_descriptor = os.open(manifest_path, os.O_RDWR | os.O_NONBLOCK)
        release_fifo = threading.Timer(0.1, os.close, args=(fifo_descriptor,))
        release_fifo.start()

    try:
        with pytest.raises(ValueError, match="missing or unsafe"):
            load_artifact(output)
    finally:
        if release_fifo is not None:
            release_fifo.cancel()
            release_fifo.join()
        if fifo_descriptor is not None:
            with suppress(OSError):
                os.close(fifo_descriptor)


def test_loader_does_not_call_unknown_manifestless_content_safe_to_remove(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "notes.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or unsafe") as raised:
        load_artifact(output)

    assert "safe to remove" not in str(raised.value)
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep\n"


def test_public_loader_always_rejects_a_hash_valid_incomplete_artifact(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)

    with pytest.raises(ValueError, match="incomplete"):
        load_artifact(output)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 2),
        ("trainer", "ppo"),
        ("profile", "checkpoint"),
        ("seed", 9),
        ("created_at_utc", "2026-08-10T11:59:00Z"),
        ("source", None),
        ("runtime", None),
        ("environment_id", "Harpy/Other-v0"),
        ("environment_contract_id", "other-contract"),
        ("train_distribution_id", "other-distribution"),
        ("evaluation_suites", []),
        ("spectrum_grid_id", "other-grid"),
        ("preprocessing_schema_id", "other-preprocessing"),
        ("architecture_schema_id", "other-architecture"),
        ("parameter_count", 57_114),
        ("training_counts", None),
    ],
)
def test_completion_refuses_every_persisted_bootstrap_field_mutation(
    tmp_path: Path, field: str, replacement: object
) -> None:
    output = tmp_path / "artifact"
    writer = ArtifactWriter.begin(output, _incomplete_manifest())
    _publish_payloads(writer)
    manifest_path = output / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    if field == "source":
        source = document["source"]
        assert isinstance(source, dict)
        source["commit"] = "2" * 40
    elif field == "runtime":
        runtime = document["runtime"]
        assert isinstance(runtime, dict)
        runtime["python_version"] = "3.12.12"
    elif field == "training_counts":
        counts = document["training_counts"]
        assert isinstance(counts, dict)
        counts["configured_training_episodes"] = 129
    else:
        document[field] = replacement  # type: ignore[assignment]
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="bootstrap"):
        writer.complete(_completion())

    persisted = decode_json_bytes(manifest_path.read_bytes())
    assert persisted["status"] == "incomplete"


def test_manifest_file_record_cannot_escape_root_even_with_valid_external_hash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    _complete_artifact(output)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    manifest_path = output / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    files = document["files"]
    assert isinstance(files, list)
    model = next(
        value for value in files if isinstance(value, dict) and value["relative_path"] == "model.pt"
    )
    model["relative_path"] = "../outside.pt"
    model["size_bytes"] = outside.stat().st_size
    model["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="relative_path"):
        load_artifact(output)


def test_unsupported_schema_rejects_without_importing_optional_v2_dependencies(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "unsupported-artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_bytes(b'{"schema_version":999}\n')
    sentinel = tmp_path / "unsupported-schema.py"
    sentinel.write_text(
        """
import importlib
import sys
from pathlib import Path

from harpy.learning.artifacts import artifact_manifest_from_document, load_artifact

real_import_module = importlib.import_module

def blocked_import(name, package=None):
    if name in {"torch", "stable_baselines3"}:
        raise ModuleNotFoundError(name)
    return real_import_module(name, package)

importlib.import_module = blocked_import
operations = (
    lambda: artifact_manifest_from_document({"schema_version": 999}),
    lambda: load_artifact(Path(sys.argv[1])),
)
for operation in operations:
    try:
        operation()
    except ValueError as error:
        assert "unsupported artifact schema_version 999" in str(error)
    else:
        raise AssertionError("unsupported schema was accepted")

forbidden = ("harpy.learning.pitch_artifacts", "harpy.learning.pitch", "torch", "stable_baselines3")
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise SystemExit(f"unsupported schema imported optional modules: {loaded}")
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(sentinel), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


_SCHEMA_V1_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "schema-v1"
)
_SCHEMA_V1_FIXTURE_SHA256 = {
    "bc-smoke/evaluation-smoke.json": (
        "361221abbd975182264259cf8fab175494720c53715ba2a05604f14ec340dd8d"
    ),
    "bc-smoke/manifest.json": ("0f775d50663312d879209d7471f60feb3f5e167123430301bde5d22e6021c356"),
    "bc-smoke/model.pt": ("615696641e2f1b82e3a8e996fe2f906eb2383d796ccbb51f688244d3a9b3dc47"),
    "bc-smoke/training-config.json": (
        "b906b2a41c0c04920152d8cff0550408319e3af1d4500a560cea420342f10a9a"
    ),
    "bc-smoke/training-summary.json": (
        "00da99e1cc93f768cc4587633316f6abc6d6fc2da32d9167e96d05e51d1428e6"
    ),
    "episode-trace.json": ("ba713f0fb5e93013b5c6d1648c4eadf761650a1e7970d944a698be50b5b2cf15"),
    "evaluation-report.json": ("b638edadedc8a088b4304f4d79bf13f5893ea87c68724a58e35e699cffa255b4"),
    "ppo-smoke/evaluation-smoke.json": (
        "c2bb46c0b4f83b28bbe501b0c3830589eee7acf5e490f9b8539438a67347fc4d"
    ),
    "ppo-smoke/manifest.json": ("5ba58331fc6c71fb4c6bab7f17c5b6a3fb8f4328963201a3cd730f4b77771bd3"),
    "ppo-smoke/model.zip": ("0629ba00eb740234e60db80abbf5e04f4dfbaea1c2e09bbd33704fe53aa5f72c"),
    "ppo-smoke/training-config.json": (
        "db44753daf819b6dbc6d2e862d00f62f0718cebc6df972d403e5203345bc7564"
    ),
    "ppo-smoke/training-summary.json": (
        "84327d2b7bcfea211f6c1e4962cab80134b91d20681105cbed6cdd3961ce04be"
    ),
}


def _schema_v1_fixture_bytes(relative_path: str) -> bytes:
    return (_SCHEMA_V1_FIXTURE_ROOT / relative_path).read_bytes()


def test_frozen_schema_v1_fixture_file_set_and_bytes_are_pinned() -> None:
    actual_paths = {
        path.relative_to(_SCHEMA_V1_FIXTURE_ROOT).as_posix()
        for path in _SCHEMA_V1_FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual_paths == set(_SCHEMA_V1_FIXTURE_SHA256)
    assert {
        relative_path: hashlib.sha256(_schema_v1_fixture_bytes(relative_path)).hexdigest()
        for relative_path in sorted(actual_paths)
    } == _SCHEMA_V1_FIXTURE_SHA256


@pytest.mark.parametrize(
    ("trainer", "fixture_directory", "model_name"),
    [
        (TrainerKind.BC, "bc-smoke", "model.pt"),
        (TrainerKind.PPO, "ppo-smoke", "model.zip"),
    ],
)
def test_frozen_schema_v1_complete_artifact_families_decode_and_reencode_exactly(
    trainer: TrainerKind,
    fixture_directory: str,
    model_name: str,
) -> None:
    from harpy.learning.evaluation import EvaluationFile

    root = _SCHEMA_V1_FIXTURE_ROOT / fixture_directory
    loaded = load_artifact(root)

    assert isinstance(loaded, LoadedArtifact)
    assert loaded.manifest.trainer is trainer
    assert loaded.manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    assert {path.name for path in root.iterdir()} == {
        "manifest.json",
        "training-config.json",
        "training-summary.json",
        model_name,
        "evaluation-smoke.json",
    }

    manifest_bytes = _schema_v1_fixture_bytes(f"{fixture_directory}/manifest.json")
    manifest = ArtifactManifest.from_document(decode_json_bytes(manifest_bytes))
    assert manifest == loaded.manifest
    assert canonical_json_bytes(manifest.to_document()) == manifest_bytes

    config_bytes = _schema_v1_fixture_bytes(f"{fixture_directory}/training-config.json")
    config = read_training_config(loaded)
    assert isinstance(config, TrainingConfigDocument)
    assert canonical_json_bytes(config.to_document()) == config_bytes

    summary_bytes = _schema_v1_fixture_bytes(f"{fixture_directory}/training-summary.json")
    summary = read_training_summary(loaded)
    assert isinstance(summary, TrainingSummaryDocument)
    assert canonical_json_bytes(summary.to_document()) == summary_bytes

    evaluation_bytes = _schema_v1_fixture_bytes(f"{fixture_directory}/evaluation-smoke.json")
    evaluation = EvaluationFile.from_document(decode_json_bytes(evaluation_bytes))
    assert evaluation.rows[0].trainer is trainer
    assert canonical_json_bytes(evaluation.to_document()) == evaluation_bytes


def test_frozen_schema_v1_models_strict_load_and_repeat_seeded_actions() -> None:
    from harpy.envs.models import PitchAction
    from harpy.envs.sine_pitch import SinePitchEnv
    from harpy.learning.bc import load_bc_actor, validate_bc_artifact
    from harpy.learning.ppo import load_ppo_actor, validate_ppo_artifact

    bc_artifact = load_artifact(_SCHEMA_V1_FIXTURE_ROOT / "bc-smoke")
    ppo_artifact = load_artifact(_SCHEMA_V1_FIXTURE_ROOT / "ppo-smoke")
    assert isinstance(bc_artifact, LoadedArtifact)
    assert isinstance(ppo_artifact, LoadedArtifact)
    validate_bc_artifact(bc_artifact)
    validate_ppo_artifact(ppo_artifact)
    actors = (
        (load_bc_actor(bc_artifact), PitchAction.CENT_DOWN),
        (load_ppo_actor(ppo_artifact), PitchAction.SUBMIT),
    )

    for actor, expected in actors:
        actions = []
        for _ in range(3):
            environment = SinePitchEnv()
            try:
                observation, _info = environment.reset(seed=123)
                actions.append(actor.act(observation))
            finally:
                environment.close()
        assert tuple(actions) == (expected, expected, expected)

    assert int(PitchAction.CENT_DOWN) == 2
    assert int(PitchAction.SUBMIT) == 3


def test_frozen_schema_v1_aggregate_report_decodes_and_reencodes_exactly() -> None:
    from harpy.learning.workflows import (
        evaluation_report_bytes,
        evaluation_report_from_bytes,
    )

    content = _schema_v1_fixture_bytes("evaluation-report.json")
    report = evaluation_report_from_bytes(content)

    assert {row.trainer for row in report.rows if row.trainer is not None} == {
        TrainerKind.BC,
        TrainerKind.PPO,
    }
    assert report.bc_criterion is not None
    assert report.ppo_criterion is not None
    assert evaluation_report_bytes(report) == content


def test_frozen_schema_v1_trace_decodes_and_reencodes_exactly() -> None:
    from harpy.envs.models import PitchAction, TerminalReason
    from harpy.learning.trace import EpisodeTrace, TraceStep, trace_json_bytes

    content = _schema_v1_fixture_bytes("episode-trace.json")
    document = decode_json_bytes(content)
    steps = document["steps"]
    assert isinstance(steps, list)
    episode = EpisodeTrace(
        environment_id=document["environment_id"],  # type: ignore[arg-type]
        distribution_id=document["distribution_id"],  # type: ignore[arg-type]
        seed=document["seed"],  # type: ignore[arg-type]
        target_note_index=document["target_note_index"],  # type: ignore[arg-type]
        target_note=document["target_note"],  # type: ignore[arg-type]
        steps=tuple(
            TraceStep(
                step=step["step"],
                action=PitchAction(step["action"]),
                reward=step["reward"],
            )
            for step in steps
            if isinstance(step, dict)
        ),  # type: ignore[arg-type]
        terminal_reason=TerminalReason(document["terminal_reason"]),  # type: ignore[arg-type]
        final_absolute_error_cents=document["final_absolute_error_cents"],  # type: ignore[arg-type]
        submitted_success=document["submitted_success"],  # type: ignore[arg-type]
        action_count=document["action_count"],  # type: ignore[arg-type]
        excess_actions=document["excess_actions"],  # type: ignore[arg-type]
        total_return=document["total_return"],  # type: ignore[arg-type]
    )

    assert trace_json_bytes(episode) == content


def test_schema_v1_loader_stays_dependency_free_after_schema_v2(
    tmp_path: Path,
) -> None:
    import sys

    sentinel = tmp_path / "v1-loader-regression.py"
    code = """
from pathlib import Path
import sys

from harpy.learning.artifacts import (
    LoadedArtifact,
    load_artifact,
    read_training_config,
    read_training_summary,
)

root = Path(sys.argv[1])
for directory in ("bc-smoke", "ppo-smoke"):
    artifact = load_artifact(root / directory)
    assert isinstance(artifact, LoadedArtifact)
    assert read_training_config(artifact).schema_version == 1
    assert read_training_summary(artifact).schema_version == 1

forbidden = ("harpy.learning.pitch_artifacts", "harpy.learning.pitch", "torch", "stable_baselines3")
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise SystemExit(f"schema-v1 path imported optional modules: {loaded}")
"""
    sentinel.write_text(code, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(sentinel), str(_SCHEMA_V1_FIXTURE_ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
