"""Paired decoder interventions on one unchanged trusted pitch-model artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode
from harpy.learning.artifacts import (
    PackageSourceStatus,
    RuntimeStatus,
    SourceProvenance,
    _capture_package_source,
    _digest,
    _relative_path,
    _runtime_from_document,
    _runtime_to_document,
    _source_from_document,
    _source_to_document,
    _verify_file_record,
    canonical_json_bytes,
    capture_source_status,
    decode_json_bytes,
)
from harpy.learning.errors import LearningContractError
from harpy.learning.experiment_results import ArtifactProvenance, _object, _pitch_provenance
from harpy.learning.models import DeviceName, ProfileName
from harpy.learning.pitch_actor import PitchDecoding
from harpy.learning.pitch_data import PitchEvaluationSuiteId, PitchTrainerKind
from harpy.learning.pitch_evaluation import PitchEvaluationRow
from harpy.learning.pitch_reports import (
    PITCH_ESTIMATOR_PARAMETER_COUNT,
    pitch_evaluation_row_from_document,
    pitch_evaluation_row_to_document,
)

COMPARISON_SCHEMA_ID = "harpy-pitch-decoder-comparison-v1"
ACTOR_SEMANTICS = MappingProxyType(
    {
        "global-argmax": PitchDecoding.GLOBAL_ARGMAX.semantics_id,
        "feasible-argmax": PitchDecoding.FEASIBLE_ARGMAX.semantics_id,
        "spectrum_peak": "harpy-sine-pitch-spectrum-peak-baseline-v0",
    }
)
_CRITERION = {
    "eligible": False,
    "criterion_met": None,
    "status": "ineligible",
    "reason": "paired_decoder_intervention",
}


def _suite_ids(profile: ProfileName) -> tuple[PitchEvaluationSuiteId, ...]:
    return (
        (PitchEvaluationSuiteId.SMOKE,)
        if profile is ProfileName.SMOKE
        else (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        )
    )


def _inventory_digest(files: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(files.items()), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PairedOutcomeCounts:
    """Terminal-outcome changes, without claiming to compare unrecorded trajectories."""

    suite_id: PitchEvaluationSuiteId
    episodes: int
    rescued: int
    regressed: int
    changed_outcome: int


@dataclass(frozen=True, slots=True)
class PitchDecoderComparison:
    """One model, two explicit actor semantics, and a matched spectrum baseline."""

    provenance: ArtifactProvenance
    evaluator_source: SourceProvenance
    evaluator_snapshot: PackageSourceStatus
    evaluator_files: Mapping[str, str]
    runtime: RuntimeStatus
    global_rows: tuple[PitchEvaluationRow, ...]
    feasible_rows: tuple[PitchEvaluationRow, ...]
    baseline_rows: tuple[PitchEvaluationRow, ...]

    def __post_init__(self) -> None:
        _pitch_provenance(self.provenance)
        if self.provenance.evaluation_device is not DeviceName.CPU:
            raise LearningContractError("decoder comparison v1 requires CPU evaluation")
        _source_from_document(_source_to_document(self.evaluator_source))
        if not isinstance(self.evaluator_snapshot, PackageSourceStatus):
            raise ValueError("evaluator_snapshot must describe the actual package contents")
        if not isinstance(self.evaluator_files, Mapping):
            raise ValueError("evaluator_files must map relative paths to SHA-256 digests")
        files = {
            _relative_path(path): _digest(digest, "source file digest")
            for path, digest in self.evaluator_files.items()
        }
        required = {"__init__.py", "learning/pitch_actor.py", "learning/pitch_comparison.py"}
        if not required <= files.keys() or _inventory_digest(files) != (
            self.evaluator_snapshot.package_sha256
        ):
            raise ValueError("evaluator file inventory must match its package snapshot")
        if isinstance(self.evaluator_source, PackageSourceStatus) and (
            self.evaluator_source != self.evaluator_snapshot
        ):
            raise ValueError("package evaluator source must match its snapshot")
        if not isinstance(self.runtime, RuntimeStatus) or (
            self.runtime.device is not self.provenance.evaluation_device
        ):
            raise ValueError("runtime device must match the recorded evaluation device")
        object.__setattr__(self, "evaluator_files", MappingProxyType(dict(sorted(files.items()))))
        suites = _suite_ids(self.provenance.profile)
        for field, actor_id, trainer in (
            (
                "global_rows",
                f"pitch-global-argmax-{self.provenance.training_seed}",
                PitchTrainerKind.PITCH,
            ),
            (
                "feasible_rows",
                f"pitch-feasible-argmax-{self.provenance.training_seed}",
                PitchTrainerKind.PITCH,
            ),
            ("baseline_rows", BaselineKind.SPECTRUM_PEAK.value, None),
        ):
            rows = tuple(getattr(self, field))
            if not all(isinstance(row, PitchEvaluationRow) for row in rows) or (
                tuple(row.suite_id for row in rows) != suites
            ):
                raise ValueError(f"{field} must contain the exact ordered profile suites")
            for row in rows:
                if row.actor_id != actor_id or row.trainer is not trainer or row.probe is not None:
                    raise ValueError(f"{field} must retain its explicit actor identity and role")
                if trainer is not None and (
                    row.seed != self.provenance.training_seed
                    or row.parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT
                    or row.training_examples
                    != (256 if self.provenance.profile is ProfileName.SMOKE else 1_400)
                ):
                    raise ValueError("learned rows must match the trained artifact")
            object.__setattr__(self, field, rows)
        if (
            len(
                {row.training_wall_time_seconds for row in (*self.global_rows, *self.feasible_rows)}
            )
            != 1
        ):
            raise ValueError("both decoders must describe the same model training")

    @property
    def paired_counts(self) -> tuple[PairedOutcomeCounts, ...]:
        counts = []
        for global_row, feasible_row in zip(self.global_rows, self.feasible_rows, strict=True):
            pairs = tuple(zip(global_row.episodes, feasible_row.episodes, strict=True))
            counts.append(
                PairedOutcomeCounts(
                    suite_id=global_row.suite_id,
                    episodes=len(pairs),
                    rescued=sum(
                        not before.submitted_success and after.submitted_success
                        for before, after in pairs
                    ),
                    regressed=sum(
                        before.submitted_success and not after.submitted_success
                        for before, after in pairs
                    ),
                    changed_outcome=sum(before != after for before, after in pairs),
                )
            )
        return tuple(counts)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_id": COMPARISON_SCHEMA_ID,
            "provenance": self.provenance.to_document(),
            "evaluator_source": _source_to_document(self.evaluator_source),
            "evaluator_snapshot": _source_to_document(self.evaluator_snapshot),
            "evaluator_files": dict(self.evaluator_files),
            "runtime": _runtime_to_document(self.runtime),
            "actor_semantics": dict(ACTOR_SEMANTICS),
            "criterion": dict(_CRITERION),
            **{
                field: [pitch_evaluation_row_to_document(row) for row in getattr(self, field)]
                for field in ("global_rows", "feasible_rows", "baseline_rows")
            },
        }

    @classmethod
    def from_document(cls, value: object) -> PitchDecoderComparison:
        document = _object(
            value,
            {
                "schema_id",
                "provenance",
                "evaluator_source",
                "evaluator_snapshot",
                "evaluator_files",
                "runtime",
                "actor_semantics",
                "criterion",
                "global_rows",
                "feasible_rows",
                "baseline_rows",
            },
            "decoder comparison",
        )
        if document["schema_id"] != COMPARISON_SCHEMA_ID:
            raise ValueError("unsupported decoder comparison schema_id")
        for field, expected in (("criterion", _CRITERION), ("actor_semantics", ACTOR_SEMANTICS)):
            if canonical_json_bytes(document[field]) != canonical_json_bytes(dict(expected)):
                raise ValueError(f"comparison {field} must retain its declared identity")
        rows = {}
        for field in ("global_rows", "feasible_rows", "baseline_rows"):
            if not isinstance(document[field], list):
                raise ValueError(f"{field} must be a JSON array")
            rows[field] = tuple(pitch_evaluation_row_from_document(row) for row in document[field])
        return cls(
            provenance=ArtifactProvenance.from_document(document["provenance"]),
            evaluator_source=_source_from_document(document["evaluator_source"]),
            evaluator_snapshot=_source_from_document(document["evaluator_snapshot"]),
            evaluator_files=document["evaluator_files"],
            runtime=_runtime_from_document(document["runtime"]),
            **rows,
        )


def comparison_bytes(report: PitchDecoderComparison) -> bytes:
    if not isinstance(report, PitchDecoderComparison):
        raise ValueError("report must be a PitchDecoderComparison")
    return canonical_json_bytes(report.to_document())


def comparison_from_bytes(content: bytes) -> PitchDecoderComparison:
    return PitchDecoderComparison.from_document(decode_json_bytes(content))


def _capture_evaluator() -> tuple[SourceProvenance, PackageSourceStatus, dict[str, str]]:
    anchor = Path(__file__).resolve()
    package = anchor.parent.parent
    snapshot = _capture_package_source(anchor)  # rejects symlinks and covers untracked source
    files = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if _inventory_digest(files) != snapshot.package_sha256:
        raise ValueError("evaluator source changed while capturing provenance")
    return capture_source_status(anchor), snapshot, files


def compare_pitch_decoders(
    artifact_path: Path,
    *,
    device: DeviceName = DeviceName.CPU,
) -> PitchDecoderComparison:
    """Evaluate both decoders on CPU using one model and identical fixed episodes."""
    if not isinstance(artifact_path, Path) or not isinstance(device, DeviceName):
        raise ValueError("artifact_path must be a Path and device must be a DeviceName")
    if device is not DeviceName.CPU:
        raise LearningContractError("decoder comparison v1 requires CPU evaluation")
    from harpy.learning.cache import SpectrumEvidenceCache
    from harpy.learning.dependencies import require_training_dependencies
    from harpy.learning.envs import make_cached_sine_pitch_env
    from harpy.learning.pitch import load_pitch_estimator_model
    from harpy.learning.pitch_actor import PitchPlannerActor
    from harpy.learning.pitch_artifacts import (
        _capture_pitch_runtime_status,
        load_pitch_artifact,
        read_pitch_training_summary,
    )
    from harpy.learning.pitch_data import fixed_pitch_evaluation_suite
    from harpy.learning.pitch_evaluation import (
        build_pitch_evaluation_row,
        evaluate_pitch_baseline_suite,
        evaluate_pitch_learned_actor,
    )
    from harpy.learning.workflows import _artifact_readout_provenance, _require_evaluation_device

    _require_evaluation_device(device)
    source, snapshot, files = _capture_evaluator()
    runtime = _capture_pitch_runtime_status(device)
    artifact = load_pitch_artifact(artifact_path.resolve(strict=True))
    provenance = _artifact_readout_provenance(artifact, device=device)
    summary = read_pitch_training_summary(artifact).summary
    torch_device = require_training_dependencies().torch.device(device.value)
    model = load_pitch_estimator_model(artifact.file("model.pt"), device=torch_device)
    suites = tuple(fixed_pitch_evaluation_suite(item) for item in _suite_ids(provenance.profile))
    cache = SpectrumEvidenceCache()
    learned = []
    for decoding in (PitchDecoding.GLOBAL_ARGMAX, PitchDecoding.FEASIBLE_ARGMAX):
        actor = PitchPlannerActor(model, device=torch_device, decoding=decoding)
        learned.append(
            tuple(
                build_pitch_evaluation_row(
                    actor_id=f"pitch-{decoding.value}-{provenance.training_seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=provenance.training_seed,
                    environment_id=artifact.manifest.environment_id,
                    observation_mode=ObservationMode.SPECTRUM,
                    suite=suite,
                    records=evaluate_pitch_learned_actor(
                        actor, suite, environment_factory=lambda: make_cached_sine_pitch_env(cache)
                    ),
                    parameter_count=artifact.manifest.parameter_count,
                    training_examples=summary.training_examples,
                    training_wall_time_seconds=summary.training_wall_time_seconds,
                )
                for suite in suites
            )
        )
    kind = BaselineKind.SPECTRUM_PEAK
    baselines = tuple(
        build_pitch_evaluation_row(
            actor_id=kind.value,
            trainer=None,
            seed=None,
            environment_id=kind.environment_id,
            observation_mode=kind.observation_mode,
            suite=suite,
            records=evaluate_pitch_baseline_suite(kind, suite, cache=cache),
        )
        for suite in suites
    )
    if _capture_evaluator() != (source, snapshot, files):
        raise ValueError("evaluator source changed during decoder comparison")
    for record in artifact.manifest.files:
        _verify_file_record(artifact.root, record)
    if _artifact_readout_provenance(artifact, device=device) != provenance:
        raise ValueError("artifact changed during decoder comparison")
    return PitchDecoderComparison(
        provenance, source, snapshot, files, runtime, learned[0], learned[1], baselines
    )


__all__ = [
    "ACTOR_SEMANTICS",
    "COMPARISON_SCHEMA_ID",
    "PairedOutcomeCounts",
    "PitchDecoderComparison",
    "compare_pitch_decoders",
    "comparison_bytes",
    "comparison_from_bytes",
]
