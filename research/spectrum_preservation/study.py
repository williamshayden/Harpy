"""Run the frozen spectrum-preservation study without changing the Harpy task.

From the checkout: python -m research.spectrum_preservation.study --help
Incomplete output directories are retained; every invocation requires a new path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from harpy.envs.models import ObservationMode
from harpy.envs.robustness import DEFAULT_CACHE_BYTES, ROBUSTNESS_CONDITIONS
from harpy.experiments import ExperimentResult, evaluate, load_result, save_result, summarize
from harpy.experiments.models import _json_loads, _plain
from harpy.experiments.runner import _artifact_files, _source
from harpy.learning.artifacts import write_new_bytes

ACTOR_NAMES = ("legacy-point", "cell-max", "quadratic-fft")


def _log(stream: TextIO, event: str, **fields: object) -> None:
    print(
        json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False),
        file=stream,
        flush=True,
    )


def _assert_unchanged(source: dict, files: list[dict], paths: tuple[Path, ...]) -> None:
    if _source() != source:
        raise ValueError("evaluator source changed during the study")
    if _artifact_files(paths) != files:
        raise ValueError("study scripts, manifest, or plan changed during the study")


def _actors(paths: tuple[Path, ...], manifest: dict):
    from research.spectrum_preservation.encoders import actor_specs

    actors = tuple(actor_specs(artifact_paths=paths))
    if tuple(actor.name for actor in actors) != ACTOR_NAMES:
        raise ValueError("study requires exactly the three declared actors in fixed order")
    if any(actor.observation_mode is not ObservationMode.WAVEFORM for actor in actors):
        raise ValueError("study actors must receive identical waveform observations")
    if len({actor.controller for actor in actors}) != 1:
        raise ValueError("study actors must share the same controller")
    return tuple(
        replace(
            actor,
            artifact_provenance={
                **_plain(actor.artifact_provenance),
                "study": {
                    "protocol_id": manifest["id"],
                    "protocol_digest_sha256": manifest["digest_sha256"],
                    "artifact_paths": [str(path) for path in paths],
                },
            },
        )
        for actor in actors
    )


def _actor_documents(actors) -> list[dict]:
    return [
        {**actor.to_document(), "artifact_files": _artifact_files(actor.artifact_paths)}
        for actor in actors
    ]


def _worker(
    index: int,
    manifest_path: Path,
    output: Path,
    paths: tuple[Path, ...],
    source: dict,
    files: list[dict],
    actor_documents: list[dict],
) -> int:
    from research.spectrum_preservation.membership import episodes_from_manifest

    condition = ROBUSTNESS_CONDITIONS[index]
    started = time.perf_counter()
    with (
        (output / f"{condition.id}.log").open("x", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            _assert_unchanged(source, files, paths)
            manifest = _json_loads(manifest_path.read_bytes())
            episodes = episodes_from_manifest(manifest)
            actors = _actors(paths, manifest)
            if _actor_documents(actors) != actor_documents:
                raise ValueError("worker actors differ from the frozen study actors")
            created = 0

            def progress(spec):
                factory = spec.factory

                def create():
                    nonlocal created
                    actor = factory()
                    created += 1
                    if created % 150 == 0:
                        _log(
                            log,
                            "actor_factories_created",
                            condition=condition.id,
                            actor_episodes_started=created,
                            elapsed_seconds=time.perf_counter() - started,
                        )
                    return actor

                return replace(spec, factory=create)

            _log(
                log,
                "condition_started",
                condition=condition.id,
                pid=os.getpid(),
                episodes=len(episodes),
                actors=len(actors),
                cache_budget_bytes=DEFAULT_CACHE_BYTES,
            )
            result = evaluate(
                episodes,
                tuple(progress(actor) for actor in actors),
                conditions=(condition,),
                protocol=manifest,
            )
            if _plain(result.actors) != actor_documents or result.provenance["source"] != source:
                raise ValueError("condition result differs from frozen actor/source identity")
            _assert_unchanged(source, files, paths)
            path = output / f"{condition.id}.json"
            save_result(result, path)
            verified = load_result(path)
            _assert_unchanged(source, files, paths)
            _log(
                log,
                "condition_complete",
                condition=condition.id,
                records=len(verified.records),
                elapsed_seconds=time.perf_counter() - started,
            )
        except BaseException as error:
            _log(
                log,
                "condition_failed",
                condition=condition.id,
                error=f"{type(error).__name__}: {error}",
            )
            raise
    return index


def merge_shards(
    parts: tuple[ExperimentResult, ...], *, workers: int, elapsed_seconds: float
) -> ExperimentResult:
    """Merge complete strict results in fixed condition order, without pooling actors."""
    if len(parts) != len(ROBUSTNESS_CONDITIONS):
        raise ValueError("study requires one shard for each of the eight conditions")
    anchor = parts[0]
    for part, condition in zip(parts, ROBUSTNESS_CONDITIONS, strict=True):
        if _plain(part.conditions) != [condition.to_document()]:
            raise ValueError("condition shards must follow the exact eight-condition order")
        if (
            part.episodes != anchor.episodes
            or part.actors != anchor.actors
            or part.protocol != anchor.protocol
            or part.provenance["source"] != anchor.provenance["source"]
            or part.provenance["renderer"] != anchor.provenance["renderer"]
        ):
            raise ValueError("shards must share episodes, actors, protocol, source, and renderer")
        runtime = _plain(part.provenance["runtime"])
        expected = _plain(anchor.provenance["runtime"])
        runtime.pop("elapsed_wall_time_seconds", None)
        expected.pop("elapsed_wall_time_seconds", None)
        if runtime != expected:
            raise ValueError("condition shards must share runtime versions and evaluation devices")
    runtime = {
        **_plain(anchor.provenance["runtime"]),
        "elapsed_wall_time_seconds": elapsed_seconds,
        "timing_scope": "study setup, eight condition runs in a bounded worker pool, shard writes, "
        "and strict reload; excludes final merge validation and exports",
        "worker_limit": workers,
        "condition_runs": len(parts),
        "cpu_threads_per_process": 1,
        "per_process_cache_budget_bytes": DEFAULT_CACHE_BYTES,
        "condition_evaluation_wall_seconds": {
            part.conditions[0]["id"]: part.provenance["runtime"]["elapsed_wall_time_seconds"]
            for part in parts
        },
        "ordering": "fixed condition, episode, actor order; paired conditions reuse nuisance seeds",
    }
    return ExperimentResult(
        anchor.episodes,
        anchor.actors,
        tuple(condition.to_document() for condition in ROBUSTNESS_CONDITIONS),
        tuple(record for part in parts for record in part.records),
        {
            "source": anchor.provenance["source"],
            "renderer": anchor.provenance["renderer"],
            "runtime": runtime,
        },
        anchor.protocol,
    )


def run_study(
    manifest_path: Path, output: Path, *, workers: int = 4, plan: Path | None = None
) -> ExperimentResult:
    """Run a previously frozen manifest once; partial work is never resumed or overwritten."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer within 1..8")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"study output must be a new directory: {output}")
    from research.spectrum_preservation.membership import episodes_from_manifest

    started = time.perf_counter()
    script = Path(__file__).resolve()
    paths = (
        script,
        script.with_name("encoders.py"),
        script.with_name("membership.py"),
        manifest_path,
    )
    if plan is not None:
        paths += (plan,)
    files = _artifact_files(paths)  # Reject symbolic links before canonicalizing paths.
    paths = tuple(path.resolve(strict=True) for path in paths)
    manifest_path = manifest_path.resolve(strict=True)
    source = _source()
    manifest = _json_loads(manifest_path.read_bytes())
    episodes = episodes_from_manifest(manifest)
    actor_documents = _actor_documents(_actors(paths, manifest))
    _assert_unchanged(source, files, paths)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    condition_output = output / "conditions"
    condition_output.mkdir()
    # Spawned processes inherit these before importing NumPy; no optional ML imports.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with (output / "study.log").open("x", encoding="utf-8") as log:
        try:
            _log(
                log,
                "study_started",
                episodes=len(episodes),
                actors=len(actor_documents),
                conditions=len(ROBUSTNESS_CONDITIONS),
                worker_limit=workers,
                cache_budget_bytes_per_process=DEFAULT_CACHE_BYTES,
                protocol_digest_sha256=manifest["digest_sha256"],
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(
                        _worker,
                        index,
                        manifest_path,
                        condition_output,
                        paths,
                        source,
                        files,
                        actor_documents,
                    )
                    for index in range(len(ROBUSTNESS_CONDITIONS))
                ]
                try:
                    for future in as_completed(futures):
                        index = future.result()
                        _assert_unchanged(source, files, paths)
                        _log(log, "condition_verified", condition=ROBUSTNESS_CONDITIONS[index].id)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            parts = tuple(
                load_result(condition_output / f"{condition.id}.json")
                for condition in ROBUSTNESS_CONDITIONS
            )
            _assert_unchanged(source, files, paths)
            result = merge_shards(
                parts, workers=workers, elapsed_seconds=time.perf_counter() - started
            )
            _assert_unchanged(source, files, paths)
            save_result(result, output / "experiment.json")
            result = load_result(output / "experiment.json")
            for format_name, suffix in (("csv", "csv"), ("markdown", "md")):
                write_new_bytes(
                    output / f"experiment.{suffix}",
                    summarize(result, format=format_name).encode("utf-8"),
                )
            _assert_unchanged(source, files, paths)
            _log(
                log,
                "study_complete",
                records=len(result.records),
                elapsed_seconds=time.perf_counter() - started,
                timing_scope="entire invocation including merge validation and exports",
            )
            return result
        except BaseException as error:
            _log(log, "study_failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory; existing or partial output is never overwritten",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--plan", type=Path, help="optional frozen design document to bind by hash")
    args = parser.parse_args(argv)
    run_study(args.manifest, args.output, workers=args.workers, plan=args.plan)
    print(f"Study complete: {args.output / 'experiment.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
