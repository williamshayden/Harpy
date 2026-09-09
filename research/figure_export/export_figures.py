"""Export the frozen website evidence; no training or new actor evaluation.

Run from this checkout with Harpy's base dependencies installed. The destination
must not exist. data-provenance.json is written last as the completion record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harpy.analysis import analyze  # noqa: E402
from harpy.envs.models import ControlState, PitchAction  # noqa: E402
from harpy.envs.robustness import ROBUSTNESS_CONDITIONS, RobustnessEvidenceCache  # noqa: E402
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, LOG_FREQUENCY_GRID_HZ  # noqa: E402
from harpy.experiments.results import load_result  # noqa: E402

STUDY = "outputs/spectrum-preservation-study/experiment.json"
DIAGNOSTIC = "outputs/v1-respec-qualification/spectrum-failure-case.json"
INPUT_HASHES = {
    STUDY: "69db356439daf33f320d7a596d236584c8053f55b1954927c452dbda8679be8f",
    DIAGNOSTIC: "d40f74c8a9b84c8e38c8e6775062c08f2133a4b1a4241754817f5c08ff92fca6",
}
REPLAY_ID = "spectrum-preservation-20260908-iid-09-01"
ACTORS = ("legacy-point", "cell-max", "quadratic-fft")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def hz(cents: float) -> float:
    return 440.0 * 2.0 ** ((cents - 6900.0) / 1200.0)


def curves(episode: dict, record: dict, low: float, high: float) -> dict:
    condition = next(c for c in ROBUSTNESS_CONDITIONS if c.id == record["condition_id"])
    evidence = RobustnessEvidenceCache(max_bytes=0).evidence(
        episode["source_pitch_cents"], condition, episode["nuisance_seed"]
    )
    saved = record["initial_evidence"]
    for key in ("waveform_sha256", "spectrum_sha256"):
        require(getattr(evidence, key) == saved[key], f"Reconstruction differs: {key}")
    require(evidence.realized_snr_db == saved["realized_snr_db"], "Realized SNR differs")
    analysis = analyze(evidence.waveform, 48000, GYM_ANALYSIS_CONFIG)
    frequencies = analysis.spectrum_frequency_hz
    levels = analysis.spectrum_level_dbfs
    sampled = np.interp(LOG_FREQUENCY_GRID_HZ, frequencies, levels)
    np.testing.assert_array_equal(
        ((np.clip(sampled, -120, 0) + 120) / 120).astype(np.float32), evidence.spectrum
    )
    linear_mask = (frequencies >= low) & (frequencies <= high)
    grid_mask = (low <= LOG_FREQUENCY_GRID_HZ) & (high >= LOG_FREQUENCY_GRID_HZ)
    return {
        "initial_evidence": {**saved, "nuisance_seed": str(saved["nuisance_seed"])},
        "fft": {
            "frequency_hz": frequencies[linear_mask].tolist(),
            "level_dbfs": levels[linear_mask].tolist(),
        },
        "spectrum": {
            "frequency_hz": LOG_FREQUENCY_GRID_HZ[grid_mask].tolist(),
            "level_dbfs": sampled[grid_mask].tolist(),
        },
    }


def actor_summary(record: dict) -> dict:
    terminal = record["terminal"]
    estimate = record["estimates"][0]["estimated_candidate_cents"]
    controls = ControlState()
    states = []
    for step, action_id in enumerate([None, *terminal["actions"]]):
        if action_id is not None and action_id != PitchAction.SUBMIT:
            controls, applied = controls.apply(PitchAction(action_id))
            require(applied, "Unexpected invalid replay action")
        states.append(
            {
                "step": step,
                "action": None if action_id is None else PitchAction(action_id).name,
                "controls": [controls.octaves, controls.semitones, controls.cents],
                "predicted_from_plan_cents": estimate + controls.offset_cents,
                "evaluator_error_cents": terminal["source_pitch_cents"]
                + controls.offset_cents
                - terminal["target_pitch_cents"],
            }
        )
    require(
        states[-1]["evaluator_error_cents"] == terminal["final_signed_error_cents"],
        "Derived replay endpoint differs from the saved result",
    )
    return {
        "actor_name": record["actor_name"],
        "estimated_candidate_cents": estimate,
        "estimated_frequency_hz": hz(estimate),
        "final_signed_error_cents": terminal["final_signed_error_cents"],
        "actions_including_submit": len(terminal["actions"]),
        "submitted_success": terminal["submitted_success"],
        "derived_states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    for path, expected in INPUT_HASHES.items():
        require(digest(ROOT / path) == expected, f"Frozen input changed: {path}")
    source_before = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src/harpy").rglob("*.py"))
    }
    # Strict reader also recomputes and checks the ordinary result's saved metrics.
    report = load_result(ROOT / STUDY).to_document()
    diagnostic = json.loads((ROOT / DIAGNOSTIC).read_text())
    require(
        len(report["episodes"]) == 600 and len(report["records"]) == 14400,
        "Unexpected study membership",
    )
    require([a["name"] for a in report["actors"]] == list(ACTORS), "Actor identities changed")
    for actor in report["actors"]:
        require(actor["observation_mode"] == "waveform", "Mixed observation tracks")
        for artifact in actor["artifact_files"]:
            relative = "research/" + artifact["path"].split("/research/", 1)[1]
            require(
                digest(ROOT / relative) == artifact["sha256"],
                f"Archived research source changed: {relative}",
            )

    groups = defaultdict(list)
    shared = defaultdict(list)
    for record in report["records"]:
        terminal = record["terminal"]
        require(record["trace"] is None, "Revisit derived-state labels for captured traces")
        require(sum(record["inference_counts"]) == 1, "Unexpected inference count")
        require(terminal["invalid_action_count"] == 0, "Unexpected invalid action")
        require(terminal["terminal_reason"].startswith("submitted_"), "Unexpected truncation")
        groups[record["condition_id"], record["actor_name"]].append(terminal)
        shared[record["condition_id"], record["episode_id"]].append(record["initial_evidence"])
    for evidence in shared.values():
        require(
            len(evidence) == 3 and evidence[0] == evidence[1] == evidence[2],
            "Actors did not receive the same initial capture",
        )
    outcomes = []
    for condition in report["conditions"]:
        for actor in ACTORS:
            rows = groups[condition["id"], actor]
            require(len(rows) == 600, "Missing condition or actor records")
            successes = sum(r["submitted_success"] for r in rows)
            outcomes.append(
                {
                    "condition_id": condition["id"],
                    "actor_name": actor,
                    "total": len(rows),
                    "successes": successes,
                    "failures": len(rows) - successes,
                    "max_error_cents": max(r["final_absolute_error_cents"] for r in rows),
                    "within_1_cent": sum(r["within_1_cent"] for r in rows),
                    "invalid_actions": sum(r["invalid_action_count"] for r in rows),
                    "truncations": 0,
                }
            )

    chosen = next(
        r
        for r in report["records"]
        if r["actor_name"] == "legacy-point"
        and r["condition_id"] == "noise-10db"
        and not r["terminal"]["submitted_success"]
    )
    require(chosen["episode_id"] == REPLAY_ID, "Frozen replay selection changed")
    episode = next(e for e in report["episodes"] if e["id"] == REPLAY_ID)
    replay_conditions = {}
    for condition_id in ("clean", "noise-10db"):
        records = [
            r
            for r in report["records"]
            if r["episode_id"] == REPLAY_ID and r["condition_id"] == condition_id
        ]
        view = curves(episode, records[0], hz(4800) - 1, hz(7200) + 1)
        view["actors"] = [actor_summary(r) for r in records]
        replay_conditions[condition_id] = view
    diagnostic_episode = diagnostic["episode"]
    require(
        (diagnostic_episode["source_pitch_cents"], diagnostic_episode["target_note_index"])
        not in {(e["source_pitch_cents"], e["target_note_index"]) for e in report["episodes"]},
        "Diagnostic and study membership overlap",
    )
    diagnostic_hz = diagnostic["true_frequency_hz"]
    diagnostic_view = curves(
        diagnostic_episode, diagnostic["record"], diagnostic_hz - 3, diagnostic_hz + 3
    )
    data = {
        "schema_id": "harpy-website-figure-data-v1",
        "study": {"conditions": report["conditions"], "actors": list(ACTORS), "outcomes": outcomes},
        "replay": {
            "episode": {
                **episode,
                "nuisance_seed": str(episode["nuisance_seed"]),
                "source_frequency_hz": hz(episode["source_pitch_cents"]),
                "target_frequency_hz": hz(4800 + 100 * episode["target_note_index"]),
            },
            "conditions": replay_conditions,
            "state_origin": "Derived from saved actions; not a captured per-step trace",
        },
        "diagnostic": {
            **diagnostic_view,
            "episode_id": diagnostic_episode["id"],
            "source_frequency_hz": diagnostic_hz,
            "wrong_peak_frequency_hz": diagnostic["public_feasible_peak"]["frequency_hz"],
            "wrong_peak_level_dbfs": diagnostic["public_feasible_peak"]["noisy_level_dbfs"],
        },
    }
    require(
        all(digest(ROOT / p) == sha for p, sha in source_before.items()),
        "Package changed during export",
    )
    require(
        all(digest(ROOT / p) == sha for p, sha in INPUT_HASHES.items()),
        "Evidence changed during export",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "data.json", data)
    write_json(
        args.output / "data-provenance.json",
        {
            "schema_id": "harpy-website-figure-provenance-v1",
            "inputs": INPUT_HASHES,
            "source_files": source_before,
            "exporter_sha256": digest(Path(__file__)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "data_sha256": digest(args.output / "data.json"),
            "study_protocol_identity": {
                "id": report["protocol"]["id"],
                "digest_sha256": report["protocol"]["digest_sha256"],
                "partition_counts": report["protocol"]["partition_counts"],
                "episode_count": len(report["episodes"]),
                "digest_scope": "Original frozen protocol including full episode membership; "
                "retained in the hashed study input. Membership is not duplicated here.",
            },
            "actor_components": report["actors"],
            "verification": {
                "strict_report_read": True,
                "terminal_records": 14400,
                "matched_initial_groups": len(shared),
                "reconstructed_captures": 3,
                "waveform_and_spectrum_hashes_match": True,
                "derived_replay_endpoints_match": True,
            },
            "scope": "Existing study and diagnostic only; no new actor evaluation or training. "
            "FFT curves reconstructed from hash-matched initial captures. "
            "Line segments join FFT bins; no smoothing. Seeds exported as decimal strings. "
            "Completion record; a directory without this file is incomplete.",
        },
    )
    print(f"Exported validated data to {args.output}; 14,400 records and 3 capture hashes checked.")


if __name__ == "__main__":
    main()
