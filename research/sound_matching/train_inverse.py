"""Run the declared inverse reference seeds, retaining every training outcome.

Execute with PYTHONPATH pointing at a preserved package source snapshot when
other development is ongoing. This script uses only train and validation
memberships; final-test actor evaluation is a separate frozen study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harpy._serialization import canonical_json_bytes, write_new_bytes
from harpy.tasks.sound_match.inverse import train_inverse
from harpy.tasks.sound_match.inverse_artifacts import (
    dataset_binding,
    require_torch,
    training_configuration,
    training_membership,
)
from harpy.tasks.sound_match.inverse_features import (
    architecture_configuration,
    decoder_configuration,
    feature_configuration,
    supervision_configuration,
)
from harpy.tasks.sound_match.runner import _source_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "reference"), default="reference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train, validation, train_entries, validation_entries = training_membership(args.profile)
    source = _source_identity()
    plan = {
        "schema_id": "harpy.sound-match.inverse-seed-study/1",
        "seeds": [0, 1, 2],
        "configuration": training_configuration(args.profile),
        "architecture": architecture_configuration(),
        "features": feature_configuration(),
        "decoder": decoder_configuration(),
        "supervision": supervision_configuration(),
        "source": source,
        "datasets": {
            "train": dataset_binding(train, train_entries),
            "validation": dataset_binding(validation, validation_entries),
        },
        "final_test_evaluation": "not-performed-by-this-training-script",
    }
    write_new_bytes(args.output / "plan.json", canonical_json_bytes(plan))
    torch = require_torch()
    torch.set_num_threads(1)
    outcomes = []
    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as journal:
        for seed in plan["seeds"]:
            if _source_identity() != source:
                raise RuntimeError("source changed after the seed-study plan was recorded")

            def progress(row, seed=seed):
                document = {"seed": seed, **row}
                encoded = json.dumps(document, sort_keys=True, allow_nan=False)
                journal.write(encoded + "\n")
                journal.flush()
                print(encoded, flush=True)

            artifact = train_inverse(
                args.output / f"seed{seed}",
                seed=seed,
                profile=args.profile,
                progress=progress,
            )
            outcome = {
                "seed": seed,
                "artifact": f"seed{seed}",
                "artifact_manifest_sha256": artifact.manifest_sha256,
                "summary": artifact.summary,
            }
            outcomes.append(outcome)
            progress({"event": "seed-completed", **outcome})
    write_new_bytes(
        args.output / "summary.json",
        canonical_json_bytes({"source_sha256": source["sha256"], "outcomes": outcomes}),
    )


if __name__ == "__main__":
    main()
