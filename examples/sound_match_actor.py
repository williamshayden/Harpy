"""Run one custom sound-matching actor from an installed base package.

Usage: python sound_match_actor.py NEW_OUTPUT_DIRECTORY
This demonstrates the public API, not a sound-quality claim.
"""

import sys
from dataclasses import replace
from pathlib import Path

from harpy.tasks.sound_match import evaluate_bundle, summarize_bundle
from harpy.tasks.sound_match.models import ActorSpec, Observation, Propose, Submit
from harpy.tasks.sound_match.protocols import smoke_episodes


class OneAudition:
    """Try one legal brightness change, then submit the heard candidate."""

    def __init__(self) -> None:
        self.proposed = False

    def decide(self, observation: Observation) -> Propose | Submit:
        if (
            self.proposed
            or "brightness_hz" not in observation.editable_fields
            or observation.render_budget_remaining == 0
            or observation.decision_budget_remaining < 2
        ):
            return Submit(model_calls=0)
        self.proposed = True
        minimum, _ = observation.allowed_values["bounds"]["brightness_hz"]
        patch = replace(
            observation.patch,
            brightness_hz=max(minimum, observation.patch.brightness_hz // 2),
        )
        return Propose(patch, model_calls=0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python sound_match_actor.py NEW_OUTPUT_DIRECTORY")
    output = Path(sys.argv[1])
    actor = ActorSpec(
        "one-audition",
        OneAudition,
        metadata={"algorithm": "halve-brightness-once-v1", "purpose": "API example"},
    )
    evaluate_bundle(smoke_episodes("tone-smoke"), (actor,), output)
    print(summarize_bundle(output), end="")
