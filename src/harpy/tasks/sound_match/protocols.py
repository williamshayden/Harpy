"""Small engineering examples, never training or qualification membership."""

from __future__ import annotations

from dataclasses import replace

from .models import EpisodeSpec
from .patch import Patch, default_patch

SMOKE_SUITES = ("tone-smoke", "chord-smoke", "shaping-smoke")


def smoke_episodes(suite: str = "tone-smoke") -> tuple[EpisodeSpec, ...]:
    """Return explicit reproducible targets under the ordinary 32/64 budgets.

    These intentionally small examples exercise the workflow. Their public source
    and consumed membership make them unsuitable for model-quality claims.
    """
    initial = default_patch()
    if suite == "tone-smoke":
        return (
            EpisodeSpec("tone-001", replace(initial, notes_cents=(6100,))),
            EpisodeSpec(
                "tone-002",
                Patch(
                    notes_cents=(6450,),
                    waveform="triangle",
                    brightness_hz=3200,
                    attack_ms=45,
                    decay_ms=180,
                    sustain_db=-9,
                    release_ms=350,
                    gain_db=-4,
                ),
            ),
        )
    chord_initial = default_patch(voice_count=3)
    if suite == "chord-smoke":
        return (
            EpisodeSpec(
                "chord-001",
                replace(chord_initial, notes_cents=(6000, 6300, 6700)),
                initial_patch=chord_initial,
            ),
            EpisodeSpec(
                "chord-002",
                replace(
                    chord_initial,
                    notes_cents=(5500, 5900, 6200),
                    waveform="saw",
                    brightness_hz=1800,
                    attack_ms=80,
                    release_ms=450,
                ),
                initial_patch=chord_initial,
            ),
        )
    if suite == "shaping-smoke":
        bright_initial = replace(initial, waveform="saw")
        return (
            EpisodeSpec(
                "shape-001",
                replace(initial, notes_cents=(6200,)),
                initial_patch=initial,
                editable_fields=("notes_cents",),
            ),
            EpisodeSpec(
                "shape-002",
                replace(chord_initial, attack_ms=200, release_ms=600),
                initial_patch=chord_initial,
                editable_fields=("attack_ms", "decay_ms", "sustain_db", "release_ms"),
            ),
            EpisodeSpec(
                "shape-003",
                replace(bright_initial, brightness_hz=700),
                initial_patch=bright_initial,
                editable_fields=("brightness_hz",),
            ),
        )
    raise ValueError("Unknown sound-matching suite. Choose: " + ", ".join(SMOKE_SUITES))
