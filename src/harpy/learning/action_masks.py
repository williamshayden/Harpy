"""Public-control-only legal action masks for learned pitch actors."""

from __future__ import annotations

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    ControlState,
)


def legal_action_mask(controls: ControlState) -> tuple[bool, ...]:
    """Return action legality in exact public ``PitchAction`` order.

    Only the visible control bounds affect mutation legality. Submit remains legal
    for every state, independent of the remaining action budget.
    """
    if not isinstance(controls, ControlState):
        raise ValueError("controls must be a ControlState")
    return (
        controls.octaves > OCTAVE_MIN,
        controls.semitones > SEMITONE_MIN,
        controls.cents > CENT_MIN,
        True,
        controls.cents < CENT_MAX,
        controls.semitones < SEMITONE_MAX,
        controls.octaves < OCTAVE_MAX,
    )
