"""Contracts for the public-control-only Milestone E action mask."""

from __future__ import annotations

import itertools

import pytest

from harpy.envs.models import ControlState, PitchAction
from harpy.learning.action_masks import legal_action_mask


def _expected_mask(controls: ControlState) -> tuple[bool, ...]:
    return (
        controls.octaves > -2,
        controls.semitones > -12,
        controls.cents > -100,
        True,
        controls.cents < 100,
        controls.semitones < 12,
        controls.octaves < 2,
    )


@pytest.mark.parametrize(
    ("octaves", "semitones", "cents"),
    itertools.product(
        (-2, -1, 0, 1, 2),
        (-12, -11, 0, 11, 12),
        (-100, -99, 0, 99, 100),
    ),
)
def test_legal_action_mask_is_exhaustive_at_and_around_every_bound(
    octaves: int, semitones: int, cents: int
) -> None:
    controls = ControlState(octaves=octaves, semitones=semitones, cents=cents)
    mask = legal_action_mask(controls)

    assert type(mask) is tuple
    assert len(mask) == len(PitchAction) == 7
    assert all(type(value) is bool for value in mask)
    assert mask == _expected_mask(controls)
    assert mask[PitchAction.SUBMIT] is True


def test_action_mask_order_matches_the_exact_public_pitch_action_order() -> None:
    assert tuple(PitchAction) == (
        PitchAction.OCTAVE_DOWN,
        PitchAction.SEMITONE_DOWN,
        PitchAction.CENT_DOWN,
        PitchAction.SUBMIT,
        PitchAction.CENT_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.OCTAVE_UP,
    )
    controls = ControlState(octaves=-2, semitones=12, cents=-100)

    assert legal_action_mask(controls) == (False, True, False, True, True, False, True)


def test_mask_legality_agrees_with_control_state_apply_for_every_mutation() -> None:
    for octaves, semitones, cents in itertools.product(
        range(-2, 3), range(-12, 13), range(-100, 101)
    ):
        controls = ControlState(octaves=octaves, semitones=semitones, cents=cents)
        mask = legal_action_mask(controls)
        for action in PitchAction:
            if action is PitchAction.SUBMIT:
                assert mask[action] is True
                continue
            _updated, applied = controls.apply(action)
            assert mask[action] is applied


@pytest.mark.parametrize("value", [None, {}, (0, 0, 0), object()])
def test_legal_action_mask_requires_a_public_control_state(value: object) -> None:
    with pytest.raises(ValueError, match="ControlState"):
        legal_action_mask(value)  # type: ignore[arg-type]
