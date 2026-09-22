"""Torch-free boundaries for inverse features and public-only patch decoding."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from harpy.tasks.sound_match.inverse_features import (
    CONDITION_SIZE,
    FEATURE_SHAPE,
    NOTE_BINS,
    conditioning,
    decode_prediction,
    denormalize_scalar,
    log_spectrogram,
    normalize_scalar,
    supervision_targets,
)
from harpy.tasks.sound_match.models import EpisodeSpec
from harpy.tasks.sound_match.patch import PARAMETER_BOUNDS, PATCH_FIELDS, SCALAR_FIELDS, Patch
from harpy.tasks.sound_match.renderer import FRAME_COUNT, render


def test_inverse_metadata_imports_do_not_load_torch():
    code = """
import sys
from harpy.tasks.sound_match import inverse, inverse_artifacts, inverse_features, inverse_cache
assert 'torch' not in sys.modules
assert 'harpy.tasks.sound_match.inverse_network' not in sys.modules
assert 'stable_baselines3' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_features_are_deterministic_finite_bounded_and_clip_complete():
    patch = Patch(notes_cents=(4800, 6000, 7200), waveform="square")
    audio = render(patch)
    first = log_spectrogram(audio)
    assert first.shape == FEATURE_SHAPE
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert 0 <= first.min() <= first.max() <= 1
    np.testing.assert_array_equal(first, log_spectrogram(audio.copy()))
    np.testing.assert_array_equal(log_spectrogram(np.zeros(FRAME_COUNT, np.float32)), 0)
    late = np.zeros(FRAME_COUNT, np.float32)
    late[-512:] = 0.5
    assert np.any(log_spectrogram(late)[:, -1] > 0)
    with pytest.raises(ValueError, match="whole-clip"):
        log_spectrogram(audio[:-1])


@pytest.mark.parametrize("name", SCALAR_FIELDS)
def test_scalar_roundtrip_preserves_every_legal_integer(name):
    low, high = PARAMETER_BOUNDS[name]
    for value in range(low, high + 1):
        assert denormalize_scalar(name, normalize_scalar(name, value)) == value
    assert denormalize_scalar(name, -1) == low
    assert denormalize_scalar(name, 2) == high


def test_decoder_ties_distinct_notes_clamps_and_locked_field_copy():
    current = Patch(notes_cents=(4900, 6400, 6900), waveform="square", decay_ms=377)
    notes, waveforms = np.zeros(NOTE_BINS), np.zeros(4)
    decoded = decode_prediction(notes, waveforms, np.ones(6), current, PATCH_FIELDS)
    assert decoded.notes_cents == (4800, 4801, 4802)
    assert decoded.waveform == "sine"
    assert decoded.sustain_db == 0 and decoded.decay_ms == 100
    locked = decode_prediction(notes, waveforms, np.ones(6), current, ("brightness_hz",))
    assert locked == replace(current, brightness_hz=12000)
    sustain_only = decode_prediction(notes, waveforms, np.ones(6), current, ("sustain_db",))
    assert sustain_only.decay_ms == 377
    assert decode_prediction(notes, waveforms, np.ones(6), current, ()) == current
    assert denormalize_scalar("gain_db", 0.25) == -14  # -13.5 ties toward lower.
    notes[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        decode_prediction(notes, waveforms, np.ones(6), current, PATCH_FIELDS)


def test_supervision_excludes_locked_and_inactive_fields():
    initial = Patch(sustain_db=0, decay_ms=377)
    target = replace(initial, decay_ms=200, attack_ms=37)
    item = EpisodeSpec("mask", target, initial, ("decay_ms", "attack_ms"))
    labels = supervision_targets((item,))
    assert labels["eligible"].sum() == 1
    assert labels["eligible"][0, PATCH_FIELDS.index("attack_ms")] == 1
    assert labels["eligible"][0, PATCH_FIELDS.index("decay_ms")] == 0
    assert labels["notes"].sum() == 1
    assert labels["counts"].tolist() == [1]
    condition = conditioning(initial, item.editable_fields)
    assert condition.shape == (CONDITION_SIZE,)
    assert condition.dtype == np.float32


@pytest.mark.parametrize("changed", ("hash", "missing", "duplicate"))
def test_validation_checkpoint_rejects_target_audio_outside_frozen_identity(monkeypatch, changed):
    from harpy.tasks.sound_match.inverse_artifacts import training_membership
    from harpy.tasks.sound_match.inverse_training import _validation

    _, _, _, entries = training_membership("quick")
    targets = [
        {
            "episode_id": entry["episode"]["episode_id"],
            "audio_sha256": entry["target_audio_sha256"],
        }
        for entry in entries
    ]
    records = [
        SimpleNamespace(
            status="submitted",
            episode_id=entry["episode"]["episode_id"],
            metrics={"submitted_distance": 1.0},
        )
        for entry in entries
    ]
    result = SimpleNamespace(targets=targets, records=records)
    monkeypatch.setattr("harpy.tasks.sound_match.runner.evaluate", lambda *a, **kw: result)
    # No model inference is needed to exercise validation-result identity checking.
    model = SimpleNamespace(state_dict=dict)
    assert _validation(model, entries, seed=0, epoch=1)["macro_distance"] == 1.0
    if changed == "hash":
        targets[0]["audio_sha256"] = "0" * 64
    elif changed == "missing":
        targets.pop()
    else:
        targets[-1] = targets[0]
    with pytest.raises(ValueError, match="frozen dataset identity"):
        _validation(model, entries, seed=0, epoch=1)
