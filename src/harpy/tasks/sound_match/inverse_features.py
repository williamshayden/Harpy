"""Frozen, Torch-free features, supervision masks, and legal inverse decoding."""

from __future__ import annotations

import math

import numpy as np

from harpy.tasks.sound_match.models import EpisodeSpec, _patch
from harpy.tasks.sound_match.patch import (
    PARAMETER_BOUNDS,
    PATCH_FIELDS,
    SCALAR_FIELDS,
    WAVEFORMS,
    Patch,
)
from harpy.tasks.sound_match.renderer import FRAME_COUNT, SAMPLE_RATE_HZ

OBSERVATION_ID = "harpy-sound-match-logspec-conditioned-v1"
DECODER_ID = "harpy-sound-match-distinct-topk-decoder-v1"
SUPERVISION_ID = "harpy-sound-match-masked-equal-field-supervision-v1"
ARCHITECTURE_ID = "harpy-sound-match-inverse-small-conv-v1"
NOTE_BINS = 2401
FEATURE_SHAPE = (128, 64)
CONDITION_SIZE = 26
FFT_SIZE = 4096
HOP_SIZE = 1024
LOG_FLOOR = 1e-7
INACTIVE_DECAY_MS = 100
LOG_FIELDS = ("brightness_hz", "attack_ms", "decay_ms", "release_ms")


def architecture_configuration() -> dict:
    return {
        "id": ARCHITECTURE_ID,
        "input_channels": 1,
        "convolutions": [[1, 4, 5, 2, 2], [4, 8, 5, 2, 2], [8, 8, 3, 2, 1]],
        "activation": "relu",
        "adaptive_average_pool": [16, 4],
        "conditioning_size": CONDITION_SIZE,
        "fusion_hidden": 64,
        "note_logits": NOTE_BINS,
        "waveform_logits": 4,
        "scalar_logits": 6,
        "normalization_layers": "none",
        "dropout": 0.0,
        "parameter_dtype": "float32",
    }


def feature_configuration() -> dict:
    return {
        "id": OBSERVATION_ID,
        "fft_size": FFT_SIZE,
        "hop_size": HOP_SIZE,
        "window": "periodic-hann",
        "centering": False,
        "padding": "right-zero-all-frame-starts-before-clip-end",
        "magnitude": "rfft-absolute-divided-by-window-sum",
        "log_floor": LOG_FLOOR,
        "frequency_bins": FEATURE_SHAPE[0],
        "frequency_grid": "geomspace-80-to-12000-hz-inclusive",
        "time_bins": FEATURE_SHAPE[1],
        "resampling": "linear-log-magnitude-frequency-then-linear-uniform-frame-index",
        "output": "clip-log-floor-to-zero-map-to-0..1-float32",
        "conditioning": [
            "voice-count-onehot-1-3",
            "editable-mask-patch-field-order",
            "current-notes-three-normalized-slots-zero-pad",
            "current-note-presence-three-slots",
            "current-waveform-onehot",
            "current-scalars-normalized-field-order",
        ],
        "conditioning_size": CONDITION_SIZE,
    }


def decoder_configuration() -> dict:
    return {
        "id": DECODER_ID,
        "note_bins": NOTE_BINS,
        "note_range_cents": list(PARAMETER_BOUNDS["notes_cents"]),
        "notes": "top-k-distinct-logits-ties-lower-then-sort",
        "waveform": "argmax-ties-earlier-declared-waveform",
        "waveforms": list(WAVEFORMS),
        "scalar_fields": list(SCALAR_FIELDS),
        "log_normalized_fields": list(LOG_FIELDS),
        "linear_normalized_fields": ["sustain_db", "gain_db"],
        "scalar_rounding": "nearest-legal-integer-ties-lower",
        "locked_fields": "copy-current-public-patch",
        "inactive_unlocked_decay_ms_at_zero_sustain_db": INACTIVE_DECAY_MS,
    }


def supervision_configuration() -> dict:
    return {
        "id": SUPERVISION_ID,
        "field_order": list(PATCH_FIELDS),
        "waveform": "cross-entropy",
        "notes": "mean-bce-with-logits-over-2401-bins",
        "note_positive_weight": "(2401-public-note-count)/public-note-count",
        "scalars": "mse-of-sigmoid-output-and-normalized-label",
        "field_weighting": "per-example-mean-of-eligible-fields-then-batch-mean",
        "eligible": "editable-and-structurally-active-under-training-label",
        "inactive": {"decay_ms": "target-sustain-db-equals-zero"},
    }


def _editable(editable_fields: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(editable_fields, (tuple, list)) or any(
        type(name) is not str or name not in PATCH_FIELDS for name in editable_fields
    ):
        raise ValueError("editable_fields must contain known field names")
    if len(set(editable_fields)) != len(editable_fields):
        raise ValueError("editable_fields must be distinct")
    return tuple(name for name in PATCH_FIELDS if name in editable_fields)


def normalize_scalar(name: str, value: int | float) -> float:
    low, high = PARAMETER_BOUNDS[name]
    if name in LOG_FIELDS:
        return math.log(float(value) / low) / math.log(high / low)
    return (float(value) - low) / (high - low)


def denormalize_scalar(name: str, value: float) -> int:
    low, high = PARAMETER_BOUNDS[name]
    value = min(1.0, max(0.0, float(value)))
    physical = low * (high / low) ** value if name in LOG_FIELDS else low + value * (high - low)
    return max(low, min(high, math.ceil(physical - 0.5)))


def log_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Encode one whole clip without retaining any caller-owned audio."""
    if (
        not isinstance(audio, np.ndarray)
        or audio.dtype != np.float32
        or audio.shape != (FRAME_COUNT,)
        or not np.isfinite(audio).all()
    ):
        raise ValueError("target audio must be a finite whole-clip float32 waveform")
    frame_count = (FRAME_COUNT + HOP_SIZE - 1) // HOP_SIZE
    padded = np.pad(
        audio.astype(np.float64), (0, (frame_count - 1) * HOP_SIZE + FFT_SIZE - FRAME_COUNT)
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::HOP_SIZE]
    window = 0.5 - 0.5 * np.cos(math.tau * np.arange(FFT_SIZE) / FFT_SIZE)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=-1)) / window.sum()
    np.maximum(magnitude, LOG_FLOOR, out=magnitude)
    np.log(magnitude, out=magnitude)
    frequencies = np.geomspace(80.0, 12000.0, FEATURE_SHAPE[0]) * FFT_SIZE / SAMPLE_RATE_HZ
    lower = np.floor(frequencies).astype(np.int64)
    fraction = frequencies - lower
    sampled = magnitude[:, lower] * (1.0 - fraction) + magnitude[:, lower + 1] * fraction
    times = np.linspace(0, frame_count - 1, FEATURE_SHAPE[1])
    earlier = np.floor(times).astype(np.int64)
    later = np.minimum(earlier + 1, frame_count - 1)
    fraction = (times - earlier)[:, None]
    sampled = sampled[earlier] * (1.0 - fraction) + sampled[later] * fraction
    result = np.clip((sampled - math.log(LOG_FLOOR)) / -math.log(LOG_FLOOR), 0.0, 1.0)
    return np.ascontiguousarray(result.T, dtype=np.float32)


def conditioning(patch: Patch, editable_fields: tuple[str, ...]) -> np.ndarray:
    """Encode only note count, editable mask, and the current public patch."""
    patch = _patch(patch)
    editable = _editable(editable_fields)
    count = len(patch.notes_cents)
    values = [float(count == 1), float(count == 3)]
    values.extend(float(name in editable) for name in PATCH_FIELDS)
    values.extend((value - 4800) / 2400 for value in patch.notes_cents)
    values.extend([0.0] * (3 - count))
    values.extend(float(index < count) for index in range(3))
    values.extend(float(patch.waveform == waveform) for waveform in WAVEFORMS)
    values.extend(normalize_scalar(name, getattr(patch, name)) for name in SCALAR_FIELDS)
    return np.asarray(values, dtype=np.float32)


def decode_prediction(
    note_logits: np.ndarray,
    waveform_logits: np.ndarray,
    scalar_values: np.ndarray,
    current_patch: Patch,
    editable_fields: tuple[str, ...],
) -> Patch:
    """Decode a legal full patch using no target parameters."""
    current_patch = _patch(current_patch)
    editable = _editable(editable_fields)
    arrays = []
    for value, length in ((note_logits, NOTE_BINS), (waveform_logits, 4), (scalar_values, 6)):
        array = np.asarray(value)
        if array.shape != (length,) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("prediction heads must have the declared numeric shapes")
        if not np.isfinite(array).all():
            raise ValueError("prediction heads must be finite")
        arrays.append(array.astype(np.float64))
    notes, waveforms, scalars = arrays
    values = current_patch.to_dict()
    if "notes_cents" in editable:
        order = np.lexsort((np.arange(NOTE_BINS), -notes))
        values["notes_cents"] = sorted(
            int(index) + 4800 for index in order[: len(current_patch.notes_cents)]
        )
    if "waveform" in editable:
        values["waveform"] = WAVEFORMS[int(np.argmax(waveforms))]
    for index, name in enumerate(SCALAR_FIELDS):
        if name in editable:
            values[name] = denormalize_scalar(name, float(scalars[index]))
    if "decay_ms" in editable and values["sustain_db"] == 0:
        values["decay_ms"] = INACTIVE_DECAY_MS
    return Patch.from_dict(values)


def supervision_targets(episodes: tuple[EpisodeSpec, ...]) -> dict[str, np.ndarray]:
    """Training-only labels and masks; never called by the online actor."""
    if not episodes or any(not isinstance(episode, EpisodeSpec) for episode in episodes):
        raise ValueError("supervision requires nonempty typed training episodes")
    notes = np.zeros((len(episodes), NOTE_BINS), dtype=np.float32)
    waveforms = np.zeros(len(episodes), dtype=np.int64)
    scalars = np.zeros((len(episodes), len(SCALAR_FIELDS)), dtype=np.float32)
    eligible = np.zeros((len(episodes), len(PATCH_FIELDS)), dtype=np.float32)
    counts = np.zeros(len(episodes), dtype=np.float32)
    for row, episode in enumerate(episodes):
        target = episode.target_patch
        notes[row, np.asarray(target.notes_cents) - 4800] = 1.0
        counts[row] = episode.voice_count
        waveforms[row] = WAVEFORMS.index(target.waveform)
        scalars[row] = [normalize_scalar(name, getattr(target, name)) for name in SCALAR_FIELDS]
        eligible[row] = [float(name in episode.editable_fields) for name in PATCH_FIELDS]
        if target.sustain_db == 0:
            eligible[row, PATCH_FIELDS.index("decay_ms")] = 0.0
    return {
        "notes": notes,
        "waveforms": waveforms,
        "scalars": scalars,
        "eligible": eligible,
        "counts": counts,
    }
