"""Fixed-schedule, fresh-state synthesis for sound-match-v1.

Harmonic, voice, filter, envelope, and gain arithmetic use float64; only the
completed clip is converted to float32. Harmonics and canonical voices are summed
in ascending order. Higher sine harmonics use the float64 recurrence
sin((h+1)*p) = 2*cos(p)*sin(h*p) - sin((h-1)*p). Oscillators continue through
release; note-off changes only
the shared envelope. No state or rendered audio survives a call.

Envelope intervals are half-open: attack [0, A), decay [A, A+D), sustain
[A+D, O), release [O, O+R). Their first samples are respectively 0, 1, sustain,
and sustain. Release reaches zero at O+R, which starts the silent tail. Adjacent
ramps do not repeat an endpoint sample. A/D/R are exact integer millisecond frame
counts and O is NOTE_OFF_FRAME. Replay qualification is scoped to the recorded
CPU/Python/NumPy environment; these constants are not a cross-platform guarantee.
"""

from __future__ import annotations

import math

import numpy as np

from harpy.tasks.sound_match.patch import Patch

RENDERER_ID = "harpy-compact-synth-v1"
SAMPLE_RATE_HZ = 48_000
FRAME_COUNT = 96_000
NOTE_OFF_FRAME = 48_000
ACCUMULATOR_DTYPE = "float64"
OUTPUT_DTYPE = "float32"
ENVELOPE_CONVENTION = "half-open-linear-ramps-start-at-endpoint-v1"
HARMONIC_METHOD = "float64-sine-recurrence-ascending-v1"


def _voice(cents: int, waveform: str, frame_count: int) -> np.ndarray:
    frequency = 440.0 * 2.0 ** ((cents - 6900) / 1200.0)
    phase_increment = math.tau * frequency / SAMPLE_RATE_HZ
    phase = np.arange(frame_count, dtype=np.float64) * phase_increment
    if waveform == "sine":
        return np.sin(phase)
    maximum = math.ceil((SAMPLE_RATE_HZ / 2) / frequency) - 1
    result = np.zeros(frame_count, dtype=np.float64)
    previous = np.zeros(frame_count, dtype=np.float64)
    current = np.sin(phase)
    twice_cosine = 2.0 * np.cos(phase)
    scratch = np.empty(frame_count, dtype=np.float64)
    for harmonic in range(1, maximum + 1):
        # Keep the strict cutoff explicit even at a floating-point Nyquist boundary.
        if harmonic * frequency >= SAMPLE_RATE_HZ / 2:
            break
        if waveform == "saw":
            weight = (-1.0) ** (harmonic + 1) / harmonic
        elif harmonic % 2 == 0:
            weight = 0.0
        elif waveform == "triangle":
            weight = (-1.0) ** ((harmonic - 1) // 2) / harmonic**2
        else:
            weight = 1.0 / harmonic
        if weight:
            np.multiply(current, weight, out=scratch)
            result += scratch
        np.multiply(current, twice_cosine, out=scratch)
        scratch -= previous
        previous, current, scratch = current, scratch, previous
    factor = {"triangle": 8.0 / math.pi**2, "square": 4.0 / math.pi, "saw": 2.0 / math.pi}
    result *= factor[waveform]
    return result


def _filter_in_place(samples: np.ndarray, brightness_hz: int) -> None:
    coefficient = math.exp(-math.tau * brightness_hz / SAMPLE_RATE_HZ)
    input_gain = 1.0 - coefficient
    first = second = 0.0
    for index, sample in enumerate(samples):
        first = input_gain * float(sample) + coefficient * first
        second = input_gain * first + coefficient * second
        samples[index] = second


def _envelope(patch: Patch, frame_count: int) -> np.ndarray:
    attack = patch.attack_ms * (SAMPLE_RATE_HZ // 1000)
    decay = patch.decay_ms * (SAMPLE_RATE_HZ // 1000)
    release = patch.release_ms * (SAMPLE_RATE_HZ // 1000)
    sustain = 10.0 ** (patch.sustain_db / 20.0)
    result = np.full(frame_count, sustain, dtype=np.float64)
    result[:attack] = np.arange(attack, dtype=np.float64) / attack
    result[attack : attack + decay] = 1.0 + (sustain - 1.0) * (
        np.arange(decay, dtype=np.float64) / decay
    )
    result[NOTE_OFF_FRAME:] = sustain * (1.0 - np.arange(release, dtype=np.float64) / release)
    return result


def render(patch: Patch) -> np.ndarray:
    """Render an owned, complete two-second clip from a validated canonical patch."""
    if not isinstance(patch, Patch):
        raise ValueError("render requires a Patch")
    active_frames = NOTE_OFF_FRAME + patch.release_ms * (SAMPLE_RATE_HZ // 1000)
    samples = np.zeros(active_frames, dtype=np.float64)
    weight = 1.0 / len(patch.notes_cents)
    for cents in patch.notes_cents:
        samples += _voice(cents, patch.waveform, active_frames) * weight
    _filter_in_place(samples, patch.brightness_hz)
    samples *= _envelope(patch, active_frames)
    samples *= 0.5 * 10.0 ** (patch.gain_db / 20.0)
    if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
        raise RuntimeError("renderer produced nonfinite or out-of-range audio")
    result = np.zeros(FRAME_COUNT, dtype=np.float32)
    result[:active_frames] = samples.astype(np.float32)
    return result
