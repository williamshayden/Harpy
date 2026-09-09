"""Paired, static single-sine corruption with a byte-bounded evidence cache.

Noise depends only on its declared nuisance seed. It is shared across conditions,
actors, candidate pitches, and revisits; no mutable RNG is consumed by evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from harpy.envs.models import ObservationMode
from harpy.envs.sine_pitch import SinePitchEnv, _CandidateEvidence, _render_settled_sine
from harpy.envs.spectrum import encode_log_spectrum

ROBUSTNESS_RENDERER_ID = "harpy-static-single-sine-corruption-v1"
WAVEFORM_FRAMES = 262_144
DEFAULT_CACHE_BYTES = 128 * 1024 * 1024
_CONDITIONS = (
    ("clean", 0.0, 0.0, None),
    ("level-minus-12db", -12.0, 0.0, None),
    ("level-minus-24db", -24.0, 0.0, None),
    ("phase-45", 0.0, 45.0, None),
    ("phase-90", 0.0, 90.0, None),
    ("noise-30db", 0.0, 0.0, 30.0),
    ("noise-10db", 0.0, 0.0, 10.0),
    ("combined", -24.0, 90.0, 10.0),
)


def _integer(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer within its declared range")
    return value


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One immutable condition in the fixed robustness protocol."""

    id: str
    gain_db: float = 0.0
    phase_degrees: float = 0.0
    snr_db: float | None = None

    def __post_init__(self) -> None:
        for name in ("gain_db", "phase_degrees", "snr_db"):
            value = getattr(self, name)
            if value is None and name == "snr_db":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if (self.id, self.gain_db, self.phase_degrees, self.snr_db) not in _CONDITIONS:
            raise ValueError("condition must exactly match one of the eight declared conditions")

    def to_document(self) -> dict:
        return {
            "id": self.id,
            "gain_db": self.gain_db,
            "phase_degrees": self.phase_degrees,
            "snr_db": self.snr_db,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> ConditionSpec:
        if not isinstance(document, Mapping) or set(document) != {
            "id",
            "gain_db",
            "phase_degrees",
            "snr_db",
        }:
            raise ValueError("condition must contain exactly id, gain_db, phase_degrees, snr_db")
        return cls(**document)


ROBUSTNESS_CONDITIONS = tuple(ConditionSpec(*values) for values in _CONDITIONS)
CLEAN_CONDITION = ROBUSTNESS_CONDITIONS[0]


@dataclass(frozen=True, slots=True, eq=False)
class _Evidence:
    waveform: np.ndarray | None
    spectrum: np.ndarray
    dry_rms: float
    realized_snr_db: float | None
    waveform_sha256: str = ""
    spectrum_sha256: str = ""

    @property
    def nbytes(self) -> int:
        return (0 if self.waveform is None else self.waveform.nbytes) + self.spectrum.nbytes


class RobustnessEvidenceCache:
    """LRU accounting for arrays, records, keys, and allocated container storage."""

    def __init__(self, max_bytes: int = DEFAULT_CACHE_BYTES) -> None:
        self._max_bytes = _integer(max_bytes, "max_bytes", 0)
        self._entries: OrderedDict[tuple, _Evidence] = OrderedDict()
        self._empty_container_bytes = sys.getsizeof(self._entries)
        self._nbytes = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def retained_bytes(self) -> int:
        return self._nbytes + sys.getsizeof(self._entries) - self._empty_container_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._nbytes = 0

    def evidence(
        self,
        candidate_cents: int,
        condition: ConditionSpec,
        nuisance_seed: int,
        *,
        include_waveform: bool = True,
    ) -> _Evidence:
        """Return owned arrays, so callers cannot mutate retained evidence."""
        candidate = _integer(candidate_cents, "candidate_cents", 1100, 10900)
        seed = _integer(nuisance_seed, "nuisance_seed", 0)
        if not isinstance(condition, ConditionSpec):
            raise ValueError("condition must be a ConditionSpec")
        if type(include_waveform) is not bool:
            raise ValueError("include_waveform must be a bool")
        # Noiseless evidence is independent of the nuisance seed; sharing it also
        # preserves the exact legacy clean cache behavior across base episodes.
        key = (
            ROBUSTNESS_RENDERER_ID,
            candidate,
            condition,
            seed if condition.snr_db is not None else None,
        )
        value = self._entries.get(key)
        if value is None or (include_waveform and value.waveform is None):
            if value is not None:
                self._nbytes -= _entry_bytes(key, self._entries.pop(key))
            value = _render(candidate, condition, seed)
            if not include_waveform:
                value = _Evidence(
                    None,
                    value.spectrum,
                    value.dry_rms,
                    value.realized_snr_db,
                    value.waveform_sha256,
                    value.spectrum_sha256,
                )
            entry_bytes = _entry_bytes(key, value)
            if entry_bytes <= self.max_bytes:
                self._entries[key] = value
                self._nbytes += entry_bytes
                while self._entries and self.retained_bytes > self.max_bytes:
                    removed_key, removed = self._entries.popitem(last=False)
                    self._nbytes -= _entry_bytes(removed_key, removed)
                if not self._entries:
                    self._entries.clear()
        else:
            self._entries.move_to_end(key)
        waveform = value.waveform.copy() if include_waveform else None
        return _Evidence(
            waveform,
            value.spectrum.copy(),
            value.dry_rms,
            value.realized_snr_db,
            value.waveform_sha256,
            value.spectrum_sha256,
        )


def _entry_bytes(key: tuple, value: _Evidence) -> int:
    # NumPy owns these contiguous arrays, so getsizeof includes payload+header.
    # Shared scalar/condition objects are deliberately charged to each entry;
    # that makes this conservative when many entries refer to the same object.
    spec = key[2]
    return (
        sys.getsizeof(key)
        + sum(sys.getsizeof(part) for part in key)
        + sum(
            sys.getsizeof(part) for part in (spec.id, spec.gain_db, spec.phase_degrees, spec.snr_db)
        )
        + sys.getsizeof(value)
        + sum(
            sys.getsizeof(part)
            for part in (
                value.waveform,
                value.spectrum,
                value.dry_rms,
                value.realized_snr_db,
                value.waveform_sha256,
                value.spectrum_sha256,
            )
        )
    )


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _render(candidate_cents: int, condition: ConditionSpec, nuisance_seed: int) -> _Evidence:
    if condition == CLEAN_CONDITION:
        audio, spectrum = SinePitchEnv()._render_candidate(candidate_cents)
        return _Evidence(
            audio,
            spectrum,
            _rms(audio),
            None,
            hashlib.sha256(audio.tobytes()).hexdigest(),
            hashlib.sha256(spectrum.tobytes()).hexdigest(),
        )
    dry = _render_settled_sine(
        candidate_cents, phase_radians=math.radians(condition.phase_degrees)
    ).astype(np.float64)
    dry *= 10.0 ** (condition.gain_db / 20.0)
    dry_rms = _rms(dry)
    requested_snr = condition.snr_db
    if requested_snr is None:
        audio = dry.astype(np.float32)
        realized = None
    else:
        # Dedicated PCG64 state is recreated from semantic identity, not traversal.
        seed_bytes = hashlib.sha256(
            f"{ROBUSTNESS_RENDERER_ID}:white-noise:{nuisance_seed}".encode("ascii")
        ).digest()
        generator = np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes, "big")))
        noise = generator.standard_normal(WAVEFORM_FRAMES)
        noise -= noise.mean()
        noise *= dry_rms * 10.0 ** (-requested_snr / 20.0) / _rms(noise)
        audio = (dry + noise).astype(np.float32)
        realized = 20.0 * math.log10(dry_rms / _rms(audio.astype(np.float64) - dry))
    if (
        audio.shape != (WAVEFORM_FRAMES,)
        or audio.dtype != np.float32
        or not np.isfinite(audio).all()
    ):
        raise RuntimeError("robustness renderer must produce a finite fixed float32 waveform")
    spectrum = encode_log_spectrum(audio)
    audio.setflags(write=False)
    spectrum.setflags(write=False)
    return _Evidence(
        audio,
        spectrum,
        dry_rms,
        realized,
        hashlib.sha256(audio.tobytes()).hexdigest(),
        hashlib.sha256(spectrum.tobytes()).hexdigest(),
    )


class RobustnessSinePitchEnv(SinePitchEnv):
    """The existing task with static corruption confined to sensory evidence."""

    def __init__(
        self,
        condition: ConditionSpec,
        *,
        nuisance_seed: int,
        observation_mode: ObservationMode = ObservationMode.SPECTRUM,
        cache: RobustnessEvidenceCache | None = None,
    ) -> None:
        if not isinstance(condition, ConditionSpec):
            raise ValueError("condition must be a ConditionSpec")
        self.condition = condition
        self.nuisance_seed = _integer(nuisance_seed, "nuisance_seed", 0)
        if cache is not None and not isinstance(cache, RobustnessEvidenceCache):
            raise ValueError("cache must be a RobustnessEvidenceCache")
        self._evidence_cache = RobustnessEvidenceCache() if cache is None else cache
        self._metadata: dict | None = None
        super().__init__(observation_mode=observation_mode)

    @property
    def evidence_metadata(self) -> dict:
        """Evaluator-owned provenance, deliberately absent from observations/info."""
        if self._metadata is None:
            raise RuntimeError("evidence_metadata is available only after reset")
        return json.loads(json.dumps(self._metadata))

    def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
        evidence = self._evidence_cache.evidence(
            candidate_cents,
            self.condition,
            self.nuisance_seed,
            include_waveform=self.observation_mode is ObservationMode.WAVEFORM,
        )
        self._metadata = {
            "renderer_id": ROBUSTNESS_RENDERER_ID,
            "condition_id": self.condition.id,
            "nuisance_seed": self.nuisance_seed,
            "dry_rms": evidence.dry_rms,
            "requested_snr_db": self.condition.snr_db,
            "realized_snr_db": evidence.realized_snr_db,
            "waveform_sha256": evidence.waveform_sha256,
            "spectrum_sha256": evidence.spectrum_sha256,
        }
        return _CandidateEvidence(evidence.waveform, evidence.spectrum)

    def reset(self, *, seed=None, options=None):
        metadata = self._metadata
        try:
            return super().reset(seed=seed, options=options)
        except Exception:
            self._metadata = metadata
            raise

    def step(self, action):
        metadata = self._metadata
        try:
            return super().step(action)
        except Exception:
            self._metadata = metadata
            raise


def make_robustness_env(
    condition: ConditionSpec,
    *,
    nuisance_seed: int,
    observation_mode: ObservationMode = ObservationMode.SPECTRUM,
    cache: RobustnessEvidenceCache | None = None,
) -> RobustnessSinePitchEnv:
    return RobustnessSinePitchEnv(
        condition, nuisance_seed=nuisance_seed, observation_mode=observation_mode, cache=cache
    )
