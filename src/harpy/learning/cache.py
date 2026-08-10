"""Bounded, learning-only cache for immutable candidate spectra."""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from harpy.envs.spectrum import LOG_SPECTRUM_SIZE

_MIN_CANDIDATE_CENTS = 1_100
_MAX_CANDIDATE_CENTS = 10_900
_SPECTRUM_SHAPE = (LOG_SPECTRUM_SIZE,)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    spectrum: np.ndarray[Any, np.dtype[np.float32]]
    sha256: str


class SpectrumEvidenceCache:
    """Store one immutable spectrum for each reachable effective pitch coordinate."""

    def __init__(self) -> None:
        self._entries: dict[int, _CacheEntry] = {}

    def get_or_create(
        self,
        candidate_pitch_cents: object,
        *,
        on_miss: Callable[[], np.ndarray[Any, np.dtype[np.float32]]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return an owned immutable spectrum, rendering and committing it on a miss."""
        key = _candidate_key(candidate_pitch_cents)
        entry = self._entries.get(key)
        if entry is None:
            spectrum = _owned_spectrum(on_miss())
            entry = _CacheEntry(spectrum=spectrum, sha256=_spectrum_sha256(spectrum))
            self._entries[key] = entry
        elif entry.sha256 != _spectrum_sha256(entry.spectrum):
            raise RuntimeError("cache digest-integrity check failed")
        return _owned_spectrum(entry.spectrum)

    def __len__(self) -> int:
        """Return the number of populated reachable pitch coordinates."""
        return len(self._entries)

    @property
    def spectrum_bytes(self) -> int:
        """Return the exact byte size of spectra only, excluding mapping overhead."""
        return sum(entry.spectrum.nbytes for entry in self._entries.values())


def _candidate_key(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("candidate_pitch_cents must be an integer within 1100..10900")
    try:
        candidate_pitch_cents = operator.index(value)
    except TypeError as error:
        raise ValueError("candidate_pitch_cents must be an integer within 1100..10900") from error
    if not _MIN_CANDIDATE_CENTS <= candidate_pitch_cents <= _MAX_CANDIDATE_CENTS:
        raise ValueError("candidate_pitch_cents must be within 1100..10900")
    return candidate_pitch_cents


def _owned_spectrum(value: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    if not isinstance(value, np.ndarray):
        raise ValueError("spectrum must be a float32 ndarray")
    if value.dtype != np.dtype(np.float32):
        raise ValueError("spectrum must have dtype float32")
    if value.shape != _SPECTRUM_SHAPE:
        raise ValueError(f"spectrum must have shape {_SPECTRUM_SHAPE}")
    if not np.isfinite(value).all():
        raise ValueError("spectrum must contain only finite values")
    spectrum = np.array(value, dtype=np.float32, copy=True, order="C")
    spectrum.setflags(write=False)
    return spectrum


def _spectrum_sha256(spectrum: np.ndarray[Any, np.dtype[np.float32]]) -> str:
    digest = hashlib.sha256()
    digest.update(spectrum.dtype.str.encode("ascii"))
    digest.update(repr(spectrum.shape).encode("ascii"))
    digest.update(spectrum.tobytes(order="C"))
    return digest.hexdigest()
